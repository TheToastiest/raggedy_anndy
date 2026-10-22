//! Durability for the live write path: op log on disk, periodic
//! snapshots, recovery. Stage 4 of docs/live-writes-plan.md.
//!
//! One directory holds:
//! - `log-{start_seq:020}.log`: op log segments (format in `oplog`). A new
//!   segment starts at every snapshot; its header carries the fingerprint
//!   of the index at `start_seq`, so every segment boundary is checked.
//! - `snap-{seq:020}.snap`: whole-index snapshots (format in `snapshot`).
//!
//! Writer thread, per tick: `apply_logged` each op (or `live::apply`, then
//! `append`), then `end_tick`. `end_tick` hands the tick's records to the
//! OS (one `write`) and fsyncs once `sync_every` has passed since the last
//! fsync, so a process crash loses at most the unflushed tick and a power
//! loss at most `sync_every` of ops. `snapshot` every so often.
//!
//! Recovery (`recover`) is read-only: newest snapshot that verifies, then
//! the log from its seq; if no snapshot verifies, the whole log on the
//! caller's base model (the index the log was created from, e.g. a fresh
//! `build` with the same inputs). It stops at the first torn, corrupt or
//! out-of-order record. `resume` then cuts that tail off and opens the
//! log for appends again.
//!
//! Sequence numbers: op 1 is the first op logged; "state at seq S" is the
//! index after ops 1..=S. The index passed to `create` is state 0 (or the
//! `seq` given).

use std::fs::{self, File, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::ivfpq::IvfPqIndex;
use crate::live::{self, Applied, Op, Scratch, WriteError};
use crate::oplog::{self, SEGMENT_HEADER_LEN};
use crate::snapshot::{self, sync_dir};

#[derive(Clone, Copy, Debug)]
pub struct DurableParams {
    /// fsync the log at most this often (1 s agreed: up to 1 s of ops may
    /// be lost on power loss).
    pub sync_every: Duration,
    /// Snapshots kept on disk, newest first; at least 1. The log is kept
    /// from the oldest kept snapshot on, so with 2 a corrupt newest
    /// snapshot still recovers from the one before it. Until `keep`
    /// snapshots exist the whole log is kept.
    pub keep_snapshots: usize,
}

pub fn default_params() -> DurableParams { DurableParams { sync_every: Duration::from_secs(1), keep_snapshots: 2 } }

/// The writer's open log. Seq counters: `last_seq` appended (in memory),
/// `written_seq` handed to the OS, `synced_seq` fsynced.
#[derive(Debug)]
pub struct LogWriter {
    pub dir: PathBuf,
    pub params: DurableParams,
    pub seg_path: PathBuf,
    pub seg_start: u64,
    pub last_seq: u64,
    pub last_tick: u64,
    pub written_seq: u64,
    pub written_len: u64,
    pub synced_seq: u64,
    pub synced_len: u64,
    pub last_sync: Instant,
    file: File,
    buf: Vec<u8>,
}

/// What `recover` found.
#[derive(Debug)]
pub struct Recovered {
    pub idx: IvfPqIndex,
    /// Last op applied (state at `seq`).
    pub seq: u64,
    /// Tick of that op (or of the snapshot, if no op was replayed).
    pub tick: u64,
    /// Seq of the snapshot used, or `None` for a full replay on the base.
    pub snapshot_seq: Option<u64>,
    pub replayed: u64,
    /// Bytes of log past the last good record: torn or corrupt tail plus
    /// segments that no longer chain on.
    pub dropped_bytes: u64,
    /// Snapshots newer than the one used that failed to verify.
    pub bad_snapshots: Vec<PathBuf>,
    /// Segment the last good record is in and its valid length; `None`
    /// if no segment starts at the recovered snapshot.
    pub tail: Option<(u64, PathBuf, u64)>,
    /// Segments after the stop point: unreachable, set aside by `resume`.
    pub unreachable: Vec<PathBuf>,
}

#[derive(Debug)]
pub enum RecoverError {
    Io(io::Error),
    /// No snapshot verifies and the log cannot be replayed on the base
    /// (no base given, its fingerprint does not match the log, or the log
    /// no longer starts at seq 0). Rebuild from the source data.
    NoState,
    /// A logged op was refused on replay: the starting state is not the
    /// one the log was written against.
    Diverged { seq: u64, err: WriteError },
}

impl std::fmt::Display for RecoverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{self:?}") }
}
impl std::error::Error for RecoverError {}
impl From<io::Error> for RecoverError { fn from(e: io::Error) -> Self { RecoverError::Io(e) } }

pub fn log_path(dir: &Path, start_seq: u64) -> PathBuf { dir.join(format!("log-{start_seq:020}.log")) }
pub fn snap_path(dir: &Path, seq: u64) -> PathBuf { dir.join(format!("snap-{seq:020}.snap")) }

/// `(seq, path)` of the files named `{prefix}{20 digits}.{ext}`, ascending.
fn list_files(dir: &Path, prefix: &str, ext: &str) -> io::Result<Vec<(u64, PathBuf)>> {
    let mut out = Vec::new();
    for e in fs::read_dir(dir)? {
        let p = e?.path();
        let Some(name) = p.file_name().and_then(|n| n.to_str()) else { continue };
        let Some(digits) = name.strip_prefix(prefix).and_then(|r| r.strip_suffix(ext)).and_then(|r| r.strip_suffix('.')) else { continue };
        if digits.len() != 20 || !digits.bytes().all(|b| b.is_ascii_digit()) { continue; }
        if let Ok(seq) = digits.parse() { out.push((seq, p)); }
    }
    out.sort();
    Ok(out)
}

fn new_segment(dir: &Path, start_seq: u64, start_fp: u64) -> io::Result<(File, PathBuf)> {
    let path = log_path(dir, start_seq);
    let mut f = OpenOptions::new().write(true).create_new(true).open(&path)?;
    f.write_all(&oplog::encode_segment_header(start_seq, start_fp))?;
    f.sync_all()?;
    sync_dir(dir)?;
    Ok((f, path))
}

fn writer(dir: &Path, params: DurableParams, file: File, seg_path: PathBuf, seg_start: u64, seq: u64, tick: u64, len: u64, now: Instant) -> LogWriter {
    LogWriter {
        dir: dir.to_path_buf(), params, seg_path, seg_start,
        last_seq: seq, last_tick: tick, written_seq: seq, written_len: len, synced_seq: seq, synced_len: len,
        last_sync: now, file, buf: Vec::with_capacity(64 * 1024),
    }
}

/// Start a new log in `dir` for `idx`, which is the state at `seq`
/// (0 for a freshly built index). Refuses a directory that already holds
/// a log or snapshot: use `recover` + `resume` there.
pub fn create(dir: &Path, idx: &IvfPqIndex, seq: u64, params: DurableParams, now: Instant) -> io::Result<LogWriter> {
    fs::create_dir_all(dir)?;
    if !list_files(dir, "log-", "log")?.is_empty() || !list_files(dir, "snap-", "snap")?.is_empty() {
        return Err(io::Error::new(io::ErrorKind::AlreadyExists, format!("{} already holds a log", dir.display())));
    }
    let (file, path) = new_segment(dir, seq, idx.fingerprint())?;
    Ok(writer(dir, params, file, path, seq, seq, 0, SEGMENT_HEADER_LEN as u64, now))
}

/// Add one op to the log buffer as the next seq. No syscall; it reaches
/// the OS at `end_tick`. Returns the op's seq.
pub fn append(w: &mut LogWriter, tick: u64, op: &Op) -> u64 {
    w.last_seq += 1;
    w.last_tick = tick;
    oplog::encode_record(&mut w.buf, w.last_seq, tick, op);
    w.last_seq
}

/// `live::apply`, and log the op if it was accepted. Refused ops change
/// nothing and are not logged, so every logged op succeeds on replay.
pub fn apply_logged(w: &mut LogWriter, idx: &mut IvfPqIndex, s: &mut Scratch, tick: u64, op: &Op) -> Result<Applied, WriteError> {
    let r = live::apply(idx, s, op)?;
    append(w, tick, op);
    Ok(r)
}

fn flush(w: &mut LogWriter) -> io::Result<()> {
    if w.buf.is_empty() { return Ok(()); }
    w.file.write_all(&w.buf)?;
    w.written_len += w.buf.len() as u64;
    w.written_seq = w.last_seq;
    w.buf.clear();
    Ok(())
}

fn fsync(w: &mut LogWriter, now: Instant) -> io::Result<bool> {
    w.last_sync = now;
    if w.synced_len == w.written_len { return Ok(false); }
    w.file.sync_data()?;
    w.synced_len = w.written_len;
    w.synced_seq = w.written_seq;
    Ok(true)
}

/// End of tick: write the tick's records, and fsync if `sync_every` has
/// passed since the last fsync. Returns whether it fsynced. An error here
/// means the log may hold a partial record; stop writing and `recover`.
pub fn end_tick(w: &mut LogWriter, now: Instant) -> io::Result<bool> {
    flush(w)?;
    if now.saturating_duration_since(w.last_sync) < w.params.sync_every { return Ok(false); }
    fsync(w, now)
}

/// Write and fsync everything appended so far.
pub fn sync(w: &mut LogWriter, now: Instant) -> io::Result<()> {
    flush(w)?;
    fsync(w, now).map(|_| ())
}

/// Snapshot `idx`, which must be the state at `w.last_seq` (true when every
/// write went through `apply_logged`), start a new log segment there, and
/// prune snapshots and segments past `keep_snapshots`. Blocks for the
/// whole write; returns the snapshot path.
pub fn snapshot(w: &mut LogWriter, idx: &IvfPqIndex, now: Instant) -> io::Result<PathBuf> {
    sync(w, now)?;
    let seq = w.last_seq;
    let path = snap_path(&w.dir, seq);
    let fp = snapshot::write(&path, idx, seq, w.last_tick)?;
    if w.seg_start != seq {
        let (file, seg_path) = new_segment(&w.dir, seq, fp)?;
        w.file = file;
        w.seg_path = seg_path;
        w.seg_start = seq;
        w.written_len = SEGMENT_HEADER_LEN as u64;
        w.synced_len = w.written_len;
    }
    prune(&w.dir, w.params.keep_snapshots)?;
    Ok(path)
}

/// Keep the newest `keep` snapshots and the log from the oldest of them on.
fn prune(dir: &Path, keep: usize) -> io::Result<()> {
    let keep = keep.max(1);
    let snaps = list_files(dir, "snap-", "snap")?;
    if snaps.len() < keep { return Ok(()); }
    let floor = snaps[snaps.len() - keep].0;
    for (seq, p) in snaps.iter().chain(list_files(dir, "log-", "log")?.iter()) {
        if *seq < floor { fs::remove_file(p)?; }
    }
    sync_dir(dir)
}

/// Replay the segments starting at or after `seq` onto `idx` (the state at
/// `seq`). Each segment must start where the previous one stopped, with
/// the matching fingerprint; the first bad record or broken link ends the
/// replay.
fn replay(logs: &[(u64, PathBuf)], mut idx: IvfPqIndex, mut seq: u64, mut tick: u64, snapshot_seq: Option<u64>, bad_snapshots: Vec<PathBuf>) -> Result<Recovered, RecoverError> {
    let mut s = live::scratch_for(&idx);
    let (mut replayed, mut dropped) = (0u64, 0u64);
    let mut tail = None;
    let mut unreachable = Vec::new();
    let mut broken = false;
    let from = seq;
    for (start, path) in logs.iter().filter(|(st, _)| *st >= from) {
        if !broken {
            let bytes = fs::read(path)?;
            if *start == seq && oplog::decode_segment_header(&bytes) == Some((*start, idx.fingerprint())) {
                let mut at = SEGMENT_HEADER_LEN;
                while let Some((rec, used)) = oplog::decode_record(&bytes[at..]) {
                    if rec.seq != seq + 1 { break; }
                    live::apply(&mut idx, &mut s, &rec.op).map_err(|err| RecoverError::Diverged { seq: rec.seq, err })?;
                    seq = rec.seq;
                    tick = rec.tick;
                    replayed += 1;
                    at += used;
                }
                tail = Some((*start, path.clone(), at as u64));
                // Nothing after a bad record can link on.
                broken = at < bytes.len();
                dropped += (bytes.len() - at) as u64;
                continue;
            }
            broken = true;
        }
        dropped += fs::metadata(path)?.len();
        unreachable.push(path.clone());
    }
    Ok(Recovered { idx, seq, tick, snapshot_seq, replayed, dropped_bytes: dropped, bad_snapshots, tail, unreachable })
}

/// Rebuild the index from `dir`. Read-only. `base` is the index the log
/// was created from (state 0), used only if no snapshot verifies.
pub fn recover(dir: &Path, base: Option<&IvfPqIndex>) -> Result<Recovered, RecoverError> {
    let snaps = list_files(dir, "snap-", "snap")?;
    let logs = list_files(dir, "log-", "log")?;
    let mut bad = Vec::new();
    for (_, path) in snaps.iter().rev() {
        match snapshot::read(path) {
            Ok(sn) => return replay(&logs, sn.idx, sn.seq, sn.tick, Some(sn.seq), bad),
            Err(_) => bad.push(path.clone()),
        }
    }
    let base = base.ok_or(RecoverError::NoState)?;
    let first = logs.first().ok_or(RecoverError::NoState)?;
    let head = fs::read(&first.1)?;
    let head = &head[..head.len().min(SEGMENT_HEADER_LEN)];
    if first.0 != 0 || oplog::decode_segment_header(head) != Some((0, base.fingerprint())) {
        return Err(RecoverError::NoState);
    }
    replay(&logs, base.clone(), 0, 0, None, bad)
}

/// Rename `p` to `p.corrupt`, keeping it for inspection.
fn set_aside(p: &Path) -> io::Result<()> {
    let mut s = p.as_os_str().to_owned();
    s.push(".corrupt");
    fs::rename(p, PathBuf::from(s))
}

/// Open the log for appends after `recover`: set bad snapshots and
/// unreachable segments aside (`.corrupt`), remove stale `.tmp` files,
/// cut the torn tail off, and continue at `rec.seq`.
pub fn resume(dir: &Path, rec: &Recovered, params: DurableParams, now: Instant) -> io::Result<LogWriter> {
    for p in rec.bad_snapshots.iter().chain(&rec.unreachable) { set_aside(p)?; }
    for e in fs::read_dir(dir)? {
        let p = e?.path();
        if p.extension().map_or(false, |x| x == "tmp") { fs::remove_file(&p)?; }
    }
    let w = match &rec.tail {
        Some((start, path, len)) => {
            let mut f = OpenOptions::new().write(true).open(path)?;
            f.set_len(*len)?;
            f.seek(SeekFrom::End(0))?;
            f.sync_all()?;
            writer(dir, params, f, path.clone(), *start, rec.seq, rec.tick, *len, now)
        }
        None => {
            let (f, path) = new_segment(dir, rec.seq, rec.idx.fingerprint())?;
            writer(dir, params, f, path, rec.seq, rec.seq, rec.tick, SEGMENT_HEADER_LEN as u64, now)
        }
    };
    sync_dir(dir)?;
    Ok(w)
}
