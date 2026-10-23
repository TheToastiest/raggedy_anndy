//! Durability for the live system: op log on disk, background snapshots,
//! recovery. Stage 4 of docs/live-writes-plan.md.
//!
//! What is durable is a `system::System`: the stage 2 writer's rows, the
//! stage 3 epoch state and the stage 5 buffer and outbox. One directory holds:
//! - `log-{start_tick:020}.log`: op log segments (format in `oplog`). A new
//!   segment starts at every snapshot; each header names the entry and tick
//!   it follows and the log's id, so every segment boundary is checked.
//! - `snap-{tick:020}.snap`: whole-system snapshots (format in `snapshot`).
//!
//! Writer thread, per server tick: `write` each op (and `swap` / `reject`
//! when a rebuild decides), then `end_tick`. `end_tick` applies and logs
//! the compactions the policy says are due, closes the tick with an
//! EndTick record, hands the tick's records to the OS in one `write`,
//! fsyncs once `sync_every` has passed since the last fsync, publishes the
//! tick's view, and every `snapshot_every_ticks` ticks starts a snapshot.
//! A process crash loses at most the tick being written; a power loss at
//! most `sync_every` of ticks. Recovery is per whole tick: a tick whose
//! EndTick is missing or wrong is dropped entirely.
//!
//! Snapshots run beside the writer. On the writer thread `start_snapshot`
//! only takes a `system::Capture` (an `Arc` of the view just published,
//! plus the non-row state: O(lists + pending + outbox)) and opens the next
//! log segment; a background thread fsyncs the old segment, writes the
//! snapshot from the view's blocks (which never move), and prunes. The
//! view stays pinned until it is done. Snapshots are only taken with no
//! rebuild pending (see `system::capture`) and only after a publish that
//! went through. The cadence is counted in ticks, never read from a clock.
//!
//! Recovery (`recover`) is read-only: newest snapshot that verifies, then
//! the log from its tick; if no snapshot verifies, the whole log on the
//! caller's base index (the one the log was created from, e.g. a fresh
//! `build` with the same inputs). Model decisions in the log are replayed
//! by rebuilding (`system::replay_decision`). It stops at the first torn,
//! corrupt or out-of-order record or tick. `resume` then cuts that tail off
//! and opens the log for appends again.
//!
//! Sequence numbers: entry 1 is the first entry logged; "state at seq S,
//! tick T" is the system after entries 1..=S, ticks through T, and the
//! publish of tick T. The system passed to `create` is state 0 (or its own
//! `st.seq` and tick).

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::gate::Gate;
use crate::ivfpq::IvfPqIndex;
use crate::live::Op;
use crate::oplog::{self, Body, SEGMENT_HEADER_LEN};
use crate::real_memory::{RealApplied, RealError, RealOp};
use crate::rebuild::{self, LogOp, RebuildError, Rebuilt};
use crate::snapshot::{self, sync_dir};
use crate::system::{self, ReplayError, System};
use crate::views::{self, CompactPolicy, Published};

/// One minute at the 60 Hz server tick (Brinn, 2026-10-23: 60 s saves).
/// At another tick rate pass `60 * rate`.
pub const SNAPSHOT_EVERY_TICKS: u64 = 60 * 60;

#[derive(Clone, Copy, Debug)]
pub struct DurableParams {
    /// fsync the log at most this often (1 s agreed: up to 1 s of ops may
    /// be lost on power loss). Measured on the `now` the caller passes.
    pub sync_every: Duration,
    /// Snapshots kept on disk, newest first; at least 1. The log is kept
    /// from the oldest kept snapshot on, so with 2 a corrupt newest
    /// snapshot still recovers from the one before it. Until `keep`
    /// snapshots exist the whole log is kept.
    pub keep_snapshots: usize,
    /// Start a snapshot this many ticks after the last one started (or
    /// after `create` / `resume`). Server ticks, counted by `end_tick`;
    /// 0 = never on its own (`start_snapshot` still works).
    pub snapshot_every_ticks: u64,
}

pub fn default_params() -> DurableParams {
    DurableParams { sync_every: Duration::from_secs(1), keep_snapshots: 2, snapshot_every_ticks: SNAPSHOT_EVERY_TICKS }
}

/// A finished snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnapDone {
    pub path: PathBuf,
    pub seq: u64,
    pub tick: u64,
    pub fp: u64,
    pub bytes: u64,
}

/// A snapshot being written on its own thread.
#[derive(Debug)]
pub struct SnapJob {
    pub seq: u64,
    pub tick: u64,
    handle: JoinHandle<io::Result<SnapDone>>,
}

/// The writer's open log. Seq counters: `last_seq` appended (in memory),
/// `written_seq` handed to the OS, `synced_seq` fsynced (in the current
/// segment; the background snapshot fsyncs the one before).
#[derive(Debug)]
pub struct LogWriter {
    pub dir: PathBuf,
    pub params: DurableParams,
    pub log_id: u64,
    pub seg_path: PathBuf,
    /// (seq, tick) the current segment starts after.
    pub seg_start: (u64, u64),
    pub last_seq: u64,
    /// Last tick closed with an EndTick.
    pub last_tick: u64,
    pub written_seq: u64,
    pub written_len: u64,
    pub synced_seq: u64,
    pub synced_len: u64,
    pub last_sync: Instant,
    /// Tick the last snapshot was started at.
    pub last_snapshot_tick: u64,
    pub job: Option<SnapJob>,
    /// Entries appended in the open tick, and the CRC of their records.
    tick_count: u32,
    tick_crc: u32,
    file: File,
    buf: Vec<u8>,
    compact: Vec<Op>,
}

/// What `end_tick` did.
#[derive(Debug)]
pub struct TickEnd {
    pub tick: u64,
    pub published: Published,
    pub synced: bool,
    /// A snapshot of this tick was started.
    pub snapshot_started: bool,
    /// A snapshot finished since the last tick (or failed: the log is
    /// intact, recovery uses an older snapshot).
    pub snapshot_done: Option<io::Result<SnapDone>>,
}

/// What `recover` found.
pub struct Recovered {
    pub sys: System,
    pub log_id: u64,
    /// Last entry applied (state at `seq`).
    pub seq: u64,
    /// Last whole tick applied.
    pub tick: u64,
    /// Tick of the snapshot used, or `None` for a full replay on the base.
    pub snapshot_tick: Option<u64>,
    /// Entries and ticks replayed from the log.
    pub replayed: u64,
    pub ticks: u64,
    /// Model decisions replayed (each one a rebuild).
    pub decisions: u64,
    /// Bytes of log past the last whole tick: torn or corrupt tail, a
    /// partial tick, plus segments that no longer chain on.
    pub dropped_bytes: u64,
    /// Snapshots newer than the one used that failed to verify.
    pub bad_snapshots: Vec<PathBuf>,
    /// Segment the last whole tick is in (its start tick, path, valid
    /// length); `None` if no segment starts at the recovered snapshot.
    pub tail: Option<(u64, PathBuf, u64)>,
    /// Segments after the stop point: unreachable, set aside by `resume`.
    pub unreachable: Vec<PathBuf>,
}

#[derive(Debug)]
pub enum RecoverError {
    Io(io::Error),
    /// No snapshot verifies and the log cannot be replayed on the base
    /// (no base given, its fingerprint does not match the log, or the log
    /// no longer starts at the beginning). Rebuild from the source data.
    NoState,
    /// A logged entry was refused on replay: the starting state is not the
    /// one the log was written against.
    Diverged { seq: u64, err: ReplayError },
}

impl std::fmt::Display for RecoverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{self:?}") }
}
impl std::error::Error for RecoverError {}
impl From<io::Error> for RecoverError { fn from(e: io::Error) -> Self { RecoverError::Io(e) } }

pub fn log_path(dir: &Path, start_tick: u64) -> PathBuf { dir.join(format!("log-{start_tick:020}.log")) }
pub fn snap_path(dir: &Path, tick: u64) -> PathBuf { dir.join(format!("snap-{tick:020}.snap")) }

/// `(tick, path)` of the files named `{prefix}{20 digits}.{ext}`, ascending.
fn list_files(dir: &Path, prefix: &str, ext: &str) -> io::Result<Vec<(u64, PathBuf)>> {
    let mut out = Vec::new();
    for e in fs::read_dir(dir)? {
        let p = e?.path();
        let Some(name) = p.file_name().and_then(|n| n.to_str()) else { continue };
        let Some(digits) = name.strip_prefix(prefix).and_then(|r| r.strip_suffix(ext)).and_then(|r| r.strip_suffix('.')) else { continue };
        if digits.len() != 20 || !digits.bytes().all(|b| b.is_ascii_digit()) { continue; }
        if let Ok(t) = digits.parse() { out.push((t, p)); }
    }
    out.sort();
    Ok(out)
}

/// Create a segment and write its header. `durable_now` also fsyncs it and
/// the directory (create, resume); a snapshot's new segment leaves that to
/// the next fsync and the snapshot thread's directory fsync.
fn new_segment(dir: &Path, start_seq: u64, start_tick: u64, log_id: u64, durable_now: bool) -> io::Result<(File, PathBuf)> {
    let path = log_path(dir, start_tick);
    let mut f = OpenOptions::new().write(true).create_new(true).open(&path)?;
    f.write_all(&oplog::encode_segment_header(start_seq, start_tick, log_id))?;
    if durable_now {
        f.sync_all()?;
        sync_dir(dir)?;
    }
    Ok((f, path))
}

#[allow(clippy::too_many_arguments)]
fn writer(dir: &Path, params: DurableParams, log_id: u64, file: File, seg_path: PathBuf, seg_start: (u64, u64), seq: u64, tick: u64, len: u64, now: Instant) -> LogWriter {
    LogWriter {
        dir: dir.to_path_buf(), params, log_id, seg_path, seg_start,
        last_seq: seq, last_tick: tick, written_seq: seq, written_len: len, synced_seq: seq, synced_len: len,
        last_sync: now, last_snapshot_tick: tick, job: None, tick_count: 0, tick_crc: 0,
        file, buf: Vec::with_capacity(64 * 1024), compact: Vec::new(),
    }
}

/// Start a new log in `dir` for `sys` (state 0 for a fresh `system::start`).
/// Its `system::fingerprint` becomes the log id. Refuses a directory that
/// already holds a log or snapshot: use `recover` + `resume` there.
pub fn create(dir: &Path, sys: &System, params: DurableParams, now: Instant) -> io::Result<LogWriter> {
    fs::create_dir_all(dir)?;
    if !list_files(dir, "log-", "log")?.is_empty() || !list_files(dir, "snap-", "snap")?.is_empty() {
        return Err(io::Error::new(io::ErrorKind::AlreadyExists, format!("{} already holds a log", dir.display())));
    }
    let log_id = system::fingerprint(sys);
    let (seq, tick) = (sys.st.seq, sys.w.seq - 1);
    let (file, path) = new_segment(dir, seq, tick, log_id, true)?;
    Ok(writer(dir, params, log_id, file, path, (seq, tick), seq, tick, SEGMENT_HEADER_LEN as u64, now))
}

fn append_record(w: &mut LogWriter, enc: impl FnOnce(&mut Vec<u8>)) {
    let at = w.buf.len();
    enc(&mut w.buf);
    w.tick_crc = oplog::crc32_update(w.tick_crc, &w.buf[at..]);
    w.tick_count += 1;
}

/// `system::write`, and log the op if it was accepted. Refused ops change
/// nothing and are not logged, so every logged op succeeds on replay. No
/// syscall; the record reaches the OS at `end_tick`.
pub fn write(w: &mut LogWriter, sys: &mut System, op: &RealOp) -> Result<RealApplied, RealError> {
    let r = system::write(sys, op)?;
    w.last_seq += 1;
    debug_assert_eq!(w.last_seq, sys.st.seq);
    let (seq, tick) = (w.last_seq, sys.w.seq);
    append_record(w, |b| oplog::encode_write(b, seq, tick, op));
    Ok(r)
}

fn log_model(w: &mut LogWriter, sys: &System, e: &LogOp) {
    w.last_seq += 1;
    debug_assert_eq!(w.last_seq, sys.st.seq);
    let (seq, tick) = (w.last_seq, sys.w.seq);
    append_record(w, |b| oplog::encode_record(b, seq, tick, &Body::Model(e.clone())));
}

/// `system::swap`, and log the `SwapModel` entry.
pub fn swap(w: &mut LogWriter, sys: &mut System, r: Rebuilt, gate: Gate) -> Result<LogOp, RebuildError> {
    let e = system::swap(sys, r, gate)?;
    log_model(w, sys, &e);
    Ok(e)
}

/// `system::reject`, and log the `RejectModel` entry.
pub fn reject(w: &mut LogWriter, sys: &mut System, r: &Rebuilt, gate: Gate) -> Result<LogOp, RebuildError> {
    let e = system::reject(sys, r, gate)?;
    log_model(w, sys, &e);
    Ok(e)
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

/// Collect a finished snapshot job, if any. Never waits.
fn poll_snapshot(w: &mut LogWriter) -> Option<io::Result<SnapDone>> {
    if !w.job.as_ref()?.handle.is_finished() { return None; }
    wait_snapshot(w)
}

/// Wait for the snapshot in flight, if any, and return how it went.
pub fn wait_snapshot(w: &mut LogWriter) -> Option<io::Result<SnapDone>> {
    let job = w.job.take()?;
    Some(job.handle.join().unwrap_or_else(|_| Err(io::Error::other("snapshot thread panicked"))))
}

/// End of tick `sys.w.seq`: apply and log the compactions `policy` says
/// are due, close the tick (EndTick), write its records, fsync if
/// `sync_every` has passed since the last fsync, publish the tick's view,
/// collect a finished snapshot and start the next one if it is due. An
/// error here means the log may hold a partial tick; stop writing and
/// `recover`.
pub fn end_tick(w: &mut LogWriter, sys: &mut System, policy: &CompactPolicy, now: Instant) -> io::Result<TickEnd> {
    views::compactions_due(policy, &sys.w, &mut w.compact);
    let mut ops = std::mem::take(&mut w.compact);
    for op in ops.drain(..) {
        write(w, sys, &RealOp::Live(op)).expect("compaction of an existing list");
    }
    w.compact = ops;
    let tick = sys.w.seq;
    let (seq, count, crc) = (w.last_seq, w.tick_count, w.tick_crc);
    oplog::encode_record(&mut w.buf, seq, tick, &Body::EndTick { count, crc });
    w.tick_count = 0;
    w.tick_crc = 0;
    w.last_tick = tick;
    flush(w)?;
    let synced = if now.saturating_duration_since(w.last_sync) >= w.params.sync_every { fsync(w, now)? } else { false };
    let published = views::publish(&mut sys.w);
    let snapshot_done = poll_snapshot(w);
    let every = w.params.snapshot_every_ticks;
    let snapshot_started = every > 0 && tick - w.last_snapshot_tick >= every && start_snapshot(w, sys)?;
    Ok(TickEnd { tick, published, synced, snapshot_started, snapshot_done })
}

/// Write and fsync everything appended so far. Records of a tick not yet
/// closed reach the disk too, but count only once `end_tick` closes it.
pub fn sync(w: &mut LogWriter, now: Instant) -> io::Result<()> {
    flush(w)?;
    fsync(w, now).map(|_| ())
}

/// Start a snapshot of the tick that just ended, if one can be taken now:
/// no snapshot in flight, no rebuild pending, and the current view is the
/// writer's state (call between `end_tick` and the next op). Writer side:
/// the capture and a new log segment (create + header, no fsync). The
/// rest runs on a new thread (`w.job`): fsync the old segment, write the
/// snapshot, prune. Returns whether it started.
pub fn start_snapshot(w: &mut LogWriter, sys: &System) -> io::Result<bool> {
    if w.job.is_some() || w.tick_count != 0 { return Ok(false); }
    let Some(cap) = system::capture(sys) else { return Ok(false) };
    let (seq, tick) = (w.last_seq, w.last_tick);
    debug_assert_eq!((seq, tick + 1), (cap.st.seq, cap.tick_next));
    flush(w)?;
    let (file, seg_path) = new_segment(&w.dir, seq, tick, w.log_id, false)?;
    let old = std::mem::replace(&mut w.file, file);
    w.seg_path = seg_path;
    w.seg_start = (seq, tick);
    w.written_len = SEGMENT_HEADER_LEN as u64;
    w.synced_len = 0; // the new header is not fsynced yet
    w.last_snapshot_tick = tick;
    let (dir, log_id, keep) = (w.dir.clone(), w.log_id, w.params.keep_snapshots);
    let handle = std::thread::Builder::new().name("raggedy-snapshot".into()).spawn(move || {
        old.sync_data()?;
        drop(old);
        let path = snap_path(&dir, tick);
        let out = snapshot::write(&path, &cap, seq, tick, log_id)?;
        drop(cap); // unpin the view
        prune(&dir, keep)?;
        Ok(SnapDone { path, seq, tick, fp: out.fp, bytes: out.bytes })
    })?;
    w.job = Some(SnapJob { seq, tick, handle });
    Ok(true)
}

/// Keep the newest `keep` snapshots and the log from the oldest of them on.
fn prune(dir: &Path, keep: usize) -> io::Result<()> {
    let keep = keep.max(1);
    let snaps = list_files(dir, "snap-", "snap")?;
    if snaps.len() < keep { return Ok(()); }
    let floor = snaps[snaps.len() - keep].0;
    for (t, p) in snaps.iter().chain(list_files(dir, "log-", "log")?.iter()) {
        if *t < floor { fs::remove_file(p)?; }
    }
    sync_dir(dir)
}

/// The whole ticks of the segments that chain on from (seq, tick): each
/// must start where the previous one stopped, with this log's id; the
/// first bad record, partial tick or broken link ends the chain.
struct Chain { ticks: Vec<oplog::Tick>, tail: Option<(u64, PathBuf, u64)>, unreachable: Vec<PathBuf>, dropped: u64 }

fn read_chain(logs: &[(u64, PathBuf)], log_id: u64, mut seq: u64, mut tick: u64) -> io::Result<Chain> {
    let mut c = Chain { ticks: Vec::new(), tail: None, unreachable: Vec::new(), dropped: 0 };
    let mut broken = false;
    let from = tick;
    for (start, path) in logs.iter().filter(|(st, _)| *st >= from) {
        if !broken {
            let bytes = fs::read(path)?;
            if *start == tick && oplog::decode_segment_header(&bytes) == Some((seq, tick, log_id)) {
                let (ticks, valid) = oplog::decode_ticks(&bytes[SEGMENT_HEADER_LEN..], seq, tick);
                for t in &ticks {
                    tick = t.tick;
                    if let Some(r) = t.entries.last() { seq = r.seq; }
                }
                let end = SEGMENT_HEADER_LEN + valid;
                c.tail = Some((*start, path.clone(), end as u64));
                c.ticks.extend(ticks);
                // Nothing after a bad record can link on.
                broken = end < bytes.len();
                c.dropped += (bytes.len() - end) as u64;
                continue;
            }
            broken = true;
        }
        c.dropped += fs::metadata(path)?.len();
        c.unreachable.push(path.clone());
    }
    Ok(c)
}

/// Replay `chain` onto `sys` (the state at `seq`, tick `tick`), tick by
/// tick, publishing after each. A decision entry rebuilds from the rows at
/// its `snapshot_seq`, which `begin_rebuild` takes when replay gets there.
#[allow(clippy::too_many_arguments)]
fn replay(mut sys: System, chain: Chain, log_id: u64, snapshot_tick: Option<u64>, bad_snapshots: Vec<PathBuf>, threads: usize) -> Result<Recovered, RecoverError> {
    let mut wanted: Vec<u64> = Vec::new();
    for r in chain.ticks.iter().flat_map(|t| &t.entries) {
        if let Body::Model(LogOp::SwapModel { snapshot_seq, .. } | LogOp::RejectModel { snapshot_seq, .. }) = &r.body {
            if *snapshot_seq < sys.st.seq || *snapshot_seq >= r.seq {
                let err = ReplayError::Model(RebuildError::BadSnapshotSeq { seq: r.seq, snapshot_seq: *snapshot_seq });
                return Err(RecoverError::Diverged { seq: r.seq, err });
            }
            wanted.push(*snapshot_seq);
        }
    }
    let (mut replayed, mut decisions) = (0u64, 0u64);
    let mut snap: Option<rebuild::Snapshot> = None;
    let ticks = chain.ticks.len() as u64;
    for t in &chain.ticks {
        if sys.w.seq != t.tick {
            return Err(RecoverError::Diverged { seq: sys.st.seq, err: ReplayError::Tick { want: sys.w.seq, got: t.tick } });
        }
        for r in &t.entries {
            let diverged = |err| RecoverError::Diverged { seq: r.seq, err };
            if snap.is_none() && wanted.first() == Some(&sys.st.seq) {
                wanted.remove(0);
                snap = Some(system::begin_rebuild(&mut sys).map_err(|e| diverged(ReplayError::Model(e)))?);
            }
            match &r.body {
                Body::Write(op) => { system::write(&mut sys, op).map_err(|e| diverged(ReplayError::Write(e)))?; }
                Body::Model(e) => {
                    let s = snap.take().ok_or_else(|| diverged(ReplayError::Model(RebuildError::NoRebuildPending)))?;
                    system::replay_decision(&mut sys, e, &s, threads).map_err(|e| diverged(ReplayError::Model(e)))?;
                    decisions += 1;
                }
                Body::EndTick { .. } => unreachable!("decode_ticks strips EndTick records"),
            }
            replayed += 1;
            debug_assert_eq!(sys.st.seq, r.seq);
        }
        views::publish(&mut sys.w);
    }
    // A rebuild started in the tail without its decision: in-flight work,
    // lost with the crash (see `system`).
    system::cancel(&mut sys);
    let (seq, tick) = (sys.st.seq, sys.w.seq - 1);
    Ok(Recovered {
        sys, log_id, seq, tick, snapshot_tick, replayed, ticks, decisions,
        dropped_bytes: chain.dropped, bad_snapshots, tail: chain.tail, unreachable: chain.unreachable,
    })
}

/// Rebuild the system from `dir`. Read-only. `base` is the index the log
/// was created from (state 0, via `system::start`), used only if no
/// snapshot verifies. Rows go into blocks of `block_rows`; model decisions
/// rebuild with `threads` (any count gives the same result).
pub fn recover(dir: &Path, base: Option<&IvfPqIndex>, block_rows: usize, threads: usize) -> Result<Recovered, RecoverError> {
    let snaps = list_files(dir, "snap-", "snap")?;
    let logs = list_files(dir, "log-", "log")?;
    let mut bad = Vec::new();
    for (_, path) in snaps.iter().rev() {
        match snapshot::read(path, block_rows) {
            Ok(sn) => {
                let chain = read_chain(&logs, sn.log_id, sn.seq, sn.tick)?;
                return replay(sn.sys, chain, sn.log_id, Some(sn.tick), bad, threads);
            }
            Err(_) => bad.push(path.clone()),
        }
    }
    let base = base.ok_or(RecoverError::NoState)?;
    let sys = system::start(base, block_rows);
    let log_id = system::fingerprint(&sys);
    let (seq, tick) = (sys.st.seq, sys.w.seq - 1);
    let first = logs.first().ok_or(RecoverError::NoState)?;
    let head = fs::read(&first.1)?;
    if first.0 != tick || oplog::decode_segment_header(&head) != Some((seq, tick, log_id)) {
        return Err(RecoverError::NoState);
    }
    let chain = read_chain(&logs, log_id, seq, tick)?;
    replay(sys, chain, log_id, None, bad, threads)
}

/// Rename `p` to `p.corrupt`, keeping it for inspection.
fn set_aside(p: &Path) -> io::Result<()> {
    let mut s = p.as_os_str().to_owned();
    s.push(".corrupt");
    fs::rename(p, PathBuf::from(s))
}

/// Open the log for appends after `recover`: set bad snapshots and
/// unreachable segments aside (`.corrupt`), remove stale `.tmp` files,
/// cut the torn tail (and any partial tick) off, and continue after
/// `rec.seq`, tick `rec.tick`. The next snapshot is due
/// `snapshot_every_ticks` after `rec.tick`.
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
            let mut h = [0u8; SEGMENT_HEADER_LEN];
            File::open(path)?.read_exact(&mut h)?;
            let (start_seq, ..) = oplog::decode_segment_header(&h).ok_or_else(|| io::Error::other("segment header changed"))?;
            writer(dir, params, rec.log_id, f, path.clone(), (start_seq, *start), rec.seq, rec.tick, *len, now)
        }
        None => {
            let (f, path) = new_segment(dir, rec.seq, rec.tick, rec.log_id, true)?;
            writer(dir, params, rec.log_id, f, path, (rec.seq, rec.tick), rec.seq, rec.tick, SEGMENT_HEADER_LEN as u64, now)
        }
    };
    sync_dir(dir)?;
    Ok(w)
}
