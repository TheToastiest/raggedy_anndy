//! On-disk op log format: segment header, records, CRC32, tick markers.
//!
//! Plain encode/decode functions; file handling lives in `durable`.
//! Everything is little-endian and fixed width.
//!
//! Segment file `log-{start_tick:020}.log`:
//!
//! | bytes | field |
//! | --- | --- |
//! | 8 | magic `RAOPLOG\0` |
//! | 4 | version (2) |
//! | 4 | reserved (0) |
//! | 8 | `start_seq`: entries in this file start at `start_seq + 1` |
//! | 8 | `start_tick`: ticks in this file start at `start_tick + 1` |
//! | 8 | `log_id`: `system::fingerprint` of the state the log was created on; same in every segment and snapshot of one log |
//! | 4 | CRC32 of the 40 bytes above |
//!
//! then records, back to back:
//!
//! | bytes | field |
//! | --- | --- |
//! | 4 | `len`: payload length |
//! | 4 | CRC32 of the payload |
//! | len | payload: `seq` u64, `tick` u64, `kind` u8, body |
//!
//! Kinds and bodies (`Body`):
//! - `RealOp::Live(op)` (a plain write): 1 Insert / 2 Upsert: `id` u64,
//!   `ts` u64, `tags` u64, `n` u32, `n` x f32 bits. 3 Delete: `id` u64.
//!   4 Expire: `before_ts` u64, `max_rows` u64 (`usize::MAX` is stored as
//!   `u64::MAX`). 5 Compact: `list` u32.
//! - 6 InsertReal / 7 UpsertReal: `id`, `ts`, `tags` u64, `flags` u8, `n`
//!   u32, `n` x f32 bits. 8 Commit: empty. 9 Ack: `through_seq` u64.
//! - 10 SwapModel / 11 RejectModel: `epoch`, `snapshot_seq`, `model_hash`
//!   u64, then the `Gate` (see `put_gate`; f64 as bits).
//! - 12 EndTick: `count` u32, `crc` u32.
//!
//! Entries (kinds 1-11) take seqs 1, 2, 3, ... with no gaps. A tick ends
//! with one EndTick record carrying the tick, the seq of the tick's last
//! entry (no new seq), the number of entries in the tick and the CRC32 of
//! their record bytes (headers included). Every tick has one, even with no
//! entries, so ticks are consecutive. A tick counts only as a whole: its
//! entries are applied only if its EndTick is there and matches, so a
//! crash part-way through writing a tick drops the whole tick.
//!
//! A record is accepted only if its length fits in what is left, its CRC
//! matches, its body parses to exactly `len` bytes and its `seq` and
//! `tick` follow on. The first record that fails ends the log, and so do
//! entries with no EndTick after them: a torn or corrupt tail is dropped,
//! never mis-parsed.

use crate::drift::{Signals, Thresholds};
use crate::gate::{GameCheck, Gate, GateParams, GateReason};
use crate::live::Op;
use crate::real_memory::RealOp;
use crate::rebuild::LogOp;

pub const SEGMENT_MAGIC: [u8; 8] = *b"RAOPLOG\0";
pub const SEGMENT_VERSION: u32 = 2;
pub const SEGMENT_HEADER_LEN: usize = 44;
/// Length prefix + CRC.
pub const RECORD_HEADER_LEN: usize = 8;
/// Largest payload accepted on read (a 1M-dim vector is 4 MB). Guards
/// against a corrupt length prefix asking for a huge read.
pub const MAX_RECORD_LEN: usize = 1 << 26;

const KIND_INSERT: u8 = 1;
const KIND_UPSERT: u8 = 2;
const KIND_DELETE: u8 = 3;
const KIND_EXPIRE: u8 = 4;
const KIND_COMPACT: u8 = 5;
const KIND_INSERT_REAL: u8 = 6;
const KIND_UPSERT_REAL: u8 = 7;
const KIND_COMMIT: u8 = 8;
const KIND_ACK: u8 = 9;
const KIND_SWAP: u8 = 10;
const KIND_REJECT: u8 = 11;
const KIND_END_TICK: u8 = 12;

/// What one record holds.
#[derive(Clone, Debug, PartialEq)]
pub enum Body {
    /// A write: a plain op (`RealOp::Live`), a real-memory op, a commit or an ack.
    Write(RealOp),
    /// A model decision: `LogOp::SwapModel` or `LogOp::RejectModel`. Never
    /// `LogOp::Write`; a plain write is `Write(RealOp::Live(op))`.
    Model(LogOp),
    /// End of tick `Record::tick`: `count` entries, CRC32 `crc` of their records.
    EndTick { count: u32, crc: u32 },
}

/// One record: its seq (entries 1, 2, 3, ...; an EndTick repeats the seq of
/// the tick's last entry), the tick it belongs to, and what it holds.
#[derive(Clone, Debug, PartialEq)]
pub struct Record { pub seq: u64, pub tick: u64, pub body: Body }

/// A whole tick read back: its number and its entries, in order (no EndTick).
#[derive(Clone, Debug, PartialEq)]
pub struct Tick { pub tick: u64, pub entries: Vec<Record> }

// ------------------------------------------------------------ CRC32

/// CRC-32 (IEEE 802.3, reflected poly 0xEDB88320), the zlib/PNG one.
const CRC_TABLE: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 { c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 }; k += 1; }
        t[i] = c;
        i += 1;
    }
    t
};

/// Continue a CRC: `crc32_update(crc32(a), b) == crc32(a ++ b)`.
pub fn crc32_update(crc: u32, bytes: &[u8]) -> u32 {
    let mut c = !crc;
    for &b in bytes { c = CRC_TABLE[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8); }
    !c
}

pub fn crc32(bytes: &[u8]) -> u32 { crc32_update(0, bytes) }

// ------------------------------------------------------------ segment header

pub fn encode_segment_header(start_seq: u64, start_tick: u64, log_id: u64) -> [u8; SEGMENT_HEADER_LEN] {
    let mut h = [0u8; SEGMENT_HEADER_LEN];
    h[0..8].copy_from_slice(&SEGMENT_MAGIC);
    h[8..12].copy_from_slice(&SEGMENT_VERSION.to_le_bytes());
    h[16..24].copy_from_slice(&start_seq.to_le_bytes());
    h[24..32].copy_from_slice(&start_tick.to_le_bytes());
    h[32..40].copy_from_slice(&log_id.to_le_bytes());
    let c = crc32(&h[..40]);
    h[40..44].copy_from_slice(&c.to_le_bytes());
    h
}

/// `(start_seq, start_tick, log_id)`, or `None` if the header is short,
/// torn or not ours.
pub fn decode_segment_header(b: &[u8]) -> Option<(u64, u64, u64)> {
    if b.len() < SEGMENT_HEADER_LEN || b[0..8] != SEGMENT_MAGIC { return None; }
    if u32::from_le_bytes(b[8..12].try_into().ok()?) != SEGMENT_VERSION { return None; }
    if u32::from_le_bytes(b[40..44].try_into().ok()?) != crc32(&b[..40]) { return None; }
    let at = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
    Some((at(16), at(24), at(32)))
}

// ------------------------------------------------------------ encode

fn put_u32(out: &mut Vec<u8>, x: u32) { out.extend_from_slice(&x.to_le_bytes()); }
fn put_u64(out: &mut Vec<u8>, x: u64) { out.extend_from_slice(&x.to_le_bytes()); }
fn put_f64(out: &mut Vec<u8>, x: f64) { put_u64(out, x.to_bits()); }
fn put_vec(out: &mut Vec<u8>, v: &[f32]) {
    put_u32(out, v.len() as u32);
    for x in v { out.extend_from_slice(&x.to_bits().to_le_bytes()); }
}

fn put_signals(out: &mut Vec<u8>, s: &Signals) {
    for x in [s.qerr_ratio, s.rnorm_ratio, s.imbalance, s.churn] { put_f64(out, x); }
    put_u64(out, s.inserts);
    out.push(s.fired);
}

/// The whole `Gate`, field by field, f64 as bits.
fn put_gate(out: &mut Vec<u8>, g: &Gate) {
    let p = &g.params;
    for x in [p.queries, p.recent, p.k] { put_u64(out, x as u64); }
    put_f64(out, p.min_gain);
    for x in [p.th.qerr_ratio, p.th.rnorm_ratio, p.th.imbalance, p.th.churn] { put_f64(out, x); }
    put_u64(out, p.th.min_inserts);
    put_u64(out, p.th.min_ops_per_epoch);
    out.push(match g.game { GameCheck::NotRun => 0, GameCheck::Pass => 1, GameCheck::Fail => 2 });
    for x in [g.queries, g.trials, g.current_hits, g.candidate_hits] { put_u32(out, x); }
    put_signals(out, &g.current);
    put_signals(out, &g.candidate);
    out.push(g.accept as u8);
    out.push(match g.reason { GateReason::RecallGain => 0, GateReason::DriftCleared => 1, GateReason::NoGain => 2, GateReason::GameCheckFailed => 3 });
}

fn put_write(out: &mut Vec<u8>, op: &RealOp) {
    match op {
        RealOp::Live(o) => match o {
            Op::Insert { id, ts, tags, vec } | Op::Upsert { id, ts, tags, vec } => {
                out.push(if matches!(o, Op::Insert { .. }) { KIND_INSERT } else { KIND_UPSERT });
                for x in [*id, *ts, *tags] { put_u64(out, x); }
                put_vec(out, vec);
            }
            Op::Delete { id } => { out.push(KIND_DELETE); put_u64(out, *id); }
            Op::Expire { before_ts, max_rows } => { out.push(KIND_EXPIRE); put_u64(out, *before_ts); put_u64(out, *max_rows as u64); }
            Op::Compact { list } => { out.push(KIND_COMPACT); put_u32(out, *list); }
        },
        RealOp::InsertReal { id, ts, tags, flags, vec } | RealOp::UpsertReal { id, ts, tags, flags, vec } => {
            out.push(if matches!(op, RealOp::InsertReal { .. }) { KIND_INSERT_REAL } else { KIND_UPSERT_REAL });
            for x in [*id, *ts, *tags] { put_u64(out, x); }
            out.push(*flags);
            put_vec(out, vec);
        }
        RealOp::Commit => out.push(KIND_COMMIT),
        RealOp::Ack { through_seq } => { out.push(KIND_ACK); put_u64(out, *through_seq); }
    }
}

fn put_body(out: &mut Vec<u8>, body: &Body) {
    match body {
        Body::Write(op) => put_write(out, op),
        Body::Model(e @ (LogOp::SwapModel { epoch, snapshot_seq, model_hash, gate } | LogOp::RejectModel { epoch, snapshot_seq, model_hash, gate })) => {
            out.push(if matches!(e, LogOp::SwapModel { .. }) { KIND_SWAP } else { KIND_REJECT });
            for x in [*epoch, *snapshot_seq, *model_hash] { put_u64(out, x); }
            put_gate(out, gate);
        }
        Body::Model(LogOp::Write(_)) => panic!("oplog: a plain write is Body::Write(RealOp::Live(op)), not Body::Model(LogOp::Write)"),
        Body::EndTick { count, crc } => { out.push(KIND_END_TICK); put_u32(out, *count); put_u32(out, *crc); }
    }
}

/// Frame what `body` puts as one record: length, CRC, seq, tick, body.
fn frame(out: &mut Vec<u8>, seq: u64, tick: u64, body: impl FnOnce(&mut Vec<u8>)) {
    let start = out.len();
    out.extend_from_slice(&[0u8; RECORD_HEADER_LEN]);
    put_u64(out, seq);
    put_u64(out, tick);
    body(out);
    let len = (out.len() - start - RECORD_HEADER_LEN) as u32;
    let c = crc32(&out[start + RECORD_HEADER_LEN..]);
    out[start..start + 4].copy_from_slice(&len.to_le_bytes());
    out[start + 4..start + 8].copy_from_slice(&c.to_le_bytes());
}

/// Append one record to `out`. No allocation once `out` has room.
/// Panics on `Body::Model(LogOp::Write(_))` (see `Body`).
pub fn encode_record(out: &mut Vec<u8>, seq: u64, tick: u64, body: &Body) { frame(out, seq, tick, |o| put_body(o, body)); }

/// `encode_record` of `Body::Write(op)` without cloning `op` into a `Body`.
pub fn encode_write(out: &mut Vec<u8>, seq: u64, tick: u64, op: &RealOp) { frame(out, seq, tick, |o| put_write(o, op)); }

// ------------------------------------------------------------ decode

fn take<'a>(b: &'a [u8], p: &mut usize, n: usize) -> Option<&'a [u8]> {
    let s = b.get(*p..p.checked_add(n)?)?;
    *p += n;
    Some(s)
}
fn u8_at(b: &[u8], p: &mut usize) -> Option<u8> { Some(take(b, p, 1)?[0]) }
fn u32_at(b: &[u8], p: &mut usize) -> Option<u32> { Some(u32::from_le_bytes(take(b, p, 4)?.try_into().ok()?)) }
fn u64_at(b: &[u8], p: &mut usize) -> Option<u64> { Some(u64::from_le_bytes(take(b, p, 8)?.try_into().ok()?)) }
fn f64_at(b: &[u8], p: &mut usize) -> Option<f64> { Some(f64::from_bits(u64_at(b, p)?)) }
fn usize_at(b: &[u8], p: &mut usize) -> Option<usize> { usize::try_from(u64_at(b, p)?).ok() }
fn vec_at(b: &[u8], p: &mut usize) -> Option<Vec<f32>> {
    let n = u32_at(b, p)? as usize;
    let raw = take(b, p, n.checked_mul(4)?)?;
    Some(raw.chunks_exact(4).map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect())
}

fn signals_at(b: &[u8], p: &mut usize) -> Option<Signals> {
    Some(Signals {
        qerr_ratio: f64_at(b, p)?, rnorm_ratio: f64_at(b, p)?, imbalance: f64_at(b, p)?, churn: f64_at(b, p)?,
        inserts: u64_at(b, p)?, fired: u8_at(b, p)?,
    })
}

fn gate_at(b: &[u8], p: &mut usize) -> Option<Gate> {
    let (queries, recent, k) = (usize_at(b, p)?, usize_at(b, p)?, usize_at(b, p)?);
    let min_gain = f64_at(b, p)?;
    let th = Thresholds {
        qerr_ratio: f64_at(b, p)?, rnorm_ratio: f64_at(b, p)?, imbalance: f64_at(b, p)?, churn: f64_at(b, p)?,
        min_inserts: u64_at(b, p)?, min_ops_per_epoch: u64_at(b, p)?,
    };
    let game = match u8_at(b, p)? { 0 => GameCheck::NotRun, 1 => GameCheck::Pass, 2 => GameCheck::Fail, _ => return None };
    let (gq, trials, current_hits, candidate_hits) = (u32_at(b, p)?, u32_at(b, p)?, u32_at(b, p)?, u32_at(b, p)?);
    let current = signals_at(b, p)?;
    let candidate = signals_at(b, p)?;
    let accept = match u8_at(b, p)? { 0 => false, 1 => true, _ => return None };
    let reason = match u8_at(b, p)? {
        0 => GateReason::RecallGain, 1 => GateReason::DriftCleared, 2 => GateReason::NoGain, 3 => GateReason::GameCheckFailed,
        _ => return None,
    };
    Some(Gate {
        params: GateParams { queries, recent, k, min_gain, th }, game,
        queries: gq, trials, current_hits, candidate_hits, current, candidate, accept, reason,
    })
}

/// Decode the record at the start of `b`: the record and the bytes it
/// used. `None` if it is torn, corrupt or malformed. Does not check `seq`
/// or tick order; see `decode_ticks`.
pub fn decode_record(b: &[u8]) -> Option<(Record, usize)> {
    let mut p = 0usize;
    let len = u32_at(b, &mut p)? as usize;
    let crc = u32_at(b, &mut p)?;
    if len > MAX_RECORD_LEN { return None; }
    let pl = take(b, &mut p, len)?;
    if crc32(pl) != crc { return None; }
    let mut q = 0usize;
    let seq = u64_at(pl, &mut q)?;
    let tick = u64_at(pl, &mut q)?;
    let body = match u8_at(pl, &mut q)? {
        k @ (KIND_INSERT | KIND_UPSERT) => {
            let (id, ts, tags) = (u64_at(pl, &mut q)?, u64_at(pl, &mut q)?, u64_at(pl, &mut q)?);
            let vec = vec_at(pl, &mut q)?;
            Body::Write(RealOp::Live(if k == KIND_INSERT { Op::Insert { id, ts, tags, vec } } else { Op::Upsert { id, ts, tags, vec } }))
        }
        KIND_DELETE => Body::Write(RealOp::Live(Op::Delete { id: u64_at(pl, &mut q)? })),
        KIND_EXPIRE => {
            let before_ts = u64_at(pl, &mut q)?;
            let max_rows = usize::try_from(u64_at(pl, &mut q)?).unwrap_or(usize::MAX);
            Body::Write(RealOp::Live(Op::Expire { before_ts, max_rows }))
        }
        KIND_COMPACT => Body::Write(RealOp::Live(Op::Compact { list: u32_at(pl, &mut q)? })),
        k @ (KIND_INSERT_REAL | KIND_UPSERT_REAL) => {
            let (id, ts, tags) = (u64_at(pl, &mut q)?, u64_at(pl, &mut q)?, u64_at(pl, &mut q)?);
            let flags = u8_at(pl, &mut q)?;
            let vec = vec_at(pl, &mut q)?;
            Body::Write(if k == KIND_INSERT_REAL { RealOp::InsertReal { id, ts, tags, flags, vec } } else { RealOp::UpsertReal { id, ts, tags, flags, vec } })
        }
        KIND_COMMIT => Body::Write(RealOp::Commit),
        KIND_ACK => Body::Write(RealOp::Ack { through_seq: u64_at(pl, &mut q)? }),
        k @ (KIND_SWAP | KIND_REJECT) => {
            let (epoch, snapshot_seq, model_hash) = (u64_at(pl, &mut q)?, u64_at(pl, &mut q)?, u64_at(pl, &mut q)?);
            let gate = gate_at(pl, &mut q)?;
            Body::Model(if k == KIND_SWAP { LogOp::SwapModel { epoch, snapshot_seq, model_hash, gate } } else { LogOp::RejectModel { epoch, snapshot_seq, model_hash, gate } })
        }
        KIND_END_TICK => Body::EndTick { count: u32_at(pl, &mut q)?, crc: u32_at(pl, &mut q)? },
        _ => return None,
    };
    if q != pl.len() { return None; }
    Some((Record { seq, tick, body }, p))
}

/// Decode whole ticks from `b` (a segment body, after its header),
/// starting after entry `after_seq` and tick `after_tick`. A tick is kept
/// only if every record decodes, its entries' seqs follow on with no gap,
/// they all carry the tick, the tick is the previous one plus 1, and its
/// EndTick has the last entry's seq, the entry count and the CRC of their
/// bytes. Returns the ticks and the length of the valid prefix (through the
/// last good EndTick); anything past it, whole records included, is a torn
/// or corrupt tail.
pub fn decode_ticks(b: &[u8], after_seq: u64, after_tick: u64) -> (Vec<Tick>, usize) {
    let mut out = Vec::new();
    let (mut valid, mut seq, mut tick) = (0usize, after_seq, after_tick);
    'ticks: loop {
        let (mut at, mut s, t) = (valid, seq, tick.wrapping_add(1));
        let mut entries = Vec::new();
        let mut crc = 0u32;
        loop {
            let Some((rec, used)) = decode_record(&b[at..]) else { break 'ticks };
            if rec.tick != t { break 'ticks; }
            if let Body::EndTick { count, crc: c } = rec.body {
                if rec.seq != s || count as usize != entries.len() || c != crc { break 'ticks; }
                valid = at + used;
                seq = s;
                tick = t;
                out.push(Tick { tick: t, entries });
                continue 'ticks;
            }
            if rec.seq != s.wrapping_add(1) { break 'ticks; }
            crc = crc32_update(crc, &b[at..at + used]);
            s = rec.seq;
            at += used;
            entries.push(rec);
        }
    }
    (out, valid)
}
