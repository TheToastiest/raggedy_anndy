//! On-disk op log format: segment header, records, CRC32.
//!
//! Plain encode/decode functions; file handling lives in `durable`.
//! Everything is little-endian and fixed width.
//!
//! Segment file `log-{start_seq:020}.log`:
//!
//! | bytes | field |
//! | --- | --- |
//! | 8 | magic `RAOPLOG\0` |
//! | 4 | version (1) |
//! | 4 | reserved (0) |
//! | 8 | `start_seq`: records in this file start at `start_seq + 1` |
//! | 8 | `start_fp`: `IvfPqIndex::fingerprint` after op `start_seq` |
//! | 4 | CRC32 of the 32 bytes above |
//!
//! then records, back to back:
//!
//! | bytes | field |
//! | --- | --- |
//! | 4 | `len`: payload length |
//! | 4 | CRC32 of the payload |
//! | len | payload: `seq` u64, `tick` u64, `kind` u8, body |
//!
//! Bodies by kind: 1 Insert / 2 Upsert: `id` u64, `ts` u64, `tags` u64,
//! `n` u32, `n` x f32 bits. 3 Delete: `id` u64. 4 Expire: `before_ts` u64,
//! `max_rows` u64 (`usize::MAX` is stored as `u64::MAX`). 5 Compact: `list` u32.
//!
//! A record is accepted only if its length fits in what is left, its CRC
//! matches, its body parses to exactly `len` bytes and its `seq` is the
//! previous one plus 1. The first record that fails ends the log: a torn
//! or corrupt tail is dropped, never mis-parsed.

use crate::live::Op;

pub const SEGMENT_MAGIC: [u8; 8] = *b"RAOPLOG\0";
pub const SEGMENT_VERSION: u32 = 1;
pub const SEGMENT_HEADER_LEN: usize = 36;
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

/// One logged op: its sequence number (1, 2, 3, ... with no gaps) and the
/// server tick it was applied in.
#[derive(Clone, Debug, PartialEq)]
pub struct Record { pub seq: u64, pub tick: u64, pub op: Op }

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

pub fn encode_segment_header(start_seq: u64, start_fp: u64) -> [u8; SEGMENT_HEADER_LEN] {
    let mut h = [0u8; SEGMENT_HEADER_LEN];
    h[0..8].copy_from_slice(&SEGMENT_MAGIC);
    h[8..12].copy_from_slice(&SEGMENT_VERSION.to_le_bytes());
    h[16..24].copy_from_slice(&start_seq.to_le_bytes());
    h[24..32].copy_from_slice(&start_fp.to_le_bytes());
    let c = crc32(&h[..32]);
    h[32..36].copy_from_slice(&c.to_le_bytes());
    h
}

/// `(start_seq, start_fp)`, or `None` if the header is short, torn or not ours.
pub fn decode_segment_header(b: &[u8]) -> Option<(u64, u64)> {
    if b.len() < SEGMENT_HEADER_LEN || b[0..8] != SEGMENT_MAGIC { return None; }
    if u32::from_le_bytes(b[8..12].try_into().ok()?) != SEGMENT_VERSION { return None; }
    if u32::from_le_bytes(b[32..36].try_into().ok()?) != crc32(&b[..32]) { return None; }
    Some((u64::from_le_bytes(b[16..24].try_into().ok()?), u64::from_le_bytes(b[24..32].try_into().ok()?)))
}

// ------------------------------------------------------------ records

/// Append one record to `out`. No allocation once `out` has room.
pub fn encode_record(out: &mut Vec<u8>, seq: u64, tick: u64, op: &Op) {
    let start = out.len();
    out.extend_from_slice(&[0u8; RECORD_HEADER_LEN]);
    out.extend_from_slice(&seq.to_le_bytes());
    out.extend_from_slice(&tick.to_le_bytes());
    match op {
        Op::Insert { id, ts, tags, vec } | Op::Upsert { id, ts, tags, vec } => {
            out.push(if matches!(op, Op::Insert { .. }) { KIND_INSERT } else { KIND_UPSERT });
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(&ts.to_le_bytes());
            out.extend_from_slice(&tags.to_le_bytes());
            out.extend_from_slice(&(vec.len() as u32).to_le_bytes());
            for x in vec { out.extend_from_slice(&x.to_bits().to_le_bytes()); }
        }
        Op::Delete { id } => { out.push(KIND_DELETE); out.extend_from_slice(&id.to_le_bytes()); }
        Op::Expire { before_ts, max_rows } => {
            out.push(KIND_EXPIRE);
            out.extend_from_slice(&before_ts.to_le_bytes());
            out.extend_from_slice(&(*max_rows as u64).to_le_bytes());
        }
        Op::Compact { list } => { out.push(KIND_COMPACT); out.extend_from_slice(&list.to_le_bytes()); }
    }
    let len = (out.len() - start - RECORD_HEADER_LEN) as u32;
    let c = crc32(&out[start + RECORD_HEADER_LEN..]);
    out[start..start + 4].copy_from_slice(&len.to_le_bytes());
    out[start + 4..start + 8].copy_from_slice(&c.to_le_bytes());
}

fn take<'a>(b: &'a [u8], p: &mut usize, n: usize) -> Option<&'a [u8]> {
    let s = b.get(*p..p.checked_add(n)?)?;
    *p += n;
    Some(s)
}
fn u8_at(b: &[u8], p: &mut usize) -> Option<u8> { Some(take(b, p, 1)?[0]) }
fn u32_at(b: &[u8], p: &mut usize) -> Option<u32> { Some(u32::from_le_bytes(take(b, p, 4)?.try_into().ok()?)) }
fn u64_at(b: &[u8], p: &mut usize) -> Option<u64> { Some(u64::from_le_bytes(take(b, p, 8)?.try_into().ok()?)) }

/// Decode the record at the start of `b`: the record and the bytes it
/// used. `None` if it is torn, corrupt or malformed. Does not check `seq`
/// order; see `decode_all`.
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
    let op = match u8_at(pl, &mut q)? {
        k @ (KIND_INSERT | KIND_UPSERT) => {
            let id = u64_at(pl, &mut q)?;
            let ts = u64_at(pl, &mut q)?;
            let tags = u64_at(pl, &mut q)?;
            let n = u32_at(pl, &mut q)? as usize;
            let raw = take(pl, &mut q, n.checked_mul(4)?)?;
            let vec = raw.chunks_exact(4).map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect();
            if k == KIND_INSERT { Op::Insert { id, ts, tags, vec } } else { Op::Upsert { id, ts, tags, vec } }
        }
        KIND_DELETE => Op::Delete { id: u64_at(pl, &mut q)? },
        KIND_EXPIRE => {
            let before_ts = u64_at(pl, &mut q)?;
            let max_rows = usize::try_from(u64_at(pl, &mut q)?).unwrap_or(usize::MAX);
            Op::Expire { before_ts, max_rows }
        }
        KIND_COMPACT => Op::Compact { list: u32_at(pl, &mut q)? },
        _ => return None,
    };
    if q != pl.len() { return None; }
    Some((Record { seq, tick, op }, p))
}

/// Decode records from `b` (a segment body, after its header) while each
/// one is valid and its seq is the previous + 1, starting at
/// `after_seq + 1`. Returns them and the length of the valid prefix;
/// anything past that prefix is a torn or corrupt tail.
pub fn decode_all(b: &[u8], after_seq: u64) -> (Vec<Record>, usize) {
    let mut out = Vec::new();
    let mut at = 0usize;
    let mut seq = after_seq;
    while let Some((rec, used)) = decode_record(&b[at..]) {
        if rec.seq != seq.wrapping_add(1) { break; }
        seq = rec.seq;
        at += used;
        out.push(rec);
    }
    (out, at)
}
