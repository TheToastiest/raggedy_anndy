//! Real memories: the pending buffer between the 1-hour short-term index
//! and the 12-hour commit to long-term storage (stage 5, index side).
//!
//! The game server decides what is real (endgame-only events) and says so
//! at insert time with `flags`. A real memory is written to the short-term
//! index like any other row, and also kept here, in RAM, with its f32
//! vector, so it stays searchable (exactly, Flat) after the 1-hour expiry
//! takes it out of the index. `commit` drains the buffer into an outbox of
//! plain `RealRecord`s for the SQL side, which is not part of this crate.
//!
//! Rules the functions here keep:
//! - `flags` is its own field, never one of the 64 tag bits. `REAL` must be
//!   set; `LEGENDARY` (never expires in SQL) is a stricter subset of real.
//! - Every check happens before any change: an `Err` leaves the index and
//!   the buffer as they were.
//! - Commit order is (ts asc, id asc), and a commit never appends an id the
//!   outbox already holds, so re-running it after a crash adds nothing.
//! - Merged search ranks by (score desc, id asc) and returns each id once.
//!   For an id in both the index and the buffer, the buffer's exact score
//!   is used (equal bits to the index's re-rank when it stores vectors).
//!
//! Long-term memories are not indexed here; another component reads them
//! with SQL at query time.

use std::collections::{HashMap, HashSet};

use crate::ivfpq::{apply_decay, norm_inplace, IvfPqIndex, RowLoc};
use crate::live::{self, Applied, Op, Scratch, WriteError};
use crate::metric::{self, Metric};
use crate::types::{stable_top_k, Hit};

/// Server says this memory is real: hold it until the commit.
pub const REAL: u8 = 1 << 0;
/// Real and never expires in SQL. Only valid together with `REAL`.
pub const LEGENDARY: u8 = 1 << 1;
const KNOWN: u8 = REAL | LEGENDARY;

/// Real memories not yet committed. Columns side by side (SoA), rows in
/// insert order, so `timestamps` never decreases.
#[derive(Clone, Debug, PartialEq)]
pub struct Pending {
    pub metric: Metric,
    pub dim: usize,
    pub ids: Vec<u64>,
    pub timestamps: Vec<u64>,
    pub tags: Vec<u64>,
    pub flags: Vec<u8>,
    /// Vectors as inserted, row-major; these go to the outbox.
    pub raw: Vec<f32>,
    /// Vectors as scored: unit length for cosine, as `raw` for L2.
    pub view: Vec<f32>,
    /// id -> row. Lookup only: never iterated.
    pub dir: HashMap<u64, u32>,
}

/// One committed real memory, for the SQL writer. `id` is the key the
/// SQL side deduplicates on (`ON CONFLICT (id) DO NOTHING`).
#[derive(Clone, Debug, PartialEq)]
pub struct RealRecord {
    pub id: u64,
    pub ts: u64,
    pub tags: u64,
    pub flags: u8,
    pub vec: Vec<f32>,
}

/// A write that may touch the pending buffer. `Live` ops behave exactly as
/// `live::apply`, except that `Insert` of a still-pending id is refused.
#[derive(Clone, Debug, PartialEq)]
pub enum RealOp {
    Live(Op),
    /// New real memory: a short-term row plus a pending row.
    InsertReal { id: u64, ts: u64, tags: u64, flags: u8, vec: Vec<f32> },
    /// Drain the pending buffer into the outbox.
    Commit,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RealApplied {
    Live(Applied),
    InsertedReal { loc: RowLoc },
    /// Records appended to the outbox (ids already there are skipped).
    Committed { appended: usize },
}

/// Why a write was refused. Nothing was changed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RealError {
    Write(WriteError),
    BadFlags { id: u64, flags: u8 },
}

impl std::fmt::Display for RealError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{self:?}") }
}
impl std::error::Error for RealError {}

impl From<WriteError> for RealError {
    fn from(e: WriteError) -> Self { RealError::Write(e) }
}

/// Empty buffer with the index's metric and dim.
pub fn pending_for(idx: &IvfPqIndex) -> Pending {
    Pending {
        metric: idx.metric, dim: idx.dim,
        ids: Vec::new(), timestamps: Vec::new(), tags: Vec::new(), flags: Vec::new(),
        raw: Vec::new(), view: Vec::new(), dir: HashMap::new(),
    }
}

/// Apply one op. See `RealOp` for each variant's meaning.
pub fn apply(idx: &mut IvfPqIndex, s: &mut Scratch, buf: &mut Pending, outbox: &mut Vec<RealRecord>, op: &RealOp) -> Result<RealApplied, RealError> {
    match op {
        RealOp::Live(op) => {
            if let Op::Insert { id, .. } = op {
                if buf.dir.contains_key(id) { return Err(WriteError::DuplicateId { id: *id }.into()); }
            }
            Ok(RealApplied::Live(live::apply(idx, s, op)?))
        }
        RealOp::InsertReal { id, ts, tags, flags, vec } => {
            insert_real(idx, s, buf, *id, *ts, *tags, *flags, vec).map(|loc| RealApplied::InsertedReal { loc })
        }
        RealOp::Commit => Ok(RealApplied::Committed { appended: commit(buf, outbox) }),
    }
}

fn check_flags(id: u64, flags: u8) -> Result<(), RealError> {
    if flags & REAL == 0 || flags & !KNOWN != 0 { return Err(RealError::BadFlags { id, flags }); }
    Ok(())
}

/// Insert a real memory: a short-term index row (same as `live::insert`)
/// and a pending row. Fails if `id` is live in the index or still pending.
#[allow(clippy::too_many_arguments)]
pub fn insert_real(idx: &mut IvfPqIndex, s: &mut Scratch, buf: &mut Pending, id: u64, ts: u64, tags: u64, flags: u8, v: &[f32]) -> Result<RowLoc, RealError> {
    check_flags(id, flags)?;
    if buf.dir.contains_key(&id) { return Err(WriteError::DuplicateId { id }.into()); }
    // live::insert checks dim, finiteness, ts order and index duplicates,
    // and changes nothing on error; the push below cannot fail.
    let loc = live::insert(idx, s, id, ts, tags, v)?;
    push(buf, id, ts, tags, flags, v);
    Ok(loc)
}

fn push(buf: &mut Pending, id: u64, ts: u64, tags: u64, flags: u8, v: &[f32]) {
    let start = buf.view.len();
    buf.dir.insert(id, buf.ids.len() as u32);
    buf.ids.push(id);
    buf.timestamps.push(ts);
    buf.tags.push(tags);
    buf.flags.push(flags);
    buf.raw.extend_from_slice(v);
    buf.view.extend_from_slice(v);
    if matches!(buf.metric, Metric::Cosine) { norm_inplace(&mut buf.view[start..]); }
}

/// Move every pending row into `outbox` in (ts asc, id asc) order, skipping
/// ids `outbox` already holds, and empty the buffer. Returns how many
/// records were appended. Running it again on a buffer rebuilt by replay
/// appends nothing new to the same outbox.
pub fn commit(buf: &mut Pending, outbox: &mut Vec<RealRecord>) -> usize {
    let dim = buf.dim;
    let mut order: Vec<usize> = (0..buf.ids.len()).collect();
    order.sort_unstable_by_key(|&r| (buf.timestamps[r], buf.ids[r]));
    let seen: HashSet<u64> = outbox.iter().map(|r| r.id).collect();
    let before = outbox.len();
    for r in order {
        if seen.contains(&buf.ids[r]) { continue; }
        outbox.push(RealRecord {
            id: buf.ids[r], ts: buf.timestamps[r], tags: buf.tags[r], flags: buf.flags[r],
            vec: buf.raw[r * dim..(r + 1) * dim].to_vec(),
        });
    }
    clear(buf);
    outbox.len() - before
}

fn clear(buf: &mut Pending) {
    buf.ids.clear();
    buf.timestamps.clear();
    buf.tags.clear();
    buf.flags.clear();
    buf.raw.clear();
    buf.view.clear();
    buf.dir.clear();
}

pub fn pending_len(buf: &Pending) -> usize { buf.ids.len() }

pub fn is_pending(buf: &Pending, id: u64) -> bool { buf.dir.contains_key(&id) }

/// `Some(true)` if `id` is pending and LEGENDARY, `None` if not pending.
pub fn is_legendary(buf: &Pending, id: u64) -> Option<bool> {
    buf.dir.get(&id).map(|&r| buf.flags[r as usize] & LEGENDARY != 0)
}

/// Exact scores of pending rows that pass the tag filter, with the same
/// score and decay rule as `IvfPqIndex::search_temporal`'s exact re-rank.
fn search_pending(buf: &Pending, q_eff: &[f32], required_tag: Option<u64>, current_time: u64, lambda: f32) -> Vec<Hit> {
    let dim = buf.dim;
    let mut hits = Vec::with_capacity(buf.ids.len());
    for r in 0..buf.ids.len() {
        if let Some(mask) = required_tag {
            if buf.tags[r] & mask == 0 { continue; }
        }
        let v = &buf.view[r * dim..(r + 1) * dim];
        let raw = match buf.metric { Metric::Cosine => metric::dot(q_eff, v), Metric::L2 => -metric::l2_distance(q_eff, v) };
        hits.push(Hit { id: buf.ids[r], score: apply_decay(raw, buf.timestamps[r], current_time, lambda) });
    }
    hits
}

/// Top-k over the short-term index and the pending buffer together, by
/// (score desc, id asc), each id at most once. Arguments as
/// `IvfPqIndex::search_temporal`. The index is asked for `k + pending`
/// hits, so dropping ids that are also pending still leaves k from it.
pub fn search(idx: &IvfPqIndex, buf: &Pending, q: &[f32], k: usize, required_tag: Option<u64>, current_time: u64, lambda: f32) -> Vec<Hit> {
    if k == 0 || q.iter().any(|x| !x.is_finite()) { return Vec::new(); }
    let mut hits = idx.search_temporal(q, k + buf.ids.len(), required_tag, current_time, lambda);
    hits.retain(|h| !buf.dir.contains_key(&h.id));
    if !buf.ids.is_empty() {
        let mut q_eff = q.to_vec();
        if matches!(buf.metric, Metric::Cosine) { norm_inplace(&mut q_eff); }
        hits.extend(search_pending(buf, &q_eff, required_tag, current_time, lambda));
    }
    stable_top_k(hits, k)
}

/// Full consistency check, O(rows). For tests and debug builds.
pub fn check_invariants(buf: &Pending) -> Result<(), String> {
    let n = buf.ids.len();
    if buf.timestamps.len() != n || buf.tags.len() != n || buf.flags.len() != n
        || buf.raw.len() != n * buf.dim || buf.view.len() != n * buf.dim {
        return Err("column lengths differ".into());
    }
    if buf.dir.len() != n { return Err(format!("{n} rows but {} directory entries", buf.dir.len())); }
    for r in 0..n {
        if buf.dir.get(&buf.ids[r]) != Some(&(r as u32)) { return Err(format!("row {r} id {}: directory mismatch", buf.ids[r])); }
        if check_flags(buf.ids[r], buf.flags[r]).is_err() { return Err(format!("row {r}: bad flags {:#b}", buf.flags[r])); }
    }
    if buf.timestamps.windows(2).any(|w| w[0] > w[1]) { return Err("timestamps not sorted".into()); }
    if buf.raw.iter().any(|x| !x.is_finite()) { return Err("non-finite vector".into()); }
    Ok(())
}
