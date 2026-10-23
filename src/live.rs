//! Live writes for IVF-PQ: insert, delete, upsert, expire, compact.
//!
//! The trained model (centroids, OPQ, PQ codebooks) is frozen; writes only
//! touch the inverted lists and the id directory. Every write is an `Op`
//! applied by `apply`, in log order, on one thread. Given the same index
//! and the same ops, the result is byte-identical (`IvfPqIndex::fingerprint`)
//! on every run and machine with the same binary and CPU features.
//!
//! Rules the functions here keep:
//! - An insert assigns, residualises and encodes exactly as `build` does, so
//!   `empty_like(build(A))` + inserts of A equals `build(A)` byte for byte.
//! - Rows are immutable once written. Upsert is delete + a new row.
//! - Timestamps never go backwards, so each list is sorted by time and
//!   expiry only ever removes a prefix (advancing `head`).
//! - Every check happens before any change: an `Err` leaves the index as it was.
//!
//! Design: "IVF-PQ live insert and delete design" (project doc), stage 1.

use crate::ivfpq::{empty_list, nearest_centroid, norm_inplace, IvfPqIndex, RowLoc};
use crate::metric::Metric;

/// One write. The log of these is the source of truth for the index.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// New row. Fails if `id` is live.
    Insert { id: u64, ts: u64, tags: u64, vec: Vec<f32> },
    /// Replace `id`'s row with a new one, or insert if `id` is not live.
    Upsert { id: u64, ts: u64, tags: u64, vec: Vec<f32> },
    /// Mark `id`'s row dead.
    Delete { id: u64 },
    /// Kill live rows with `ts < before_ts`, at most `max_rows` rows walked
    /// this call. Lists are walked in index order; repeat until `done`.
    Expire { before_ts: u64, max_rows: usize },
    /// Rewrite one list without its dead rows.
    Compact { list: u32 },
}

/// What a successful `apply` did.
#[derive(Clone, Debug, PartialEq)]
pub enum Applied {
    Inserted { loc: RowLoc },
    Upserted { loc: RowLoc, replaced: Option<RowLoc> },
    Deleted { loc: RowLoc },
    /// `ids` of the rows this call expired, in walk order.
    Expired { ids: Vec<u64>, done: bool },
    Compacted { list: u32, dropped: usize },
}

/// Why a write was refused. Nothing was changed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WriteError {
    WrongDim { id: u64, got: usize, want: usize },
    NonFinite { id: u64 },
    DuplicateId { id: u64 },
    TsBackwards { id: u64, ts: u64, last: u64 },
    NotFound { id: u64 },
    NoSuchList { list: u32 },
}

impl std::fmt::Display for WriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{self:?}") }
}
impl std::error::Error for WriteError {}

/// Per-writer buffers so an insert does not allocate.
#[derive(Clone, Debug)]
pub struct Scratch { pub(crate) view: Vec<f32>, res: Vec<f32>, rot: Vec<f32>, pub(crate) codes: Vec<u8> }

pub fn scratch_for(idx: &IvfPqIndex) -> Scratch {
    Scratch { view: vec![0.0; idx.dim], res: vec![0.0; idx.dim], rot: vec![0.0; idx.dim], codes: vec![0; idx.params.m] }
}

/// Apply one op. See `Op` for each variant's meaning.
pub fn apply(idx: &mut IvfPqIndex, s: &mut Scratch, op: &Op) -> Result<Applied, WriteError> {
    match op {
        Op::Insert { id, ts, tags, vec } => insert(idx, s, *id, *ts, *tags, vec).map(|loc| Applied::Inserted { loc }),
        Op::Upsert { id, ts, tags, vec } => upsert(idx, s, *id, *ts, *tags, vec).map(|(loc, replaced)| Applied::Upserted { loc, replaced }),
        Op::Delete { id } => delete(idx, *id).map(|loc| Applied::Deleted { loc }),
        Op::Expire { before_ts, max_rows } => {
            let (ids, done) = expire(idx, *before_ts, *max_rows);
            Ok(Applied::Expired { ids, done })
        }
        Op::Compact { list } => compact_list(idx, *list).map(|dropped| Applied::Compacted { list: *list, dropped }),
    }
}

/// Row checks shared by every write path. `max_ts` is the writer's newest timestamp.
pub(crate) fn check_row(dim: usize, max_ts: u64, id: u64, ts: u64, v: &[f32]) -> Result<(), WriteError> {
    if v.len() != dim { return Err(WriteError::WrongDim { id, got: v.len(), want: dim }); }
    if v.iter().any(|x| !x.is_finite()) { return Err(WriteError::NonFinite { id }); }
    if ts < max_ts { return Err(WriteError::TsBackwards { id, ts, last: max_ts }); }
    Ok(())
}

/// Nearest list for an already-normalised vector: `build`'s assignment
/// (`nearest_centroid`, 4-wide, lower list index wins ties).
fn nearest(metric: Metric, v: &[f32], centroids: &[Vec<f32>]) -> usize {
    nearest_centroid(metric, v, centroids.len(), |j| &centroids[j][..])
}

/// List an insert of `v` would land in.
pub fn nearest_list(idx: &IvfPqIndex, v: &[f32]) -> usize {
    let mut view = v.to_vec();
    if matches!(idx.metric, Metric::Cosine) { norm_inplace(&mut view); }
    nearest(idx.metric, &view, &idx.centroids)
}

/// Append a new row. Cost: one centroid scan + one PQ encode, no
/// allocation once `reserve` has made room.
pub fn insert(idx: &mut IvfPqIndex, s: &mut Scratch, id: u64, ts: u64, tags: u64, v: &[f32]) -> Result<RowLoc, WriteError> {
    check_row(idx.dim, idx.max_ts, id, ts, v)?;
    if idx.dir.contains_key(&id) { return Err(WriteError::DuplicateId { id }); }
    Ok(append(idx, s, id, ts, tags, v))
}

/// Normalise (cosine), assign and encode `v` exactly as `build` does, with
/// `model`'s centroids, OPQ and codebooks. Leaves the stored vector in
/// `s.view` and the codes in `s.codes`; returns the list. No allocation.
pub(crate) fn encode_row(model: &IvfPqIndex, s: &mut Scratch, v: &[f32]) -> usize {
    s.view.copy_from_slice(v);
    if matches!(model.metric, Metric::Cosine) { norm_inplace(&mut s.view); }
    let li = nearest(model.metric, &s.view, &model.centroids);
    for ((r, &x), &c) in s.res.iter_mut().zip(&s.view).zip(&model.centroids[li]) { *r = x - c; }
    let to_encode: &[f32] = match &model.opq {
        Some(o) => { o.apply_to(&s.res, &mut s.rot); &s.rot }
        None => &s.res,
    };
    model.pq.encode_into(to_encode, &mut s.codes);
    li
}

/// Writes the row; all checks already passed.
fn append(idx: &mut IvfPqIndex, s: &mut Scratch, id: u64, ts: u64, tags: u64, v: &[f32]) -> RowLoc {
    let dim = idx.dim;
    let li = encode_row(idx, s, v);

    let list = &mut idx.lists[li];
    let loc = RowLoc { list: li as u32, row: list.ids.len() as u32 };
    list.ids.push(id);
    list.timestamps.push(ts);
    list.tomb.push(0);
    list.tags.push(tags);
    list.codes.extend_from_slice(&s.codes);
    if let Some(vecs) = &mut list.vecs { vecs.extend_from_slice(&s.view[..dim]); }
    idx.dir.insert(id, loc);
    idx.max_ts = ts;
    loc
}

/// Mark `id` dead. O(1).
pub fn delete(idx: &mut IvfPqIndex, id: u64) -> Result<RowLoc, WriteError> {
    let loc = idx.dir.remove(&id).ok_or(WriteError::NotFound { id })?;
    let list = &mut idx.lists[loc.list as usize];
    list.tomb[loc.row as usize] = 1;
    list.dead += 1;
    Ok(loc)
}

/// Delete `id` if live, then append the new row (never in place, even in
/// the same list). Returns the new location and the replaced one.
pub fn upsert(idx: &mut IvfPqIndex, s: &mut Scratch, id: u64, ts: u64, tags: u64, v: &[f32]) -> Result<(RowLoc, Option<RowLoc>), WriteError> {
    check_row(idx.dim, idx.max_ts, id, ts, v)?;
    let replaced = delete(idx, id).ok();
    Ok((append(idx, s, id, ts, tags, v), replaced))
}

/// Expire live rows with `ts < before_ts`. Walks lists in index order from
/// each list's `head`, at most `max_rows` rows per call, and returns the ids
/// it killed and whether every list is now clear of rows older than the
/// cutoff. A capped run repeated until `done` ends in the same state as
/// one uncapped run.
pub fn expire(idx: &mut IvfPqIndex, before_ts: u64, max_rows: usize) -> (Vec<u64>, bool) {
    let mut ids = Vec::new();
    let mut budget = max_rows;
    for list in idx.lists.iter_mut() {
        while list.head < list.ids.len() && list.timestamps[list.head] < before_ts {
            if budget == 0 { return (ids, false); }
            budget -= 1;
            let r = list.head;
            if list.tomb[r] == 0 {
                list.tomb[r] = 1;
                list.dead += 1;
                idx.dir.remove(&list.ids[r]);
                ids.push(list.ids[r]);
            }
            list.head += 1;
        }
    }
    (ids, true)
}

/// Rewrite `list` keeping only live rows, in their current order, and
/// repoint their directory entries. Search results do not change.
pub fn compact_list(idx: &mut IvfPqIndex, list: u32) -> Result<usize, WriteError> {
    let li = list as usize;
    if li >= idx.lists.len() { return Err(WriteError::NoSuchList { list }); }
    let m = idx.params.m;
    let dim = idx.dim;
    let old = std::mem::replace(&mut idx.lists[li], empty_list(idx.params.store_vecs));
    let live = old.ids.len() - old.dead;
    let new = &mut idx.lists[li];
    new.ids.reserve_exact(live);
    new.timestamps.reserve_exact(live);
    new.tomb.reserve_exact(live);
    new.tags.reserve_exact(live);
    new.codes.reserve_exact(live * m);
    if let Some(v) = &mut new.vecs { v.reserve_exact(live * dim); }
    for r in old.head..old.ids.len() {
        if old.tomb[r] == 1 { continue; }
        idx.dir.insert(old.ids[r], RowLoc { list, row: new.ids.len() as u32 });
        new.ids.push(old.ids[r]);
        new.timestamps.push(old.timestamps[r]);
        new.tomb.push(0);
        new.tags.push(old.tags[r]);
        new.codes.extend_from_slice(&old.codes[r * m..(r + 1) * m]);
        if let (Some(nv), Some(ov)) = (&mut new.vecs, &old.vecs) { nv.extend_from_slice(&ov[r * dim..(r + 1) * dim]); }
    }
    Ok(old.dead)
}

/// Lists whose dead share is above `max_dead_ratio`, ascending.
pub fn lists_to_compact(idx: &IvfPqIndex, max_dead_ratio: f32) -> Vec<u32> {
    idx.lists.iter().enumerate()
        .filter(|(_, l)| !l.ids.is_empty() && l.dead as f32 > max_dead_ratio * l.ids.len() as f32)
        .map(|(i, _)| i as u32).collect()
}

/// Same model, no rows. The starting point for replaying a log.
pub fn empty_like(idx: &IvfPqIndex) -> IvfPqIndex {
    IvfPqIndex {
        metric: idx.metric, dim: idx.dim, params: idx.params,
        centroids: idx.centroids.clone(), centroids_u: idx.centroids_u.clone(),
        pq: idx.pq.clone(), opq: idx.opq.clone(),
        lists: (0..idx.lists.len()).map(|_| empty_list(idx.params.store_vecs)).collect(),
        dir: Default::default(), max_ts: 0,
    }
}

/// Room for `rows_per_list` more rows in every list and in the directory,
/// so the next inserts do not allocate. Call between ticks.
pub fn reserve(idx: &mut IvfPqIndex, rows_per_list: usize) {
    let (m, dim) = (idx.params.m, idx.dim);
    for l in idx.lists.iter_mut() {
        l.ids.reserve(rows_per_list);
        l.timestamps.reserve(rows_per_list);
        l.tomb.reserve(rows_per_list);
        l.tags.reserve(rows_per_list);
        l.codes.reserve(rows_per_list * m);
        if let Some(v) = &mut l.vecs { v.reserve(rows_per_list * dim); }
    }
    idx.dir.reserve(rows_per_list * idx.lists.len());
}

pub fn locate(idx: &IvfPqIndex, id: u64) -> Option<RowLoc> { idx.dir.get(&id).copied() }

pub fn live_rows(idx: &IvfPqIndex) -> usize { idx.dir.len() }

/// Full consistency check, O(rows). For tests and debug builds.
pub fn check_invariants(idx: &IvfPqIndex) -> Result<(), String> {
    let (m, dim) = (idx.params.m, idx.dim);
    let mut live = 0usize;
    for (li, l) in idx.lists.iter().enumerate() {
        let n = l.ids.len();
        if l.timestamps.len() != n || l.tomb.len() != n || l.tags.len() != n || l.codes.len() != n * m {
            return Err(format!("list {li}: column lengths differ"));
        }
        if let Some(v) = &l.vecs { if v.len() != n * dim { return Err(format!("list {li}: vecs length")); } }
        if l.head > n { return Err(format!("list {li}: head {} > len {n}", l.head)); }
        if l.tomb[..l.head].iter().any(|&t| t != 1) { return Err(format!("list {li}: live row before head")); }
        if l.timestamps.windows(2).any(|w| w[0] > w[1]) { return Err(format!("list {li}: timestamps not sorted")); }
        let dead = l.tomb.iter().filter(|&&t| t == 1).count();
        if dead != l.dead { return Err(format!("list {li}: dead {} but {dead} tombstones", l.dead)); }
        for r in 0..n {
            if l.tomb[r] == 1 { continue; }
            live += 1;
            match idx.dir.get(&l.ids[r]) {
                Some(loc) if loc.list as usize == li && loc.row as usize == r => {}
                other => return Err(format!("list {li} row {r} id {}: directory has {other:?}", l.ids[r])),
            }
            if l.timestamps[r] > idx.max_ts { return Err(format!("list {li} row {r}: ts above max_ts")); }
        }
    }
    if live != idx.dir.len() { return Err(format!("{live} live rows but {} directory entries", idx.dir.len())); }
    Ok(())
}
