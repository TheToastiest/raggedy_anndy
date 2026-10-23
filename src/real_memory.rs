//! Real memories: the pending buffer between the 1-hour short-term index
//! and the 12-hour commit to long-term storage (stage 5, index side).
//!
//! The game server decides what is real (endgame-only events) and says so
//! at insert time with `flags`. A real memory is written to the short-term
//! index like any other row, and also kept here, in RAM, with its f32
//! vector, so it stays searchable (exactly, Flat) after the 1-hour expiry
//! takes it out of the index. `RealOp::Commit` drains the buffer into an
//! `Outbox` of plain `RealRecord`s for the SQL side, which is not part of
//! this crate; `RealOp::Ack` tells this side the SQL writer has them.
//!
//! The server owns the clock: there is no timer and no wall-clock read
//! here. Commit and ack happen only when the server sends the op, and are
//! applied in log order like every other write.
//!
//! Rules the functions here keep:
//! - `flags` is its own field, never one of the 64 tag bits. `REAL` must be
//!   set; `LEGENDARY` (never expires in SQL) is a stricter subset of real.
//! - Every check happens before any change: an `Err` leaves the index, the
//!   buffer and the outbox as they were.
//! - Delete of a pending id drops its pending row: it is never committed.
//!   Upsert of a pending id replaces its pending row (the latest version is
//!   what gets committed) and keeps its flags; `UpsertReal` sets them.
//! - An id committed but not yet acked is "handed". Delete of a handed id
//!   appends a `Delete` record to the outbox at once; upsert of a handed id
//!   makes it pending again, so the next commit appends a newer `Put`.
//! - Every record carries `seq`, from one counter in `Pending`. The outbox
//!   skips a record whose `seq` it has already taken, so replaying ops from
//!   a snapshot (e.g. a commit re-run after a crash) appends nothing twice.
//! - Commit order is (ts asc, id asc).
//! - Merged search ranks by (score desc, id asc) and returns each id once.
//!   For an id in both the index and the buffer, the buffer's exact score
//!   is used (equal bits to the index's re-rank when it stores vectors).
//!
//! Long-term memories are not indexed here; another component reads them
//! with SQL at query time.

use std::collections::HashMap;

use crate::ivfpq::{apply_decay, norm_inplace, IvfPqIndex, RowLoc};
use crate::live::{self, Applied, Op, Scratch, WriteError};
use crate::metric::{self, Metric};
use crate::types::{stable_top_k, Hit};
use crate::views::{self, Writer};

/// Server says this memory is real: hold it until the commit.
pub const REAL: u8 = 1 << 0;
/// Real and never expires in SQL. Only valid together with `REAL`.
pub const LEGENDARY: u8 = 1 << 1;
const KNOWN: u8 = REAL | LEGENDARY;

/// A committed id the SQL side has not acked yet: its flags and the `seq`
/// of its latest `Put`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Handed {
    pub flags: u8,
    pub seq: u64,
}

/// Real memories not yet committed, plus what this side must remember about
/// committed ones until they are acked. Columns side by side (SoA), rows in
/// insert order, so `timestamps` never decreases. All of it is rebuilt by
/// replaying the op log.
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
    /// Committed, not yet acked. Lookup and `retain` only.
    pub handed: HashMap<u64, Handed>,
    /// `seq` of the next record this side emits.
    pub next_seq: u64,
}

/// What a record asks the SQL side to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordKind {
    /// Insert, or replace the row with a lower `seq`.
    Put,
    /// Remove the row. Only `seq`, `id` and `flags` (the deleted row's)
    /// are meaningful; `ts` and `tags` are 0 and `vec` is empty.
    Delete,
}

/// One change for the SQL writer, plain data. Apply in `seq` order; a
/// record with a `seq` at or below the last one applied is a resend.
#[derive(Clone, Debug, PartialEq)]
pub struct RealRecord {
    pub seq: u64,
    pub kind: RecordKind,
    pub id: u64,
    pub ts: u64,
    pub tags: u64,
    pub flags: u8,
    pub vec: Vec<f32>,
}

/// Records handed to the SQL side and not yet acked, in `seq` order.
/// `next_seq` is one past the highest `seq` ever taken, so acked and
/// trimmed records are still known and a replay cannot re-add them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Outbox {
    pub records: Vec<RealRecord>,
    pub next_seq: u64,
}

/// A write that may touch the pending buffer. `Live` ops behave as
/// `live::apply` on the index; for a pending or handed id they also update
/// the buffer (see the module rules), and `Insert` of such an id is refused.
#[derive(Clone, Debug, PartialEq)]
pub enum RealOp {
    Live(Op),
    /// New real memory: a short-term row plus a pending row.
    InsertReal { id: u64, ts: u64, tags: u64, flags: u8, vec: Vec<f32> },
    /// Upsert with explicit flags: the row becomes (or stays) a real memory
    /// with exactly these flags, whatever it was before.
    UpsertReal { id: u64, ts: u64, tags: u64, flags: u8, vec: Vec<f32> },
    /// Drain the pending buffer into the outbox. Sent by the server.
    Commit,
    /// The SQL side has every record with `seq <= through_seq`.
    Ack { through_seq: u64 },
}

#[derive(Clone, Debug, PartialEq)]
pub enum RealApplied {
    Live(Applied),
    InsertedReal { loc: RowLoc },
    /// A pending or handed id was upserted, or `UpsertReal` ran. `flags`
    /// is what the pending row now carries.
    UpsertedReal { loc: RowLoc, replaced: Option<RowLoc>, flags: u8 },
    /// A pending or handed id was deleted. `loc` is the index row, if it
    /// was still live; `outboxed` is true if a `Delete` record was emitted.
    DeletedReal { loc: Option<RowLoc>, was_pending: bool, outboxed: bool },
    /// Records appended to the outbox (records it already took are skipped).
    Committed { appended: usize },
    /// Outbox records dropped by the ack.
    Acked { dropped: usize },
}

/// Why a write was refused. Nothing was changed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RealError {
    Write(WriteError),
    BadFlags { id: u64, flags: u8 },
    /// Ack of a `seq` this side has not emitted.
    AckAhead { through_seq: u64, next_seq: u64 },
}

impl std::fmt::Display for RealError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{self:?}") }
}
impl std::error::Error for RealError {}

impl From<WriteError> for RealError {
    fn from(e: WriteError) -> Self { RealError::Write(e) }
}

/// The short-term rows a real-memory op writes to: a plain index with its
/// scratch (stage 1), or the stage 2 writer. Same results either way
/// (`views::apply` matches `live::apply`), so the buffer and outbox rules
/// below are written once.
pub enum Rows<'a> {
    Index(&'a mut IvfPqIndex, &'a mut Scratch),
    Writer(&'a mut Writer),
}

fn rows_insert(rows: &mut Rows, id: u64, ts: u64, tags: u64, v: &[f32]) -> Result<RowLoc, WriteError> {
    match rows { Rows::Index(idx, s) => live::insert(idx, s, id, ts, tags, v), Rows::Writer(w) => views::insert(w, id, ts, tags, v) }
}

fn rows_upsert(rows: &mut Rows, id: u64, ts: u64, tags: u64, v: &[f32]) -> Result<(RowLoc, Option<RowLoc>), WriteError> {
    match rows { Rows::Index(idx, s) => live::upsert(idx, s, id, ts, tags, v), Rows::Writer(w) => views::upsert(w, id, ts, tags, v) }
}

fn rows_delete(rows: &mut Rows, id: u64) -> Result<RowLoc, WriteError> {
    match rows { Rows::Index(idx, _) => live::delete(idx, id), Rows::Writer(w) => views::delete(w, id) }
}

fn rows_apply(rows: &mut Rows, op: &Op) -> Result<Applied, WriteError> {
    match rows { Rows::Index(idx, s) => live::apply(idx, s, op), Rows::Writer(w) => views::apply(w, op) }
}

/// Empty buffer with the index's metric and dim.
pub fn pending_for(idx: &IvfPqIndex) -> Pending {
    Pending {
        metric: idx.metric, dim: idx.dim,
        ids: Vec::new(), timestamps: Vec::new(), tags: Vec::new(), flags: Vec::new(),
        raw: Vec::new(), view: Vec::new(), dir: HashMap::new(),
        handed: HashMap::new(), next_seq: 0,
    }
}

/// Apply one op. See `RealOp` for each variant's meaning.
pub fn apply(idx: &mut IvfPqIndex, s: &mut Scratch, buf: &mut Pending, outbox: &mut Outbox, op: &RealOp) -> Result<RealApplied, RealError> {
    apply_rows(&mut Rows::Index(idx, s), buf, outbox, op)
}

/// `apply` on the stage 2 writer's rows.
pub fn apply_writer(w: &mut Writer, buf: &mut Pending, outbox: &mut Outbox, op: &RealOp) -> Result<RealApplied, RealError> {
    apply_rows(&mut Rows::Writer(w), buf, outbox, op)
}

/// `apply` on either storage.
pub fn apply_rows(rows: &mut Rows, buf: &mut Pending, outbox: &mut Outbox, op: &RealOp) -> Result<RealApplied, RealError> {
    match op {
        RealOp::Live(Op::Insert { id, .. }) if is_known(buf, *id) => Err(WriteError::DuplicateId { id: *id }.into()),
        RealOp::Live(Op::Upsert { id, ts, tags, vec }) => upsert_in(rows, buf, *id, *ts, *tags, None, vec),
        RealOp::Live(Op::Delete { id }) => delete_in(|id| rows_delete(rows, id), buf, outbox, *id),
        RealOp::Live(op) => Ok(RealApplied::Live(rows_apply(rows, op)?)),
        RealOp::InsertReal { id, ts, tags, flags, vec } => {
            insert_real_in(rows, buf, *id, *ts, *tags, *flags, vec).map(|loc| RealApplied::InsertedReal { loc })
        }
        RealOp::UpsertReal { id, ts, tags, flags, vec } => upsert_in(rows, buf, *id, *ts, *tags, Some(*flags), vec),
        RealOp::Commit => Ok(RealApplied::Committed { appended: commit(buf, outbox) }),
        RealOp::Ack { through_seq } => ack(buf, outbox, *through_seq).map(|dropped| RealApplied::Acked { dropped }),
    }
}

fn check_flags(id: u64, flags: u8) -> Result<(), RealError> {
    if flags & REAL == 0 || flags & !KNOWN != 0 { return Err(RealError::BadFlags { id, flags }); }
    Ok(())
}

/// Pending, or committed and not yet acked.
fn is_known(buf: &Pending, id: u64) -> bool { buf.dir.contains_key(&id) || buf.handed.contains_key(&id) }

/// Insert a real memory: a short-term index row (same as `live::insert`)
/// and a pending row. Fails if `id` is live in the index, pending, or
/// handed and not yet acked.
#[allow(clippy::too_many_arguments)]
pub fn insert_real(idx: &mut IvfPqIndex, s: &mut Scratch, buf: &mut Pending, id: u64, ts: u64, tags: u64, flags: u8, v: &[f32]) -> Result<RowLoc, RealError> {
    insert_real_in(&mut Rows::Index(idx, s), buf, id, ts, tags, flags, v)
}

#[allow(clippy::too_many_arguments)]
fn insert_real_in(rows: &mut Rows, buf: &mut Pending, id: u64, ts: u64, tags: u64, flags: u8, v: &[f32]) -> Result<RowLoc, RealError> {
    check_flags(id, flags)?;
    if is_known(buf, id) { return Err(WriteError::DuplicateId { id }.into()); }
    // live::insert checks dim, finiteness, ts order and index duplicates,
    // and changes nothing on error; the push below cannot fail.
    let loc = rows_insert(rows, id, ts, tags, v)?;
    push(buf, id, ts, tags, flags, v);
    Ok(loc)
}

/// Upsert in the index, and in the buffer if the id is real. `flags`
/// `Some` sets them (and makes any id real); `None` keeps the flags of the
/// pending or handed row, or, for an id that is neither, is a plain
/// `live::upsert`. The new pending row goes at the end (its ts is the
/// newest), the old one, if any, is removed.
#[allow(clippy::too_many_arguments)]
pub fn upsert(idx: &mut IvfPqIndex, s: &mut Scratch, buf: &mut Pending, id: u64, ts: u64, tags: u64, flags: Option<u8>, v: &[f32]) -> Result<RealApplied, RealError> {
    upsert_in(&mut Rows::Index(idx, s), buf, id, ts, tags, flags, v)
}

#[allow(clippy::too_many_arguments)]
fn upsert_in(rows: &mut Rows, buf: &mut Pending, id: u64, ts: u64, tags: u64, flags: Option<u8>, v: &[f32]) -> Result<RealApplied, RealError> {
    if let Some(f) = flags { check_flags(id, f)?; }
    let row = buf.dir.get(&id).map(|&r| r as usize);
    let kept = row.map(|r| buf.flags[r]).or_else(|| buf.handed.get(&id).map(|h| h.flags));
    let Some(new_flags) = flags.or(kept) else {
        let (loc, replaced) = rows_upsert(rows, id, ts, tags, v)?;
        return Ok(RealApplied::Live(Applied::Upserted { loc, replaced }));
    };
    // live::upsert does every check before any change.
    let (loc, replaced) = rows_upsert(rows, id, ts, tags, v)?;
    if let Some(r) = row { remove_row(buf, r); }
    push(buf, id, ts, tags, new_flags, v);
    Ok(RealApplied::UpsertedReal { loc, replaced, flags: new_flags })
}

/// Delete in the index, and in the buffer if the id is real: a pending row
/// is dropped (never committed), a handed id gets a `Delete` record in the
/// outbox. A real id already expired from the index still deletes; an id
/// that is neither real nor live is `NotFound`.
pub fn delete(idx: &mut IvfPqIndex, buf: &mut Pending, outbox: &mut Outbox, id: u64) -> Result<RealApplied, RealError> {
    delete_in(|id| live::delete(idx, id), buf, outbox, id)
}

/// `delete` with the index-side delete passed in (`live::delete` or `views::delete`).
fn delete_in(mut del: impl FnMut(u64) -> Result<RowLoc, WriteError>, buf: &mut Pending, outbox: &mut Outbox, id: u64) -> Result<RealApplied, RealError> {
    let row = buf.dir.get(&id).map(|&r| r as usize);
    let handed = buf.handed.get(&id).copied();
    if row.is_none() && handed.is_none() {
        return Ok(RealApplied::Live(Applied::Deleted { loc: del(id)? }));
    }
    let loc = del(id).ok();
    let mut flags = handed.map(|h| h.flags).unwrap_or(0);
    if let Some(r) = row {
        flags = buf.flags[r];
        remove_row(buf, r);
    }
    if handed.is_some() {
        buf.handed.remove(&id);
        let rec = RealRecord { seq: buf.next_seq, kind: RecordKind::Delete, id, ts: 0, tags: 0, flags, vec: Vec::new() };
        buf.next_seq += 1;
        emit(outbox, rec);
    }
    Ok(RealApplied::DeletedReal { loc, was_pending: row.is_some(), outboxed: handed.is_some() })
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

/// Remove row `r`, keeping the others in order (so `timestamps` stays
/// sorted), and repoint the directory for the rows after it. O(pending);
/// the buffer holds only endgame events, so it is small.
fn remove_row(buf: &mut Pending, r: usize) {
    let dim = buf.dim;
    buf.dir.remove(&buf.ids[r]);
    buf.ids.remove(r);
    buf.timestamps.remove(r);
    buf.tags.remove(r);
    buf.flags.remove(r);
    buf.raw.drain(r * dim..(r + 1) * dim);
    buf.view.drain(r * dim..(r + 1) * dim);
    for (i, id) in buf.ids.iter().enumerate().skip(r) { buf.dir.insert(*id, i as u32); }
}

/// Append `rec` unless the outbox already took its `seq`. Returns whether
/// it was appended.
fn emit(outbox: &mut Outbox, rec: RealRecord) -> bool {
    if rec.seq < outbox.next_seq { return false; }
    outbox.next_seq = rec.seq + 1;
    outbox.records.push(rec);
    true
}

/// Move every pending row into `outbox` as a `Put`, in (ts asc, id asc)
/// order, each with the next `seq`; mark each id handed and empty the
/// buffer. Returns how many records were appended: a re-run from a buffer
/// restored to before the commit appends nothing to the same outbox.
pub fn commit(buf: &mut Pending, outbox: &mut Outbox) -> usize {
    let dim = buf.dim;
    let mut order: Vec<usize> = (0..buf.ids.len()).collect();
    order.sort_unstable_by_key(|&r| (buf.timestamps[r], buf.ids[r]));
    let mut appended = 0;
    for r in order {
        let seq = buf.next_seq;
        buf.next_seq += 1;
        buf.handed.insert(buf.ids[r], Handed { flags: buf.flags[r], seq });
        let rec = RealRecord {
            seq, kind: RecordKind::Put, id: buf.ids[r], ts: buf.timestamps[r], tags: buf.tags[r], flags: buf.flags[r],
            vec: buf.raw[r * dim..(r + 1) * dim].to_vec(),
        };
        appended += emit(outbox, rec) as usize;
    }
    clear(buf);
    appended
}

/// The SQL side has every record with `seq <= through_seq`: forget those
/// handed ids (unless a later `Put` is still unacked) and drop those
/// records from the outbox. Returns how many records were dropped.
pub fn ack(buf: &mut Pending, outbox: &mut Outbox, through_seq: u64) -> Result<usize, RealError> {
    if through_seq >= buf.next_seq { return Err(RealError::AckAhead { through_seq, next_seq: buf.next_seq }); }
    buf.handed.retain(|_, h| h.seq > through_seq);
    let before = outbox.records.len();
    outbox.records.retain(|r| r.seq > through_seq);
    Ok(before - outbox.records.len())
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

/// Committed and not yet acked.
pub fn is_handed(buf: &Pending, id: u64) -> bool { buf.handed.contains_key(&id) }

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
    for (id, h) in &buf.handed {
        if h.seq >= buf.next_seq { return Err(format!("handed id {id}: seq {} not below next_seq {}", h.seq, buf.next_seq)); }
        if check_flags(*id, h.flags).is_err() { return Err(format!("handed id {id}: bad flags {:#b}", h.flags)); }
    }
    Ok(())
}
