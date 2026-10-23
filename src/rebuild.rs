//! Background rebuild with a deterministic, logged model swap (stage 3).
//!
//! An epoch is one frozen model (centroids, OPQ, PQ codebooks). When drift
//! signals fire (see `drift`), the writer takes a `snapshot` of the live rows
//! at log position `seq`. A worker trains the next epoch's model from it
//! (`rebuild`): seed `epoch_seed(base_seed, epoch + 1)`, rows in ascending id
//! order, the same deterministic build as epoch 0, so the result is identical
//! for any thread count. The writer keeps applying ops to the old index and
//! appending them to the log; `write` also records what each one did as
//! layout-independent ops (`Carry`). When the worker is done, `catch_up`
//! applies those to the new index, and `swap` replaces the old index and
//! returns a `SwapModel` entry for the log.
//!
//! Why carry effects and not re-run the ops: a capped `Expire` walks lists
//! in index order, so on a different layout it would kill different rows
//! than the ones the live index killed and reported. Carried, an expiry is a
//! `Delete` of each id it killed; a `Compact` (storage only) carries nothing;
//! a refused op carries nothing (it is refused on both: same live ids, same
//! `max_ts`). Inserts, upserts and deletes carry as they are.
//!
//! The new model depends only on the snapshot (that is, on the log up to
//! `snapshot_seq`), never on when the worker finishes. Before a swap the
//! worker runs the swap gate on it (`gate::evaluate`, also a pure function of
//! the snapshot): a candidate that is not better is dropped with `reject`,
//! which logs a `RejectModel` entry and keeps the old model. Where the swap lands
//! is up to the writer and is recorded in the log, so `replay` of a log that
//! contains `SwapModel` entries rebuilds the same model at the same place
//! and ends in the same state, byte for byte.
//!
//! Positions: `seq` counts log entries applied. A snapshot at `seq = s` sees
//! entries `log[..s]`; the carry covers `log[s..]` up to the swap. A
//! `SwapModel` or `RejectModel` entry takes one position like any other.
//!
//! Cadence: `due` holds a fired signal until `Thresholds::min_ops_per_epoch`
//! writes have been applied since the last swap or reject. Triggers are
//! not logged (the decision entries name their `snapshot_seq`), so the
//! trigger policy, e.g. later hysteresis, can change without changing the
//! log format or replay.

use std::collections::BTreeMap;

use crate::drift::{self, DriftBaseline, DriftStats, Signals, Thresholds};
use crate::gate::{self, Gate};
use crate::ivfpq::{build_with_threads, IvfPqIndex, IvfPqParams};
use crate::live::{self, Applied, Op, Scratch, WriteError};
use crate::metric::Metric;
use crate::seed::SplitMix64;

/// One log entry: a write, or a decision on a rebuilt model.
#[derive(Clone, Debug, PartialEq)]
pub enum LogOp {
    Write(Op),
    /// Replace the model with the one trained from the snapshot at
    /// `snapshot_seq`, caught up to this entry. `model_hash` and `gate`
    /// (which accepted it) are checked on replay.
    SwapModel { epoch: u64, snapshot_seq: u64, model_hash: u64, gate: Gate },
    /// The model trained from the snapshot at `snapshot_seq` lost the gate
    /// and was dropped; the old model stays. Checked on replay like a swap.
    RejectModel { epoch: u64, snapshot_seq: u64, model_hash: u64, gate: Gate },
}

/// Why a rebuild, swap or replay was refused. Nothing was changed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RebuildError {
    /// The index does not keep raw vectors (`store_vecs = false`).
    NoRawVectors,
    TooFewRows { rows: usize, nlist: usize },
    /// A snapshot was taken and neither swapped nor cancelled: one rebuild at a time.
    RebuildPending { snapshot_seq: u64 },
    /// Swap or catch-up without a matching snapshot.
    NoRebuildPending,
    WrongSnapshot { got: u64, want: u64 },
    /// The rebuilt index has applied `carried` of the `pending` carried ops.
    NotCaughtUp { carried: usize, pending: usize },
    WrongEpoch { got: u64, want: u64 },
    /// A carried op was refused by the rebuilt index. Never expected: it
    /// means the two indexes disagree on the live set.
    CarryRefused { at: usize, err: WriteError },
    HashMismatch { seq: u64, logged: u64, rebuilt: u64 },
    /// The decision entry at `seq` names a snapshot position not before it.
    BadSnapshotSeq { seq: u64, snapshot_seq: u64 },
    /// The gate runs on the rebuild as built; this one has applied `carried` ops.
    GateAfterCatchUp { carried: usize },
    /// `swap` with a rejecting gate, or `reject` with an accepting one.
    GateDecision { accept: bool },
    /// The gate recomputed on replay differs from the logged one.
    GateMismatch { seq: u64 },
}

impl std::fmt::Display for RebuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{self:?}") }
}
impl std::error::Error for RebuildError {}

/// The writer's state: the index, its drift stats, and its log position.
#[derive(Clone, Debug)]
pub struct Live {
    pub idx: IvfPqIndex,
    pub scratch: Scratch,
    /// Log entries applied so far.
    pub seq: u64,
    /// `seq` just after the last `SwapModel` or `RejectModel`; 0 in epoch 0
    /// until the first decision. `due` counts ops from here.
    pub model_seq: u64,
    pub epoch: u64,
    /// Seed of epoch 0; every epoch's seed derives from it.
    pub base_seed: u64,
    pub baseline: DriftBaseline,
    pub stats: DriftStats,
    /// Set by `snapshot` until `swap` or `cancel`.
    pub carry: Option<Carry>,
}

/// What the writes since a snapshot did, as ops that mean the same on any
/// layout. Applied in order by `catch_up`.
#[derive(Clone, Debug, PartialEq)]
pub struct Carry {
    pub snapshot_seq: u64,
    pub ops: Vec<Op>,
}

/// Live rows at one log position, ascending id, raw (stored) vectors, plus
/// the current model and its baseline for the swap gate.
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub seq: u64,
    pub epoch: u64,
    /// The current model with no rows (`live::empty_like`).
    pub model: IvfPqIndex,
    pub baseline: DriftBaseline,
    /// `max_ts` of the index; the rebuilt index keeps it so the same writes
    /// are refused before and after the swap.
    pub max_ts: u64,
    pub metric: Metric,
    pub dim: usize,
    pub params: IvfPqParams,
    pub ids: Vec<u64>,
    pub ts: Vec<u64>,
    pub tags: Vec<u64>,
    /// Row-major, `dim` per row.
    pub vecs: Vec<f32>,
}

/// A next-epoch index, where its snapshot was taken, and how many carried
/// ops it has applied. `next.seq` becomes the log position at `swap`.
#[derive(Clone, Debug)]
pub struct Rebuilt {
    pub next: Live,
    pub snapshot_seq: u64,
    pub carried: usize,
}

/// Epoch 0 on a freshly built index (needs `store_vecs`).
pub fn start(idx: IvfPqIndex) -> Live {
    Live {
        scratch: live::scratch_for(&idx), seq: 0, model_seq: 0, epoch: 0, base_seed: idx.params.seed,
        baseline: drift::baseline(&idx), stats: drift::new_stats(&idx), carry: None, idx,
    }
}

/// Apply one write and account for it (drift stats, and the carry while a
/// rebuild is pending). The caller appends `LogOp::Write(op.clone())` to its log.
pub fn write(st: &mut Live, op: &Op) -> Result<Applied, WriteError> {
    let r = live::apply(&mut st.idx, &mut st.scratch, op);
    drift::record(&mut st.stats, &st.idx, &st.scratch, &r);
    if let Some(c) = &mut st.carry { carry_effect(&mut c.ops, op, &r); }
    st.seq += 1;
    r
}

/// Writes applied since the last swap or reject (since the start in epoch 0).
pub fn ops_since_model(st: &Live) -> u64 { st.seq - st.model_seq }

/// Whether to take a snapshot now: no rebuild pending, a signal fired, and
/// at least `th.min_ops_per_epoch` writes since the last model decision.
/// Pure function of the state, so it holds at the same op on every run.
pub fn due(st: &Live, sig: &Signals, th: &Thresholds) -> bool {
    st.carry.is_none() && sig.fired != 0 && ops_since_model(st) >= th.min_ops_per_epoch
}

/// Append what `op` did (`result`) as layout-independent ops.
fn carry_effect(out: &mut Vec<Op>, op: &Op, result: &Result<Applied, WriteError>) {
    match (op, result) {
        (Op::Insert { .. }, Ok(_)) | (Op::Upsert { .. }, Ok(_)) | (Op::Delete { .. }, Ok(_)) => out.push(op.clone()),
        (Op::Expire { .. }, Ok(Applied::Expired { ids, .. })) => out.extend(ids.iter().map(|&id| Op::Delete { id })),
        _ => {}
    }
}

/// Seed of `epoch`'s model. Epoch 0 is the base seed itself.
pub fn epoch_seed(base_seed: u64, epoch: u64) -> u64 {
    if epoch == 0 { return base_seed; }
    SplitMix64::new(base_seed ^ epoch.wrapping_mul(0x9E37_79B9_7F4A_7C15)).next_u64()
}

/// Copy of the live rows at the current position, and start carrying writes
/// for the rebuild. O(live rows); runs on the writer thread.
pub fn snapshot(st: &mut Live) -> Result<Snapshot, RebuildError> {
    if let Some(c) = &st.carry { return Err(RebuildError::RebuildPending { snapshot_seq: c.snapshot_seq }); }
    let idx = &st.idx;
    if !idx.params.store_vecs { return Err(RebuildError::NoRawVectors); }
    let dim = idx.dim;
    let mut rows: Vec<(u64, u32, u32)> = Vec::with_capacity(live::live_rows(idx));
    for (li, l) in idx.lists.iter().enumerate() {
        for r in l.head..l.len() {
            if l.tomb[r] == 0 { rows.push((l.ids[r], li as u32, r as u32)); }
        }
    }
    rows.sort_unstable_by_key(|r| r.0); // ids of live rows are unique
    let n = rows.len();
    let mut snap = Snapshot {
        seq: st.seq, epoch: st.epoch, model: live::empty_like(idx), baseline: st.baseline.clone(), max_ts: idx.max_ts, metric: idx.metric, dim, params: idx.params,
        ids: Vec::with_capacity(n), ts: Vec::with_capacity(n), tags: Vec::with_capacity(n), vecs: Vec::with_capacity(n * dim),
    };
    for (id, li, r) in rows {
        let l = &idx.lists[li as usize];
        snap.ids.push(id);
        snap.ts.push(l.timestamps[r as usize]);
        snap.tags.push(l.tags[r as usize]);
        snap.vecs.extend_from_slice(l.vec_row(dim, r as usize));
    }
    st.carry = Some(Carry { snapshot_seq: st.seq, ops: Vec::new() });
    Ok(snap)
}

/// Drop a pending rebuild (e.g. `rebuild` failed). The index is untouched.
pub fn cancel(st: &mut Live) { st.carry = None; }

/// Train the next epoch's model on `snap` and encode it. Safe to run on a
/// worker thread; the result is identical for every `threads` value.
pub fn rebuild(snap: &Snapshot, base_seed: u64, threads: usize) -> Result<Rebuilt, RebuildError> {
    let n = snap.ids.len();
    if !snap.params.store_vecs { return Err(RebuildError::NoRawVectors); }
    if n == 0 || n < snap.params.nlist { return Err(RebuildError::TooFewRows { rows: n, nlist: snap.params.nlist }); }
    let dim = snap.dim;
    let epoch = snap.epoch + 1;
    let mut params = snap.params;
    params.seed = epoch_seed(base_seed, epoch);
    let data: Vec<(u64, Vec<f32>, u64)> = (0..n).map(|i| (snap.ids[i], snap.vecs[i * dim..(i + 1) * dim].to_vec(), snap.ts[i])).collect();
    let mut idx = build_with_threads(snap.metric, dim, &data, params, threads.max(1));
    drop(data);
    // Build writes no tags; put them back.
    for i in 0..n {
        let loc = idx.dir[&snap.ids[i]];
        idx.lists[loc.list as usize].tags[loc.row as usize] = snap.tags[i];
    }
    idx.max_ts = snap.max_ts;
    let next = Live {
        scratch: live::scratch_for(&idx), seq: snap.seq, model_seq: snap.seq, epoch, base_seed,
        baseline: drift::baseline(&idx), stats: drift::new_stats(&idx), carry: None, idx,
    };
    Ok(Rebuilt { next, snapshot_seq: snap.seq, carried: 0 })
}

/// Apply the carried ops `r` has not applied yet. May be called repeatedly
/// as the carry grows; the result does not depend on how it is split.
/// They count towards the new epoch's drift stats.
pub fn catch_up(r: &mut Rebuilt, carry: &Carry) -> Result<(), RebuildError> {
    if carry.snapshot_seq != r.snapshot_seq { return Err(RebuildError::WrongSnapshot { got: carry.snapshot_seq, want: r.snapshot_seq }); }
    for (at, op) in carry.ops.iter().enumerate().skip(r.carried) {
        if let Err(err) = write(&mut r.next, op) { return Err(RebuildError::CarryRefused { at, err }); }
        r.carried = at + 1;
    }
    Ok(())
}

/// Replace `st` with the caught-up rebuild that `gate` accepted and return
/// the `SwapModel` entry to append to the log (it takes position `st.seq`).
pub fn swap(st: &mut Live, r: Rebuilt, gate: Gate) -> Result<LogOp, RebuildError> {
    if !gate.accept { return Err(RebuildError::GateDecision { accept: false }); }
    let carry = st.carry.as_ref().ok_or(RebuildError::NoRebuildPending)?;
    if carry.snapshot_seq != r.snapshot_seq { return Err(RebuildError::WrongSnapshot { got: r.snapshot_seq, want: carry.snapshot_seq }); }
    if r.carried != carry.ops.len() { return Err(RebuildError::NotCaughtUp { carried: r.carried, pending: carry.ops.len() }); }
    if r.next.epoch != st.epoch + 1 { return Err(RebuildError::WrongEpoch { got: r.next.epoch, want: st.epoch + 1 }); }
    let entry = LogOp::SwapModel { epoch: r.next.epoch, snapshot_seq: r.snapshot_seq, model_hash: model_hash(&r.next.idx), gate };
    let seq = st.seq + 1;
    *st = r.next;
    st.seq = seq;
    st.model_seq = seq;
    Ok(entry)
}

/// Drop a rebuild that `gate` rejected: end the carry, keep the old model,
/// and return the `RejectModel` entry for the log (position `st.seq`). No
/// catch-up needed. Drift stats carry on, and `due` waits another
/// `min_ops_per_epoch` writes before the next try.
pub fn reject(st: &mut Live, r: &Rebuilt, gate: Gate) -> Result<LogOp, RebuildError> {
    let carry = st.carry.as_ref().ok_or(RebuildError::NoRebuildPending)?;
    if carry.snapshot_seq != r.snapshot_seq { return Err(RebuildError::WrongSnapshot { got: r.snapshot_seq, want: carry.snapshot_seq }); }
    if r.next.epoch != st.epoch + 1 { return Err(RebuildError::WrongEpoch { got: r.next.epoch, want: st.epoch + 1 }); }
    if gate.accept { return Err(RebuildError::GateDecision { accept: true }); }
    let entry = LogOp::RejectModel { epoch: r.next.epoch, snapshot_seq: r.snapshot_seq, model_hash: model_hash(&r.next.idx), gate };
    st.carry = None;
    st.seq += 1;
    st.model_seq = st.seq;
    Ok(entry)
}

/// Replay `log` from the epoch-0 index `base`. Snapshots are taken at the
/// positions the decision entries name; each one rebuilds, re-runs the gate
/// with the logged parameters and game verdict, and checks the model hash
/// and the whole gate against the log, then swaps (after catch-up) or
/// rejects as logged. Same log, same state as the live run.
pub fn replay(base: IvfPqIndex, log: &[LogOp], threads: usize) -> Result<Live, RebuildError> {
    let mut wanted: Vec<u64> = Vec::new();
    for (i, e) in log.iter().enumerate() {
        if let LogOp::SwapModel { snapshot_seq, .. } | LogOp::RejectModel { snapshot_seq, .. } = e {
            if *snapshot_seq > i as u64 { return Err(RebuildError::BadSnapshotSeq { seq: i as u64, snapshot_seq: *snapshot_seq }); }
            wanted.push(*snapshot_seq);
        }
    }
    let mut st = start(base);
    let mut snaps: BTreeMap<u64, Snapshot> = BTreeMap::new();
    for (i, e) in log.iter().enumerate() {
        if wanted.contains(&st.seq) && !snaps.contains_key(&st.seq) { snaps.insert(st.seq, snapshot(&mut st)?); }
        match e {
            LogOp::Write(op) => { let _ = write(&mut st, op); }
            LogOp::SwapModel { epoch, snapshot_seq, model_hash: logged, gate: lg }
            | LogOp::RejectModel { epoch, snapshot_seq, model_hash: logged, gate: lg } => {
                let seq = i as u64;
                let snap = snaps.remove(snapshot_seq).ok_or(RebuildError::BadSnapshotSeq { seq, snapshot_seq: *snapshot_seq })?;
                let mut r = rebuild(&snap, st.base_seed, threads)?;
                if r.next.epoch != *epoch { return Err(RebuildError::WrongEpoch { got: *epoch, want: r.next.epoch }); }
                let h = model_hash(&r.next.idx);
                if h != *logged { return Err(RebuildError::HashMismatch { seq, logged: *logged, rebuilt: h }); }
                let g = gate::evaluate(&snap, &r, &lg.params, lg.game, threads)?;
                drop(snap);
                if g != *lg { return Err(RebuildError::GateMismatch { seq }); }
                if matches!(e, LogOp::SwapModel { .. }) {
                    catch_up(&mut r, st.carry.as_ref().ok_or(RebuildError::NoRebuildPending)?)?;
                    swap(&mut st, r, g)?;
                } else {
                    reject(&mut st, &r, g)?;
                }
            }
        }
    }
    Ok(st)
}

/// FNV-1a fold.
#[inline] fn fnv(h: u64, x: u64) -> u64 { (h ^ x).wrapping_mul(0x100000001b3) }

/// FNV-1a over the model only: shape, centroids, OPQ, codebooks. Rows are
/// not included (see `IvfPqIndex::fingerprint`).
pub fn model_hash(idx: &IvfPqIndex) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    h = fnv(h, matches!(idx.metric, Metric::Cosine) as u64);
    h = fnv(h, idx.dim as u64);
    h = fnv(h, idx.centroids.len() as u64);
    h = fnv(h, idx.params.m as u64);
    h = fnv(h, idx.pq.nbits as u64);
    for c in &idx.centroids { for &x in c { h = fnv(h, x.to_bits() as u64); } }
    if let Some(o) = &idx.opq { for x in o.fingerprint_bits() { h = fnv(h, x); } }
    for cb in &idx.pq.codebooks { for &x in cb { h = fnv(h, x.to_bits() as u64); } }
    h
}
