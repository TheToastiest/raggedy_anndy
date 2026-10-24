//! The writer thread's whole state, stages 2, 3 and 5 together: what the
//! op log (stage 4) replays and what a durable snapshot captures.
//!
//! - Rows: the stage 2 `views::Writer` (arena blocks, list metadata,
//!   `dead_since`, the id directory, the tick `w.seq`). Queries read its
//!   published views.
//! - Epoch: the stage 3 model state (`State::epoch`, `base_seed`,
//!   `model_seq`, drift `baseline` and `stats`) and, while a rebuild is
//!   pending, its `carry`. The model itself is `w.model`.
//! - Real memories: the stage 5 pending buffer and outbox, with their seq
//!   counters.
//!
//! Log entries: every write is a `RealOp` (`RealOp::Live(op)` for a plain
//! write, policy-issued `Compact`s included); model decisions are
//! `rebuild::LogOp::SwapModel` / `RejectModel`. `State::seq` counts the
//! entries applied; refused writes change nothing and are not entries.
//!
//! A pending rebuild (`carry`) is in-flight work, not log state: triggers
//! are not logged (stage 3), so a rebuild whose decision never reached the
//! log is lost in a crash and the trigger fires again. `fingerprint`
//! covers everything else.

use std::sync::Arc;

use crate::drift::{self, DriftBaseline, DriftStats};
use crate::gate::{self, Gate};
use crate::ivfpq::{fnv, IvfPqIndex};
use crate::live::{self, Applied, Op};
use crate::metric::Metric;
use crate::real_memory::{self, Outbox, Pending, RealApplied, RealError, RealOp, RecordKind};
use crate::rebuild::{self, Carry, LogOp, RebuildError, Rebuilt};
use crate::views::{self, View, Writer, NEVER};

/// Everything but the rows and the pending rebuild. Plain data.
#[derive(Clone, Debug, PartialEq)]
pub struct State {
    /// Log entries applied.
    pub seq: u64,
    /// `seq` just after the last `SwapModel` or `RejectModel` (0 until the
    /// first); `rebuild::due`'s op count starts here.
    pub model_seq: u64,
    pub epoch: u64,
    /// Seed of epoch 0; every epoch's seed derives from it.
    pub base_seed: u64,
    pub baseline: DriftBaseline,
    pub stats: DriftStats,
    pub buf: Pending,
    pub outbox: Outbox,
}

/// The writer thread's state. Owned by the writer thread.
pub struct System {
    pub w: Writer,
    pub st: State,
    /// Set by `begin_rebuild` until `swap`, `reject` or `cancel`.
    pub carry: Option<Carry>,
}

/// What a durable snapshot takes on the writer thread: the view published
/// for the tick that just ended (rows never move, so another thread can
/// read them while the writer goes on) plus everything outside the rows.
/// O(lists + pending + outbox); no row is copied.
#[derive(Clone)]
pub struct Capture {
    pub view: Arc<View>,
    /// The writer's tick (`w.seq`): the tick after the view's.
    pub tick_next: u64,
    pub dead_since: Vec<u64>,
    pub st: State,
}

/// Why a logged entry could not be replayed.
#[derive(Clone, Debug, PartialEq)]
pub enum ReplayError {
    Write(RealError),
    Model(RebuildError),
    /// The log's tick is not the writer's.
    Tick { want: u64, got: u64 },
}

/// Epoch 0 on a freshly built index (needs `store_vecs`, as stage 3 does),
/// rows in blocks of `block_rows`. Nothing applied: `st.seq` 0, tick 1.
pub fn start(idx: &IvfPqIndex, block_rows: usize) -> System {
    System {
        w: views::writer_from(idx, block_rows),
        st: State {
            seq: 0, model_seq: 0, epoch: 0, base_seed: idx.params.seed,
            baseline: drift::baseline(idx), stats: drift::new_stats(idx),
            buf: real_memory::pending_for(idx), outbox: Outbox::default(),
        },
        carry: None,
    }
}

/// A system from its parts (a loaded snapshot): rows `idx` (directory
/// rebuilt), the writer's tick and `dead_since`, and the rest. The
/// pending buffer's rows go to the writer's view side too, so the first
/// view already searches them.
pub fn from_parts(idx: &IvfPqIndex, block_rows: usize, tick_next: u64, dead_since: Vec<u64>, st: State) -> System {
    let b = &st.buf;
    let mut w = views::writer_with_pending(idx, block_rows, &b.ids, &b.timestamps, &b.tags, &b.view);
    w.seq = tick_next;
    w.dead_since = dead_since;
    System { w, st, carry: None }
}

/// What `r` did to the index rows, as `live::apply` would have said it;
/// `None` if it touched no row (commit, ack, delete of a real id whose row
/// already expired).
fn index_effect(r: &RealApplied) -> Option<Applied> {
    match r {
        RealApplied::Live(a) => Some(a.clone()),
        RealApplied::InsertedReal { loc } => Some(Applied::Inserted { loc: *loc }),
        RealApplied::UpsertedReal { loc, replaced, .. } => Some(Applied::Upserted { loc: *loc, replaced: *replaced }),
        RealApplied::DeletedReal { loc: Some(loc), .. } => Some(Applied::Deleted { loc: *loc }),
        RealApplied::DeletedReal { loc: None, .. } | RealApplied::Committed { .. } | RealApplied::Acked { .. } => None,
    }
}

/// Append what `op` did to the index (`a`) as layout-independent ops, as
/// `rebuild::write` does for plain ops: inserts and upserts as plain
/// inserts and upserts (the buffer is not part of the rebuild), deletes as
/// they are, an expiry as a delete of each id it killed, nothing for a
/// compaction.
fn carry_effect(out: &mut Vec<Op>, op: &RealOp, a: &Applied) {
    match (op, a) {
        (RealOp::Live(op), Applied::Inserted { .. } | Applied::Upserted { .. } | Applied::Deleted { .. }) => out.push(op.clone()),
        (RealOp::InsertReal { id, ts, tags, vec, .. }, Applied::Inserted { .. }) => out.push(Op::Insert { id: *id, ts: *ts, tags: *tags, vec: vec.clone() }),
        (RealOp::UpsertReal { id, ts, tags, vec, .. }, Applied::Upserted { .. }) => out.push(Op::Upsert { id: *id, ts: *ts, tags: *tags, vec: vec.clone() }),
        (_, Applied::Expired { ids, .. }) => out.extend(ids.iter().map(|&id| Op::Delete { id })),
        _ => {}
    }
}

/// Apply one write: rows (stage 2), buffer and outbox (stage 5), drift
/// stats and the carry (stage 3). A refused write changes nothing and is
/// not counted. The caller logs accepted ones (`durable::write` does both).
pub fn write(sys: &mut System, op: &RealOp) -> Result<RealApplied, RealError> {
    let r = real_memory::apply_writer(&mut sys.w, &mut sys.st.buf, &mut sys.st.outbox, op)?;
    if let Some(a) = index_effect(&r) {
        let a = Ok(a);
        drift::record(&mut sys.st.stats, &sys.w.model, views::scratch(&sys.w), &a);
        if let (Some(c), Ok(a)) = (&mut sys.carry, &a) { carry_effect(&mut c.ops, op, a); }
    }
    sys.st.seq += 1;
    Ok(r)
}

/// Copy of the live rows for a rebuild (as `rebuild::snapshot`, from the
/// writer's blocks), and start carrying writes. O(live rows); writer thread.
pub fn begin_rebuild(sys: &mut System) -> Result<rebuild::Snapshot, RebuildError> {
    if let Some(c) = &sys.carry { return Err(RebuildError::RebuildPending { snapshot_seq: c.snapshot_seq }); }
    let (w, model) = (&sys.w, &*sys.w.model);
    if !model.params.store_vecs { return Err(RebuildError::NoRawVectors); }
    let (dim, cap) = (model.dim, views::block_rows(w));
    let mut rows: Vec<(u64, u32, u32)> = Vec::with_capacity(views::live_rows(w));
    for (li, l) in w.lists.iter().enumerate() {
        for r in l.head..l.len {
            if l.tomb[r] == 0 { rows.push((l.blocks[r / cap].id(r % cap), li as u32, r as u32)); }
        }
    }
    rows.sort_unstable_by_key(|r| r.0); // ids of live rows are unique
    let n = rows.len();
    let mut snap = rebuild::Snapshot {
        seq: sys.st.seq, epoch: sys.st.epoch, model: model.clone(), baseline: sys.st.baseline.clone(), max_ts: w.max_ts,
        metric: model.metric, dim, params: model.params,
        ids: Vec::with_capacity(n), ts: Vec::with_capacity(n), tags: Vec::with_capacity(n), vecs: Vec::with_capacity(n * dim),
    };
    for (id, li, r) in rows {
        let (b, s) = (&w.lists[li as usize].blocks[r as usize / cap], r as usize % cap);
        snap.ids.push(id);
        snap.ts.push(b.ts(s));
        snap.tags.push(b.tags(s));
        snap.vecs.extend_from_slice(b.vec_row(s));
    }
    sys.carry = Some(Carry { snapshot_seq: sys.st.seq, ops: Vec::new() });
    Ok(snap)
}

/// Drop a pending rebuild (e.g. `rebuild::rebuild` failed). Nothing logged.
pub fn cancel(sys: &mut System) { sys.carry = None; }

/// Swap in the caught-up rebuild `gate` accepted (as `rebuild::swap`):
/// the writer takes its model and rows (`views::rebase`), the epoch state
/// its baseline and stats. Returns the `SwapModel` entry to log; it takes
/// position `st.seq`.
pub fn swap(sys: &mut System, r: Rebuilt, gate: Gate) -> Result<LogOp, RebuildError> {
    if !gate.accept { return Err(RebuildError::GateDecision { accept: false }); }
    let carry = sys.carry.as_ref().ok_or(RebuildError::NoRebuildPending)?;
    if carry.snapshot_seq != r.snapshot_seq { return Err(RebuildError::WrongSnapshot { got: r.snapshot_seq, want: carry.snapshot_seq }); }
    if r.carried != carry.ops.len() { return Err(RebuildError::NotCaughtUp { carried: r.carried, pending: carry.ops.len() }); }
    if r.next.epoch != sys.st.epoch + 1 { return Err(RebuildError::WrongEpoch { got: r.next.epoch, want: sys.st.epoch + 1 }); }
    let entry = LogOp::SwapModel { epoch: r.next.epoch, snapshot_seq: r.snapshot_seq, model_hash: rebuild::model_hash(&r.next.idx), gate };
    views::rebase(&mut sys.w, &r.next.idx);
    let st = &mut sys.st;
    st.seq += 1;
    st.model_seq = st.seq;
    st.epoch = r.next.epoch;
    st.baseline = r.next.baseline;
    st.stats = r.next.stats;
    sys.carry = None;
    Ok(entry)
}

/// Drop a rebuild `gate` rejected (as `rebuild::reject`): end the carry,
/// keep the model. Returns the `RejectModel` entry to log (position `st.seq`).
pub fn reject(sys: &mut System, r: &Rebuilt, gate: Gate) -> Result<LogOp, RebuildError> {
    let carry = sys.carry.as_ref().ok_or(RebuildError::NoRebuildPending)?;
    if carry.snapshot_seq != r.snapshot_seq { return Err(RebuildError::WrongSnapshot { got: r.snapshot_seq, want: carry.snapshot_seq }); }
    if r.next.epoch != sys.st.epoch + 1 { return Err(RebuildError::WrongEpoch { got: r.next.epoch, want: sys.st.epoch + 1 }); }
    if gate.accept { return Err(RebuildError::GateDecision { accept: true }); }
    let entry = LogOp::RejectModel { epoch: r.next.epoch, snapshot_seq: r.snapshot_seq, model_hash: rebuild::model_hash(&r.next.idx), gate };
    sys.carry = None;
    sys.st.seq += 1;
    sys.st.model_seq = sys.st.seq;
    Ok(entry)
}

/// Replay a logged decision (as in `rebuild::replay`): rebuild from `snap`
/// (taken by `begin_rebuild` at the entry's `snapshot_seq`), check the
/// epoch and model hash, re-run the gate with the logged parameters and
/// game verdict and check it equals the logged one, then swap (after
/// catch-up) or reject as logged.
pub fn replay_decision(sys: &mut System, e: &LogOp, snap: &rebuild::Snapshot, threads: usize) -> Result<(), RebuildError> {
    let seq = sys.st.seq;
    let (LogOp::SwapModel { epoch, snapshot_seq, model_hash: logged, gate: lg } | LogOp::RejectModel { epoch, snapshot_seq, model_hash: logged, gate: lg }) = e else {
        return Err(RebuildError::NoRebuildPending);
    };
    if snap.seq != *snapshot_seq { return Err(RebuildError::WrongSnapshot { got: snap.seq, want: *snapshot_seq }); }
    let mut r = rebuild::rebuild(snap, sys.st.base_seed, threads)?;
    if r.next.epoch != *epoch { return Err(RebuildError::WrongEpoch { got: *epoch, want: r.next.epoch }); }
    let h = rebuild::model_hash(&r.next.idx);
    if h != *logged { return Err(RebuildError::HashMismatch { seq, logged: *logged, rebuilt: h }); }
    let g = gate::evaluate(snap, &r, &lg.params, lg.game, threads)?;
    if g != *lg { return Err(RebuildError::GateMismatch { seq }); }
    if matches!(e, LogOp::SwapModel { .. }) {
        rebuild::catch_up(&mut r, sys.carry.as_ref().ok_or(RebuildError::NoRebuildPending)?)?;
        swap(sys, r, g)?;
    } else {
        reject(sys, &r, g)?;
    }
    Ok(())
}

/// The capture for a durable snapshot, or `None` if the current view is
/// not the writer's state (a publish was deferred, or ops were applied
/// since) or a rebuild is pending. Call right after `views::publish`.
/// Snapshots are only taken with no rebuild pending: then every decision
/// in the log after a snapshot names a `snapshot_seq` at or after it, and
/// replay can take the rebuild's rows itself. Simpler than storing the
/// carry and the rebuild's rows, and just as deterministic.
pub fn capture(sys: &System) -> Option<Capture> {
    if sys.carry.is_some() { return None; }
    if !views::is_current(&sys.w) { return None; }
    let view = views::current(&views::latest(&sys.w));
    Some(Capture { view, tick_next: sys.w.seq, dead_since: sys.w.dead_since.clone(), st: sys.st.clone() })
}

/// FNV-1a over `st`: every field, f32/f64 by bits, the handed map by id.
pub fn state_hash(st: &State) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let f = |h: u64, xs: &[f64]| xs.iter().fold(h, |h, x| fnv(h, x.to_bits()));
    let u = |h: u64, xs: &[u64]| xs.iter().fold(h, |h, &x| fnv(h, x));
    let v = |h: u64, xs: &[f32]| xs.iter().fold(h, |h, x| fnv(h, x.to_bits() as u64));
    for x in [st.seq, st.model_seq, st.epoch, st.base_seed] { h = fnv(h, x); }
    h = f(h, &st.baseline.qerr);
    h = f(h, &st.baseline.rnorm);
    h = u(h, &st.baseline.rows);
    h = f(h, &st.stats.qerr);
    h = f(h, &st.stats.rnorm);
    h = u(h, &st.stats.inserts);
    h = fnv(fnv(h, st.stats.deletes), st.stats.live_at_start);
    let b = &st.buf;
    h = fnv(fnv(h, matches!(b.metric, Metric::Cosine) as u64), b.dim as u64);
    h = fnv(h, b.ids.len() as u64);
    h = u(u(u(h, &b.ids), &b.timestamps), &b.tags);
    for &x in &b.flags { h = fnv(h, x as u64); }
    h = v(v(h, &b.raw), &b.view);
    let mut handed: Vec<_> = b.handed.iter().collect();
    handed.sort_unstable_by_key(|(id, _)| **id);
    h = fnv(h, handed.len() as u64);
    for (id, x) in handed { h = fnv(fnv(fnv(h, *id), x.flags as u64), x.seq); }
    h = fnv(h, b.next_seq);
    h = fnv(fnv(h, st.outbox.next_seq), st.outbox.records.len() as u64);
    for r in &st.outbox.records {
        h = fnv(fnv(h, r.seq), matches!(r.kind, RecordKind::Delete) as u64);
        h = fnv(fnv(fnv(fnv(h, r.id), r.ts), r.tags), r.flags as u64);
        h = fnv(h, r.vec.len() as u64);
        h = v(h, &r.vec);
    }
    h
}

/// Fingerprint of a whole system from its parts: the rows'
/// `IvfPqIndex::fingerprint`, the writer's tick and `dead_since`, `state_hash`.
pub fn fingerprint_parts(rows_fp: u64, tick_next: u64, dead_since: &[u64], st: &State) -> u64 {
    let mut h = fnv(fnv(0xcbf29ce484222325u64, rows_fp), tick_next);
    for &d in dead_since { h = fnv(h, d); }
    fnv(h, state_hash(st))
}

/// Fingerprint of everything that replays: rows, tick, compaction state,
/// epoch, drift, buffer, outbox. Not the pending rebuild (see module doc).
/// O(rows), no copy.
pub fn fingerprint(sys: &System) -> u64 {
    fingerprint_parts(views::writer_fingerprint(&sys.w), sys.w.seq, &sys.w.dead_since, &sys.st)
}

/// `fingerprint` of the state a capture holds. O(rows); any thread.
pub fn capture_fingerprint(c: &Capture) -> u64 {
    fingerprint_parts(views::fingerprint(&c.view), c.tick_next, &c.dead_since, &c.st)
}

/// Full consistency check, O(rows). For tests and snapshot loads.
pub fn check_invariants(sys: &System) -> Result<(), String> {
    live::check_invariants(&views::writer_index(&sys.w))?;
    real_memory::check_invariants(&sys.st.buf)?;
    if sys.w.dead_since.len() != sys.w.lists.len() { return Err("dead_since length".into()); }
    for (li, (l, &d)) in sys.w.lists.iter().zip(&sys.w.dead_since).enumerate() {
        if (l.dead == 0) != (d == NEVER) || (d != NEVER && d > sys.w.seq) {
            return Err(format!("list {li}: {} dead rows, dead_since {d}, tick {}", l.dead, sys.w.seq));
        }
    }
    let nlist = sys.w.lists.len();
    let (b, s) = (&sys.st.baseline, &sys.st.stats);
    if [b.qerr.len(), b.rnorm.len(), b.rows.len(), s.qerr.len(), s.rnorm.len(), s.inserts.len()].iter().any(|&n| n != nlist) {
        return Err("drift vectors length".into());
    }
    if sys.st.model_seq > sys.st.seq { return Err("model_seq past seq".into()); }
    if sys.st.buf.dim != sys.w.model.dim || sys.st.buf.metric != sys.w.model.metric { return Err("buffer shape".into()); }
    let ob = &sys.st.outbox;
    if ob.records.windows(2).any(|w| w[0].seq >= w[1].seq) || ob.records.last().map_or(false, |r| r.seq >= ob.next_seq) {
        return Err("outbox seq order".into());
    }
    if ob.next_seq > sys.st.buf.next_seq { return Err("outbox ahead of buffer".into()); }
    Ok(())
}
