//! Swap gate: is a rebuilt model better than the one it would replace?
//!
//! Runs on the worker, right after `rebuild::rebuild` and before catch-up,
//! and reads only the snapshot and the candidate, so it is a pure function
//! of the log up to `snapshot_seq`. Both models are judged on the same rows:
//!
//! - current: `snap.model` (the live model, no rows) with every snapshot row
//!   inserted in (ts, id) order. Same rows, same codes as the live index at
//!   the snapshot, so the same search results.
//! - candidate: the rebuilt index as built, before catch-up.
//!
//! Evaluation set: the `recent` newest snapshot rows by (ts, id); `queries`
//! of them, picked with a seed derived from the candidate's epoch seed, are
//! the queries. Ground truth is an exact `FlatIndex` over all snapshot rows.
//! Drift: the recent rows are inserted into an empty copy of each model and
//! measured against that model's own baseline (`drift::check`). Churn says
//! nothing about a model, so only `QERR | RNORM | IMBALANCE` count here.
//!
//! Accept when candidate recall >= current recall + `min_gain`, or the
//! current model's drift signals fire on the recent rows and the
//! candidate's do not. A `GameCheck::Fail` from an outside game-level test
//! rejects regardless. The whole `Gate` goes in the log (`SwapModel` or
//! `RejectModel`), and `rebuild::replay` recomputes it and checks equality.

use crate::drift::{self, DriftBaseline, Signals, Thresholds, IMBALANCE, QERR, RNORM};
use crate::flat::FlatIndex;
use crate::ivfpq::IvfPqIndex;
use crate::live;
use crate::par;
use crate::types::Hit;
use crate::rebuild::{RebuildError, Rebuilt, Snapshot};
use crate::seed::SplitMix64;

/// The drift signals that describe how well a model fits rows.
pub const MODEL_SIGNALS: u8 = QERR | RNORM | IMBALANCE;

/// Salt for the query pick, so it never shares a stream with the build.
const GATE_SALT: u64 = 0x6A7E_5EED_6A7E_5EED;

/// How the gate measures and decides. Logged inside every `Gate`, so replay
/// re-runs it with the same values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateParams {
    /// Queries drawn from the recent rows.
    pub queries: usize,
    /// Newest snapshot rows the queries and drift measure come from.
    pub recent: usize,
    pub k: usize,
    /// Recall@k the candidate must gain to win on recall alone.
    pub min_gain: f64,
    /// Drift thresholds both models are judged with (churn ignored).
    pub th: Thresholds,
}

/// 256 queries x k 10 = 2560 trials: the standard error of one recall is at
/// most 0.01, and both models answer the same queries, so a 0.02 gain is
/// well outside noise. 4096 recent rows = `min_ops_per_epoch`: roughly what
/// arrived since the last model decision.
pub fn default_gate_params() -> GateParams {
    GateParams { queries: 256, recent: 4096, k: 10, min_gain: 0.02, th: drift::default_thresholds() }
}

/// Extension point for a game-level test of the candidate, run by the
/// caller on the worker before `evaluate` and passed in as its verdict
/// (e.g. replay a recorded "Best Time" run on both models and compare).
/// No such test exists yet. The verdict is logged with the gate and replay
/// takes it from the log, so the test itself must be deterministic to be
/// worth logging. `Fail` rejects; `Pass` and `NotRun` leave the decision
/// to the offline checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameCheck { NotRun, Pass, Fail }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateReason {
    /// Candidate recall >= current + `min_gain`.
    RecallGain,
    /// Current model's drift signals fire on the recent rows, candidate's do not.
    DriftCleared,
    /// Neither: the old model stays.
    NoGain,
    /// The game-level check failed.
    GameCheckFailed,
}

/// What the gate measured and decided. Plain data; logged as is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gate {
    pub params: GateParams,
    pub game: GameCheck,
    /// Queries run and truth ids checked (queries x k, fewer if rows < k).
    pub queries: u32,
    pub trials: u32,
    /// Truth ids found in the top k, per model.
    pub current_hits: u32,
    pub candidate_hits: u32,
    /// Drift of each model on the recent rows, against its own baseline.
    pub current: Signals,
    pub candidate: Signals,
    pub accept: bool,
    pub reason: GateReason,
}

/// Hits / trials; 0 when nothing was checked.
pub fn recall(hits: u32, trials: u32) -> f64 { if trials == 0 { 0.0 } else { hits as f64 / trials as f64 } }

/// Accept or reject from the measured values. Game veto first, then recall,
/// then drift.
pub fn decide(p: &GateParams, game: GameCheck, trials: u32, current_hits: u32, candidate_hits: u32, current_fired: u8, candidate_fired: u8) -> (bool, GateReason) {
    if game == GameCheck::Fail { return (false, GateReason::GameCheckFailed); }
    if trials > 0 && recall(candidate_hits, trials) >= recall(current_hits, trials) + p.min_gain { return (true, GateReason::RecallGain); }
    if current_fired & MODEL_SIGNALS != 0 && candidate_fired & MODEL_SIGNALS == 0 { return (true, GateReason::DriftCleared); }
    (false, GateReason::NoGain)
}

/// Rows encoded per round of `reencode`: bounds the encoded buffers
/// (rows x (dim floats + m bytes)) while keeping every thread busy.
const ENCODE_CHUNK: usize = 1 << 14;

/// One thread's share of a round, encoded with the model, as columns.
struct EncodedRows {
    list: Vec<u32>,
    codes: Vec<u8>,
    /// Stored (normalised) vectors, `dim` per row.
    view: Vec<f32>,
    /// `drift::row_err` of each row.
    qerr: Vec<f64>,
    rnorm: Vec<f64>,
}

/// Encode snapshot rows `rows` with `model`, as `live::apply` would.
/// Pure per row, so it can run on any thread.
fn encode_rows(model: &IvfPqIndex, snap: &Snapshot, rows: &[usize]) -> EncodedRows {
    let (dim, m, n) = (snap.dim, model.params.m, rows.len());
    let mut s = live::scratch_for(model);
    let mut out = EncodedRows {
        list: Vec::with_capacity(n), codes: Vec::with_capacity(n * m), view: Vec::with_capacity(n * dim),
        qerr: Vec::with_capacity(n), rnorm: Vec::with_capacity(n),
    };
    for &i in rows {
        let li = live::encode_row(model, &mut s, &snap.vecs[i * dim..(i + 1) * dim]);
        let (r, code) = live::last_encoded(model, &s);
        let (e, nrm) = drift::row_err(&model.pq, r, code);
        out.list.push(li as u32);
        out.codes.extend_from_slice(code);
        out.view.extend_from_slice(&s.view);
        out.qerr.push(e);
        out.rnorm.push(nrm);
    }
    out
}

/// Insert `rows` (indexes into the snapshot, in (ts, id) order) into an
/// empty copy of `model`; the result and its drift stats. Rows are encoded
/// on `threads` threads, a round of `ENCODE_CHUNK` at a time, then
/// appended and recorded in `rows` order, so the index and the stats are
/// those of `live::apply` + `drift::record` row by row, for any `threads`.
fn reencode(model: &IvfPqIndex, snap: &Snapshot, rows: &[usize], threads: usize) -> (IvfPqIndex, drift::DriftStats) {
    let (dim, m) = (snap.dim, model.params.m);
    let threads = threads.max(1);
    let mut idx = live::empty_like(model);
    let mut stats = drift::new_stats(&idx);
    for round in rows.chunks(ENCODE_CHUNK) {
        let parts: Vec<&[usize]> = round.chunks(round.len().div_ceil(threads)).collect();
        let encoded = par::parallel_map_indexed(&parts, threads, |part, _| encode_rows(model, snap, part));
        for (part, e) in parts.iter().zip(&encoded) {
            for (j, &i) in part.iter().enumerate() {
                let li = e.list[j] as usize;
                let raw = &snap.vecs[i * dim..(i + 1) * dim];
                let r = live::insert_encoded(&mut idx, snap.ids[i], snap.ts[i], snap.tags[i], raw, li, &e.codes[j * m..(j + 1) * m], &e.view[j * dim..(j + 1) * dim]);
                assert!(r.is_ok(), "gate re-encode refused id {}: {r:?}", snap.ids[i]);
                drift::record_row(&mut stats, li, e.qerr[j], e.rnorm[j]);
            }
        }
    }
    (idx, stats)
}

/// Drift of `model` on `recent`, judged against `baseline`; imbalance from
/// `full` (the model holding every snapshot row).
fn model_signals(model: &IvfPqIndex, baseline: &DriftBaseline, full: &IvfPqIndex, snap: &Snapshot, recent: &[usize], th: &Thresholds, threads: usize) -> Signals {
    let (_, stats) = reencode(model, snap, recent, threads);
    let mut sig = drift::check(&stats, baseline, full, th);
    sig.fired &= MODEL_SIGNALS;
    sig
}

/// Run the gate for `cand`, rebuilt from `snap` and not yet caught up.
/// Worker side; O(snapshot rows) encodes and `queries` exact scans, both
/// spread over `threads`. Encoded rows are appended and per-query counts
/// summed in a fixed order, so the result is the same for every `threads`
/// value.
pub fn evaluate(snap: &Snapshot, cand: &Rebuilt, p: &GateParams, game: GameCheck, threads: usize) -> Result<Gate, RebuildError> {
    if cand.snapshot_seq != snap.seq { return Err(RebuildError::WrongSnapshot { got: cand.snapshot_seq, want: snap.seq }); }
    if cand.carried != 0 { return Err(RebuildError::GateAfterCatchUp { carried: cand.carried }); }
    let n = snap.ids.len();
    let dim = snap.dim;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by_key(|&i| (snap.ts[i], snap.ids[i]));
    let recent = &order[n - p.recent.min(n)..];

    // Queries: a seeded pick without replacement from the recent rows.
    let mut pool: Vec<usize> = recent.to_vec();
    let q = p.queries.min(pool.len());
    let mut rng = SplitMix64::new(cand.next.idx.params.seed ^ GATE_SALT);
    for i in 0..q {
        let j = i + rng.gen_range(pool.len() - i);
        pool.swap(i, j);
    }
    pool.truncate(q);

    let (current, _) = reencode(&snap.model, snap, &order, threads);
    let cidx = &cand.next.idx;
    let mut flat = FlatIndex::new(dim, snap.metric);
    for i in 0..n { flat.add(snap.ids[i], &snap.vecs[i * dim..(i + 1) * dim]); }
    let per_query = par::parallel_map_indexed(&pool, threads.max(1), |&i, _| {
        let v = &snap.vecs[i * dim..(i + 1) * dim];
        let truth = flat.search(v, p.k);
        let found = |hits: Vec<Hit>| truth.iter().filter(|t| hits.iter().any(|h| h.id == t.id)).count() as u32;
        (truth.len() as u32, found(current.search(v, p.k)), found(cidx.search(v, p.k)))
    });
    let (mut trials, mut current_hits, mut candidate_hits) = (0u32, 0u32, 0u32);
    for (t, c, n) in per_query { trials += t; current_hits += c; candidate_hits += n; }

    let cur_sig = model_signals(&snap.model, &snap.baseline, &current, snap, recent, &p.th, threads);
    let cand_sig = model_signals(cidx, &cand.next.baseline, cidx, snap, recent, &p.th, threads);
    let (accept, reason) = decide(p, game, trials, current_hits, candidate_hits, cur_sig.fired, cand_sig.fired);
    Ok(Gate {
        params: *p, game, queries: q as u32, trials, current_hits, candidate_hits,
        current: cur_sig, candidate: cand_sig, accept, reason,
    })
}
