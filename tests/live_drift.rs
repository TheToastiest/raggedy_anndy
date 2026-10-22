//! Stage 3: drift measurement and background rebuild with a logged swap.
//!
//! The model (centroids, OPQ, PQ codebooks) is frozen per epoch. Drift stats
//! are f64 sums accumulated in op order, so signals fire at the same op on
//! every run. A rebuild trains the next epoch's model from a snapshot of the
//! live rows, replays the ops that came after the snapshot, and swaps in with
//! a `SwapModel` log entry. These tests pin:
//! - signals fire at the same op count on every run for shifted content, and
//!   stay quiet for more of the training distribution;
//! - a rebuild from one snapshot is reproducible (model hash and fingerprint),
//!   on a worker thread or inline, with any thread count;
//! - replaying a log that contains `SwapModel` gives the live run's state;
//! - recall lost to shifted content comes back after the rebuild.

mod common;
use common::*;

use std::collections::BTreeMap;

use raggedy_anndy::drift::{self, Signals, Thresholds};
use raggedy_anndy::live::{self, Op};
use raggedy_anndy::rebuild::{self, Live, LogOp};
use raggedy_anndy::seed::SplitMix64;
use raggedy_anndy::{FlatIndex, Hit, IvfPqIndex, Metric};

const DIM: usize = 32;
const HALF: usize = DIM / 2;
const NLIST: usize = 32;
const M: usize = 8;
const N: usize = 20_000;
const CLUSTERS: usize = 64;
const SPREAD: f32 = 0.08;

// ------------------------------------------------------------ data

/// Cluster centres that live in one half of the dimensions. Half 0 is the
/// training distribution; half 1 is the shifted one: the same structure in
/// the other dimensions, so the epoch-0 coarse lists and codebooks fit it badly.
fn half_centres(half: usize, seed: u64) -> Vec<f32> {
    let c = cluster_centres(CLUSTERS, HALF, seed);
    let mut out = vec![0.0f32; CLUSTERS * DIM];
    for k in 0..CLUSTERS {
        out[k * DIM + half * HALF..k * DIM + (half + 1) * HALF].copy_from_slice(&c[k * HALF..(k + 1) * HALF]);
    }
    out
}

fn train_data(n: usize, seed: u64) -> Dataset { clustered(n, DIM, &half_centres(0, 1), SPREAD, seed) }
fn shifted_data(n: usize, seed: u64) -> Dataset { clustered(n, DIM, &half_centres(1, 2), SPREAD, seed) }

/// Epoch-0 index over `N` training rows, ids 0..N, ts = id.
fn base_index() -> (IvfPqIndex, Dataset) {
    let ds = train_data(N, 10);
    let rows: Vec<(u64, Vec<f32>, u64)> = (0..N).map(|i| (ds.ids[i], row(&ds, i).to_vec(), i as u64)).collect();
    (IvfPqIndex::build(Metric::Cosine, DIM, &rows, ivfpq_params(NLIST, 4, 64, M, 7)), ds)
}

/// Inserts of `ds`, ids from `first_id`, ts from `first_ts`, one per op.
fn insert_ops(ds: &Dataset, first_id: u64, first_ts: u64) -> Vec<Op> {
    (0..ds.ids.len()).map(|i| Op::Insert { id: first_id + i as u64, ts: first_ts + i as u64, tags: (i as u64) & 0x7, vec: row(ds, i).to_vec() }).collect()
}

/// Shifted inserts with a delete of an old training row every fourth op.
fn shifted_mix(n: usize, seed: u64) -> Vec<Op> {
    let ds = shifted_data(n, seed);
    let mut ops = Vec::with_capacity(n + n / 4);
    for (i, op) in insert_ops(&ds, 1_000_000, N as u64).into_iter().enumerate() {
        ops.push(op);
        if i % 4 == 3 { ops.push(Op::Delete { id: (i as u64 * 7) % N as u64 }); }
    }
    ops
}

fn assert_ok(idx: &IvfPqIndex) {
    if let Err(e) = live::check_invariants(idx) { panic!("invariant broken: {e}"); }
}

fn bits(hits: &[Hit]) -> Vec<(u64, u32)> { hits.iter().map(|h| (h.id, h.score.to_bits())).collect() }

// ------------------------------------------------------------ driver

/// Applies `ops` one at a time, logging each, and returns the first op
/// (count of ops applied) after which any signal fired, with the signals.
fn first_fire(base: &IvfPqIndex, ops: &[Op], th: &Thresholds) -> Option<(usize, Signals)> {
    let mut st = rebuild::start(base.clone());
    for (n, op) in ops.iter().enumerate() {
        let _ = rebuild::write(&mut st, op);
        let sig = drift::check(&st.stats, &st.baseline, &st.idx, th);
        if sig.fired != 0 { return Some((n + 1, sig)); }
    }
    None
}

/// When the writer swaps a finished rebuild in.
#[derive(Clone, Copy)]
enum SwapWhen {
    /// As soon as the worker is seen to be done (depends on timing).
    WorkerDone,
    /// Exactly `lag` ops after the snapshot (worker joined if needed).
    AfterOps(u64),
}

/// Rebuilds per test run. The design has no cooldown yet, and while a shift
/// is still arriving each new model is trained on only part of it, so the
/// signals fire again soon after every swap.
const MAX_REBUILDS: u64 = 2;

/// The writer loop: log, apply, check, snapshot on a signal, rebuild on a
/// worker thread, catch up and swap. Returns the final state and the log.
fn live_run(base: &IvfPqIndex, ops: &[Op], th: &Thresholds, when: SwapWhen) -> (Live, Vec<LogOp>) {
    let base_seed = base.params.seed;
    let mut st = rebuild::start(base.clone());
    let mut log: Vec<LogOp> = Vec::new();
    let mut pending: Option<(u64, std::thread::JoinHandle<rebuild::Rebuilt>)> = None;
    let finish = |st: &mut Live, log: &mut Vec<LogOp>, h: std::thread::JoinHandle<rebuild::Rebuilt>| {
        let mut r = h.join().unwrap();
        rebuild::catch_up(&mut r, st.carry.as_ref().unwrap()).unwrap();
        let entry = rebuild::swap(st, r).unwrap();
        log.push(entry);
    };
    for op in ops {
        log.push(LogOp::Write(op.clone()));
        let _ = rebuild::write(&mut st, op);
        assert_eq!(st.seq as usize, log.len());
        if let Some((snap_seq, h)) = pending.take() {
            let due = match when {
                SwapWhen::WorkerDone => h.is_finished(),
                SwapWhen::AfterOps(lag) => st.seq >= snap_seq + lag,
            };
            if due { finish(&mut st, &mut log, h); } else { pending = Some((snap_seq, h)); }
        } else if st.epoch < MAX_REBUILDS && drift::check(&st.stats, &st.baseline, &st.idx, th).fired != 0 {
            let snap = rebuild::snapshot(&mut st).unwrap();
            let seq = snap.seq;
            pending = Some((seq, std::thread::spawn(move || rebuild::rebuild(&snap, base_seed, 2).unwrap())));
        }
    }
    if let Some((_, h)) = pending { finish(&mut st, &mut log, h); }
    (st, log)
}

// ------------------------------------------------------------ signals

#[test]
fn signals_fire_at_the_same_op_for_shifted_inserts() {
    let (base, _) = base_index();
    let ops = shifted_mix(4_000, 20);
    let th = drift::default_thresholds();
    let a = first_fire(&base, &ops, &th);
    let b = { let (base, ops, th) = (base.clone(), ops.clone(), th); std::thread::spawn(move || first_fire(&base, &ops, &th)).join().unwrap() };
    let c = first_fire(&base, &ops, &th);
    println!("shifted: first fire {a:?}");
    let (n, sig) = a.clone().expect("no drift signal for shifted inserts");
    assert_eq!(a, b);
    assert_eq!(a, c);
    assert!(n < ops.len(), "fired only at the last op");
    assert_eq!(sig.fired & drift::CHURN, 0, "churn alone should not explain a shift this early");
}

#[test]
fn signals_stay_quiet_for_more_training_data() {
    let (base, _) = base_index();
    // 40% more rows of the training distribution: churn stays under 0.5.
    let ds = train_data(N * 2 / 5, 21);
    let ops = insert_ops(&ds, 1_000_000, N as u64);
    let th = drift::default_thresholds();
    let mut st = rebuild::start(base.clone());
    let mut worst = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for op in &ops {
        rebuild::write(&mut st, op).unwrap();
        let sig = drift::check(&st.stats, &st.baseline, &st.idx, &th);
        assert_eq!(sig.fired, 0, "fired on in-distribution data: {sig:?}");
        // The ratios are checked from `min_inserts` on; before that one row moves them a lot.
        if sig.inserts >= th.min_inserts { worst.0 = worst.0.max(sig.qerr_ratio); worst.1 = worst.1.max(sig.rnorm_ratio); }
        worst.2 = worst.2.max(sig.imbalance);
        worst.3 = worst.3.max(sig.churn);
    }
    let end = drift::check(&st.stats, &st.baseline, &st.idx, &th);
    println!("in-distribution worst (qerr, rnorm, imbalance, churn) = {worst:?}, at the end {end:?}");
}

#[test]
fn thresholds_are_parameters() {
    let (base, _) = base_index();
    let ds = train_data(2_000, 22);
    let ops = insert_ops(&ds, 1_000_000, N as u64);
    let mut th = drift::default_thresholds();
    assert_eq!(first_fire(&base, &ops, &th), None);
    th.churn = 0.05; // 1000 inserts on 20000 rows
    let (n, sig) = first_fire(&base, &ops, &th).expect("lowered churn threshold did not fire");
    assert_eq!(sig.fired, drift::CHURN);
    assert_eq!(n, 1_001); // churn > 0.05 first holds after 1001 inserts
}

#[test]
fn baseline_matches_stats_of_inserting_the_training_rows() {
    // Inserting the training rows into an empty copy of the model measures
    // the same per-list sums as the build-time baseline, list by list.
    let (base, ds) = base_index();
    let b = drift::baseline(&base);
    let mut st = rebuild::start(live::empty_like(&base));
    for i in 0..N { rebuild::write(&mut st, &Op::Insert { id: ds.ids[i], ts: i as u64, tags: 0, vec: row(&ds, i).to_vec() }).unwrap(); }
    assert_eq!(st.idx.fingerprint(), base.fingerprint());
    assert_eq!(st.stats.inserts, b.rows);
    for l in 0..NLIST {
        assert!((st.stats.qerr[l] - b.qerr[l]).abs() <= 1e-9 * b.qerr[l].max(1.0), "list {l} qerr");
        assert!((st.stats.rnorm[l] - b.rnorm[l]).abs() <= 1e-9 * b.rnorm[l].max(1.0), "list {l} rnorm");
    }
}

#[test]
fn baseline_matches_insert_stats_with_opq() {
    // Same check through the OPQ path: the hook reads the rotated residual.
    let n = 4_000;
    let ds = train_data(n, 11);
    let rows: Vec<(u64, Vec<f32>, u64)> = (0..n).map(|i| (ds.ids[i], row(&ds, i).to_vec(), i as u64)).collect();
    let mut p = ivfpq_params(16, 4, 64, M, 7);
    p.use_opq = true;
    p.opq_mode = raggedy_anndy::ivfpq::OpqMode::PcaPerm;
    p.opq_sweeps = 4;
    let base = IvfPqIndex::build(Metric::Cosine, DIM, &rows, p);
    assert!(base.opq.is_some());
    let b = drift::baseline(&base);
    let mut st = rebuild::start(live::empty_like(&base));
    for (id, v, ts) in &rows { rebuild::write(&mut st, &Op::Insert { id: *id, ts: *ts, tags: 0, vec: v.clone() }).unwrap(); }
    assert_eq!(st.idx.fingerprint(), base.fingerprint());
    assert_eq!(st.stats.inserts, b.rows);
    assert_eq!(st.stats.qerr, b.qerr);
    assert_eq!(st.stats.rnorm, b.rnorm);
    let sig = drift::check(&st.stats, &b, &st.idx, &drift::default_thresholds());
    assert!((sig.qerr_ratio - 1.0).abs() < 1e-12 && (sig.rnorm_ratio - 1.0).abs() < 1e-12, "{sig:?}");
}

// ------------------------------------------------------------ rebuild

fn drifted_state() -> Live {
    let (base, _) = base_index();
    let mut st = rebuild::start(base);
    for op in &shifted_mix(6_000, 30) { let _ = rebuild::write(&mut st, op); }
    st
}

#[test]
fn rebuild_from_one_snapshot_is_reproducible() {
    let mut st = drifted_state();
    let base_seed = st.base_seed;
    let snap = rebuild::snapshot(&mut st).unwrap();
    assert_eq!(snap.seq, st.seq);
    assert_eq!(snap.ids.len(), live::live_rows(&st.idx));
    assert!(snap.ids.windows(2).all(|w| w[0] < w[1]), "snapshot not in ascending id order");

    let a = rebuild::rebuild(&snap, base_seed, 1).unwrap().next;
    let b = rebuild::rebuild(&snap, base_seed, 5).unwrap().next;
    assert_eq!(rebuild::model_hash(&a.idx), rebuild::model_hash(&b.idx));
    assert_eq!(a.idx.fingerprint(), b.idx.fingerprint());
    assert_eq!(a.epoch, 1);
    assert_eq!(a.idx.params.seed, rebuild::epoch_seed(base_seed, 1));
    assert_ne!(rebuild::model_hash(&a.idx), rebuild::model_hash(&st.idx));
    assert_ok(&a.idx);

    // Same live rows, ids, timestamps and tags; lists sorted by time; max_ts kept.
    assert_eq!(live::live_rows(&a.idx), live::live_rows(&st.idx));
    assert_eq!(a.idx.max_ts, st.idx.max_ts);
    let tag_of = |idx: &IvfPqIndex, id: u64| { let l = live::locate(idx, id).unwrap(); (idx.lists[l.list as usize].timestamps[l.row as usize], idx.lists[l.list as usize].tags[l.row as usize]) };
    for &id in snap.ids.iter().step_by(97) { assert_eq!(tag_of(&a.idx, id), tag_of(&st.idx, id), "id {id}"); }
}

#[test]
fn rebuild_on_a_worker_equals_inline() {
    let mut st = drifted_state();
    let base_seed = st.base_seed;
    let snap = rebuild::snapshot(&mut st).unwrap();
    // Ops that land while the rebuild runs, including an expiry and a compaction.
    let mut more = shifted_mix(1_500, 31);
    for op in more.iter_mut() { if let Op::Insert { id, ts, .. } = op { *id += 500_000; *ts += 100_000; } }
    more.push(Op::Expire { before_ts: 5_000, max_rows: 700 });
    more.push(Op::Compact { list: 3 });
    let mut expired = Vec::new();
    for op in &more {
        if let Ok(live::Applied::Expired { ids, .. }) = rebuild::write(&mut st, op) { expired.extend(ids); }
    }
    assert!(!expired.is_empty());
    let carry = st.carry.clone().unwrap();

    let worker = { let snap = snap.clone(); std::thread::spawn(move || rebuild::rebuild(&snap, base_seed, 3).unwrap()) };
    let mut inline = rebuild::rebuild(&snap, base_seed, 1).unwrap();
    rebuild::catch_up(&mut inline, &carry).unwrap();
    let mut threaded = worker.join().unwrap();
    rebuild::catch_up(&mut threaded, &carry).unwrap();

    let (inline, threaded) = (&inline.next, &threaded.next);
    assert_eq!(inline.idx.fingerprint(), threaded.idx.fingerprint());
    assert_eq!(inline.stats, threaded.stats);
    assert_eq!(inline.baseline, threaded.baseline);
    assert_ok(&inline.idx);
    // The rebuilt index holds exactly the live index's live ids, including
    // after a capped expiry that walked the old layout.
    assert_eq!(live::live_rows(&inline.idx), live::live_rows(&st.idx));
    for l in &st.idx.lists {
        for r in 0..l.len() { if l.tomb[r] == 0 { assert!(live::locate(&inline.idx, l.ids[r]).is_some(), "id {} lost", l.ids[r]); } }
    }
    for id in &expired { assert!(live::locate(&inline.idx, *id).is_none(), "expired id {id} came back"); }

    // Catching up in pieces equals catching up at once.
    let mut pieces = rebuild::rebuild(&snap, base_seed, 2).unwrap();
    let part = rebuild::Carry { snapshot_seq: carry.snapshot_seq, ops: carry.ops[..400].to_vec() };
    rebuild::catch_up(&mut pieces, &part).unwrap();
    rebuild::catch_up(&mut pieces, &carry).unwrap();
    assert_eq!(pieces.next.idx.fingerprint(), inline.idx.fingerprint());
    assert_eq!(pieces.next.stats, inline.stats);
}

// ------------------------------------------------------------ swap + replay

fn assert_same_state(a: &Live, b: &Live) {
    assert_eq!(a.seq, b.seq);
    assert_eq!(a.epoch, b.epoch);
    assert_eq!(rebuild::model_hash(&a.idx), rebuild::model_hash(&b.idx));
    assert_eq!(a.idx.fingerprint(), b.idx.fingerprint());
    assert_eq!(a.stats, b.stats);
    assert_eq!(a.baseline, b.baseline);
}

#[test]
fn replaying_a_log_with_swap_model_equals_the_live_run() {
    let (base, _) = base_index();
    let ops = shifted_mix(6_000, 40);
    let th = drift::default_thresholds();
    for when in [SwapWhen::WorkerDone, SwapWhen::AfterOps(300), SwapWhen::AfterOps(0)] {
        let (st, log) = live_run(&base, &ops, &th, when);
        let swaps: Vec<&LogOp> = log.iter().filter(|e| matches!(e, LogOp::SwapModel { .. })).collect();
        println!("swaps: {swaps:?}");
        assert!(!swaps.is_empty(), "no rebuild happened");
        assert!(st.epoch >= 1);
        assert_ok(&st.idx);

        let replayed = rebuild::replay(base.clone(), &log, 4).unwrap();
        assert_same_state(&st, &replayed);
        let mut rng = SplitMix64::new(41);
        for _ in 0..20 {
            let mut q: Vec<f32> = (0..DIM).map(|_| gauss(&mut rng)).collect();
            normalize(&mut q);
            assert_eq!(bits(&st.idx.search(&q, 10)), bits(&replayed.idx.search(&q, 10)));
        }
    }
}

#[test]
fn swap_model_with_a_wrong_hash_is_refused() {
    let (base, _) = base_index();
    let ops = shifted_mix(4_000, 42);
    let (_, mut log) = live_run(&base, &ops, &drift::default_thresholds(), SwapWhen::AfterOps(10));
    let at = log.iter().position(|e| matches!(e, LogOp::SwapModel { .. })).unwrap();
    if let LogOp::SwapModel { model_hash, .. } = &mut log[at] { *model_hash ^= 1; }
    assert!(rebuild::replay(base, &log, 1).is_err());
}

#[test]
fn swap_is_refused_until_caught_up() {
    let mut st = drifted_state();
    let snap = rebuild::snapshot(&mut st).unwrap();
    let r = rebuild::rebuild(&snap, st.base_seed, 2).unwrap();
    let ds = shifted_data(1, 43);
    let ts = st.idx.max_ts;
    rebuild::write(&mut st, &Op::Insert { id: 9_000_000, ts, tags: 0, vec: row(&ds, 0).to_vec() }).unwrap();
    let fp = st.idx.fingerprint();
    assert!(matches!(rebuild::snapshot(&mut st), Err(rebuild::RebuildError::RebuildPending { .. })));
    assert!(matches!(rebuild::swap(&mut st, r.clone()), Err(rebuild::RebuildError::NotCaughtUp { carried: 0, pending: 1 })));
    assert_eq!(st.idx.fingerprint(), fp);
    assert_eq!(st.epoch, 0);
    let mut r = r;
    rebuild::catch_up(&mut r, st.carry.as_ref().unwrap()).unwrap();
    let seq = st.seq;
    assert!(matches!(rebuild::swap(&mut st, r), Ok(LogOp::SwapModel { epoch: 1, .. })));
    assert_eq!((st.epoch, st.seq), (1, seq + 1));
    assert!(st.carry.is_none());
    assert_eq!(live::locate(&st.idx, 9_000_000).map(|_| ()), Some(()));
}

// ------------------------------------------------------------ recall

fn recall_at_10(idx: &IvfPqIndex, rows: &BTreeMap<u64, Vec<f32>>, queries: &Dataset) -> f64 {
    let mut flat = FlatIndex::new(DIM, Metric::Cosine);
    for (id, v) in rows { flat.add(*id, v); }
    let truth = truth_sets(&flat, queries, 10);
    recall(ivfpq_recall(idx, queries, &truth, 10))
}

#[test]
fn recall_drops_with_shifted_content_and_comes_back_after_rebuild() {
    let (base, ds) = base_index();
    let mut rows: BTreeMap<u64, Vec<f32>> = (0..N).map(|i| (ds.ids[i], row(&ds, i).to_vec())).collect();
    // Queries never copy stored rows. Each phase is measured on queries
    // from the content it holds: epoch 0 on the training distribution, after
    // the shift on both halves.
    let qa = train_data(200, 50);
    let qb = shifted_data(200, 51);
    let qmix = Dataset { dim: DIM, ids: (0..400).collect(), vecs: [qa.vecs.clone(), qb.vecs.clone()].concat() };

    let r0 = recall_at_10(&base, &rows, &qa);
    let r0_mix = recall_at_10(&base, &rows, &qmix);

    // 50% of the content becomes shifted: the oldest half of the training
    // rows expire and as many shifted rows arrive, so the row count holds.
    let mut st = rebuild::start(base.clone());
    match rebuild::write(&mut st, &Op::Expire { before_ts: (N / 2) as u64, max_rows: usize::MAX }).unwrap() {
        live::Applied::Expired { ids, done } => { assert!(done); for id in ids { rows.remove(&id); } }
        other => panic!("unexpected {other:?}"),
    }
    let sd = shifted_data(N / 2, 52);
    for (i, op) in insert_ops(&sd, 1_000_000, N as u64).iter().enumerate() {
        rebuild::write(&mut st, op).unwrap();
        rows.insert(1_000_000 + i as u64, row(&sd, i).to_vec());
    }
    assert_eq!(live::live_rows(&st.idx), N);
    let sig = drift::check(&st.stats, &st.baseline, &st.idx, &drift::default_thresholds());
    let r_before = recall_at_10(&st.idx, &rows, &qmix);
    let r_before_b = recall_at_10(&st.idx, &rows, &qb);

    let t = std::time::Instant::now();
    let snap = rebuild::snapshot(&mut st).unwrap();
    let r = rebuild::rebuild(&snap, st.base_seed, 4).unwrap();
    let took = t.elapsed();
    rebuild::swap(&mut st, r).unwrap();
    let r_after = recall_at_10(&st.idx, &rows, &qmix);
    let (r_after_a, r_after_b) = (recall_at_10(&st.idx, &rows, &qa), recall_at_10(&st.idx, &rows, &qb));

    println!("recall@10 epoch 0 = {r0:.4} (mixed queries {r0_mix:.4})");
    println!("recall@10 before rebuild = {r_before:.4} (shifted queries {r_before_b:.4}), signals {sig:?}");
    println!("recall@10 after rebuild = {r_after:.4} (training {r_after_a:.4}, shifted {r_after_b:.4}); snapshot + rebuild of {} rows took {took:?}", snap.ids.len());
    assert!(sig.fired != 0);
    assert!(r_before < r0 - 0.01, "shifted content did not cost recall: {r_before} vs {r0}");
    assert!(r_after >= r0 - 0.01, "recall after rebuild {r_after} vs epoch 0 {r0}");
    assert_ok(&st.idx);
}

// ------------------------------------------------------------ timing

/// Snapshot + rebuild time for a 100k-row index. Not a gate.
/// cargo test --release --test live_drift rebuild_time -- --ignored --nocapture
#[test]
#[ignore = "timing aid"]
fn rebuild_time() {
    for &(n, dim, nlist, m) in &[(100_000usize, 128usize, 256usize, 32usize), (100_000, 256, 1024, 32)] {
        let ds = uniform_sphere(n, dim, 60);
        let rows: Vec<(u64, Vec<f32>, u64)> = (0..n).map(|i| (ds.ids[i], row(&ds, i).to_vec(), i as u64)).collect();
        let idx = IvfPqIndex::build(Metric::Cosine, dim, &rows, ivfpq_params(nlist, 8, 64, m, 7));
        let mut st = rebuild::start(idx);
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let t = std::time::Instant::now();
        let snap = rebuild::snapshot(&mut st).unwrap();
        let t_snap = t.elapsed();
        let t = std::time::Instant::now();
        let r = rebuild::rebuild(&snap, st.base_seed, threads).unwrap();
        println!("n={n} dim={dim} nlist={nlist} m={m}: snapshot {t_snap:?}, rebuild threads={threads} {:?} (hash {:x})", t.elapsed(), rebuild::model_hash(&r.next.idx));
    }
}
