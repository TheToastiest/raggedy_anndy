//! Recall + latency gates for game memory (approved by Brinn 2026-10-21).
//!
//! Corpus size per shard and k are not known yet, so every combination of
//!     N in {10k, 100k, 1M}  x  k in {5, 10}
//! is a cell, and each cell must pass both gates:
//!   - recall@k Wilson lb95 >= 0.95 against exact FlatIndex, on two query
//!     sets: held-out (fresh draws, never stored) and near-copy (stored
//!     rows + N(0, 0.01) noise, the `freeze` query style);
//!   - single-query search p95 < 2 ms (one lookup on one thread).
//!
//! 2 ms is a per-lookup ceiling, not a per-tick budget: how many lookups run
//! per 60 Hz tick is still unknown and decides the real budget.
//!
//! Data: dim 128, unit-norm Gaussian blobs around 256 "zones" (fixed seeds).
//! Latency is wall-clock and machine-dependent; it is only asserted in
//! release builds. Debug builds print it and gate recall only.
//!
//! Cost: the 10k cells run in every `cargo test`. 100k and 1M cells are
//! `#[ignore = "SLOW ..."]`; run them with
//!     cargo test --release --test gates -- --ignored --test-threads=1
//! All six cells pass on master @ 4088841 (4-core container, release):
//! tightest margin is N=10k k=10 at held-out lb95 0.9586. A cell that
//! starts failing gets `KNOWN-FAIL` in its ignore reason (see recall.rs).

mod common;
use common::*;

use raggedy_anndy::eval::wilson_lower_bound;
use raggedy_anndy::par::parallel_map_indexed;
use raggedy_anndy::seed::SplitMix64;
use raggedy_anndy::{FlatIndex, Hit, IvfPqIndex, IvfPqParams, Metric};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

const DIM: usize = 128;
const ZONES: usize = 256;
const SPREAD: f32 = 0.15;
const NQ: usize = 400;
const NEAR_SIGMA: f32 = 0.01;
const K_MAX: usize = 10;
const MIN_LB95: f64 = 0.95;
const MAX_P95_US: f64 = 2_000.0;
const WARMUP: usize = 50;

// ------------------------------------------------------------------ params

/// Index shape per corpus size. These are the knobs a fix PR tunes; the
/// gates themselves (MIN_LB95, MAX_P95_US) do not move.
fn params_for(n: usize) -> IvfPqParams {
    // (nlist, nprobe, refine, m). Picked from `sweep_search_params` on this
    // data (2026-10-21, 4-core container): recall here is bound by
    // partitioning, not PQ, so nprobe is the lever and refine barely moves.
    //   10k:  nprobe 32 -> 0.892 held-out; 64 -> 0.964 at ~0.45 ms p95.
    //   100k: nprobe 16 -> 0.978 held-out at ~0.55 ms p95.
    //   1M:   nprobe 8, refine 400 -> 0.982 held-out at ~1.1 ms p95
    //         (refine 100 drops to 0.952; nprobe 16 already costs ~2.1 ms).
    let (nlist, nprobe, refine, m) = match n {
        10_000 => (128, 64, 200, 32),
        100_000 => (512, 16, 200, 32),
        1_000_000 => (1024, 8, 400, 32),
        _ => panic!("no params for N={n}"),
    };
    let mut p = ivfpq_params(nlist, nprobe, refine, m, 7);
    p.iters = 10;
    p
}

// ------------------------------------------------------------------ fixture

struct Fixture {
    index: IvfPqIndex,
    held_out: Dataset,
    near: Dataset,
    truth_held_out: Vec<Vec<Hit>>, // exact top-K_MAX; top-5 is its prefix
    truth_near: Vec<Vec<Hit>>,
    build_s: f64,
}

/// Stored rows (every N/NQ-th) plus per-dim N(0, sigma) noise, renormalised.
fn near_copies(ds: &Dataset, nq: usize, sigma: f32, seed: u64) -> Dataset {
    let mut rng = SplitMix64::new(seed);
    let step = (ds.ids.len() / nq).max(1);
    let mut vecs = Vec::with_capacity(nq * ds.dim);
    for j in 0..nq {
        let start = vecs.len();
        for &x in row(ds, (j * step) % ds.ids.len()) { vecs.push(x + sigma * gauss(&mut rng)); }
        normalize(&mut vecs[start..]);
    }
    Dataset { dim: ds.dim, ids: (0..nq as u64).collect(), vecs }
}

fn exact_truth(flat: &FlatIndex, qs: &Dataset) -> Vec<Vec<Hit>> {
    let idx: Vec<usize> = (0..qs.ids.len()).collect();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    parallel_map_indexed(&idx, threads, |&i, _| flat.search(row(qs, i), K_MAX))
}

fn build_fixture(n: usize) -> Fixture {
    let centres = cluster_centres(ZONES, DIM, 1234);
    let data = clustered(n, DIM, &centres, SPREAD, 42);
    let held_out = clustered(NQ, DIM, &centres, SPREAD, 999);
    let near = near_copies(&data, NQ, NEAR_SIGMA, 555);
    let flat = flat_from(&data, Metric::Cosine);
    let truth_held_out = exact_truth(&flat, &held_out);
    let truth_near = exact_truth(&flat, &near);
    drop(flat);
    let rows = ivfpq_rows(&data);
    drop(data);
    let t0 = Instant::now();
    let index = IvfPqIndex::build(Metric::Cosine, DIM, &rows, params_for(n));
    let build_s = t0.elapsed().as_secs_f64();
    Fixture { index, held_out, near, truth_held_out, truth_near, build_s }
}

/// One fixture per N, shared by both k cells.
fn fixture(n: usize) -> &'static Fixture {
    static F10K: OnceLock<Fixture> = OnceLock::new();
    static F100K: OnceLock<Fixture> = OnceLock::new();
    static F1M: OnceLock<Fixture> = OnceLock::new();
    let cell = match n { 10_000 => &F10K, 100_000 => &F100K, 1_000_000 => &F1M, _ => unreachable!() };
    cell.get_or_init(|| build_fixture(n))
}

/// Serialises cells in this binary: a build on all cores running next to a
/// timing loop would inflate p95.
static SERIAL: Mutex<()> = Mutex::new(());

// ------------------------------------------------------------------ measure

#[derive(Debug)]
struct CellResult {
    n: usize, k: usize,
    recall_held_out: f64, lb_held_out: f64,
    recall_near: f64, lb_near: f64,
    p50_us: f64, p95_us: f64,
    build_s: f64,
}

fn tally(idx: &IvfPqIndex, qs: &Dataset, truth: &[Vec<Hit>], k: usize) -> RecallTally {
    let mut t = RecallTally::default();
    for i in 0..qs.ids.len() { tally_one(&truth[i][..k.min(truth[i].len())], &idx.search(row(qs, i), k), k, &mut t); }
    t
}

/// Nearest-rank percentile over per-query wall-clock times.
fn percentile_us(sorted_us: &[f64], p: f64) -> f64 {
    let rank = ((p * sorted_us.len() as f64).ceil() as usize).clamp(1, sorted_us.len());
    sorted_us[rank - 1]
}

fn latency_us(idx: &IvfPqIndex, qs: &Dataset, k: usize) -> (f64, f64) {
    for i in 0..WARMUP.min(qs.ids.len()) { std::hint::black_box(idx.search(row(qs, i), k)); }
    let mut us: Vec<f64> = (0..qs.ids.len()).map(|i| {
        let t0 = Instant::now();
        std::hint::black_box(idx.search(row(qs, i), k));
        t0.elapsed().as_secs_f64() * 1e6
    }).collect();
    us.sort_by(|a, b| a.total_cmp(b));
    (percentile_us(&us, 0.50), percentile_us(&us, 0.95))
}

fn measure(n: usize, k: usize) -> CellResult {
    let f = fixture(n);
    let th = tally(&f.index, &f.held_out, &f.truth_held_out, k);
    let tn = tally(&f.index, &f.near, &f.truth_near, k);
    let (p50_us, p95_us) = latency_us(&f.index, &f.held_out, k);
    CellResult {
        n, k,
        recall_held_out: recall(th), lb_held_out: wilson_lower_bound(th.hits, th.possible, 1.96),
        recall_near: recall(tn), lb_near: wilson_lower_bound(tn.hits, tn.possible, 1.96),
        p50_us, p95_us, build_s: f.build_s,
    }
}

fn gate(n: usize, k: usize) {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let r = measure(n, k);
    let p = params_for(n);
    eprintln!(
        "GATE N={} k={} nlist={} nprobe={} refine={} m={} | held-out recall={:.4} lb95={:.4} | near recall={:.4} lb95={:.4} | p50={:.0}us p95={:.0}us | build={:.1}s",
        r.n, r.k, p.nlist, p.nprobe, p.refine, p.m,
        r.recall_held_out, r.lb_held_out, r.recall_near, r.lb_near, r.p50_us, r.p95_us, r.build_s);

    let mut fails = Vec::new();
    if r.lb_held_out < MIN_LB95 { fails.push(format!("held-out lb95 {:.4} < {MIN_LB95}", r.lb_held_out)); }
    if r.lb_near < MIN_LB95 { fails.push(format!("near-copy lb95 {:.4} < {MIN_LB95}", r.lb_near)); }
    if cfg!(debug_assertions) {
        eprintln!("  (debug build: p95 not gated; use --release)");
    } else if r.p95_us >= MAX_P95_US {
        fails.push(format!("p95 {:.0}us >= {MAX_P95_US}us", r.p95_us));
    }
    assert!(fails.is_empty(), "N={n} k={k}: {}", fails.join("; "));
}

// ------------------------------------------------------------------ cells

#[test] fn gate_n10k_k5() { gate(10_000, 5); }
#[test] fn gate_n10k_k10() { gate(10_000, 10); }

#[test]
#[ignore = "SLOW: 100k build (~20 s) + exact truth"]
fn gate_n100k_k5() { gate(100_000, 5); }

#[test]
#[ignore = "SLOW: 100k build (~20 s) + exact truth"]
fn gate_n100k_k10() { gate(100_000, 10); }

#[test]
#[ignore = "SLOW: 1M build (~2 min) + exact truth, ~2.5 GB RAM"]
fn gate_n1m_k5() { gate(1_000_000, 5); }

#[test]
#[ignore = "SLOW: 1M build (~2 min) + exact truth, ~2.5 GB RAM"]
fn gate_n1m_k10() { gate(1_000_000, 10); }

// ------------------------------------------------------------------ helpers

#[test]
fn percentile_is_nearest_rank() {
    let v: Vec<f64> = (1..=100).map(|x| x as f64).collect();
    assert_eq!(percentile_us(&v, 0.95), 95.0);
    assert_eq!(percentile_us(&v, 0.50), 50.0);
    assert_eq!(percentile_us(&[7.0], 0.95), 7.0);
}

/// Tuning aid, not a gate: prints recall/p95 over an nprobe x refine grid
/// for the N in RA_SWEEP_N (default 10000), k=10. Search knobs only; the
/// build is shared.
#[test]
#[ignore = "tuning aid: RA_SWEEP_N=100000 cargo test --release --test gates sweep -- --ignored --nocapture"]
fn sweep_search_params() {
    let n: usize = std::env::var("RA_SWEEP_N").ok().and_then(|s| s.parse().ok()).unwrap_or(10_000);
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let f = fixture(n);
    let nlist = f.index.params.nlist;
    for np in [8usize, 16, 32, 64, 128, 256, 512].into_iter().filter(|&x| x <= nlist) {
        for rf in [100usize, 400, 1600] {
            let idx = with_search_params(&f.index, np, rf);
            let h = recall(tally(&idx, &f.held_out, &f.truth_held_out, K_MAX));
            let nr = recall(tally(&idx, &f.near, &f.truth_near, K_MAX));
            let (_, p95) = latency_us(&idx, &f.held_out, K_MAX);
            eprintln!("N={n} nlist={nlist} nprobe={np:4} refine={rf:5} held-out={h:.4} near={nr:.4} p95={p95:.0}us");
        }
    }
}
