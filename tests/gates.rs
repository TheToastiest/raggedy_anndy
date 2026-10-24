//! Recall + latency gates for game memory (approved by Brinn 2026-10-21).
//!
//! Corpus size per shard and k are not known yet, so every combination of
//!     N in {10k, 100k, 1M}  x  k in {5, 10}
//! is a cell, and each cell must pass both gates:
//!   - recall@k Wilson lb95 >= 0.95 against exact FlatIndex, on two query
//!     sets: held-out (fresh draws, never stored) and near-copy (stored
//!     rows + N(0, 0.01) noise, the `freeze` query style);
//!   - latency, for a budget of LOOKUPS_PER_TICK = 2 sequential lookups per
//!     60 Hz server tick inside 2 ms (Brinn, 2026-10-21; 5 per tick failed
//!     every cell, see PR #7):
//!       * single-query search p95 <= 1000 us (2 ms / 2);
//!       * tick p95 < 2 ms, where a tick is 2 distinct queries run back to
//!         back on one thread and timed as one unit.
//!     Timing uses its own 2000 held-out queries (1000 distinct ticks), so no
//!     query repeats and the caches are not flattered.
//!
//! Data: dim 128, unit-norm Gaussian blobs around 256 "zones" (fixed seeds).
//! Latency is wall-clock and machine-dependent; it is only asserted in
//! release builds. Debug builds print it and gate recall only.
//!
//! Cost: the 10k cells run in every `cargo test`. 100k and 1M cells are
//! `#[ignore = "SLOW ..."]`; run them with
//!     cargo test --release --test gates -- --ignored --test-threads=1
//! Recall passes in all six cells on master @ 3a25945 (tightest: N=10k k=10,
//! held-out lb95 0.9586). Latency passes in all six since search scores 4
//! rows per kernel call (numbers under "cells" below).

mod common;
use common::*;

use raggedy_anndy::eval::wilson_lower_bound;
use raggedy_anndy::par::parallel_map_indexed;
use raggedy_anndy::pick::{pick_ranked, pick_seed, PICK_DEFAULT, TOP_K_DEFAULT};
use raggedy_anndy::seed::SplitMix64;
use raggedy_anndy::views;
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
const LOOKUPS_PER_TICK: usize = 2;
const MAX_TICK_P95_US: f64 = 2_000.0;
const MAX_QUERY_P95_US: f64 = MAX_TICK_P95_US / LOOKUPS_PER_TICK as f64; // 1000 us
const NQ_TIMING: usize = 2_000;
const WARMUP: usize = 50;

// ------------------------------------------------------------------ params

/// Index shape per corpus size. These are the knobs a fix PR tunes; the
/// gates themselves (MIN_LB95, MAX_QUERY_P95_US, MAX_TICK_P95_US) do not move.
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
    timing: Dataset,               // held-out, no truth needed
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
    let timing = clustered(NQ_TIMING, DIM, &centres, SPREAD, 777);
    let flat = flat_from(&data, Metric::Cosine);
    let truth_held_out = exact_truth(&flat, &held_out);
    let truth_near = exact_truth(&flat, &near);
    drop(flat);
    let rows = ivfpq_rows(&data);
    drop(data);
    let t0 = Instant::now();
    let index = IvfPqIndex::build(Metric::Cosine, DIM, &rows, params_for(n));
    let build_s = t0.elapsed().as_secs_f64();
    Fixture { index, held_out, near, timing, truth_held_out, truth_near, build_s }
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
    tick_p50_us: f64, tick_p95_us: f64,
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

/// Per-tick wall-clock: LOOKUPS_PER_TICK distinct queries back to back.
fn tick_latency_us(idx: &IvfPqIndex, qs: &Dataset, k: usize) -> (f64, f64) {
    for i in 0..WARMUP.min(qs.ids.len()) { std::hint::black_box(idx.search(row(qs, i), k)); }
    let ticks = qs.ids.len() / LOOKUPS_PER_TICK;
    let mut us: Vec<f64> = (0..ticks).map(|t| {
        let t0 = Instant::now();
        for j in 0..LOOKUPS_PER_TICK { std::hint::black_box(idx.search(row(qs, t * LOOKUPS_PER_TICK + j), k)); }
        t0.elapsed().as_secs_f64() * 1e6
    }).collect();
    us.sort_by(|a, b| a.total_cmp(b));
    (percentile_us(&us, 0.50), percentile_us(&us, 0.95))
}

fn measure(n: usize, k: usize) -> CellResult {
    let f = fixture(n);
    let th = tally(&f.index, &f.held_out, &f.truth_held_out, k);
    let tn = tally(&f.index, &f.near, &f.truth_near, k);
    let (p50_us, p95_us) = latency_us(&f.index, &f.timing, k);
    let (tick_p50_us, tick_p95_us) = tick_latency_us(&f.index, &f.timing, k);
    CellResult {
        n, k,
        recall_held_out: recall(th), lb_held_out: wilson_lower_bound(th.hits, th.possible, 1.96),
        recall_near: recall(tn), lb_near: wilson_lower_bound(tn.hits, tn.possible, 1.96),
        p50_us, p95_us, tick_p50_us, tick_p95_us, build_s: f.build_s,
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Gate { Recall, Latency }

fn gate(n: usize, k: usize, which: Gate) {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let r = measure(n, k);
    let p = params_for(n);
    eprintln!(
        "GATE N={} k={} nlist={} nprobe={} refine={} m={} | held-out recall={:.4} lb95={:.4} | near recall={:.4} lb95={:.4} | query p50={:.0}us p95={:.0}us | tick({LOOKUPS_PER_TICK}) p50={:.0}us p95={:.0}us | build={:.1}s",
        r.n, r.k, p.nlist, p.nprobe, p.refine, p.m,
        r.recall_held_out, r.lb_held_out, r.recall_near, r.lb_near,
        r.p50_us, r.p95_us, r.tick_p50_us, r.tick_p95_us, r.build_s);

    let mut fails = Vec::new();
    if which == Gate::Recall {
    if r.lb_held_out < MIN_LB95 { fails.push(format!("held-out lb95 {:.4} < {MIN_LB95}", r.lb_held_out)); }
    if r.lb_near < MIN_LB95 { fails.push(format!("near-copy lb95 {:.4} < {MIN_LB95}", r.lb_near)); }
    } else if cfg!(debug_assertions) {
        eprintln!("  (debug build: latency not gated; use --release)");
    } else {
        if r.p95_us > MAX_QUERY_P95_US { fails.push(format!("query p95 {:.0}us > {MAX_QUERY_P95_US}us", r.p95_us)); }
        if r.tick_p95_us >= MAX_TICK_P95_US { fails.push(format!("tick p95 {:.0}us >= {MAX_TICK_P95_US}us", r.tick_p95_us)); }
    }
    assert!(fails.is_empty(), "N={n} k={k}: {}", fails.join("; "));
}

// ------------------------------------------------------------------ cells
// Recall and latency are separate tests so a latency regression does not
// hide recall results. 4-core container, release, 2 lookups/tick
// (query p95 / tick p95):
//            master @ 3a25945        4-row kernels (this branch)
//   10k  k5   522 us / 1088 us        263 us /  636 us
//   10k  k10  506 us / 1066 us        310 us /  506 us
//   100k k5   585 us /  873 us        256 us /  572 us
//   100k k10  570 us /  935 us        320 us /  552 us
//   1M   k5  1451 us / 2767 us        643 us / 1324 us
//   1M   k10 1472 us / 3107 us        666 us / 1212 us
// Results (ids and score bits) are unchanged: tests/search_equivalence.rs.

#[test] fn recall_n10k_k5() { gate(10_000, 5, Gate::Recall); }
#[test] fn recall_n10k_k10() { gate(10_000, 10, Gate::Recall); }

#[test]
#[ignore = "SLOW: 100k build (~20 s) + exact truth"]
fn recall_n100k_k5() { gate(100_000, 5, Gate::Recall); }

#[test]
#[ignore = "SLOW: 100k build (~20 s) + exact truth"]
fn recall_n100k_k10() { gate(100_000, 10, Gate::Recall); }

#[test]
#[ignore = "SLOW: 1M build (~2 min) + exact truth, ~2.5 GB RAM"]
fn recall_n1m_k5() { gate(1_000_000, 5, Gate::Recall); }

#[test]
#[ignore = "SLOW: 1M build (~2 min) + exact truth, ~2.5 GB RAM"]
fn recall_n1m_k10() { gate(1_000_000, 10, Gate::Recall); }

#[test]
fn latency_n10k_k5() { gate(10_000, 5, Gate::Latency); }

#[test]
fn latency_n10k_k10() { gate(10_000, 10, Gate::Latency); }

#[test]
#[ignore = "SLOW: 100k build (~20 s) + exact truth"]
fn latency_n100k_k5() { gate(100_000, 5, Gate::Latency); }

#[test]
#[ignore = "SLOW: 100k build (~20 s) + exact truth"]
fn latency_n100k_k10() { gate(100_000, 10, Gate::Latency); }

#[test]
#[ignore = "SLOW: 1M build (~2 min) + exact truth, ~2.5 GB RAM"]
fn latency_n1m_k5() { gate(1_000_000, 5, Gate::Latency); }

#[test]
#[ignore = "SLOW: 1M build (~2 min) + exact truth, ~2.5 GB RAM"]
fn latency_n1m_k10() { gate(1_000_000, 10, Gate::Latency); }

// ------------------------------------------------------------------ pick
// Seeded 2-of-5 (Brinn, 2026-10-21): one k=5 lookup, then pick 2. Same
// latency gates as above, measured on `search_pick`, plus the pick alone.

fn percentiles(mut us: Vec<f64>) -> (f64, f64) {
    us.sort_by(|a, b| a.total_cmp(b));
    (percentile_us(&us, 0.50), percentile_us(&us, 0.95))
}

fn pick_gate(n: usize) {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let f = fixture(n);
    let (idx, qs) = (&f.index, &f.timing);
    let (k, p) = (TOP_K_DEFAULT, PICK_DEFAULT);
    let seed = |i: usize| pick_seed((i / LOOKUPS_PER_TICK) as u64, 7, (i % LOOKUPS_PER_TICK) as u64);
    let sp = |i: usize| idx.search_pick(row(qs, i), k, p, None, 0, 0.0, seed(i));
    for i in 0..WARMUP.min(qs.ids.len()) { std::hint::black_box(sp(i)); }

    let (q50, q95) = percentiles((0..qs.ids.len()).map(|i| {
        let t0 = Instant::now();
        std::hint::black_box(sp(i));
        t0.elapsed().as_secs_f64() * 1e6
    }).collect());
    let ticks = qs.ids.len() / LOOKUPS_PER_TICK;
    let (t50, t95) = percentiles((0..ticks).map(|t| {
        let t0 = Instant::now();
        for j in 0..LOOKUPS_PER_TICK { std::hint::black_box(sp(t * LOOKUPS_PER_TICK + j)); }
        t0.elapsed().as_secs_f64() * 1e6
    }).collect());

    // The pick by itself, on real top-5 results. Batched: one pick is
    // below Instant's resolution.
    let tops: Vec<Vec<Hit>> = (0..qs.ids.len()).map(|i| idx.search(row(qs, i), k)).collect();
    const REPS: usize = 1_000;
    let t0 = Instant::now();
    for r in 0..REPS { for (i, top) in tops.iter().enumerate() {
        std::hint::black_box(pick_ranked(std::hint::black_box(top), p, pick_seed(r as u64, i as u64, 0)));
    } }
    let pick_ns = t0.elapsed().as_secs_f64() * 1e9 / (REPS * tops.len()) as f64;

    eprintln!("PICK N={n} k={k} pick={p} | search_pick p50={q50:.0}us p95={q95:.0}us | tick({LOOKUPS_PER_TICK}) p50={t50:.0}us p95={t95:.0}us | pick alone mean={pick_ns:.1}ns");
    if cfg!(debug_assertions) { eprintln!("  (debug build: latency not gated; use --release)"); return; }
    let mut fails = Vec::new();
    if q95 > MAX_QUERY_P95_US { fails.push(format!("search_pick p95 {q95:.0}us > {MAX_QUERY_P95_US}us")); }
    if t95 >= MAX_TICK_P95_US { fails.push(format!("tick p95 {t95:.0}us >= {MAX_TICK_P95_US}us")); }
    assert!(fails.is_empty(), "N={n} pick: {}", fails.join("; "));
}

#[test]
fn pick_latency_n10k() { pick_gate(10_000); }

#[test]
#[ignore = "SLOW: 100k build (~20 s) + exact truth"]
fn pick_latency_n100k() { pick_gate(100_000); }

#[test]
#[ignore = "SLOW: 1M build (~2 min) + exact truth, ~2.5 GB RAM"]
fn pick_latency_n1m() { pick_gate(1_000_000); }

// ------------------------------------------------------------------ pending
// View queries also scan the pending real-memory buffer (exact, every
// pending row) and ask the index for k + (pending ids still in the index)
// hits, so pending ids are dropped from the index side without leaving
// fewer than k. Same gates, measured on `views::search_temporal` over a
// published view of the fixture plus PENDING_GATE real memories, twice:
// with their short-term rows still live (first hour: the index is asked
// for k + PENDING_GATE, the worst case) and expired (after the hour).
// Expected pending size is not known yet (real memories per 12 h), so the
// gate uses PENDING_GATE and `pending_sweep` prints larger sizes.

// 4-core container, release, N=1M k=5, tick p95 (2 lookups) by pending size:
//   pending   rows live   rows expired
//      0       848 us      1059 us   (no pending: the plain path, noise)
//    256      1132 us      1170 us
//   1024      1676 us       929 us
//   4096      3306 us      1638 us
//  16384      6648 us      3505 us
// The live case costs more because the index is asked for k + pending.
const PENDING_GATE: usize = 1_024;

/// A writer over the fixture with `p` pending real memories (vectors from
/// the corpus distribution, seeded), published. Returns the view, the
/// writer (which owns the blocks) and the pending vectors.
fn pending_view(n: usize, p: usize, expired: bool) -> (std::sync::Arc<views::View>, views::Writer, Dataset) {
    use raggedy_anndy::real_memory::{self as real, Outbox, RealOp, REAL};
    let f = fixture(n);
    let centres = cluster_centres(ZONES, DIM, 1234);
    let pend = clustered(p, DIM, &centres, SPREAD, 4242);
    let mut w = views::writer_from(&f.index, views::DEFAULT_BLOCK_ROWS);
    let mut buf = real::pending_for(&f.index);
    let mut outbox = Outbox::default();
    for i in 0..p {
        let op = RealOp::InsertReal { id: 50_000_000 + i as u64, ts: 1, tags: 1, flags: REAL, vec: row(&pend, i).to_vec() };
        real::apply_writer(&mut w, &mut buf, &mut outbox, &op).unwrap();
    }
    // Take their short-term rows out of the index (as the 1-hour expiry
    // would) by id: an expiry by ts would take the corpus (ts 0) too.
    if expired { for i in 0..p { views::delete(&mut w, 50_000_000 + i as u64).unwrap(); } }
    views::publish(&mut w);
    let v = views::current(&views::latest(&w));
    assert_eq!(views::pending_rows(&v), p);
    (v, w, pend)
}

fn pending_gate(n: usize, k: usize, p: usize, gated: bool) {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for expired in [false, true] { pending_cell(n, k, p, expired, gated); }
}

fn pending_cell(n: usize, k: usize, p: usize, expired: bool, gated: bool) {
    let f = fixture(n);
    let (v, _w, pend) = pending_view(n, p, expired);
    let qs = &f.timing;
    let s = |i: usize| views::search_temporal(&v, row(qs, i), k, None, 0, 0.0);
    // Found: each pending row is the top hit for its own vector.
    let probe = p.min(200);
    let found = (0..probe).filter(|&i| {
        views::search(&v, row(&pend, i), 1).first().map(|h| h.id) == Some(50_000_000 + i as u64)
    }).count();
    for i in 0..WARMUP.min(qs.ids.len()) { std::hint::black_box(s(i)); }
    let (q50, q95) = percentiles((0..qs.ids.len()).map(|i| {
        let t0 = Instant::now();
        std::hint::black_box(s(i));
        t0.elapsed().as_secs_f64() * 1e6
    }).collect());
    let ticks = qs.ids.len() / LOOKUPS_PER_TICK;
    let (t50, t95) = percentiles((0..ticks).map(|t| {
        let t0 = Instant::now();
        for j in 0..LOOKUPS_PER_TICK { std::hint::black_box(s(t * LOOKUPS_PER_TICK + j)); }
        t0.elapsed().as_secs_f64() * 1e6
    }).collect());
    let rows = if expired { "expired" } else { "live" };
    eprintln!("PENDING N={n} k={k} pending={p} rows={rows} | found {found}/{probe} | view query p50={q50:.0}us p95={q95:.0}us | tick({LOOKUPS_PER_TICK}) p50={t50:.0}us p95={t95:.0}us");
    assert_eq!(found, probe, "N={n} pending={p} rows={rows}: a pending memory was not its own top hit");
    if !gated { return; }
    if cfg!(debug_assertions) { eprintln!("  (debug build: latency not gated; use --release)"); return; }
    let mut fails = Vec::new();
    if q95 > MAX_QUERY_P95_US { fails.push(format!("view query p95 {q95:.0}us > {MAX_QUERY_P95_US}us")); }
    if t95 >= MAX_TICK_P95_US { fails.push(format!("tick p95 {t95:.0}us >= {MAX_TICK_P95_US}us")); }
    assert!(fails.is_empty(), "N={n} k={k} pending={p} rows={rows}: {}", fails.join("; "));
}

#[test] fn pending_latency_n10k_k5() { pending_gate(10_000, 5, PENDING_GATE, true); }
#[test] fn pending_latency_n10k_k10() { pending_gate(10_000, 10, PENDING_GATE, true); }

#[test]
#[ignore = "SLOW: 100k build (~20 s) + exact truth"]
fn pending_latency_n100k_k5() { pending_gate(100_000, 5, PENDING_GATE, true); }

#[test]
#[ignore = "SLOW: 100k build (~20 s) + exact truth"]
fn pending_latency_n100k_k10() { pending_gate(100_000, 10, PENDING_GATE, true); }

#[test]
#[ignore = "SLOW: 1M build (~2 min) + exact truth, ~3 GB RAM"]
fn pending_latency_n1m_k5() { pending_gate(1_000_000, 5, PENDING_GATE, true); }

#[test]
#[ignore = "SLOW: 1M build (~2 min) + exact truth, ~3 GB RAM"]
fn pending_latency_n1m_k10() { pending_gate(1_000_000, 10, PENDING_GATE, true); }

/// Sizing aid, not a gate: view query latency as the pending buffer grows,
/// for the N in RA_SWEEP_N (default 10000), k=5.
#[test]
#[ignore = "sizing aid: RA_SWEEP_N=1000000 cargo test --release --test gates pending_sweep -- --ignored --nocapture"]
fn pending_sweep() {
    let n: usize = std::env::var("RA_SWEEP_N").ok().and_then(|s| s.parse().ok()).unwrap_or(10_000);
    for p in [0usize, 256, 1_024, 4_096, 16_384, 65_536] { pending_gate(n, 5, p, false); }
}

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

/// Profiling aid, not a gate: for the N in RA_SWEEP_N (default 1M), prints
/// list-size spread, rows scanned per query, and where search time goes,
/// by replaying the search phases over the index's public data with the
/// same kernels `IvfPqIndex::search_temporal` uses (rerank here is scalar).
#[test]
#[ignore = "profiling aid: RA_SWEEP_N=1000000 cargo test --release --test gates profile -- --ignored --nocapture"]
fn profile_search_phases() {
    use raggedy_anndy::metric;
    let n: usize = std::env::var("RA_SWEEP_N").ok().and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let f = fixture(n);
    let idx = &f.index;
    let mut lens: Vec<usize> = idx.lists.iter().map(|l| l.len()).collect();
    lens.sort();
    let pct = |v: &[usize], p: f64| v[((p * v.len() as f64).ceil() as usize).clamp(1, v.len()) - 1];
    eprintln!("lists={} len min={} p50={} p95={} max={}", lens.len(), lens[0], pct(&lens, 0.5), pct(&lens, 0.95), lens[lens.len() - 1]);

    let (nprobe, m, k) = (idx.params.nprobe, idx.params.m, 10usize);
    let (mut t_coarse, mut t_lut, mut t_scan, mut t_sel, mut t_rerank) = (vec![], vec![], vec![], vec![], vec![]);
    let mut scanned = vec![];
    for i in 0..f.timing.ids.len() {
        let mut q = row(&f.timing, i).to_vec(); normalize(&mut q);
        let t0 = Instant::now();
        let mut probe: Vec<(usize, f32)> = Vec::with_capacity(idx.centroids.len());
        for (b, c) in idx.centroids.chunks(4).enumerate() {
            if c.len() == 4 {
                let s = metric::dot_x4(&q, [&c[0][..], &c[1][..], &c[2][..], &c[3][..]]);
                for t in 0..4 { probe.push((b * 4 + t, s[t])); }
            } else { for (t, cc) in c.iter().enumerate() { probe.push((b * 4 + t, metric::dot(&q, cc))); } }
        }
        probe.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        probe.truncate(nprobe);
        let t1 = Instant::now();
        let lut = idx.pq.adc_lut_ip(&q);
        let t2 = Instant::now();
        let mut approx: Vec<(f32, u64, usize, usize)> = Vec::new();
        let mut sums = Vec::new();
        for &(li, cs) in &probe {
            let l = &idx.lists[li];
            sums.resize(l.len(), 0.0);
            idx.pq.adc_sum_rows(&lut, &l.codes[..l.len() * m], &mut sums);
            for r in 0..l.len() {
                if l.tomb[r] == 1 { continue; }
                approx.push((cs + sums[r], l.ids[r], li, r));
            }
        }
        let t3 = Instant::now();
        scanned.push(approx.len());
        let keep = idx.params.refine.max(k);
        if approx.len() > keep { approx.select_nth_unstable_by(keep - 1, |a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))); approx.truncate(keep); }
        let t4 = Instant::now();
        let hits: Vec<Hit> = approx.iter().map(|&(_, id, li, r)| Hit { id, score: metric::dot(&q, idx.lists[li].vec_row(DIM, r)) }).collect();
        std::hint::black_box(raggedy_anndy::stable_top_k(hits, k));
        let t5 = Instant::now();
        for (v, a, b) in [(&mut t_coarse, t0, t1), (&mut t_lut, t1, t2), (&mut t_scan, t2, t3), (&mut t_sel, t3, t4), (&mut t_rerank, t4, t5)] {
            v.push((b - a).as_secs_f64() * 1e6);
        }
    }
    scanned.sort();
    eprintln!("rows scanned/query p50={} p95={} max={}", pct(&scanned, 0.5), pct(&scanned, 0.95), scanned[scanned.len() - 1]);
    for (name, mut v) in [("coarse", t_coarse), ("lut", t_lut), ("scan", t_scan), ("select", t_sel), ("rerank", t_rerank)] {
        v.sort_by(|a, b| a.total_cmp(b));
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        eprintln!("{name:7} mean={mean:7.1}us p50={:7.1}us p95={:7.1}us", percentile_us(&v, 0.5), percentile_us(&v, 0.95));
    }
}
