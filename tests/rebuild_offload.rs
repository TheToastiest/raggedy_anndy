//! Moving rebuild work off the writer thread without changing any result.
//!
//! Measured on master (tests/rebuild_cost.rs, 128-dim, nlist 1024, m 32):
//! `begin_rebuild` copied every live row on the writer (70 ms at 100k rows,
//! 1.2 s at 2.16M) and `swap` re-copied every row into new blocks (87 ms,
//! 3.5 s); the tick budget at 60 Hz is 16.7 ms. The swap gate re-encoded
//! every snapshot row on one thread (250 s at 2.16M).
//!
//! Now:
//! - `capture_rows` (writer) takes the rows by reference (rows never move,
//!   so holding the block handles is enough) and `snapshot_of` (any thread)
//!   copies them out. Same `Snapshot` as `begin_rebuild`, whatever the
//!   writer does in between.
//! - `prepare` (any thread) builds the next writer's blocks from the
//!   rebuilt index, `catch_up_prepared` applies the carried ops to it in
//!   any number of pieces, and `swap_prepared` (writer) installs it.
//!   Same system state, views and log entry as `catch_up` + `swap`.
//! - The gate encodes rows on `threads` threads and appends them in order:
//!   the same `Gate`, bit for bit, for every thread count, equal to the
//!   value master logged for the same fixture.

mod common;
use common::*;

use raggedy_anndy::drift::Signals;
use raggedy_anndy::gate::{self, GameCheck, Gate, GateParams, GateReason};

use raggedy_anndy::live::Op;
use raggedy_anndy::real_memory::{RealOp, REAL};
use raggedy_anndy::rebuild::{self, LogOp, RebuildError, Snapshot};
use raggedy_anndy::seed::SplitMix64;
use raggedy_anndy::system::{self, System};
use raggedy_anndy::views::{self, CompactPolicy};
use raggedy_anndy::durable::{self, DurableParams};

use std::time::{Duration, Instant};
use raggedy_anndy::{IvfPqIndex, Metric};

const DIM: usize = 32;
const NLIST: usize = 16;
const M: usize = 8;
const BLOCK: usize = 64;
const BASE_N: usize = 3000;

/// Compacts often, so retired blocks are in play while a capture is held.
const POLICY: CompactPolicy = CompactPolicy { window_ticks: 20, max_dead_share: 0.2, max_dead_rows: 48, max_per_tick: 3 };

/// FNV-1a fold.
fn fnv(h: u64, x: u64) -> u64 { (h ^ x).wrapping_mul(0x100000001b3) }

// ------------------------------------------------------------ fixture

fn base_index(opq: bool) -> IvfPqIndex {
    let ds = uniform_sphere(BASE_N, DIM, 5);
    let rows: Vec<(u64, Vec<f32>, u64)> = (0..BASE_N).map(|i| (ds.ids[i], row(&ds, i).to_vec(), i as u64)).collect();
    let mut p = ivfpq_params(NLIST, 4, 64, M, 7);
    p.use_opq = opq;
    IvfPqIndex::build(Metric::Cosine, DIM, &rows, p)
}

fn unit(rng: &mut SplitMix64) -> Vec<f32> {
    let mut v: Vec<f32> = (0..DIM).map(|_| gauss(rng)).collect();
    normalize(&mut v);
    v
}

/// Seeded mix for one tick: inserts, upserts, deletes, capped expiry, real
/// memories, explicit compactions.
fn tick_ops(rng: &mut SplitMix64, tick: u64, next_id: &mut u64) -> Vec<RealOp> {
    let ts = 10_000 + tick;
    let mut ops = Vec::new();
    for _ in 0..1 + rng.gen_range(10) {
        let any = |rng: &mut SplitMix64, next: u64| rng.gen_range(next as usize + 1) as u64;
        ops.push(match rng.gen_range(20) {
            0..=8 => { *next_id += 1; RealOp::Live(Op::Insert { id: *next_id, ts, tags: rng.next_u64() & 0xff, vec: unit(rng) }) }
            9..=10 => RealOp::Live(Op::Upsert { id: any(rng, *next_id), ts, tags: 1, vec: unit(rng) }),
            11..=13 => RealOp::Live(Op::Delete { id: any(rng, *next_id) }),
            14 => RealOp::Live(Op::Expire { before_ts: (tick * 3).min(ts), max_rows: 30 }),
            15 => { *next_id += 1; RealOp::InsertReal { id: *next_id, ts, tags: 2, flags: REAL, vec: unit(rng) } }
            16 => RealOp::Live(Op::Compact { list: rng.gen_range(NLIST) as u32 }),
            _ => RealOp::Live(Op::Delete { id: any(rng, *next_id) }),
        });
    }
    ops
}

/// The writer's loop without a log: ops, policy compactions, publish.
struct Driver { rng: SplitMix64, next_id: u64 }

fn driver(seed: u64) -> Driver { Driver { rng: SplitMix64::new(seed), next_id: BASE_N as u64 } }

fn run_ticks(sys: &mut System, d: &mut Driver, ticks: u64) {
    let mut compact = Vec::new();
    for _ in 0..ticks {
        let tick = sys.w.seq;
        for op in tick_ops(&mut d.rng, tick, &mut d.next_id) { let _ = system::write(sys, &op); }
        views::compactions_due(&POLICY, &sys.w, &mut compact);
        for op in compact.drain(..) { system::write(sys, &RealOp::Live(op)).unwrap(); }
        views::publish(&mut sys.w);
    }
}

fn assert_ok(sys: &System) {
    if let Err(e) = system::check_invariants(sys) { panic!("invariant broken: {e}"); }
}

/// FNV-1a over everything in a snapshot; the model by `model_hash`.
fn snapshot_hash(s: &Snapshot) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for x in [s.seq, s.epoch, s.max_ts, s.dim as u64, matches!(s.metric, Metric::Cosine) as u64, rebuild::model_hash(&s.model)] { h = fnv(h, x); }
    for x in s.baseline.qerr.iter().chain(&s.baseline.rnorm) { h = fnv(h, x.to_bits()); }
    for &x in &s.baseline.rows { h = fnv(h, x); }
    h = fnv(h, s.ids.len() as u64);
    for i in 0..s.ids.len() { h = fnv(fnv(fnv(h, s.ids[i]), s.ts[i]), s.tags[i]); }
    for x in &s.vecs { h = fnv(h, x.to_bits() as u64); }
    h
}

fn signals_hash(h: u64, s: &Signals) -> u64 {
    let mut h = h;
    for x in [s.qerr_ratio, s.rnorm_ratio, s.imbalance, s.churn] { h = fnv(h, x.to_bits()); }
    fnv(fnv(h, s.inserts), s.fired as u64)
}

/// FNV-1a over every measured field of a gate.
fn gate_hash(g: &Gate) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for x in [g.queries, g.trials, g.current_hits, g.candidate_hits] { h = fnv(h, x as u64); }
    h = signals_hash(h, &g.current);
    h = signals_hash(h, &g.candidate);
    fnv(h, g.accept as u64)
}

/// A gate that accepts; the gate's measurements play no part in a swap.
fn accepting() -> Gate {
    let s = Signals { qerr_ratio: 0.0, rnorm_ratio: 0.0, imbalance: 0.0, churn: 0.0, inserts: 0, fired: 0 };
    Gate {
        params: gate::default_gate_params(), game: GameCheck::NotRun, queries: 0, trials: 0, current_hits: 0, candidate_hits: 0,
        current: s, candidate: s, accept: true, reason: GateReason::RecallGain,
    }
}

/// Small gate so the fixture runs quickly; every code path still runs.
fn small_gate() -> GateParams { GateParams { queries: 64, recent: 1000, ..gate::default_gate_params() } }

/// A system that has run `ticks` ticks of the seeded mix.
fn running(opq: bool, seed: u64, ticks: u64) -> (System, Driver) {
    let base = base_index(opq);
    let mut sys = system::start(&base, BLOCK);
    let mut d = driver(seed);
    run_ticks(&mut sys, &mut d, ticks);
    (sys, d)
}

// ------------------------------------------------------------ golden values (from master)

/// Pinned on master before any of this change: the snapshot and the gate
/// for the fixture. The change must not move them.
const GOLDEN_SNAPSHOT: [u64; 2] = [0xba1ebc9c9968327f, 0xe671a2092852451a];
const GOLDEN_GATE: [u64; 2] = [0x701bb1dd96b1adf7, 0x7af9d36c8a3155b5];
const GOLDEN_SWAP: [u64; 2] = [0x04c78127c51993ae, 0x2e2c8673ce1171fe];

#[test]
fn snapshot_and_gate_match_master() {
    let mut got = Vec::new();
    for (k, opq) in [false, true].into_iter().enumerate() {
        let (mut sys, _) = running(opq, 11, 400);
        let snap = system::begin_rebuild(&mut sys).unwrap();
        let r = rebuild::rebuild(&snap, sys.st.base_seed, 2).unwrap();
        let g = gate::evaluate(&snap, &r, &small_gate(), GameCheck::NotRun, 1).unwrap();
        println!("opq={opq}: snapshot {:#018x} rows {}, gate {:#018x} {g:?}", snapshot_hash(&snap), snap.ids.len(), gate_hash(&g));
        got.push((k, opq, snapshot_hash(&snap), gate_hash(&g)));
    }
    for (k, opq, s, g) in got {
        assert_eq!(s, GOLDEN_SNAPSHOT[k], "opq={opq}: snapshot");
        assert_eq!(g, GOLDEN_GATE[k], "opq={opq}: gate");
    }
}

/// Snapshot, 150 ticks of writes while the worker trains, catch-up, swap,
/// 100 more ticks: the system fingerprint, pinned on master.
#[test]
fn swap_matches_master() {
    let mut got = Vec::new();
    for (k, opq) in [false, true].into_iter().enumerate() {
        let (mut sys, mut d) = running(opq, 11, 400);
        let snap = system::begin_rebuild(&mut sys).unwrap();
        run_ticks(&mut sys, &mut d, 150);
        let mut r = rebuild::rebuild(&snap, sys.st.base_seed, 2).unwrap();
        rebuild::catch_up(&mut r, sys.carry.as_ref().unwrap()).unwrap();
        let e = system::swap(&mut sys, r, accepting()).unwrap();
        assert!(matches!(e, LogOp::SwapModel { epoch: 1, .. }));
        views::publish(&mut sys.w);
        run_ticks(&mut sys, &mut d, 100);
        assert_ok(&sys);
        println!("opq={opq}: swap fingerprint {:#018x}", system::fingerprint(&sys));
        got.push((k, opq, system::fingerprint(&sys)));
    }
    for (k, opq, f) in got { assert_eq!(f, GOLDEN_SWAP[k], "opq={opq}"); }
}

// ------------------------------------------------------------ capture on the writer, copy on a worker

#[test]
fn capture_then_snapshot_equals_begin_rebuild() {
    for (k, opq) in [false, true].into_iter().enumerate() {
        let (mut sys, mut d) = running(opq, 11, 400);
        let cap = system::capture_rows(&mut sys).unwrap();
        assert!(matches!(system::capture_rows(&mut sys), Err(RebuildError::RebuildPending { .. })));
        assert!(matches!(system::begin_rebuild(&mut sys), Err(RebuildError::RebuildPending { .. })));
        // The writer goes on: inserts into the same blocks, deletes,
        // expiry, compactions that retire blocks the capture still holds.
        run_ticks(&mut sys, &mut d, 200);
        assert_ok(&sys);
        let snap = std::thread::spawn(move || system::snapshot_of(&cap)).join().unwrap();
        assert_eq!(snapshot_hash(&snap), GOLDEN_SNAPSHOT[k], "opq={opq}");
        assert_eq!(sys.carry.as_ref().unwrap().snapshot_seq, snap.seq);
    }
}

#[test]
fn cancel_ends_a_capture() {
    let (mut sys, _) = running(false, 3, 20);
    system::capture_rows(&mut sys).unwrap();
    system::cancel(&mut sys);
    assert!(sys.carry.is_none());
    assert!(system::capture_rows(&mut sys).is_ok());
}

// ------------------------------------------------------------ prepare on a worker, install on the writer

/// The worker's side, run on its own thread: rebuild, gate, prepare, and
/// catch up on what the writer has carried so far.
fn worker_prepare(snap: Snapshot, base_seed: u64, threads: usize, carried: Vec<Op>) -> (system::Prepared, Gate) {
    std::thread::spawn(move || {
        let r = rebuild::rebuild(&snap, base_seed, threads).unwrap();
        let g = gate::evaluate(&snap, &r, &small_gate(), GameCheck::NotRun, threads).unwrap();
        let mut p = system::prepare(r, BLOCK);
        system::catch_up_prepared(&mut p, 0, &carried).unwrap();
        (p, g)
    }).join().unwrap()
}

/// The prepared writer handed back by a refused swap, after checking why.
fn refused(r: Result<(LogOp, views::Retired), system::SwapRefused>, why: impl Fn(&RebuildError) -> bool) -> system::Prepared {
    match r {
        Err(e) => { assert!(why(&e.err), "refused for {:?}", e.err); *e.prepared }
        Ok((e, _)) => panic!("swap went through: {e:?}"),
    }
}

#[test]
fn prepared_swap_equals_swap() {
    for (k, opq) in [false, true].into_iter().enumerate() {
        for threads in [1usize, 3] {
            // The current path, on a twin.
            let (mut a, mut da) = running(opq, 11, 400);
            let snap_a = system::begin_rebuild(&mut a).unwrap();
            run_ticks(&mut a, &mut da, 150);
            let mut r = rebuild::rebuild(&snap_a, a.st.base_seed, 2).unwrap();
            let ga = gate::evaluate(&snap_a, &r, &small_gate(), GameCheck::NotRun, 1).unwrap();
            rebuild::catch_up(&mut r, a.carry.as_ref().unwrap()).unwrap();
            let ea = system::swap(&mut a, r, ga).unwrap();

            // Off the writer: capture, 50 ticks, the worker prepares and
            // catches up on those, 100 more ticks, the writer applies the
            // rest and installs.
            let (mut b, mut db) = running(opq, 11, 400);
            let cap = system::capture_rows(&mut b).unwrap();
            let snap_b = std::thread::spawn(move || system::snapshot_of(&cap)).join().unwrap();
            run_ticks(&mut b, &mut db, 50);
            let so_far = system::carry_since(&b, 0).unwrap().to_vec();
            assert!(!so_far.is_empty());
            let (p, gb) = worker_prepare(snap_b, b.st.base_seed, threads, so_far);
            assert_eq!(gb, ga, "opq={opq} threads={threads}: gate");
            run_ticks(&mut b, &mut db, 100);
            let mut p = refused(system::swap_prepared(&mut b, p, gb), |e| matches!(e, RebuildError::NotCaughtUp { .. }));
            let at = p.carried;
            let rest = system::carry_since(&b, at).unwrap().to_vec();
            assert!(!rest.is_empty());
            system::catch_up_prepared(&mut p, at, &rest).unwrap();
            let (eb, retired) = system::swap_prepared(&mut b, p, gb).unwrap();

            assert_eq!(eb, ea, "opq={opq} threads={threads}: log entry");
            assert_eq!(b.st, a.st, "opq={opq} threads={threads}: state");
            assert_eq!(b.w.dead_since, a.w.dead_since);
            assert_eq!(system::fingerprint(&b), system::fingerprint(&a), "opq={opq} threads={threads}: system");
            assert_ok(&b);

            views::publish(&mut a.w);
            views::publish(&mut b.w);
            let (va, vb) = (views::current(&views::latest(&a.w)), views::current(&views::latest(&b.w)));
            assert_eq!(views::fingerprint(&vb), views::fingerprint(&va), "opq={opq} threads={threads}: view");
            let mut rng = SplitMix64::new(77);
            for _ in 0..32 {
                let q = unit(&mut rng);
                let bits = |v: &views::View| views::search(v, &q, 10).iter().map(|h| (h.id, h.score.to_bits())).collect::<Vec<_>>();
                assert_eq!(bits(&vb), bits(&va), "opq={opq} threads={threads}: query");
            }
            drop((va, vb));
            // Freed off the writer thread, after the publish.
            std::thread::spawn(move || drop(retired)).join().unwrap();
            run_ticks(&mut a, &mut da, 100);
            run_ticks(&mut b, &mut db, 100);
            assert_eq!(system::fingerprint(&b), system::fingerprint(&a));
            assert_eq!(system::fingerprint(&b), GOLDEN_SWAP[k], "opq={opq} threads={threads}: after 100 more ticks");
            assert_ok(&b);
        }
    }
}

#[test]
fn prepared_catch_up_and_swap_refuse_mismatches() {
    let (mut sys, mut d) = running(false, 5, 300);
    let snap = system::begin_rebuild(&mut sys).unwrap();
    run_ticks(&mut sys, &mut d, 20);
    let n = system::carry_since(&sys, 0).unwrap().len();
    assert!(n > 2);
    assert!(matches!(system::carry_since(&sys, n + 1), Err(RebuildError::CarryGap { .. })));
    let r = rebuild::rebuild(&snap, sys.st.base_seed, 1).unwrap();
    let mut p = system::prepare(r, BLOCK);
    let ops = system::carry_since(&sys, 0).unwrap().to_vec();
    // Not where it left off: refused, nothing applied.
    assert!(matches!(system::catch_up_prepared(&mut p, 1, &ops[1..]), Err(RebuildError::CarryGap { at: 1, carried: 0 })));
    assert_eq!(p.carried, 0);
    system::catch_up_prepared(&mut p, 0, &ops[..2]).unwrap();
    assert_eq!(p.carried, 2);
    let before = (system::fingerprint(&sys), sys.st.clone());
    let mut p = refused(system::swap_prepared(&mut sys, p, accepting()), |e| matches!(e, RebuildError::NotCaughtUp { carried: 2, .. }));
    let mut rejecting = accepting();
    rejecting.accept = false;
    let mut p = refused(system::swap_prepared(&mut sys, p, rejecting), |e| matches!(e, RebuildError::GateDecision { accept: false }));
    system::catch_up_prepared(&mut p, 2, &ops[2..]).unwrap();
    // A different rebuild's preparation.
    p.snapshot_seq += 1;
    let mut p = refused(system::swap_prepared(&mut sys, p, accepting()), |e| matches!(e, RebuildError::WrongSnapshot { .. }));
    p.snapshot_seq -= 1;
    p.epoch += 1;
    let mut p = refused(system::swap_prepared(&mut sys, p, accepting()), |e| matches!(e, RebuildError::WrongEpoch { .. }));
    p.epoch -= 1;
    assert_eq!((system::fingerprint(&sys), sys.st.clone()), before, "refused swaps change nothing");
    let (e, _) = system::swap_prepared(&mut sys, p, accepting()).unwrap();
    assert!(matches!(e, LogOp::SwapModel { epoch: 1, .. }));
    assert_eq!(sys.st.epoch, 1);
    assert!(sys.carry.is_none());
    assert!(matches!(system::carry_since(&sys, 0), Err(RebuildError::NoRebuildPending)));
    assert_ok(&sys);
}

/// A prepared swap logged through `durable` recovers to the live state;
/// recovery replays the decision the old way (rebuild from the log), so
/// this also checks the two paths agree.
#[test]
fn durable_prepared_swap_recovers() {
    let dir = std::env::temp_dir().join(format!("raggedy_offload_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let base = base_index(false);
    let mut sys = system::start(&base, BLOCK);
    let mut now = Instant::now();
    let tick = Duration::from_micros(16_667);
    let mut w = durable::create(&dir, &sys, DurableParams { sync_every: Duration::from_secs(1), keep_snapshots: 2, snapshot_every_ticks: 0 }, now).unwrap();
    let mut d = driver(11);
    let mut ticks = |w: &mut durable::LogWriter, sys: &mut System, n: u64, now: &mut Instant| {
        for _ in 0..n {
            let t = sys.w.seq;
            for op in tick_ops(&mut d.rng, t, &mut d.next_id) { let _ = durable::write(w, sys, &op); }
            *now += tick;
            durable::end_tick(w, sys, &POLICY, *now).unwrap();
        }
    };
    ticks(&mut w, &mut sys, 400, &mut now);
    let cap = system::capture_rows(&mut sys).unwrap();
    let snap = system::snapshot_of(&cap);
    drop(cap);
    ticks(&mut w, &mut sys, 30, &mut now);
    let (mut p, g) = worker_prepare(snap, sys.st.base_seed, 2, system::carry_since(&sys, 0).unwrap().to_vec());
    assert!(g.accept, "{g:?}");
    ticks(&mut w, &mut sys, 30, &mut now);
    let at = p.carried;
    let rest = system::carry_since(&sys, at).unwrap().to_vec();
    system::catch_up_prepared(&mut p, at, &rest).unwrap();
    durable::swap_prepared(&mut w, &mut sys, p, g).unwrap();
    ticks(&mut w, &mut sys, 30, &mut now);
    durable::sync(&mut w, now).unwrap();
    drop(w);
    let rec = durable::recover(&dir, Some(&base), BLOCK, 2).unwrap();
    assert_eq!(rec.decisions, 1);
    assert_eq!(rec.sys.st, sys.st);
    assert_eq!(system::fingerprint(&rec.sys), system::fingerprint(&sys));
    let _ = std::fs::remove_dir_all(&dir);
}

// ------------------------------------------------------------ gate on any number of threads

#[test]
fn gate_is_the_same_for_any_thread_count() {
    for (k, opq) in [false, true].into_iter().enumerate() {
        let (mut sys, _) = running(opq, 11, 400);
        let snap = system::begin_rebuild(&mut sys).unwrap();
        let r = rebuild::rebuild(&snap, sys.st.base_seed, 2).unwrap();
        for threads in [1usize, 2, 3, 5, 8] {
            let g = gate::evaluate(&snap, &r, &small_gate(), GameCheck::NotRun, threads).unwrap();
            assert_eq!(gate_hash(&g), GOLDEN_GATE[k], "opq={opq} threads={threads}");
        }
    }
}
