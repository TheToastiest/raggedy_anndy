//! View queries cover the pending real-memory buffer.
//!
//! A real memory sits in the pending buffer from its insert until the
//! server's 12-hour commit. Queries on a published view must find it the
//! whole time, also after the 1-hour expiry has taken its short-term row
//! out of the index, and must stop finding it (from RAM) once it is
//! committed. These tests pin:
//! - the view of tick T sees exactly the buffer as it was at the end of T:
//!   `views::search_temporal` on it equals `real_memory::search` on
//!   (`views::to_index(view)`, buffer at T), bit for bit, every tick of a
//!   seeded churn log (real and plain inserts, upserts, deletes, expiry,
//!   commit, ack), with tag filters, decay, cosine and L2;
//! - a held view does not change while the writer goes on (pending writes,
//!   commit);
//! - each id comes back at most once, a pending id with its exact score;
//! - a snapshot load (`system::from_parts`) and a model swap
//!   (`views::rebase`) keep the buffer searchable from the first view on;
//! - the same log gives the same results on a second run.
//!
//! No-allocation of real-memory writes and publish is checked in
//! tests/live_alloc.rs, latency in tests/gates.rs.

mod common;
use common::*;

use std::collections::BTreeSet;

use raggedy_anndy::live::Op;
use raggedy_anndy::real_memory::{self as real, Pending, RealOp, LEGENDARY, REAL};
use raggedy_anndy::seed::SplitMix64;
use raggedy_anndy::system::{self, System};
use raggedy_anndy::views::{self, View};
use raggedy_anndy::{Hit, IvfPqIndex, Metric};

const DIM: usize = 32;
const NLIST: usize = 16;
const M: usize = 8;
const BLOCK: usize = 16;
/// One hour in the test's time unit (seconds).
const HOUR: u64 = 3_600;

/// Rows with ts = 0..n, as if one arrived per second.
fn base_index(metric: Metric, n: usize, seed: u64) -> IvfPqIndex {
    let ds = uniform_sphere(n, DIM, seed);
    let rows: Vec<(u64, Vec<f32>, u64)> = (0..n).map(|i| (ds.ids[i], row(&ds, i).to_vec(), i as u64)).collect();
    IvfPqIndex::build(metric, DIM, &rows, ivfpq_params(NLIST, 4, 64, M, 7))
}

fn random_unit(rng: &mut SplitMix64) -> Vec<f32> {
    let mut v: Vec<f32> = (0..DIM).map(|_| gauss(rng)).collect();
    normalize(&mut v);
    v
}

fn bits(hits: &[Hit]) -> Vec<(u64, u32)> { hits.iter().map(|h| (h.id, h.score.to_bits())).collect() }

fn write(sys: &mut System, op: RealOp) { system::write(sys, &op).unwrap_or_else(|e| panic!("{op:?}: {e}")); }

fn now_view(sys: &System) -> std::sync::Arc<View> { views::current(&views::latest(&sys.w)) }

fn top1(v: &View, q: &[f32]) -> Option<u64> { views::search(v, q, 1).first().map(|h| h.id) }

/// Query shapes compared every tick: (k, tag, current_time, lambda).
const SHAPES: [(usize, Option<u64>, u64, f32); 5] = [
    (1, None, 0, 0.0),
    (5, None, 0, 0.0),
    (10, Some(0b0101), 0, 0.0),
    (5, None, 9_000, 1e-4),
    (64, Some(0b0010), 9_000, 1e-3),
];

/// The view of tick T searched through `views` must equal the stage-5
/// merged search on the same rows and the buffer as it was at T.
fn assert_view_matches(v: &View, buf: &Pending, qs: &[Vec<f32>], ctx: &str) {
    let idx = views::to_index(v);
    for (qi, q) in qs.iter().enumerate() {
        for &(k, tag, now, lambda) in &SHAPES {
            let got = views::search_temporal(v, q, k, tag, now, lambda);
            let want = real::search(&idx, buf, q, k, tag, now, lambda);
            assert_eq!(bits(&got), bits(&want), "{ctx}: query {qi} k={k} tag={tag:?} now={now} lambda={lambda}");
            let uniq: BTreeSet<u64> = got.iter().map(|h| h.id).collect();
            assert_eq!(uniq.len(), got.len(), "{ctx}: query {qi}: an id came back twice");
        }
    }
}

/// Seeded churn, `ticks` ticks of `per_tick` ops each, one second apart:
/// plain and real inserts, upserts of pending, handed and plain ids,
/// `UpsertReal`, deletes of each kind, a capped expiry each tick (the
/// 1-hour window), a commit every 9 ticks and an ack 3 ticks after it.
fn churn(seed: u64, ticks: usize, per_tick: usize, base_n: u64) -> Vec<Vec<RealOp>> {
    let mut rng = SplitMix64::new(seed);
    let mut next_id = 1_000_000u64;
    let mut ts = base_n;
    let mut live: Vec<u64> = Vec::new();
    let mut real_ids: Vec<u64> = Vec::new();
    let mut out = Vec::with_capacity(ticks);
    let mut last_commit_seq: Option<usize> = None;
    for t in 0..ticks {
        let mut ops = Vec::with_capacity(per_tick + 3);
        for _ in 0..per_tick {
            ts += 40; // hours pass over the run, so expiry kills real rows too
            let tags = rng.next_u64() & 0xF;
            let vec = random_unit(&mut rng);
            match rng.next_u64() % 10 {
                0..=2 => {
                    let flags = if rng.next_u64() % 3 == 0 { REAL | LEGENDARY } else { REAL };
                    ops.push(RealOp::InsertReal { id: next_id, ts, tags, flags, vec });
                    real_ids.push(next_id);
                    live.push(next_id);
                    next_id += 1;
                }
                3 if !real_ids.is_empty() => {
                    let id = real_ids[(rng.next_u64() % real_ids.len() as u64) as usize];
                    ops.push(RealOp::Live(Op::Upsert { id, ts, tags, vec }));
                }
                4 if !live.is_empty() => {
                    let id = live[(rng.next_u64() % live.len() as u64) as usize];
                    ops.push(RealOp::UpsertReal { id, ts, tags, flags: REAL, vec });
                    if !real_ids.contains(&id) { real_ids.push(id); }
                }
                5 if !real_ids.is_empty() => {
                    let i = (rng.next_u64() % real_ids.len() as u64) as usize;
                    let id = real_ids.swap_remove(i);
                    live.retain(|&x| x != id);
                    ops.push(RealOp::Live(Op::Delete { id }));
                }
                6 if !live.is_empty() => {
                    let i = (rng.next_u64() % live.len() as u64) as usize;
                    let id = live[i];
                    if !real_ids.contains(&id) {
                        live.swap_remove(i);
                        ops.push(RealOp::Live(Op::Delete { id }));
                    }
                }
                _ => {
                    ops.push(RealOp::Live(Op::Insert { id: next_id, ts, tags, vec }));
                    live.push(next_id);
                    next_id += 1;
                }
            }
        }
        ops.push(RealOp::Live(Op::Expire { before_ts: ts.saturating_sub(HOUR), max_rows: 64 }));
        if t % 9 == 8 { ops.push(RealOp::Commit); last_commit_seq = Some(t); }
        if last_commit_seq.map_or(false, |c| t == c + 3) { ops.push(RealOp::Ack { through_seq: u64::MAX }); }
        out.push(ops);
    }
    out
}

/// Apply a churn log, publishing after each tick; ack is turned into an
/// ack of everything emitted so far. Returns the per-tick results of `qs`.
fn run_churn(sys: &mut System, log: &[Vec<RealOp>], qs: &[Vec<f32>], check: bool) -> Vec<Vec<(u64, u32)>> {
    let mut results = Vec::new();
    for (t, ops) in log.iter().enumerate() {
        for op in ops {
            let op = match op {
                RealOp::Ack { .. } if sys.st.buf.next_seq == 0 => continue,
                RealOp::Ack { .. } => RealOp::Ack { through_seq: sys.st.buf.next_seq - 1 },
                op => op.clone(),
            };
            // Deletes of real ids that churn already dropped from its lists can
            // race an expiry; a refused write changes nothing, so skip it.
            let _ = system::write(sys, &op);
        }
        let p = views::publish(&mut sys.w);
        assert_eq!(p.deferred, 0);
        let v = now_view(sys);
        if check {
            assert_view_matches(&v, &sys.st.buf, qs, &format!("tick {t}"));
        }
        for q in qs {
            for &(k, tag, now, lambda) in &SHAPES { results.push(bits(&views::search_temporal(&v, q, k, tag, now, lambda))); }
        }
    }
    results
}

#[test]
fn real_memory_is_found_in_the_view_after_expiry_until_commit() {
    let idx = base_index(Metric::Cosine, 2_000, 1);
    let mut sys = system::start(&idx, BLOCK);
    let mut rng = SplitMix64::new(5);
    let v = random_unit(&mut rng);
    let id = 777_777;
    write(&mut sys, RealOp::InsertReal { id, ts: 2_000, tags: 0b1, flags: REAL | LEGENDARY, vec: v.clone() });
    views::publish(&mut sys.w);
    assert_eq!(top1(&now_view(&sys), &v), Some(id), "not found the tick it was inserted");

    // The hour passes: its short-term row expires, the buffer still has it.
    write(&mut sys, RealOp::Live(Op::Expire { before_ts: 2_000 + HOUR + 1, max_rows: usize::MAX }));
    views::publish(&mut sys.w);
    let after_expiry = now_view(&sys);
    assert!(views::locate(&sys.w, id).is_none(), "short-term row should be expired");
    assert_eq!(top1(&after_expiry, &v), Some(id), "real memory lost at the 1-hour expiry");
    let hit = views::search(&after_expiry, &v, 1)[0];
    assert!((hit.score - 1.0).abs() < 1e-5, "pending score is exact cosine: {}", hit.score);
    assert_eq!(views::search_temporal(&after_expiry, &v, 3, Some(0b10), 0, 0.0).iter().filter(|h| h.id == id).count(), 0,
        "tag filter applies to pending rows");

    // Commit: it moves to the outbox (SQL), so RAM no longer returns it.
    write(&mut sys, RealOp::Commit);
    views::publish(&mut sys.w);
    assert_ne!(top1(&now_view(&sys), &v), Some(id), "committed memory still served from RAM");
    // The view held from before the commit still has it.
    assert_eq!(top1(&after_expiry, &v), Some(id), "held view changed");
    assert_eq!(views::pending_rows(&after_expiry), 1);
    assert_eq!(views::pending_rows(&now_view(&sys)), 0);
}

#[test]
fn not_visible_before_publish_and_visible_after() {
    let idx = base_index(Metric::Cosine, 1_000, 2);
    let mut sys = system::start(&idx, BLOCK);
    let before = now_view(&sys);
    let mut rng = SplitMix64::new(6);
    let v = random_unit(&mut rng);
    write(&mut sys, RealOp::InsertReal { id: 9_000_042, ts: 1_000, tags: 0, flags: REAL, vec: v.clone() });
    // Expire it out of the index in the same tick: only the buffer has it.
    write(&mut sys, RealOp::Live(Op::Expire { before_ts: u64::MAX, max_rows: usize::MAX }));
    assert_ne!(top1(&now_view(&sys), &v), Some(9_000_042), "seen before publish");
    views::publish(&mut sys.w);
    assert_eq!(top1(&now_view(&sys), &v), Some(9_000_042));
    assert_ne!(top1(&before, &v), Some(9_000_042), "old view changed");
}

#[test]
fn pending_delete_and_upsert_show_in_the_next_view() {
    let idx = base_index(Metric::Cosine, 1_000, 3);
    let mut sys = system::start(&idx, BLOCK);
    let mut rng = SplitMix64::new(7);
    let (a, b) = (random_unit(&mut rng), random_unit(&mut rng));
    write(&mut sys, RealOp::InsertReal { id: 9_000_001, ts: 1_000, tags: 0, flags: REAL, vec: a.clone() });
    write(&mut sys, RealOp::InsertReal { id: 9_000_002, ts: 1_000, tags: 0, flags: REAL, vec: b.clone() });
    write(&mut sys, RealOp::Live(Op::Expire { before_ts: u64::MAX, max_rows: usize::MAX }));
    views::publish(&mut sys.w);
    let v1 = now_view(&sys);
    assert_eq!((top1(&v1, &a), top1(&v1, &b)), (Some(9_000_001), Some(9_000_002)));

    // Upsert 1 to b's vector (pending -> new pending row), delete 2.
    write(&mut sys, RealOp::Live(Op::Upsert { id: 9_000_001, ts: 1_001, tags: 0, vec: b.clone() }));
    write(&mut sys, RealOp::Live(Op::Delete { id: 9_000_002 }));
    views::publish(&mut sys.w);
    let v2 = now_view(&sys);
    let hits = views::search(&v2, &b, 10);
    assert_eq!(hits[0].id, 9_000_001);
    assert_eq!(hits.iter().filter(|h| h.id == 9_000_001).count(), 1);
    assert!(hits.iter().all(|h| h.id != 9_000_002), "deleted pending id still found");
    assert_eq!(views::pending_rows(&v2), 1);
    assert_view_matches(&v2, &sys.st.buf, &[a.clone(), b.clone()], "after upsert+delete");
    // v1 is unchanged.
    assert_eq!((top1(&v1, &a), top1(&v1, &b)), (Some(9_000_001), Some(9_000_002)));
}

#[test]
fn view_search_equals_merged_reference_every_tick_cosine_and_l2() {
    for metric in [Metric::Cosine, Metric::L2] {
        let idx = base_index(metric, 1_500, 11);
        let mut sys = system::start(&idx, BLOCK);
        let mut rng = SplitMix64::new(12);
        let qs: Vec<Vec<f32>> = (0..6).map(|_| random_unit(&mut rng)).collect();
        let log = churn(99, 40, 12, 1_500);
        run_churn(&mut sys, &log, &qs, true);
        system::check_invariants(&sys).unwrap();
        assert!(real::pending_len(&sys.st.buf) > 0 || sys.st.outbox.next_seq > 0, "{metric:?}: churn made no real memories");
    }
}

#[test]
fn same_log_same_results() {
    let idx = base_index(Metric::Cosine, 1_500, 21);
    let mut rng = SplitMix64::new(22);
    let qs: Vec<Vec<f32>> = (0..4).map(|_| random_unit(&mut rng)).collect();
    let log = churn(23, 30, 10, 1_500);
    let a = run_churn(&mut system::start(&idx, BLOCK), &log, &qs, false);
    // A different block size is a storage choice only.
    let b = run_churn(&mut system::start(&idx, 64), &log, &qs, false);
    assert_eq!(a, b);
}

#[test]
fn snapshot_load_and_model_swap_keep_pending_searchable() {
    let idx = base_index(Metric::Cosine, 1_500, 31);
    let mut sys = system::start(&idx, BLOCK);
    let mut rng = SplitMix64::new(32);
    let qs: Vec<Vec<f32>> = (0..4).map(|_| random_unit(&mut rng)).collect();
    run_churn(&mut sys, &churn(33, 20, 10, 1_500), &qs, false);
    // Leave a non-empty buffer whose rows are partly expired from the index.
    let mut pend_vecs = Vec::new();
    for i in 0..5u64 {
        let v = random_unit(&mut rng);
        let ts = sys.w.max_ts + 1;
        write(&mut sys, RealOp::InsertReal { id: 5_000_000 + i, ts, tags: 0, flags: REAL, vec: v.clone() });
        pend_vecs.push(v);
    }
    let before_ts = sys.w.max_ts;
    write(&mut sys, RealOp::Live(Op::Expire { before_ts, max_rows: usize::MAX }));
    views::publish(&mut sys.w);
    assert!(real::pending_len(&sys.st.buf) >= 5);

    // Snapshot load: the first view of the loaded system already has the buffer.
    let cap = system::capture(&sys).expect("capture");
    let loaded = system::from_parts(&views::to_index(&cap.view), BLOCK, cap.tick_next, cap.dead_since.clone(), cap.st.clone());
    let lv = now_view(&loaded);
    assert_eq!(views::pending_rows(&lv), real::pending_len(&sys.st.buf));
    assert_view_matches(&lv, &loaded.st.buf, &qs, "loaded");
    for (i, v) in pend_vecs.iter().enumerate() { assert_eq!(top1(&lv, v), Some(5_000_000 + i as u64), "loaded: pending {i}"); }
    let orig = now_view(&sys);
    for q in &qs {
        assert_eq!(bits(&views::search(&lv, q, 10)), bits(&views::search(&orig, q, 10)), "loaded differs from original");
    }

    // Model swap (same rows, new model object): the next view still has the buffer.
    let rows = views::writer_index(&sys.w);
    views::rebase(&mut sys.w, &rows);
    views::publish(&mut sys.w);
    let sv = now_view(&sys);
    assert_eq!(views::pending_rows(&sv), real::pending_len(&sys.st.buf));
    assert_view_matches(&sv, &sys.st.buf, &qs, "after rebase");
    for (i, v) in pend_vecs.iter().enumerate() { assert_eq!(top1(&sv, v), Some(5_000_000 + i as u64), "rebase: pending {i}"); }

    // Writes after the swap still reach the buffer's view side.
    write(&mut sys, RealOp::Live(Op::Delete { id: 5_000_000 }));
    write(&mut sys, RealOp::Commit);
    views::publish(&mut sys.w);
    assert_eq!(views::pending_rows(&now_view(&sys)), 0);
    assert_view_matches(&now_view(&sys), &sys.st.buf, &qs, "after rebase + commit");
}

#[test]
fn search_pick_draws_from_the_merged_top_k() {
    let idx = base_index(Metric::Cosine, 1_000, 41);
    let mut sys = system::start(&idx, BLOCK);
    let mut rng = SplitMix64::new(42);
    let v = random_unit(&mut rng);
    write(&mut sys, RealOp::InsertReal { id: 9_000_009, ts: 1_000, tags: 0, flags: REAL, vec: v.clone() });
    write(&mut sys, RealOp::Live(Op::Expire { before_ts: u64::MAX, max_rows: 999 }));
    views::publish(&mut sys.w);
    let view = now_view(&sys);
    let top = views::search_temporal(&view, &v, 5, None, 0, 0.0);
    assert_eq!(top[0].id, 9_000_009);
    for seed in 0..32u64 {
        let p = views::search_pick(&view, &v, 5, 2, None, 0, 0.0, seed);
        assert_eq!(p, raggedy_anndy::pick::pick_ranked(&top, 2, seed));
    }
}
