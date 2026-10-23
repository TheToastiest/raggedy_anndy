//! Live writes, stage 2: one writer thread, query threads reading views.
//!
//! Pins the guarantees from docs/live-writes-plan.md, stage 2:
//! - a view never changes: queries on view T-1 return the same bits
//!   before, during and after the writer's tick T, including deletes,
//!   upserts, expiry and compaction in T;
//! - blocks retired by compaction are not reused while a view that can see
//!   them is alive, and are reused once none is;
//! - the writer gives the same results (`Applied`), the same bytes
//!   (`fingerprint`) and the same search bits per tick as `live::apply`
//!   on an `IvfPqIndex` for the same op log;
//! - with a writer thread and several query threads, every result equals
//!   the single-threaded reference for the view's seq;
//! - when every pooled view is held, publish defers (the current view
//!   stays) and catches up once one is free;
//! - the compaction policy picks the same lists for the same log, at most
//!   `max_per_tick` per tick, frees the blocks of expired rows, and does
//!   not change search results.
//!
//! Loom is not used: it needs its own `Arc`/`Mutex`/atomics threaded
//! through the crate, and `--cfg loom` also switches tokio (a dependency)
//! into loom mode. Races are covered by std-thread stress here plus Miri
//! (data-race detector, weak-memory emulation) on the arena and on
//! `tiny_writer_and_readers_under_miri`:
//!     cargo +nightly miri test --lib arena
//!     MIRIFLAGS=-Zmiri-deterministic-floats cargo +nightly miri test --test live_views tiny_
//! (Miri adds random error to `exp` unless told not to; decay uses `exp`.)
//! Insert and publish allocation after `views::reserve` is checked in tests/live_alloc.rs.

mod common;
use common::*;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use raggedy_anndy::live::{self, Applied, Op};
use raggedy_anndy::seed::SplitMix64;
use raggedy_anndy::views::{self, View, Writer};
use raggedy_anndy::{Hit, IvfPqIndex, Metric};

const DIM: usize = 32;
const NLIST: usize = 16;
const M: usize = 8;

fn base_index(n: usize, seed: u64) -> IvfPqIndex {
    let ds = uniform_sphere(n, DIM, seed);
    let rows: Vec<(u64, Vec<f32>, u64)> = (0..n).map(|i| (ds.ids[i], row(&ds, i).to_vec(), i as u64)).collect();
    IvfPqIndex::build(Metric::Cosine, DIM, &rows, ivfpq_params(NLIST, 4, 64, M, 7))
}

fn random_unit(rng: &mut SplitMix64, dim: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim).map(|_| gauss(rng)).collect();
    normalize(&mut v);
    v
}

fn bits(hits: &[Hit]) -> Vec<(u64, u32)> { hits.iter().map(|h| (h.id, h.score.to_bits())).collect() }

/// Seeded op log with inserts, upserts, deletes, expiry and compaction,
/// timestamps moving forward (as `tests/live_writes.rs`).
fn random_log(n_ops: usize, seed: u64, first_id: u64, first_ts: u64, dim: usize, nlist: usize) -> Vec<Op> {
    let mut rng = SplitMix64::new(seed);
    let mut live_ids: Vec<u64> = Vec::new();
    let mut next_id = first_id;
    let mut ts = first_ts;
    let mut ops = Vec::with_capacity(n_ops);
    for _ in 0..n_ops {
        ts += rng.gen_range(3) as u64;
        let roll = rng.gen_range(100);
        if roll < 55 || live_ids.is_empty() {
            let id = next_id; next_id += 1;
            live_ids.push(id);
            ops.push(Op::Insert { id, ts, tags: rng.next_u64() & 0xF, vec: random_unit(&mut rng, dim) });
        } else if roll < 70 {
            let id = live_ids[rng.gen_range(live_ids.len())];
            ops.push(Op::Upsert { id, ts, tags: 0, vec: random_unit(&mut rng, dim) });
        } else if roll < 90 {
            let id = live_ids.swap_remove(rng.gen_range(live_ids.len()));
            ops.push(Op::Delete { id });
        } else if roll < 97 {
            ops.push(Op::Expire { before_ts: ts.saturating_sub(400), max_rows: 64 });
        } else {
            ops.push(Op::Compact { list: rng.gen_range(nlist) as u32 });
        }
    }
    ops
}

/// Cut `ops` into ticks of 1..=max ops (seeded).
fn ticks(ops: &[Op], max: usize, seed: u64) -> Vec<&[Op]> {
    let mut rng = SplitMix64::new(seed);
    let mut out = Vec::new();
    let mut i = 0;
    while i < ops.len() {
        let n = (1 + rng.gen_range(max)).min(ops.len() - i);
        out.push(&ops[i..i + n]);
        i += n;
    }
    out
}

/// What every query records: plain top-10, and tag-filtered with decay.
fn probe(q: &[f32], s: impl Fn(&[f32], usize, Option<u64>, u64, f32) -> Vec<Hit>) -> [Vec<(u64, u32)>; 2] {
    [bits(&s(q, 10, None, 0, 0.0)), bits(&s(q, 10, Some(0x3), 120_000, 1e-4))]
}

fn probe_view(v: &View, q: &[f32]) -> [Vec<(u64, u32)>; 2] { probe(q, |q, k, t, now, l| views::search_temporal(v, q, k, t, now, l)) }

fn probe_index(idx: &IvfPqIndex, q: &[f32]) -> [Vec<(u64, u32)>; 2] { probe(q, |q, k, t, now, l| idx.search_temporal(q, k, t, now, l)) }

fn assert_view_ok(v: &View) {
    if let Err(e) = live::check_invariants(&views::to_index(v)) { panic!("view {}: {e}", v.seq); }
}

// ------------------------------------------------------------ equivalence

#[test]
fn view_zero_equals_the_index_for_any_block_size() {
    let idx = base_index(3000, 1);
    let mut rng = SplitMix64::new(2);
    let qs: Vec<Vec<f32>> = (0..30).map(|_| random_unit(&mut rng, DIM)).collect();
    for block_rows in [1usize, 3, 4, 64, 4096] {
        let w = views::writer_from(&idx, block_rows);
        let v = views::current(&views::latest(&w));
        assert_eq!(v.seq, 0);
        assert_eq!(views::fingerprint(&v), idx.fingerprint(), "block_rows={block_rows}");
        for q in &qs { assert_eq!(probe_view(&v, q), probe_index(&idx, q), "block_rows={block_rows}"); }
    }
}

#[test]
fn writer_matches_live_apply_tick_by_tick() {
    for (seed, block_rows) in [(3u64, 1usize), (4, 5), (5, 64)] {
        let mut idx = base_index(2000, seed);
        let mut w = views::writer_from(&idx, block_rows);
        let latest = views::latest(&w);
        let ops = random_log(4000, seed * 17, 1_000_000, 100_000, DIM, NLIST);
        let mut s = live::scratch_for(&idx);
        let mut rng = SplitMix64::new(seed + 100);
        let qs: Vec<Vec<f32>> = (0..8).map(|_| random_unit(&mut rng, DIM)).collect();
        for (t, tick) in ticks(&ops, 40, seed).into_iter().enumerate() {
            for op in tick {
                assert_eq!(views::apply(&mut w, op), live::apply(&mut idx, &mut s, op), "seed {seed} tick {t} {op:?}");
            }
            assert_eq!(views::publish(&mut w).seq, t as u64 + 1);
            let v = views::current(&latest);
            assert_eq!(v.seq, t as u64 + 1);
            for q in &qs { assert_eq!(probe_view(&v, q), probe_index(&idx, q), "seed {seed} tick {t}"); }
            if t % 10 == 0 {
                assert_eq!(views::fingerprint(&v), idx.fingerprint(), "seed {seed} tick {t}");
                assert_view_ok(&v);
            }
        }
        let v = views::current(&latest);
        assert_eq!(views::fingerprint(&v), idx.fingerprint());
        assert_eq!(views::live_rows(&w), live::live_rows(&idx));
    }
}

#[test]
fn writer_without_raw_vectors_matches_live_apply() {
    let ds = uniform_sphere(2000, DIM, 6);
    let rows: Vec<(u64, Vec<f32>, u64)> = (0..2000).map(|i| (ds.ids[i], row(&ds, i).to_vec(), i as u64)).collect();
    let mut p = ivfpq_params(NLIST, 4, 64, M, 7);
    p.store_vecs = false;
    let mut idx = IvfPqIndex::build(Metric::Cosine, DIM, &rows, p);
    let mut w = views::writer_from(&idx, 32);
    let ops = random_log(2000, 61, 1_000_000, 100_000, DIM, NLIST);
    let mut s = live::scratch_for(&idx);
    let mut rng = SplitMix64::new(62);
    let qs: Vec<Vec<f32>> = (0..8).map(|_| random_unit(&mut rng, DIM)).collect();
    for tick in ticks(&ops, 50, 63) {
        for op in tick { assert_eq!(views::apply(&mut w, op), live::apply(&mut idx, &mut s, op)); }
        views::publish(&mut w);
        let v = views::current(&views::latest(&w));
        for q in &qs { assert_eq!(probe_view(&v, q), probe_index(&idx, q)); }
    }
    assert_eq!(views::fingerprint(&views::current(&views::latest(&w))), idx.fingerprint());
}

// ------------------------------------------------------------ isolation

/// Tick T deletes view T-1's top hits, upserts and inserts rows that would
/// rank first, expires half the rows and compacts every list. Queries on
/// view T-1 return the same bits after every op and after publish.
#[test]
fn old_view_is_unchanged_by_the_tick_in_progress() {
    let idx = base_index(3000, 7);
    let mut w = views::writer_from(&idx, 16);
    let latest = views::latest(&w);
    let mut rng = SplitMix64::new(8);
    let qs: Vec<Vec<f32>> = (0..20).map(|_| random_unit(&mut rng, DIM)).collect();

    let old = views::current(&latest);
    let want: Vec<_> = qs.iter().map(|q| probe_view(&old, q)).collect();
    let fp = views::fingerprint(&old);
    let check = |what: &str| for (i, q) in qs.iter().enumerate() { assert_eq!(probe_view(&old, q), want[i], "{what}: query {i}"); };

    let mut tick: Vec<Op> = Vec::new();
    for (i, q) in qs.iter().enumerate() {
        let top = views::search(&old, q, 3);
        tick.push(Op::Delete { id: top[0].id });
        tick.push(Op::Upsert { id: top[1].id, ts: 5_000, tags: 0x1, vec: random_unit(&mut rng, DIM) });
        tick.push(Op::Insert { id: 900_000 + i as u64, ts: 5_000, tags: 0x3, vec: q.clone() });
    }
    tick.push(Op::Expire { before_ts: 1_500, max_rows: usize::MAX });
    for l in 0..NLIST as u32 { tick.push(Op::Compact { list: l }); }

    for op in &tick {
        // Some deletes target ids an earlier op already removed: refusals are fine here.
        let _ = views::apply(&mut w, op);
        check(&format!("after {op:?}"));
    }
    views::publish(&mut w);
    check("after publish");
    let new = views::current(&latest);
    assert_eq!(new.seq, 1);
    for (i, q) in qs.iter().enumerate() {
        assert_eq!(views::search(&new, q, 1)[0].id, 900_000 + i as u64, "new view misses tick 1's insert");
    }
    // More ticks, including fresh inserts into the blocks compaction freed.
    for t in 0..5u64 {
        for j in 0..200u64 { views::insert(&mut w, 2_000_000 + t * 1000 + j, 6_000 + t, 0, &random_unit(&mut rng, DIM)).unwrap(); }
        views::publish(&mut w);
    }
    check("five ticks later");
    assert_eq!(views::fingerprint(&old), fp);
}

/// The same, with the writer on its own thread: a reader on view T-1 sees
/// identical bits at whatever point the writer is in tick T.
#[test]
fn old_view_is_unchanged_while_writer_thread_runs() {
    let idx = base_index(3000, 9);
    let mut w = views::writer_from(&idx, 16);
    let latest = views::latest(&w);
    let mut rng = SplitMix64::new(10);
    let qs: Vec<Vec<f32>> = (0..10).map(|_| random_unit(&mut rng, DIM)).collect();
    let old = views::current(&latest);
    let want: Vec<_> = qs.iter().map(|q| probe_view(&old, q)).collect();
    let ops = random_log(6000, 11, 1_000_000, 100_000, DIM, NLIST);

    let done = Arc::new(AtomicBool::new(false));
    let writer = {
        let done = done.clone();
        std::thread::spawn(move || {
            for tick in ticks(&ops, 60, 12) {
                for op in tick { let _ = views::apply(&mut w, op); }
                views::publish(&mut w);
            }
            done.store(true, Ordering::Release);
            w
        })
    };
    let mut rounds = 0;
    while !done.load(Ordering::Acquire) || rounds < 3 {
        for (i, q) in qs.iter().enumerate() { assert_eq!(probe_view(&old, q), want[i], "round {rounds} query {i}"); }
        rounds += 1;
    }
    let w = writer.join().unwrap();
    assert!(views::current(&latest).seq > 50);
    drop(w);
}

// ------------------------------------------------------------ reclamation

fn block_ptrs(lists: &[views::ListMeta]) -> Vec<*const raggedy_anndy::arena::Block> {
    lists.iter().flat_map(|l| l.blocks.iter().map(Arc::as_ptr)).collect()
}

#[test]
fn retired_blocks_are_not_reused_while_an_old_view_lives() {
    let idx = base_index(2000, 13);
    let mut w = views::writer_from(&idx, 8);
    let latest = views::latest(&w);
    let mut rng = SplitMix64::new(14);
    let qs: Vec<Vec<f32>> = (0..10).map(|_| random_unit(&mut rng, DIM)).collect();

    // Tick 1: delete a third of the rows and compact every list.
    for id in (0..2000u64).step_by(3) { views::delete(&mut w, id).unwrap(); }
    for l in 0..NLIST as u32 { views::compact_list(&mut w, l).unwrap(); }
    let old = views::current(&latest); // view 0: sees every retired block
    let old_blocks = block_ptrs(&old.lists);
    let want: Vec<_> = qs.iter().map(|q| probe_view(&old, q)).collect();
    let fp = views::fingerprint(&old);
    views::publish(&mut w);

    // Ticks 2..: enough inserts to need many blocks.
    let insert_tick = |w: &mut Writer, t: u64, rng: &mut SplitMix64| {
        for j in 0..100u64 { views::insert(w, 3_000_000 + t * 1000 + j, 10_000 + t, 0, &random_unit(rng, DIM)).unwrap(); }
        views::publish(w);
    };
    for t in 0..10 { insert_tick(&mut w, t, &mut rng); }
    let st = views::arena_stats(&w);
    assert!(st.retired >= old_blocks.len(), "{st:?}: retired blocks were reclaimed while view 0 lives");
    let now = block_ptrs(&w.lists);
    assert!(!now.iter().any(|p| old_blocks.contains(p)), "a block view 0 can see is back in a list");
    for (i, q) in qs.iter().enumerate() { assert_eq!(probe_view(&old, q), want[i], "query {i}"); }
    assert_eq!(views::fingerprint(&old), fp);

    // Drop the old view: within a few publishes every retired block is free
    // and new rows go into reused blocks instead of new allocations.
    drop(old);
    for t in 10..13 { insert_tick(&mut w, t, &mut rng); }
    let st = views::arena_stats(&w);
    assert_eq!(st.retired, 0, "{st:?}");
    assert!(st.ready > 0, "{st:?}");
    let before = st.allocated;
    for t in 13..16 { insert_tick(&mut w, t, &mut rng); }
    let st = views::arena_stats(&w);
    assert_eq!(st.allocated, before, "{st:?}: allocated while reusable blocks were ready");
    assert_view_ok(&views::current(&latest));
}

// ------------------------------------------------------------ view pool

#[test]
fn default_block_size_is_256_rows() {
    assert_eq!(views::DEFAULT_BLOCK_ROWS, 256);
    let w = views::writer_from(&base_index(600, 50), views::DEFAULT_BLOCK_ROWS);
    assert_eq!(views::current(&views::latest(&w)).block_rows, 256);
}

/// Pool of 3 views. A query keeps every view it sees for a while, so the
/// pool runs out and `publish` defers: the current view stays, held views
/// keep their bits, and the writer keeps the same results as `live::apply`.
/// Once the query lets go, the next publish shows the newest tick.
#[test]
fn publish_defers_while_every_view_is_held() {
    let mut idx = base_index(2000, 40);
    let mut w = views::writer_from(&idx, views::DEFAULT_BLOCK_ROWS);
    views::reserve(&mut w, 64, 3);
    let latest = views::latest(&w);
    let ops = random_log(1200, 41, 1_000_000, 100_000, DIM, NLIST);
    let mut s = live::scratch_for(&idx);
    let mut rng = SplitMix64::new(42);
    let qs: Vec<Vec<f32>> = (0..6).map(|_| random_unit(&mut rng, DIM)).collect();

    let hold = |v: Arc<View>| { let fp = views::fingerprint(&v); let p: Vec<_> = qs.iter().map(|q| probe_view(&v, q)).collect(); (v, fp, p) };
    let mut held = vec![hold(views::current(&latest))];
    let mut deferred = 0u64;
    let tick_ops = ticks(&ops, 20, 43);
    for (t, tick) in tick_ops.iter().enumerate() {
        for op in *tick { assert_eq!(views::apply(&mut w, op), live::apply(&mut idx, &mut s, op), "tick {t} {op:?}"); }
        let p = views::publish(&mut w);
        assert_eq!(p.seq, t as u64 + 1);
        assert_eq!(p.deferred, p.seq - p.current);
        let v = views::current(&latest);
        assert_eq!(v.seq, p.current);
        if p.deferred == 0 {
            for q in &qs { assert_eq!(probe_view(&v, q), probe_index(&idx, q), "tick {t}"); }
        } else {
            deferred += 1;
        }
        // Hold every view of ticks 0..8 (more than the pool), release at 12 and 30.
        if t < 8 || (20..26).contains(&t) { held.push(hold(v)); }
        if t == 12 || t == 30 { held.clear(); }
        for (h, fp, want) in &held {
            for (i, q) in qs.iter().enumerate() { assert_eq!(&probe_view(h, q), &want[i], "tick {t}: held view {} changed", h.seq); }
            assert_eq!(views::fingerprint(h), *fp);
        }
    }
    assert!(deferred >= 4, "pool never ran out ({deferred} deferred)");
    assert_eq!(views::arena_stats(&w).deferred, deferred);
    assert_eq!(views::arena_stats(&w).grown, 0);
    let v = views::current(&latest);
    assert_eq!(v.seq, tick_ops.len() as u64);
    assert_eq!(views::fingerprint(&v), idx.fingerprint());
}

// ------------------------------------------------------------ compaction policy

/// The writer's tick as a server runs it: the tick's ops, then the
/// policy's compactions (logged with them), then publish.
fn tick_with_policy(w: &mut Writer, p: &views::CompactPolicy, tick: &[Op], log: &mut Vec<Op>, buf: &mut Vec<Op>) -> Vec<Applied> {
    let mut out: Vec<Applied> = tick.iter().map(|op| { log.push(op.clone()); views::apply(w, op) }).filter_map(Result::ok).collect();
    views::compactions_due(p, w, buf);
    for op in buf.iter() { log.push(op.clone()); out.push(views::apply(w, op).unwrap()); }
    views::publish(w);
    out
}

/// Independent statement of the rule: due lists sorted by (dead_since, list).
fn due_lists(p: &views::CompactPolicy, w: &Writer) -> Vec<u32> {
    let mut due: Vec<(u64, u32)> = w.lists.iter().enumerate()
        .filter(|(li, l)| l.dead > 0 && (w.seq - w.dead_since[*li] >= p.window_ticks || l.dead >= p.max_dead_rows || l.dead as f32 > p.max_dead_share * l.len as f32))
        .map(|(li, _)| (w.dead_since[li], li as u32)).collect();
    due.sort();
    due.into_iter().map(|(_, li)| li).collect()
}

/// Same log, same choice on every run; each tick's choice is the first
/// `max_per_tick` due lists by (oldest dead row, list index); the logged
/// compactions replay through `live::apply` to the same bytes; and search
/// results equal an index that never compacts.
#[test]
fn compaction_policy_is_deterministic_and_does_not_change_results() {
    let p = views::CompactPolicy { window_ticks: 30, max_dead_share: 0.2, max_dead_rows: 40, max_per_tick: 2 };
    let base = base_index(2000, 60);
    // No Compact ops of its own, and uncapped expiry: a capped Expire counts
    // the dead rows it walks, so after a compaction it would reach further,
    // and the never-compacted index below would no longer be comparable.
    let ops: Vec<Op> = random_log(5000, 61, 1_000_000, 100_000, DIM, NLIST).into_iter()
        .filter(|op| !matches!(op, Op::Compact { .. }))
        .map(|op| match op { Op::Expire { before_ts, .. } => Op::Expire { before_ts, max_rows: usize::MAX }, op => op })
        .collect();
    let tick_ops = ticks(&ops, 30, 62);
    let mut rng = SplitMix64::new(63);
    let qs: Vec<Vec<f32>> = (0..8).map(|_| random_unit(&mut rng, DIM)).collect();

    let run = || {
        let mut w = views::writer_from(&base, 16);
        let (mut log, mut buf, mut chosen) = (Vec::new(), Vec::with_capacity(p.max_per_tick), Vec::new());
        let mut plain = base.clone(); // never compacted
        let mut s = live::scratch_for(&plain);
        for (t, tick) in tick_ops.iter().enumerate() {
            for op in *tick { let _ = live::apply(&mut plain, &mut s, op); }
            for op in *tick { log.push(op.clone()); let _ = views::apply(&mut w, op); }
            let mut want = due_lists(&p, &w);
            want.truncate(p.max_per_tick);
            views::compactions_due(&p, &w, &mut buf);
            let got: Vec<u32> = buf.iter().map(|op| match op { Op::Compact { list } => *list, _ => unreachable!() }).collect();
            assert_eq!(got, want, "tick {t}");
            for op in buf.iter() { log.push(op.clone()); views::apply(&mut w, op).unwrap(); }
            views::publish(&mut w);
            chosen.push(got);
            let v = views::current(&views::latest(&w));
            for q in &qs { assert_eq!(probe_view(&v, q), probe_index(&plain, q), "tick {t}"); }
        }
        (chosen, log, views::fingerprint(&views::current(&views::latest(&w))))
    };
    let (a, log, fp) = run();
    let (b, _, fp2) = run();
    assert_eq!(a, b);
    assert_eq!(fp, fp2);
    assert!(a.iter().map(Vec::len).sum::<usize>() >= 20, "policy barely ran: {a:?}");

    // The log (with the policy's compactions in it) replays to the same bytes.
    let mut idx = base.clone();
    let mut s = live::scratch_for(&idx);
    for op in &log { let _ = live::apply(&mut idx, &mut s, op); }
    assert_eq!(idx.fingerprint(), fp);
}

/// Every list over budget at once: the policy takes `max_per_tick` per
/// tick in list order (all died in the same tick), until none is due.
/// A lone dead row is only collected once the window has passed.
#[test]
fn compaction_policy_caps_per_tick_and_waits_for_the_window() {
    let p = views::CompactPolicy { window_ticks: 50, max_dead_share: 0.25, max_dead_rows: usize::MAX, max_per_tick: 3 };
    let idx = base_index(2000, 70);
    let mut w = views::writer_from(&idx, 16);
    let (mut log, mut buf) = (Vec::new(), Vec::new());
    // Tick 1: kill half of every list.
    let kill: Vec<Op> = w.lists.iter().flat_map(|l| (0..l.len).step_by(2).map(|r| l.blocks[r / 16].id(r % 16)).collect::<Vec<_>>())
        .map(|id| Op::Delete { id }).collect();
    let mut order = Vec::new();
    let mut t = 0;
    let mut first = true;
    loop {
        let tick: &[Op] = if first { &kill } else { &[] };
        first = false;
        let done = tick_with_policy(&mut w, &p, tick, &mut log, &mut buf);
        let lists: Vec<u32> = done.iter().filter_map(|a| match a { Applied::Compacted { list, .. } => Some(*list), _ => None }).collect();
        if lists.is_empty() { break; }
        assert_eq!(lists.len(), 3.min(NLIST - order.len()), "tick {t}");
        order.extend(lists);
        t += 1;
    }
    assert_eq!(order, (0..NLIST as u32).collect::<Vec<_>>());
    assert_eq!(t, (NLIST + 2) / 3);
    assert!(w.lists.iter().all(|l| l.dead == 0));

    // One dead row in list 5: under every budget, so due only after 50 ticks.
    let l5 = &w.lists[5];
    let id = l5.blocks[0].id(0);
    let died = w.seq;
    tick_with_policy(&mut w, &p, &[Op::Delete { id }], &mut log, &mut buf);
    let mut waited = 0;
    while buf.is_empty() {
        tick_with_policy(&mut w, &p, &[], &mut log, &mut buf);
        waited += 1;
        assert!(waited <= 60, "lone dead row never collected");
    }
    assert_eq!(buf, vec![Op::Compact { list: 5 }]);
    assert_eq!(w.seq - 1 - died, p.window_ticks);
}

/// Expiry only advances `head`; the policy's compactions are what free the
/// blocks. After expiring most rows and letting the policy run, the lists
/// hold far fewer blocks, and once no view sees the old ones they are
/// reused by new inserts instead of new allocations.
#[test]
fn expired_rows_are_freed_by_policy_compaction() {
    let p = views::CompactPolicy { window_ticks: 10, max_dead_share: 0.25, max_dead_rows: 64, max_per_tick: 4 };
    let idx = base_index(4000, 80); // timestamps 0..4000
    let mut w = views::writer_from(&idx, 16);
    let latest = views::latest(&w);
    let blocks = |w: &Writer| w.lists.iter().map(|l| l.blocks.len()).sum::<usize>();
    let before = blocks(&w);
    let (mut log, mut buf) = (Vec::new(), Vec::with_capacity(p.max_per_tick));
    let mut rng = SplitMix64::new(81);
    let qs: Vec<Vec<f32>> = (0..8).map(|_| random_unit(&mut rng, DIM)).collect();

    // Expire 90% of rows, 500 rows walked per tick, policy every tick.
    let mut t = 0;
    loop {
        let applied = tick_with_policy(&mut w, &p, &[Op::Expire { before_ts: 3600, max_rows: 500 }], &mut log, &mut buf);
        let expired_all = applied.iter().any(|a| matches!(a, Applied::Expired { done: true, .. }));
        t += 1;
        if expired_all && w.lists.iter().all(|l| l.dead == 0) { break; }
        assert!(t < 200);
    }
    assert_eq!(views::live_rows(&w), 400);
    let after = blocks(&w);
    assert!(after * 4 < before, "blocks {before} -> {after}");
    assert!(w.lists.iter().all(|l| l.dead == 0 && l.head == 0));

    // Search on the compacted view equals the log replayed on an index.
    let mut ref_idx = idx.clone();
    let mut s = live::scratch_for(&ref_idx);
    for op in &log { let _ = live::apply(&mut ref_idx, &mut s, op); }
    let v = views::current(&latest);
    assert_eq!(views::fingerprint(&v), ref_idx.fingerprint());
    for q in &qs { assert_eq!(probe_view(&v, q), probe_index(&ref_idx, q)); }
    drop(v);

    // A pool's worth of publishes later no view sees the old blocks: all reclaimed.
    for _ in 0..views::DEFAULT_VIEWS { tick_with_policy(&mut w, &p, &[], &mut log, &mut buf); }
    let st = views::arena_stats(&w);
    assert_eq!(st.retired, 0, "{st:?}");
    assert!(st.ready >= before - after - 1, "{st:?}");
    // New rows go into reclaimed blocks.
    for j in 0..1000u64 { views::insert(&mut w, 5_000_000 + j, 5_000, 0, &random_unit(&mut rng, DIM)).unwrap(); }
    views::publish(&mut w);
    assert_eq!(views::arena_stats(&w).allocated, st.allocated, "allocated new blocks while reclaimed ones were ready");
}

// ------------------------------------------------------------ stress

/// Reference: results per (view seq, query), from `live::apply` on an
/// `IvfPqIndex`, single-threaded.
fn reference(base: &IvfPqIndex, tick_ops: &[&[Op]], qs: &[Vec<f32>]) -> Vec<Vec<[Vec<(u64, u32)>; 2]>> {
    let mut idx = base.clone();
    let mut s = live::scratch_for(&idx);
    let mut out = vec![qs.iter().map(|q| probe_index(&idx, q)).collect::<Vec<_>>()];
    for tick in tick_ops {
        for op in *tick { let _ = live::apply(&mut idx, &mut s, op); }
        out.push(qs.iter().map(|q| probe_index(&idx, q)).collect());
    }
    out
}

/// Writer thread plus `readers` query threads. Each reader takes the
/// newest view, runs seeded queries on it, sometimes keeps it and queries
/// it again ticks later. Every result must equal the reference for that
/// view's seq. The pool has `pool` views, fewer than the readers can hold,
/// so some publishes are deferred; after the log the writer publishes
/// empty ticks until one goes through (their seqs show the last tick's
/// state). Returns how many results were checked.
fn stress(base: &IvfPqIndex, ops: &[Op], max_tick: usize, block_rows: usize, readers: usize, pool: usize, seed: u64, dim: usize) -> usize {
    let tick_ops = ticks(ops, max_tick, seed);
    let mut rng = SplitMix64::new(seed ^ 0xabc);
    let qs: Vec<Vec<f32>> = (0..6).map(|_| random_unit(&mut rng, dim)).collect();
    let want = reference(base, &tick_ops, &qs);
    let last = tick_ops.len() as u64;

    let mut w = views::writer_from(base, block_rows);
    views::reserve(&mut w, 0, pool);
    let latest = views::latest(&w);
    let checked = AtomicUsize::new(0);
    let state = |seq: u64| seq.min(last) as usize;
    std::thread::scope(|sc| {
        let tick_ops = &tick_ops;
        sc.spawn(move || {
            for tick in tick_ops {
                for op in *tick { let _ = views::apply(&mut w, op); }
                views::publish(&mut w);
                std::thread::yield_now();
            }
            while views::publish(&mut w).deferred > 0 { std::thread::yield_now(); }
        });
        for r in 0..readers {
            let (latest, qs, want, checked, state) = (&latest, &qs, &want, &checked, &state);
            sc.spawn(move || {
                let mut rng = SplitMix64::new(seed + 1 + r as u64);
                let mut kept: Vec<Arc<View>> = Vec::new();
                loop {
                    let v = views::current(latest);
                    let qi = rng.gen_range(qs.len());
                    assert_eq!(probe_view(&v, &qs[qi]), want[state(v.seq)][qi], "reader {r} seq {} query {qi}", v.seq);
                    checked.fetch_add(1, Ordering::Relaxed);
                    if rng.gen_range(4) == 0 { kept.push(v.clone()); }
                    if kept.len() > 3 {
                        let old = kept.remove(rng.gen_range(kept.len()));
                        let qi = rng.gen_range(qs.len());
                        assert_eq!(probe_view(&old, &qs[qi]), want[state(old.seq)][qi], "reader {r} kept seq {} query {qi}", old.seq);
                        checked.fetch_add(1, Ordering::Relaxed);
                    }
                    if v.seq >= last { break; }
                }
            });
        }
    });
    checked.into_inner()
}

#[test]
fn writer_and_query_threads_agree_with_reference_per_view() {
    let base = base_index(2000, 20);
    for (seed, block_rows) in [(21u64, 8usize), (22, 64)] {
        let ops = random_log(3000, seed, 1_000_000, 100_000, DIM, NLIST);
        let n = stress(&base, &ops, 20, block_rows, 4, 6, seed, DIM);
        assert!(n > 100, "only {n} results checked");
    }
}

/// Small enough for Miri, which checks the whole publish / reclaim path
/// for data races:
/// MIRIFLAGS=-Zmiri-deterministic-floats cargo +nightly miri test --test live_views tiny_
#[test]
fn tiny_writer_and_readers_under_miri() {
    let (n, dim, nlist) = (64usize, 8usize, 4usize);
    let ds = uniform_sphere(n, dim, 30);
    let rows: Vec<(u64, Vec<f32>, u64)> = (0..n).map(|i| (ds.ids[i], row(&ds, i).to_vec(), i as u64)).collect();
    let mut p = ivfpq_params(nlist, 2, 16, 2, 7);
    p.nbits = 4;
    p.iters = 2;
    let base = IvfPqIndex::build(Metric::Cosine, dim, &rows, p);
    let ops = random_log(if cfg!(miri) { 40 } else { 400 }, 31, 1_000_000, 1_000, dim, nlist);
    assert!(stress(&base, &ops, 4, 2, 2, 3, 31, dim) > 0);
}

// ------------------------------------------------------------ timing

fn pct(sorted: &[f64], p: f64) -> f64 { sorted[((p * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len()) - 1] }

/// Query latency on a reader thread while a writer thread inserts 10 rows
/// per 60 Hz tick and publishes. Same data and params as the 100k gate.
/// Not a gate: prints p50/p95 for the index alone, an idle view, and a
/// view with the writer running.
/// cargo test --release --test live_views concurrent_query_latency -- --ignored --nocapture
#[test]
#[ignore = "timing aid"]
fn concurrent_query_latency() {
    const D: usize = 128;
    let n = 100_000;
    let centres = cluster_centres(256, D, 1234);
    let data = clustered(n, D, &centres, 0.15, 42);
    let timing = clustered(2_000, D, &centres, 0.15, 777);
    let fresh = clustered(20_000, D, &centres, 0.15, 4242);
    let mut p = ivfpq_params(512, 16, 200, 32, 7);
    p.iters = 10;
    let idx = IvfPqIndex::build(Metric::Cosine, D, &ivfpq_rows(&data), p);
    let block_rows: usize = std::env::var("RA_BLOCK_ROWS").ok().and_then(|s| s.parse().ok()).unwrap_or(views::DEFAULT_BLOCK_ROWS);

    // `passes` runs over the 2000 timing queries.
    let measure = |s: &dyn Fn(&[f32]) -> Vec<Hit>, passes: usize| {
        for i in 0..50 { std::hint::black_box(s(row(&timing, i))); }
        let nq = timing.ids.len();
        let mut us: Vec<f64> = (0..nq * passes).map(|i| {
            let t0 = Instant::now();
            std::hint::black_box(s(row(&timing, i % nq)));
            t0.elapsed().as_secs_f64() * 1e6
        }).collect();
        us.sort_by(|a, b| a.total_cmp(b));
        (pct(&us, 0.5), pct(&us, 0.95))
    };
    let (a50, a95) = measure(&|q| idx.search(q, 10), 1);
    eprintln!("IvfPqIndex::search            k=10 p50={a50:.0}us p95={a95:.0}us");

    let mut w = views::writer_from(&idx, block_rows);
    views::reserve(&mut w, 64, views::DEFAULT_VIEWS);
    let latest = views::latest(&w);
    let v = views::current(&latest);
    let (b50, b95) = measure(&|q| views::search(&v, q, 10), 1);
    eprintln!("view, writer idle (block {block_rows}) k=10 p50={b50:.0}us p95={b95:.0}us");
    drop(v);

    let stop = Arc::new(AtomicBool::new(false));
    let writer = {
        let stop = stop.clone();
        std::thread::spawn(move || {
            let tick = std::time::Duration::from_micros(16_667);
            let mut next = Instant::now();
            let (mut i, mut ticks, mut busy, mut publish) = (0usize, 0usize, Vec::new(), Vec::new());
            while !stop.load(Ordering::Relaxed) && i + 10 <= fresh.ids.len() {
                let t0 = Instant::now();
                for _ in 0..10 { views::insert(&mut w, 1_000_000 + i as u64, 1, 0, row(&fresh, i)).unwrap(); i += 1; }
                let t1 = Instant::now();
                views::publish(&mut w);
                publish.push(t1.elapsed().as_secs_f64() * 1e6);
                busy.push(t0.elapsed().as_secs_f64() * 1e6);
                ticks += 1;
                next += tick;
                let now = Instant::now();
                if next > now { std::thread::sleep(next - now); }
            }
            busy.sort_by(|a, b| a.total_cmp(b));
            publish.sort_by(|a, b| a.total_cmp(b));
            (ticks, pct(&busy, 0.5), pct(&busy, 0.95), pct(&publish, 0.95))
        })
    };
    // Each query takes the newest view, as a server query thread would.
    let (c50, c95) = measure(&|q| views::search(&views::current(&latest), q, 10), 10);
    stop.store(true, Ordering::Relaxed);
    let (ticks, w50, w95, pub95) = writer.join().unwrap();
    eprintln!("view, writer 10 inserts/tick   k=10 p50={c50:.0}us p95={c95:.0}us | writer ticks={ticks} tick busy p50={w50:.0}us p95={w95:.0}us, publish p95={pub95:.0}us");
}
