//! An insert on the writer thread must not allocate once `live::reserve`
//! has made room: an allocation is an unbounded pause inside a server tick.
//! Own test binary because it installs a counting global allocator.

mod common;
use common::*;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use raggedy_anndy::live::{self, Op};
use raggedy_anndy::views;
use raggedy_anndy::seed::SplitMix64;
use raggedy_anndy::{IvfPqIndex, Metric};

struct Counting;
static ARMED: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) { ALLOCS.fetch_add(1, Ordering::Relaxed); }
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) { System.dealloc(p, l) }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) { ALLOCS.fetch_add(1, Ordering::Relaxed); }
        System.realloc(p, l, n)
    }
}

#[global_allocator]
static A: Counting = Counting;

/// The counter is global: tests in this binary take turns.
static SERIAL: Mutex<()> = Mutex::new(());

#[test]
fn insert_delete_and_expire_do_not_allocate_after_reserve() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dim = 32;
    let ds = uniform_sphere(2000, dim, 1);
    let rows: Vec<(u64, Vec<f32>, u64)> = (0..2000).map(|i| (ds.ids[i], row(&ds, i).to_vec(), i as u64)).collect();
    for use_opq in [false, true] {
        let mut p = ivfpq_params(16, 4, 64, 8, 7);
        p.use_opq = use_opq;
        let mut idx = IvfPqIndex::build(Metric::Cosine, dim, &rows, p);
        live::reserve(&mut idx, 500);
        let mut s = live::scratch_for(&idx);
        let mut rng = SplitMix64::new(2);
        let vs: Vec<Vec<f32>> = (0..200).map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| gauss(&mut rng)).collect(); normalize(&mut v); v
        }).collect();

        ARMED.store(true, Ordering::SeqCst);
        for (i, v) in vs.iter().enumerate() {
            live::insert(&mut idx, &mut s, 100_000 + i as u64, 5_000, 0, v).unwrap();
        }
        for i in 0..50u64 { live::delete(&mut idx, 100_000 + i).unwrap(); }
        for i in 50..100u64 { live::upsert(&mut idx, &mut s, 100_000 + i, 5_001, 0, &vs[0]).unwrap(); }
        ARMED.store(false, Ordering::SeqCst);
        let n = ALLOCS.swap(0, Ordering::SeqCst);
        assert_eq!(n, 0, "use_opq={use_opq}: {n} allocations on the write path");

        // Expire returns the killed ids, so it allocates one Vec; the walk itself does not.
        let _ = live::apply(&mut idx, &mut s, &Op::Expire { before_ts: 1000, max_rows: 0 });
    }
}

/// The same for the stage-2 writer (`views`): inserts land in blocks that
/// `views::reserve` attached, so the writer thread never allocates mid-tick.
#[test]
fn writer_insert_delete_upsert_do_not_allocate_after_reserve() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dim = 32;
    let ds = uniform_sphere(2000, dim, 1);
    let rows: Vec<(u64, Vec<f32>, u64)> = (0..2000).map(|i| (ds.ids[i], row(&ds, i).to_vec(), i as u64)).collect();
    for (use_opq, block_rows) in [(false, 16usize), (true, 64)] {
        let mut p = ivfpq_params(16, 4, 64, 8, 7);
        p.use_opq = use_opq;
        let idx = IvfPqIndex::build(Metric::Cosine, dim, &rows, p);
        let mut w = views::writer_from(&idx, block_rows);
        views::reserve(&mut w, 500, views::DEFAULT_VIEWS);
        let mut rng = SplitMix64::new(2);
        let vs: Vec<Vec<f32>> = (0..200).map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| gauss(&mut rng)).collect(); normalize(&mut v); v
        }).collect();

        ARMED.store(true, Ordering::SeqCst);
        for (i, v) in vs.iter().enumerate() {
            views::insert(&mut w, 100_000 + i as u64, 5_000, 0, v).unwrap();
        }
        for i in 0..50u64 { views::delete(&mut w, 100_000 + i).unwrap(); }
        for i in 50..100u64 { views::upsert(&mut w, 100_000 + i, 5_001, 0, &vs[0]).unwrap(); }
        ARMED.store(false, Ordering::SeqCst);
        let n = ALLOCS.swap(0, Ordering::SeqCst);
        assert_eq!(n, 0, "use_opq={use_opq} block_rows={block_rows}: {n} allocations on the writer's insert path");
        views::publish(&mut w);
    }
}

/// `views::publish` must not allocate either: it copies changed lists into
/// a pooled view whose buffers `views::reserve` sized, and when every view
/// is held by a query it defers instead of making a new one. Also checks
/// that a compacted list keeps the room `reserve` made.
#[test]
fn writer_publish_does_not_allocate_after_reserve() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dim = 32;
    let ds = uniform_sphere(2000, dim, 1);
    let rows: Vec<(u64, Vec<f32>, u64)> = (0..2000).map(|i| (ds.ids[i], row(&ds, i).to_vec(), i as u64)).collect();
    for (use_opq, block_rows) in [(false, 16usize), (true, views::DEFAULT_BLOCK_ROWS)] {
        let mut p = ivfpq_params(16, 4, 64, 8, 7);
        p.use_opq = use_opq;
        let idx = IvfPqIndex::build(Metric::Cosine, dim, &rows, p);
        let mut w = views::writer_from(&idx, block_rows);
        let pool = views::DEFAULT_VIEWS + 2; // `reserve` only grows the pool
        views::reserve(&mut w, 400, pool);
        let latest = views::latest(&w);
        let mut rng = SplitMix64::new(3);
        let vs: Vec<Vec<f32>> = (0..300).map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| gauss(&mut rng)).collect(); normalize(&mut v); v
        }).collect();
        // Warm-up: arc-swap gives each thread a debt slot on first use (once per
        // thread, reused after), and the view that was current during the first
        // `reserve` is fitted by the second.
        drop(views::current(&latest));
        views::publish(&mut w);
        views::reserve(&mut w, 400, pool);
        let mut held: Vec<Arc<views::View>> = Vec::with_capacity(2 * pool);
        let mut next = 0usize;
        let mut tick = |w: &mut views::Writer, n: usize| {
            for _ in 0..n {
                views::insert(w, 100_000 + next as u64, 5_000, 0, &vs[next]).unwrap();
                next += 1;
            }
            views::delete(w, 100_000 + next as u64 - 1).unwrap();
            views::upsert(w, 100_000 + next as u64 - 2, 5_000, 0, &vs[0]).unwrap();
            views::publish(w)
        };

        ARMED.store(true, Ordering::SeqCst);
        for _ in 0..5 { assert_eq!(tick(&mut w, 20).deferred, 0); }
        // A query keeps every view it gets until the pool runs out.
        let mut last = views::publish(&mut w);
        while last.deferred == 0 {
            held.push(views::current(&latest));
            last = tick(&mut w, 10);
        }
        let stuck = views::current(&latest).seq;
        let again = tick(&mut w, 10);
        assert_eq!((again.deferred, again.current), (2, stuck));
        ARMED.store(false, Ordering::SeqCst);
        let n = ALLOCS.swap(0, Ordering::SeqCst);
        assert_eq!(n, 0, "use_opq={use_opq} block_rows={block_rows}: {n} allocations in ticks with publish");
        assert_eq!(held.len(), pool, "deferred before every view was held");

        // Released: the next publish goes through and shows the newest tick.
        held.clear();
        let p = views::publish(&mut w);
        assert_eq!((p.deferred, views::current(&latest).seq), (0, p.seq));

        // Compaction allocates its new list, but keeps the reserved room, so the
        // inserts and publishes after it still do not allocate.
        for l in 0..16u32 { views::compact_list(&mut w, l).unwrap(); }
        views::publish(&mut w);
        ARMED.store(true, Ordering::SeqCst);
        for _ in 0..3 { assert_eq!(tick(&mut w, 20).deferred, 0); }
        ARMED.store(false, Ordering::SeqCst);
        let n = ALLOCS.swap(0, Ordering::SeqCst);
        assert_eq!(n, 0, "use_opq={use_opq} block_rows={block_rows}: {n} allocations after compaction");
        assert_eq!(views::arena_stats(&w).grown, 0);
    }
}
