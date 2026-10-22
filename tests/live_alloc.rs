//! An insert on the writer thread must not allocate once `live::reserve`
//! has made room: an allocation is an unbounded pause inside a server tick.
//! Own test binary because it installs a counting global allocator.

mod common;
use common::*;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

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
        views::reserve(&mut w, 500);
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
