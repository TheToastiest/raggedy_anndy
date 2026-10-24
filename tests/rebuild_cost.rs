//! Where a rebuild's cost lands: on the writer thread (`capture_rows`,
//! `swap_prepared`) or on the worker (`snapshot_of`, `rebuild`, the gate,
//! `prepare`). The old writer-side copies (`begin_rebuild`, `swap`) are
//! timed too, for comparison. Timing aid, not a gate:
//! cargo test --release --test rebuild_cost -- --ignored --nocapture

mod common;
use common::*;

use std::time::Instant;

use raggedy_anndy::gate::{self, GameCheck};
use raggedy_anndy::rebuild;
use raggedy_anndy::system;
use raggedy_anndy::views::{self, DEFAULT_BLOCK_ROWS};
use raggedy_anndy::{IvfPqIndex, Metric};

#[test]
#[ignore = "timing aid"]
fn writer_side_rebuild_cost() {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    // 2.16M = 1 h at the 10 inserts/tick cap, 60 Hz.
    for &(n, dim, nlist, m) in &[(100_000usize, 128usize, 1024usize, 32usize), (500_000, 128, 1024, 32), (2_160_000, 128, 1024, 32)] {
        let ds = uniform_sphere(n, dim, 60);
        let rows: Vec<(u64, Vec<f32>, u64)> = (0..n).map(|i| (ds.ids[i], row(&ds, i).to_vec(), i as u64)).collect();
        drop(ds);
        let mut p = ivfpq_params(nlist, 8, 64, m, 7);
        p.iters = 1; // build time is not what is measured here
        let idx = IvfPqIndex::build(Metric::Cosine, dim, &rows, p);
        drop(rows);
        let mut sys = system::start(&idx, DEFAULT_BLOCK_ROWS);
        drop(idx);
        views::publish(&mut sys.w);

        // Old: the whole copy on the writer.
        let t = Instant::now();
        let snap = system::begin_rebuild(&mut sys).unwrap();
        let t_begin = t.elapsed();
        system::cancel(&mut sys);
        drop(snap);

        // New: handles on the writer, copy on the worker.
        let t = Instant::now();
        let cap = system::capture_rows(&mut sys).unwrap();
        let t_capture = t.elapsed();
        let t = Instant::now();
        let snap = system::snapshot_of(&cap);
        let t_snapshot_of = t.elapsed();
        drop(cap);

        let t = Instant::now();
        let r = rebuild::rebuild(&snap, sys.st.base_seed, threads).unwrap();
        let t_build = t.elapsed();
        let mut gate_times = Vec::new();
        let mut g = None;
        for th in [1usize, threads] {
            let t = Instant::now();
            g = Some(gate::evaluate(&snap, &r, &gate::default_gate_params(), GameCheck::NotRun, th).unwrap());
            gate_times.push((th, t.elapsed()));
        }
        let mut g = g.unwrap();
        g.accept = true; // timing only: install whatever the gate said
        drop(snap);

        // Old: rebase copies every row into new blocks on the writer.
        let t = Instant::now();
        views::rebase(&mut sys.w, &r.next.idx);
        let t_rebase = t.elapsed();

        // New: prepare on the worker, install on the writer.
        let t = Instant::now();
        let prep = system::prepare(r, DEFAULT_BLOCK_ROWS);
        let t_prepare = t.elapsed();
        let t = Instant::now();
        let (_, retired) = system::swap_prepared(&mut sys, prep, g).unwrap();
        let t_swap = t.elapsed();
        let t = Instant::now();
        views::publish(&mut sys.w);
        let t_publish = t.elapsed();
        let t = Instant::now();
        drop(retired);
        let t_retire = t.elapsed();

        println!("n={n} dim={dim} nlist={nlist} m={m} threads={threads}");
        println!("  writer, old: begin_rebuild {t_begin:?}, swap (rebase) {t_rebase:?}");
        println!("  writer, new: capture_rows {t_capture:?}, swap_prepared {t_swap:?}, first publish after {t_publish:?}");
        println!("  worker: snapshot_of {t_snapshot_of:?}, rebuild {t_build:?}, prepare {t_prepare:?}, drop retired {t_retire:?}");
        for (th, t) in gate_times { println!("  worker: gate threads={th} {t:?}"); }
    }
}
