//! Where a rebuild's cost lands: on the writer thread (`begin_rebuild`,
//! `swap`) or on the worker (`rebuild`, `gate::evaluate`).
//! Timing aid, not a gate:
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
        let mut p = ivfpq_params(nlist, 8, 64, m, 7);
        p.iters = 1;
        let idx = IvfPqIndex::build(Metric::Cosine, dim, &rows, p);
        drop(rows);
        let mut sys = system::start(&idx, DEFAULT_BLOCK_ROWS);

        let t = Instant::now();
        let snap = system::begin_rebuild(&mut sys).unwrap();
        let t_begin = t.elapsed();
        system::cancel(&mut sys);

        // `swap` = checks + `views::rebase` onto the rebuilt index; the
        // copy cost depends on rows, not on which model, so rebase onto `idx`.
        let t = Instant::now();
        views::rebase(&mut sys.w, &idx);
        let t_rebase = t.elapsed();

        let t = Instant::now();
        let r = rebuild::rebuild(&snap, sys.st.base_seed, threads).unwrap();
        let t_build = t.elapsed();
        let gp = gate::default_gate_params();
        let t = Instant::now();
        let _ = gate::evaluate(&snap, &r, &gp, GameCheck::NotRun, threads).unwrap();
        let t_gate = t.elapsed();
        println!(
            "n={n} dim={dim} nlist={nlist} m={m} threads={threads}: WRITER begin_rebuild {t_begin:?}, rebase(swap) {t_rebase:?} | WORKER rebuild {t_build:?}, gate {t_gate:?}"
        );
    }
}
