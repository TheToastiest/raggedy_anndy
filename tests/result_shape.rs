//! Search results must own exactly what they hold: len == min(k, rows
//! scored) and capacity == len. A result that keeps the scratch buffer it
//! was sorted in carries N*16 B per query; at N=1M that is 16 MB per
//! result, and holding a few hundred results ran a 15 GB box out of memory.

mod common;
use common::*;

use raggedy_anndy::{stable_top_k, Hit, IvfIndex, IvfParams, IvfPqIndex, Metric};

const DIM: usize = 32;
const N: usize = 4_000;

fn assert_exact(hits: &[Hit], cap: usize, want_len: usize, what: &str) {
    assert_eq!(hits.len(), want_len, "{what}: len");
    assert_eq!(cap, hits.len(), "{what}: capacity {cap} for {} hits", hits.len());
}

#[test]
fn stable_top_k_returns_exact_capacity_in_score_then_id_order() {
    let hits: Vec<Hit> = (0..1000u64).map(|i| Hit { id: 999 - i, score: (i % 7) as f32 }).collect();
    let top = stable_top_k(hits.clone(), 5);
    assert_exact(&top, top.capacity(), 5, "stable_top_k");
    // Score 6 is the max; ids holding it, ascending.
    let mut want: Vec<u64> = hits.iter().filter(|h| h.score == 6.0).map(|h| h.id).collect();
    want.sort();
    assert_eq!(top.iter().map(|h| h.id).collect::<Vec<_>>(), want[..5].to_vec());

    let all = stable_top_k(hits.clone(), 5000);
    assert_exact(&all, all.capacity(), 1000, "stable_top_k k > len");
    assert!(all.windows(2).all(|w| w[0].score > w[1].score || (w[0].score == w[1].score && w[0].id < w[1].id)));

    let none = stable_top_k(hits, 0);
    assert_exact(&none, none.capacity(), 0, "stable_top_k k = 0");
}

#[test]
fn flat_search_result_has_exact_capacity() {
    let data = uniform_sphere(N, DIM, 1);
    let flat = flat_from(&data, Metric::Cosine);
    let q = row(&data, 17).to_vec();
    for k in [1usize, 10, N + 5] {
        let h = flat.search(&q, k);
        assert_exact(&h, h.capacity(), k.min(N), &format!("Flat k={k}"));
    }
}

#[test]
fn ivf_flat_search_result_has_exact_capacity() {
    let data = uniform_sphere(N, DIM, 1);
    let rows: Vec<(u64, Vec<f32>)> = (0..N).map(|i| (data.ids[i], row(&data, i).to_vec())).collect();
    let ivf = IvfIndex::build(Metric::Cosine, DIM, &rows, IvfParams { nlist: 16, nprobe: 16, refine: 0, seed: 7 });
    let q = row(&data, 17).to_vec();
    for k in [1usize, 10] {
        let h = ivf.search(&q, k);
        assert_exact(&h, h.capacity(), k, &format!("IvfIndex::search k={k}"));
        for t in [1usize, 4] {
            let h = ivf.search_parallel(&q, k, t);
            assert_exact(&h, h.capacity(), k, &format!("IvfIndex::search_parallel k={k} t={t}"));
        }
    }
}

#[test]
fn ivfpq_search_result_has_exact_capacity() {
    let data = uniform_sphere(N, DIM, 1);
    let idx = IvfPqIndex::build(Metric::Cosine, DIM, &ivfpq_rows(&data), ivfpq_params(16, 16, 500, 8, 7));
    let q = row(&data, 17).to_vec();
    for k in [1usize, 10] {
        let h = idx.search(&q, k);
        assert_exact(&h, h.capacity(), k, &format!("IvfPqIndex::search k={k}"));
    }
}
