//! IvfPqIndex::search must return bit-identical results (ids and score
//! bits) to a plain reference scan written from the documented scoring
//! rules. Guards search-speed work: any reordering of float sums would
//! change score bits and fail here.

mod common;
use common::*;

use raggedy_anndy::metric;
use raggedy_anndy::{stable_top_k, Hit, IvfPqIndex, Metric};

/// Straight-line reference for search_temporal with no tag filter and no
/// decay: rank lists by coarse score (score desc, list asc), score every
/// live row of the top nprobe lists with ADC, keep the best
/// max(refine, k) by (score desc, id asc), re-rank exactly, take top k.
fn reference_search(idx: &IvfPqIndex, q: &[f32], k: usize) -> Vec<Hit> {
    assert!(idx.opq.is_none(), "reference does not apply OPQ");
    let cosine = matches!(idx.metric, Metric::Cosine);
    let mut qe = q.to_vec();
    if cosine { metric::normalize_inplace(&mut qe); }
    let coarse = |c: &[f32]| if cosine { metric::dot(&qe, c) } else { -metric::l2_distance(&qe, c) };

    let mut probe: Vec<(usize, f32)> = idx.centroids.iter().enumerate().map(|(i, c)| (i, coarse(c))).collect();
    probe.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    probe.truncate(idx.params.nprobe.max(1));

    let m = idx.params.m;
    let lut_ip = idx.pq.adc_lut_ip(&qe);
    let mut approx: Vec<(f32, u64, usize, usize)> = Vec::new();
    for &(li, cs) in &probe {
        let l = &idx.lists[li];
        let lut_l2 = if cosine { Vec::new() } else {
            let r: Vec<f32> = qe.iter().zip(&idx.centroids[li]).map(|(a, b)| a - b).collect();
            idx.pq.adc_lut_l2(&r)
        };
        for r in 0..l.len() {
            if l.tomb[r] == 1 { continue; }
            let code = l.code_row(m, r);
            let s = if cosine { cs + idx.pq.adc_score_ip(&lut_ip, code) } else { -idx.pq.adc_distance_l2(&lut_l2, code) };
            approx.push((s, l.ids[r], li, r));
        }
    }
    approx.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    approx.truncate(idx.params.refine.max(k));
    let hits: Vec<Hit> = approx.iter().map(|&(_, id, li, r)| {
        let v = idx.lists[li].vec_row(idx.dim, r);
        Hit { id, score: if cosine { metric::dot(&qe, v) } else { -metric::l2_distance(&qe, v) } }
    }).collect();
    stable_top_k(hits, k)
}

fn bits(h: &[Hit]) -> Vec<(u64, u32)> { h.iter().map(|x| (x.id, x.score.to_bits())).collect() }

fn check(metric_kind: Metric, dim: usize, m: usize) {
    let centres = cluster_centres(32, dim, 11);
    let data = clustered(6_000, dim, &centres, 0.2, 12);
    let qs = clustered(150, dim, &centres, 0.2, 13);
    let mut idx = IvfPqIndex::build(metric_kind, dim, &ivfpq_rows(&data), ivfpq_params(48, 6, 120, m, 7));
    // Tombstone a few rows so the skip path is covered.
    for l in idx.lists.iter_mut().step_by(5) { if !l.tomb.is_empty() { l.tomb[0] = 1; } }
    for (np, rf) in [(1usize, 10usize), (6, 120), (13, 7), (48, 500)] {
        let ix = with_search_params(&idx, np, rf);
        for i in 0..qs.ids.len() {
            let q = row(&qs, i);
            for k in [1usize, 5, 10] {
                assert_eq!(bits(&ix.search(q, k)), bits(&reference_search(&ix, q, k)),
                    "{metric_kind:?} dim={dim} nprobe={np} refine={rf} query={i} k={k}");
            }
        }
    }
}

#[test] fn cosine_search_matches_reference_dim64() { check(Metric::Cosine, 64, 16); }
#[test] fn cosine_search_matches_reference_dim30() { check(Metric::Cosine, 30, 6); }
#[test] fn l2_search_matches_reference_dim64() { check(Metric::L2, 64, 16); }
#[test] fn l2_search_matches_reference_dim30() { check(Metric::L2, 30, 6); }
