use crate::metric::Metric;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit { pub id: u64, pub score: f32 }

/// Total order for results: score desc, then id asc. NaN-safe.
#[inline]
pub fn by_score_then_id(a: &Hit, b: &Hit) -> std::cmp::Ordering {
    b.score.total_cmp(&a.score).then(a.id.cmp(&b.id))
}

/// Top-k by (score desc, id asc). O(n) select + O(k log k) sort, and the
/// result is a fresh Vec of exactly k: it never keeps the n-sized scratch
/// buffer, which at n = 1M would pin 16 MB per result.
pub fn stable_top_k(mut hits: Vec<Hit>, k: usize) -> Vec<Hit> {
    let k = k.min(hits.len());
    if k == 0 { return Vec::new(); }
    if k < hits.len() { hits.select_nth_unstable_by(k - 1, by_score_then_id); }
    let mut top = hits[..k].to_vec();
    top.sort_unstable_by(by_score_then_id);
    top
}

#[derive(Clone, Debug)]
pub struct QueryOptions { pub k: usize }

#[derive(Clone, Copy, Debug)]
pub struct BuildSeeds { pub global: u64 }

#[derive(Clone, Debug)]
pub struct BuildManifest {
    pub dim: usize,
    pub metric: Metric,
    pub seed: u64,
    pub nlist: Option<usize>,
}