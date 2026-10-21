use crate::metric::Metric;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit { pub id: u64, pub score: f32 }

/// Stable top‑k by (score desc, id asc) for determinism.
pub fn stable_top_k(mut hits: Vec<Hit>, k: usize) -> Vec<Hit> {
    hits.sort_by(|a, b| {
        // score desc
        b.score.total_cmp(&a.score).then(a.id.cmp(&b.id))
    });
    hits.truncate(k.min(hits.len()));
    hits
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