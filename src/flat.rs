use crate::metric::{self, Metric};
use crate::types::{Hit, stable_top_k};

/// Exact, row‑major flat index.
pub struct FlatIndex {
    metric: Metric,
    dim: usize,
    ids: Vec<u64>,
    vecs: Vec<f32>, // concatenated rows of length `dim`
}

impl FlatIndex {
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self { metric, dim, ids: Vec::new(), vecs: Vec::new() }
    }
    pub fn len(&self) -> usize { self.ids.len() }
    pub fn dim(&self) -> usize { self.dim }

    pub fn add(&mut self, id: u64, v: &[f32]) {
        assert_eq!(v.len(), self.dim);
        self.ids.push(id);
        self.vecs.extend_from_slice(v);
    }

    #[inline]
    fn row(&self, i: usize) -> &[f32] {
        let start = i * self.dim; let end = start + self.dim; &self.vecs[start..end]
    }

    pub fn search(&self, q: &[f32], k: usize) -> Vec<Hit> {
        assert_eq!(q.len(), self.dim);
        let mut hits = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            let s = metric::score(self.metric, q, self.row(i));
            hits.push(Hit { id: self.ids[i], score: s });
        }
        stable_top_k(hits, k)
    }
}