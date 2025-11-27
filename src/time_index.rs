use std::sync::Arc;

use crate::types::Hit;
use crate::{FlatIndex, IvfIndex, IvfPqIndex};

pub trait AnnIndex: Sync {
    fn search(&self, q: &[f32], k: usize) -> Vec<Hit>;
}

impl AnnIndex for FlatIndex {
    fn search(&self, q: &[f32], k: usize) -> Vec<Hit> {
        FlatIndex::search(self, q, k)
    }
}

impl AnnIndex for IvfIndex {
    fn search(&self, q: &[f32], k: usize) -> Vec<Hit> {
        IvfIndex::search(self, q, k)
    }
}

impl AnnIndex for IvfPqIndex {
    fn search(&self, q: &[f32], k: usize) -> Vec<Hit> {
        IvfPqIndex::search(self, q, k)
    }
}

// If you still want a wrapper type, keep it but remove time-based state.
pub struct TimedIndex<I: AnnIndex> {
    pub inner: I,
}

impl<I: AnnIndex> TimedIndex<I> {
    pub fn search(&self, q: &[f32], k: usize) -> Vec<Hit> {
        let mut hits = self.inner.search(q, k);

        // Keep deterministic sort; doesn't modify geometry, just ordering on ties.
        use std::cmp::Ordering;
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then(a.id.cmp(&b.id))
        });

        hits
    }
}
