// src/recall.rs
use std::collections::HashSet;
use crate::types::Hit;

pub struct RecallStats {
    pub recall: f32,
    pub p95_ms: f64,
}

pub fn calculate_recall(truth: &[Hit], ann: &[Hit], k: usize) -> f32 {
    if truth.is_empty() || ann.is_empty() { return 0.0; }
    let ann_set: HashSet<u64> = ann.iter().take(k).map(|h| h.id).collect();
    let hits = truth.iter().take(k).filter(|h| ann_set.contains(&h.id)).count();
    hits as f32 / k as f32
}