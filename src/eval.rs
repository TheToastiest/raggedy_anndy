/// Recall@k: fraction of true neighbors (from exact Flat) recovered by an ANN result set.
pub fn recall_at_k(true_ids: &[u64], ann_ids: &[u64]) -> f32 {
    let k = true_ids.len().min(ann_ids.len());
    let mut hit = 0usize;
    for i in 0..k {
        if ann_ids.contains(&true_ids[i]) { hit += 1; }
    }
    (hit as f32) / (k as f32)
}

/// Wilson score 95% lower bound for a Bernoulli proportion.
pub fn wilson_lower_bound(successes: usize, trials: usize, z: f64) -> f64 {
    assert!(trials > 0);
    let n = trials as f64;
    let phat = (successes as f64) / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = phat + z2 / (2.0 * n);
    let margin = z * ((phat * (1.0 - phat) + z2 / (4.0 * n)) / n).sqrt();
    (center - margin) / denom
}

// Recall and determinism tests over real indexes live in tests/recall.rs and
// tests/determinism.rs. These cover the two pure functions above.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_at_k_counts_overlap_regardless_of_order() {
        assert_eq!(recall_at_k(&[1, 2, 3, 4], &[4, 3, 2, 1]), 1.0);
        assert_eq!(recall_at_k(&[1, 2, 3, 4], &[1, 9, 3, 8]), 0.5);
        assert_eq!(recall_at_k(&[1, 2, 3, 4], &[5, 6, 7, 8]), 0.0);
    }

    #[test]
    fn recall_at_k_uses_the_shorter_list_as_k() {
        // Only the first 2 truth ids are checked when ANN returned 2.
        assert_eq!(recall_at_k(&[1, 2, 3, 4], &[2, 1]), 1.0);
        assert_eq!(recall_at_k(&[1, 2], &[2, 9, 1, 8]), 1.0);
    }

    #[test]
    fn wilson_lower_bound_matches_reference_values() {
        // Reference: z=1.96. 950/1000 -> 0.93469, 90/100 -> 0.82564, 0/10 -> 0.
        assert!((wilson_lower_bound(950, 1000, 1.96) - 0.934_69).abs() < 1e-4);
        assert!((wilson_lower_bound(90, 100, 1.96) - 0.825_64).abs() < 1e-4);
        assert!(wilson_lower_bound(0, 10, 1.96).abs() < 1e-12);
    }

    #[test]
    fn wilson_lower_bound_is_below_point_estimate_and_tightens_with_n() {
        let small = wilson_lower_bound(9, 10, 1.96);
        let large = wilson_lower_bound(900, 1000, 1.96);
        assert!(small < 0.9 && large < 0.9);
        assert!(large > small);
    }
}
