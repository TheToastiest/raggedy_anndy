// src/kmeans.rs

use crate::metric::{self, Metric};
use crate::seed::SplitMix64;

/// Deterministic k-means++ (fixed seed, fixed iteration cap, id-based tie-breaks).
/// Uses a flat memory layout to prevent allocator thrashing during the build phase.
pub fn kmeans_seeded(
    data: &[Vec<f32>],
    k: usize,
    metric: Metric,
    seed: u64,
    max_iters: usize,
) -> Vec<Vec<f32>> {
    assert!(k > 0 && k <= data.len());
    let n = data.len();
    let dim = data[0].len();
    for v in data { assert_eq!(v.len(), dim); }

    let mut rng = SplitMix64::new(seed);

    // --- 1. Init: K-Means++ Seeding ---
    // We use a flat array for centroids: index `c * dim + d`
    let mut centers = vec![0.0f32; k * dim];
    let first = rng.gen_range(n);
    centers[0..dim].copy_from_slice(&data[first]);

    let mut nearest: Vec<(usize, f64)> = vec![(0, f64::INFINITY); n];
    let mut num_centers = 1;

    while num_centers < k {
        let last_c_idx = num_centers - 1;
        let last_c = &centers[last_c_idx * dim .. (last_c_idx + 1) * dim];

        let mut sum = 0.0f64;
        for i in 0..n {
            // Note: If metric is Cosine, data should already be L2-normalized
            // by the caller (IvfIndex::build), so L2 distance here effectively
            // performs Spherical K-Means.
            let d = match metric {
                Metric::L2 => metric::l2_distance(&data[i], last_c) as f64,
                Metric::Cosine => 1.0 - metric::cosine_sim(&data[i], last_c) as f64,
            };
            if d < nearest[i].1 { nearest[i] = (last_c_idx, d); }
            sum += nearest[i].1 * nearest[i].1;
        }

        let mut r = rng.next_f64() * sum;
        let mut idx = 0usize;
        for i in 0..n {
            let w = nearest[i].1 * nearest[i].1;
            if r <= w { idx = i; break; } else { r -= w; }
        }

        let offset = num_centers * dim;
        centers[offset .. offset + dim].copy_from_slice(&data[idx]);
        num_centers += 1;
    }

    // --- 2. Lloyd Iterations (Zero-Allocation Hot Loop) ---
    let mut assign = vec![usize::MAX; n];
    let mut new_centers = vec![0.0f32; k * dim];
    let mut counts = vec![0usize; k];

    for _iter in 0..max_iters {
        let mut changes = 0;

        // Step A: Assign vectors to nearest centroid
        for i in 0..n {
            let mut best_c = 0usize;
            let mut best_d = f64::INFINITY;

            for c in 0..k {
                let c_vec = &centers[c * dim .. (c + 1) * dim];
                let d = match metric {
                    Metric::L2 => metric::l2_distance(&data[i], c_vec) as f64,
                    Metric::Cosine => 1.0 - metric::cosine_sim(&data[i], c_vec) as f64,
                };

                if d < best_d {
                    best_d = d;
                    best_c = c;
                }
            }

            if assign[i] != best_c {
                assign[i] = best_c;
                changes += 1;
            }
        }

        // Convergence Check: If nothing moved, we are done.
        if changes == 0 { break; }

        // Step B: Update Centroids
        new_centers.fill(0.0);
        counts.fill(0);

        for i in 0..n {
            let c = assign[i];
            counts[c] += 1;
            let offset = c * dim;
            for d in 0..dim {
                new_centers[offset + d] += data[i][d];
            }
        }

        // Step C: Average and Normalize
        for c in 0..k {
            let offset = c * dim;
            let c_vec = &mut new_centers[offset .. offset + dim];

            if counts[c] > 0 {
                let inv = 1.0f32 / (counts[c] as f32);
                for d in 0..dim { c_vec[d] *= inv; }

                // If Cosine, re-normalize the centroid back to the unit sphere
                if metric == Metric::Cosine {
                    let mut n_len = 0.0f32;
                    for d in 0..dim { n_len += c_vec[d] * c_vec[d]; }
                    if n_len > 0.0 {
                        let n_inv = n_len.sqrt().recip();
                        for d in 0..dim { c_vec[d] *= n_inv; }
                    }
                }
            } else {
                // Empty cluster logic: pull a deterministic vector from data
                let fallback_idx = (c * 17) % n;
                c_vec.copy_from_slice(&data[fallback_idx]);
            }
        }

        // Swap buffers
        centers.copy_from_slice(&new_centers);
    }

    // --- 3. Output Translation ---
    // Convert flat buffer back to Vec<Vec<f32>> for API compatibility
    let mut out = Vec::with_capacity(k);
    for c in 0..k {
        let offset = c * dim;
        out.push(centers[offset .. offset + dim].to_vec());
    }
    out
}