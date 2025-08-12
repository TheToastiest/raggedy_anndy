use crate::metric::{self, Metric};
use crate::seed::SplitMix64;

/// Deterministic k‑means++ (fixed seed, fixed iteration cap, id‑based tie‑breaks).
pub fn kmeans_seeded(
    data: &[Vec<f32>],
    k: usize,
    metric: Metric,
    seed: u64,
    max_iters: usize,
) -> Vec<Vec<f32>> {
    assert!(k > 0 && k <= data.len());
    let dim = data[0].len();
    for v in data { assert_eq!(v.len(), dim); }

    let mut rng = SplitMix64::new(seed);
    let n = data.len();

    // --- init: first center by seeded pick (not data‑dependent randomness)
    let first = rng.gen_range(n);
    let mut centers: Vec<Vec<f32>> = vec![data[first].to_vec()];
    let mut nearest: Vec<(usize, f64)> = vec![(0, std::f64::INFINITY); n];

    // --- k‑means++ seeding
    while centers.len() < k {
        // compute D(x)^2 to current nearest center
        let last = centers.last().unwrap();
        for i in 0..n {
            // distance in f64 for numeric stability
            let d = match metric {
                Metric::L2 => metric::l2_distance(&data[i], last) as f64,
                Metric::Cosine => 1.0 - metric::cosine_sim(&data[i], last) as f64,
            };
            let cur = nearest[i].1;
            if d < cur { nearest[i] = (centers.len() - 1, d); }
        }
        let mut sum = 0.0f64;
        for (_, d) in &nearest { sum += (*d) * (*d); }
        // select next center via deterministic weighted sampling
        let mut r = rng.next_f64() * sum;
        let mut idx = 0usize;
        for i in 0..n {
            let w = nearest[i].1 * nearest[i].1;
            if r <= w { idx = i; break; } else { r -= w; }
        }
        centers.push(data[idx].to_vec());
    }

    // --- Lloyd iterations
    let mut assign: Vec<usize> = vec![0; n];
    for _ in 0..max_iters {
        // Assign
        for i in 0..n {
            let mut best_c = 0usize;
            let mut best_d = std::f64::INFINITY;
            for (c, center) in centers.iter().enumerate() {
                let d = match metric {
                    Metric::L2 => metric::l2_distance(&data[i], center) as f64,
                    Metric::Cosine => 1.0 - metric::cosine_sim(&data[i], center) as f64,
                };
                if d < best_d || (d == best_d && c < best_c) {
                    best_d = d; best_c = c;
                }
            }
            assign[i] = best_c;
        }
        // Update
        let mut new_centers = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = assign[i]; counts[c] += 1;
            for d in 0..dim { new_centers[c][d] += data[i][d]; }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0f32 / (counts[c] as f32);
                for d in 0..dim { new_centers[c][d] *= inv; }
            } else {
                // empty cluster: re‑seed deterministically
                let idx = c % n; new_centers[c] = data[idx].to_vec();
            }
        }
        if almost_eq_mat(&centers, &new_centers) { break; }
        centers = new_centers;
    }

    centers
}

fn almost_eq_mat(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>) -> bool {
    if a.len() != b.len() || a[0].len() != b[0].len() { return false; }
    for i in 0..a.len() { for j in 0..a[0].len() {
        if (a[i][j] - b[i][j]).abs() > 1e-6 { return false; }
    }}
    true
}