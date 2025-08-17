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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ivf::{IvfIndex, IvfParams}};
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    fn random_unit_vec(rng: &mut StdRng, dim: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        let n = v.iter().map(|x| (*x as f64)*(*x as f64)).sum::<f64>().sqrt() as f32;
        if n > 0.0 { for x in v.iter_mut() { *x /= n; } }
        v
    }

    #[test]
    fn determinism_and_recall_cosine() {
        let dim = 32usize; let n = 4000usize; let k = 10usize;
        let mut rng = StdRng::seed_from_u64(42);
        let mut flat = FlatIndex::new(dim, Metric::Cosine);
        let mut data: Vec<(u64, Vec<f32>)> = Vec::new();
        for i in 0..n {
            let v = random_unit_vec(&mut rng, dim);
            flat.add(i as u64, &v);
            data.push((i as u64, v));
        }

        let params = IvfParams { nlist: 256, nprobe: 64, refine: 200, seed: 7 };
        let ivf = IvfIndex::build(Metric::Cosine, dim, &data, params);

        let mut q_rng = StdRng::seed_from_u64(999);
        let mut total_hits = 0usize;
        let mut total_possible = 0usize;
        for _ in 0..100 {
            let q = random_unit_vec(&mut q_rng, dim);
            let truth = flat.search(&q, k);
            let ann = ivf.search(&q, k);

            // determinism
            let ann2 = ivf.search(&q, k);
            assert_eq!(ann, ann2, "non‑deterministic results");

            // accumulate
            let truth_ids: Vec<u64> = truth.iter().map(|h| h.id).collect();
            let ann_ids: Vec<u64> = ann.iter().map(|h| h.id).collect();
            for id in truth_ids.iter().take(k) {
                if ann_ids.contains(id) { total_hits += 1; }
                total_possible += 1;
            }
        }
        let recall = (total_hits as f32) / (total_possible as f32);
        let lb = wilson_lower_bound(total_hits, total_possible, 1.96);
        assert!(recall >= 0.90 && lb >= 0.90, "recall/lower‑bound too low: {:.3}/{:.3}", recall, lb);
    }

    #[test]
    fn recall_monotonicity_with_nprobe_and_refine() {
        let dim = 32usize; let n = 3000usize; let k = 10usize;
        let mut rng = StdRng::seed_from_u64(123);
        let mut flat = FlatIndex::new(dim, Metric::Cosine);
        let mut data: Vec<(u64, Vec<f32>)> = Vec::new();
        for i in 0..n { let v = random_unit_vec(&mut rng, dim); flat.add(i as u64, &v); data.push((i as u64, v)); }

        let base = IvfParams { nlist: 128, nprobe: 8, refine: 50, seed: 99 };
        let hi   = IvfParams { nlist: 128, nprobe: 64, refine: 200, seed: 99 };
        let ivf_lo = IvfIndex::build(Metric::Cosine, dim, &data, base);
        let ivf_hi = IvfIndex::build(Metric::Cosine, dim, &data, hi);

        let mut q_rng = StdRng::seed_from_u64(456);
        let mut hits_lo = 0usize; let mut hits_hi = 0usize; let mut poss = 0usize;
        for _ in 0..50 {
            let q = random_unit_vec(&mut q_rng, dim);
            let truth = flat.search(&q, k);
            let lo = ivf_lo.search(&q, k);
            let hi = ivf_hi.search(&q, k);
            let truth_ids: Vec<u64> = truth.iter().map(|h| h.id).collect();
            let lo_ids: Vec<u64> = lo.iter().map(|h| h.id).collect();
            let hi_ids: Vec<u64> = hi.iter().map(|h| h.id).collect();
            for id in truth_ids.iter().take(k) {
                if lo_ids.contains(id) { hits_lo += 1; }
                if hi_ids.contains(id) { hits_hi += 1; }
                poss += 1;
            }
        }
        assert!(hits_hi >= hits_lo, "recall should be monotonic when increasing nprobe/refine");
    }

    #[test]
    fn stable_ties_and_zero_vectors_cosine() {
        // Create identical vectors and a zero vector; ensure stable id ordering.
        let dim = 8usize; let k = 3usize;
        let mut flat = FlatIndex::new(dim, Metric::Cosine);
        let v = vec![1.0f32; dim];
        let z = vec![0.0f32; dim];
        flat.add(2, &v); // higher id but same vec
        flat.add(1, &v); // lower id, same vec
        flat.add(3, &z); // zero vector

        // Query equal to v → two identical scores; expect id=1 before id=2
        let hits = flat.search(&v, k);
        assert_eq!(hits[0].id, 1);
        assert_eq!(hits[1].id, 2);

        // Query zero vector → cosine sim 0 for all; expect id ascending (1,2,3)
        let hits0 = flat.search(&z, k);
        assert_eq!(hits0[0].id, 1);
        assert_eq!(hits0[1].id, 2);
        assert_eq!(hits0[2].id, 3);
    }
}