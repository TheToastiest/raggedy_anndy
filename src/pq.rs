// src/pq.rs
// ProductQuantizer with both L2 (residual) and IP/Cosine ADC paths.

use crate::kmeans::kmeans_seeded;
use crate::metric::Metric;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, Clone, Debug))]
pub struct ProductQuantizer {
    pub d: usize,            // full dimension
    pub m: usize,            // number of sub-quantizers
    pub dsub: usize,         // d / m
    pub nbits: u8,           // currently 8
    pub k: usize,            // 1 << nbits (codewords per sub)
    pub iters: usize,
    pub seed: u64,
    /// For each sub-quantizer j, a flat array of size k * dsub storing codewords in row-major:
    /// codeword t for sub j lives at codebooks[j][t*dsub .. (t+1)*dsub]
    pub codebooks: Vec<Vec<f32>>,
}

impl ProductQuantizer {
    pub fn new(d: usize, m: usize, nbits: u8, iters: usize, seed: u64) -> Self {
        assert!(d % m == 0, "d must be divisible by m");
        let dsub = d / m;
        let k = 1usize << nbits;
        Self { d, m, dsub, nbits, k, iters, seed, codebooks: vec![Vec::new(); m] }
    }

    /// Train per-sub codebooks with k-means (L2) on residual training vectors in the *training space*.
    /// `train_rows` are full-dimensional residuals (len == d).
    pub fn train(&mut self, train_rows: &[Vec<f32>]) {
        let max_samples = 5000;
        let n = train_rows.len();

        // 1. Deterministic Selection of Training Subset
        let subset: Vec<&Vec<f32>> = if n > max_samples {
            // We use a deterministic stride to pick 5000 vectors across the whole range
            // This is faster than a full shuffle and keeps the distribution stable.
            let stride = n / max_samples;
            (0..max_samples).map(|i| &train_rows[i * stride]).collect()
        } else {
            train_rows.iter().collect()
        };

        // 2. Transpose data into sub-quantizer chunks
        let mut per_sub: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(subset.len()); self.m];
        for row in subset {
            for j in 0..self.m {
                let s = j * self.dsub;
                per_sub[j].push(row[s..s + self.dsub].to_vec());
            }
        }

        // 3. Parallelize the training of sub-quantizers
        // Since each sub-quantizer is independent, we use your par.rs logic
        use crate::par::parallel_map_indexed;

        let m = self.m;
        let k = self.k;
        let iters = self.iters;
        let seed = self.seed;
        let dsub = self.dsub;

        // We use a local parallel map to train all 64 quantizers across your 16 cores
        let results = parallel_map_indexed(&per_sub, 12, |sub_data, j| {
            // Fewer training rows than codewords: train what the data supports and pad by
            // repeating the last codeword (encode picks the first nearest, so padding is never chosen).
            let kc = k.min(sub_data.len());
            let centers = kmeans_seeded(
                sub_data,
                kc,
                Metric::L2,
                seed ^ (j as u64),
                iters
            );
            let mut flat = Vec::with_capacity(k * dsub);
            for c in &centers { flat.extend_from_slice(c); }
            for _ in kc..k { flat.extend_from_slice(&centers[kc - 1]); }
            flat
        });

        self.codebooks = results;
    }

    /// Encode a full vector into m codes (one per sub) using nearest L2 codeword in each sub.
    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        debug_assert_eq!(v.len(), self.d);
        let mut codes = vec![0u8; self.m];
        for j in 0..self.m {
            let s = j * self.dsub;
            let q = &v[s..s + self.dsub];
            let cb = &self.codebooks[j];
            let mut best = 0usize; let mut bestd = f32::INFINITY;
            for t in 0..self.k {
                let start = t * self.dsub;
                let dist = l2sq(q, &cb[start..start + self.dsub]);
                if dist < bestd { bestd = dist; best = t; }
            }
            codes[j] = best as u8;
        }
        codes
    }

    // ========================== ADC LUTs ==========================
    // We implement both L2-residual and IP/Cosine (dot-product) LUTs.

    /// Build an L2-residual ADC LUT for a query residual `qres` (full length `d`).
    /// Returns a flat table of size m * k where entry (j,t) is ||qres_j - codeword_{j,t}||^2.
    pub fn adc_lut_l2(&self, qres: &[f32]) -> Vec<f32> {
        debug_assert_eq!(qres.len(), self.d);
        let mut lut = vec![0f32; self.m * self.k];
        for j in 0..self.m {
            let s = j * self.dsub;
            let q = &qres[s..s + self.dsub];
            let cb = &self.codebooks[j];
            let base = j * self.k;
            for t in 0..self.k {
                let start = t * self.dsub;
                lut[base + t] = l2sq(q, &cb[start..start + self.dsub]);
            }
        }
        lut
    }

    /// Sum L2 contributions for the code vector using the provided LUT.
    pub fn adc_distance_l2(&self, lut: &[f32], codes: &[u8]) -> f32 {
        debug_assert_eq!(codes.len(), self.m);
        let mut acc = 0f32;
        for j in 0..self.m {
            let idx = (j * self.k) + (codes[j] as usize);
            acc += unsafe { *lut.get_unchecked(idx) };
        }
        acc
    }

    /// Build an IP/Cosine ADC LUT for query `qprime` (full length `d`) where each entry is dot(q_j, codeword_{j,t}).
    /// This is intended for Cosine/IP scoring. Larger is better. (No residual!)
    pub fn adc_lut_ip(&self, qprime: &[f32]) -> Vec<f32> {
        debug_assert_eq!(qprime.len(), self.d);
        let mut lut = vec![0f32; self.m * self.k];
        for j in 0..self.m {
            let s = j * self.dsub;
            let q = &qprime[s..s + self.dsub];
            let cb = &self.codebooks[j];
            let base = j * self.k;
            for t in 0..self.k {
                let start = t * self.dsub;
                lut[base + t] = dot(q, &cb[start..start + self.dsub]);
            }
        }
        lut
    }

    /// Sum IP/Cosine contributions using the provided LUT (Σ_j dot(q_j, codeword_{j,codes[j]})).
    pub fn adc_score_ip(&self, lut: &[f32], codes: &[u8]) -> f32 {
        debug_assert_eq!(codes.len(), self.m);
        let mut acc = 0f32;
        for j in 0..self.m {
            let idx = (j * self.k) + (codes[j] as usize);
            acc += unsafe { *lut.get_unchecked(idx) };
        }
        acc
    }

    /// LUT sum for every row of `codes` (row-major, `m` bytes per row) into
    /// `out` (one f32 per row). Works for both LUT kinds: it is the sum
    /// `adc_score_ip` and `adc_distance_l2` compute. Rows are scored 4 at a
    /// time with one sequential sum each, so every value is bit-identical
    /// to the per-row functions.
    pub fn adc_sum_rows(&self, lut: &[f32], codes: &[u8], out: &mut [f32]) {
        let m = self.m;
        let k = self.k;
        let rows = out.len();
        debug_assert_eq!(codes.len(), rows * m);
        debug_assert!(lut.len() >= m * k);
        let blocks = rows / 4;
        for b in 0..blocks {
            let r = b * 4;
            let c0 = &codes[r * m..(r + 1) * m];
            let c1 = &codes[(r + 1) * m..(r + 2) * m];
            let c2 = &codes[(r + 2) * m..(r + 3) * m];
            let c3 = &codes[(r + 3) * m..(r + 4) * m];
            let (mut s0, mut s1, mut s2, mut s3) = (0f32, 0f32, 0f32, 0f32);
            for j in 0..m {
                let base = j * k;
                // codes are < k by construction (nbits <= 8, k = 1 << nbits).
                unsafe {
                    s0 += *lut.get_unchecked(base + c0[j] as usize);
                    s1 += *lut.get_unchecked(base + c1[j] as usize);
                    s2 += *lut.get_unchecked(base + c2[j] as usize);
                    s3 += *lut.get_unchecked(base + c3[j] as usize);
                }
            }
            out[r] = s0; out[r + 1] = s1; out[r + 2] = s2; out[r + 3] = s3;
        }
        for r in blocks * 4..rows {
            out[r] = self.adc_score_ip(lut, &codes[r * m..(r + 1) * m]);
        }
    }

    // -------- Backward-compatible aliases (optional) --------
    /// Historical name used in L2 path.
    pub fn adc_lut(&self, qres: &[f32]) -> Vec<f32> { self.adc_lut_l2(qres) }
    pub fn adc_distance(&self, lut: &[f32], codes: &[u8]) -> f32 { self.adc_distance_l2(lut, codes) }
}

// ========================== small math helpers ==========================
#[inline]
fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f32;
    for i in 0..a.len() { let d = a[i] - b[i]; s += d*d; }
    s
}
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f32;
    for i in 0..a.len() { s += a[i] * b[i]; }
    s
}

#[cfg(test)]
mod adc_rows_tests {
    use super::*;
    use crate::seed::SplitMix64;

    #[test]
    fn adc_sum_rows_is_bit_identical_to_per_row_scores() {
        for (d, m, nbits) in [(64usize, 16usize, 8u8), (30, 6, 8), (32, 8, 4)] {
            let mut rng = SplitMix64::new(d as u64);
            let mut pq = ProductQuantizer::new(d, m, nbits, 1, 7);
            let dsub = d / m;
            pq.codebooks = (0..m).map(|_| (0..pq.k * dsub).map(|_| rng.next_f64() as f32 - 0.5).collect()).collect();
            let q: Vec<f32> = (0..d).map(|_| rng.next_f64() as f32 - 0.5).collect();
            let lut_ip = pq.adc_lut_ip(&q);
            let lut_l2 = pq.adc_lut_l2(&q);
            for rows in [0usize, 1, 3, 4, 5, 8, 37] {
                let codes: Vec<u8> = (0..rows * m).map(|_| rng.gen_range(pq.k) as u8).collect();
                let mut out = vec![0f32; rows];
                pq.adc_sum_rows(&lut_ip, &codes, &mut out);
                for r in 0..rows {
                    assert_eq!(out[r].to_bits(), pq.adc_score_ip(&lut_ip, &codes[r * m..(r + 1) * m]).to_bits(), "ip d={d} rows={rows} r={r}");
                }
                pq.adc_sum_rows(&lut_l2, &codes, &mut out);
                for r in 0..rows {
                    assert_eq!(out[r].to_bits(), pq.adc_distance_l2(&lut_l2, &codes[r * m..(r + 1) * m]).to_bits(), "l2 d={d} rows={rows} r={r}");
                }
            }
        }
    }
}
