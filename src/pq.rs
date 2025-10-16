// src/pq.rs
// ProductQuantizer with both L2 (residual) and IP/Cosine ADC paths.

use crate::kmeans::kmeans_seeded;
use crate::metric::Metric;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
        // collect subvectors per sub-quantizer
        let mut per_sub: Vec<Vec<Vec<f32>>> = vec![Vec::new(); self.m];
        for row in train_rows {
            assert_eq!(row.len(), self.d);
            for j in 0..self.m {
                let s = j * self.dsub;
                per_sub[j].push(row[s..s + self.dsub].to_vec());
            }
        }
        // kmeans for each sub
        for j in 0..self.m {
            let centers = kmeans_seeded(&per_sub[j], self.k, Metric::L2, self.seed ^ (j as u64), self.iters);
            let mut flat = Vec::with_capacity(self.k * self.dsub);
            for c in centers { flat.extend_from_slice(&c); }
            self.codebooks[j] = flat;
        }
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
