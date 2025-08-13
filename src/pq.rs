// src/pq.rs
// Deterministic Product Quantization (PQ) with 8-bit codebooks (k = 256).
// Trains per-subspace codebooks using seeded k-means on residuals.

use crate::kmeans::kmeans_seeded;
use crate::metric;
use crate::metric::Metric;

#[derive(Clone, Debug)]
pub struct ProductQuantizer {
    pub dim: usize,      // full dimensionality
    pub m: usize,        // number of sub-quantizers
    pub dsub: usize,     // subspace dim = dim / m
    pub nbits: u8,       // bits per code (supported: 8)
    pub k: usize,        // number of centroids per subspace (2^nbits)
    pub iters: usize,    // k-means iterations
    pub seed: u64,       // base seed for determinism
    // codebooks[j] is a flattened [k * dsub] array for subspace j
    pub codebooks: Vec<Vec<f32>>,
}

impl ProductQuantizer {
    pub fn new(dim: usize, m: usize, nbits: u8, iters: usize, seed: u64) -> Self {
        assert!(m > 0 && dim > 0 && dim % m == 0, "PQ: dim must be divisible by m");
        assert!(nbits == 8, "PQ: only 8-bit (nbits=8) supported for now");
        let dsub = dim / m; let k = 1usize << nbits;
        Self { dim, m, dsub, nbits, k, iters, seed, codebooks: vec![vec![0.0; k * dsub]; m] }
    }

    /// Train codebooks on residuals (N x dim). Uses L2 within subspaces.
    pub fn train(&mut self, residuals: &[Vec<f32>]) {
        assert!(!residuals.is_empty());
        for v in residuals { assert_eq!(v.len(), self.dim); }
        // For each subspace j, slice residuals into Vec<Vec<f32>> of length dsub and run k-means.
        for j in 0..self.m {
            let start = j * self.dsub; let end = start + self.dsub;
            let mut sub: Vec<Vec<f32>> = Vec::with_capacity(residuals.len());
            for r in residuals { sub.push(r[start..end].to_vec()); }
            let sub_seed = self.seed ^ ((j as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let centers = kmeans_seeded(&sub, self.k, Metric::L2, sub_seed, self.iters);
            // flatten into codebooks[j]
            let cb = &mut self.codebooks[j];
            for (c_idx, c) in centers.iter().enumerate() {
                let base = c_idx * self.dsub;
                cb[base..base + self.dsub].copy_from_slice(&c[..]);
            }
        }
    }

    /// Encode a full-dim vector into m bytes (one per subspace).
    pub fn encode(&self, x: &[f32]) -> Vec<u8> {
        assert_eq!(x.len(), self.dim);
        let mut codes = vec![0u8; self.m];
        for j in 0..self.m { codes[j] = self.encode_subspace(j, &x[j*self.dsub..(j+1)*self.dsub]); }
        codes
    }

    /// Build ADC LUT for a query residual q (flattened length m*k).
    /// Each subspace j contributes k entries of squared L2 distance to its codewords.
    pub fn adc_lut(&self, q: &[f32]) -> Vec<f32> {
        assert_eq!(q.len(), self.dim);
        let mut lut = vec![0.0f32; self.m * self.k];
        for j in 0..self.m {
            let qj = &q[j*self.dsub..(j+1)*self.dsub];
            let cb = &self.codebooks[j];
            let row = &mut lut[j*self.k..(j+1)*self.k];
            for c in 0..self.k { row[c] = l2_sq_to_codeword(qj, &cb[c*self.dsub..(c+1)*self.dsub]); }
        }
        lut
    }

    /// Sum-of-LUT ADC distance for a single code row.
    #[inline]
    pub fn adc_distance(&self, lut: &[f32], codes: &[u8]) -> f32 {
        debug_assert_eq!(lut.len(), self.m * self.k);
        debug_assert_eq!(codes.len(), self.m);
        let mut s = 0.0f32;
        for j in 0..self.m { s += lut[j*self.k + (codes[j] as usize)]; }
        s
    }

    #[inline]
    fn encode_subspace(&self, j: usize, xj: &[f32]) -> u8 {
        let cb = &self.codebooks[j];
        let mut best = 0usize; let mut bestd = f32::INFINITY;
        for c in 0..self.k {
            let d = l2_sq_to_codeword(xj, &cb[c*self.dsub..(c+1)*self.dsub]);
            if d < bestd { bestd = d; best = c; }
        }
        best as u8
    }
}

#[inline]
fn l2_sq_to_codeword(x: &[f32], cw: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), cw.len());
    let mut s = 0.0f32;
    for i in 0..x.len() { let d = x[i] - cw[i]; s += d * d; }
    s
}
