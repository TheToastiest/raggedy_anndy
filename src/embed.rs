// src/embed.rs
// Deterministic text -> vector “toy” embedder for scaffolding and tests.
// Uses a base seed XOR FNV-1a(text) to seed SplitMix64 per input string.
// This makes embeddings reproducible across runs and independent of call order.

use crate::seed::SplitMix64;

pub trait Embedder {
    fn dim(&self) -> usize;
    fn embed(&mut self, text: &str) -> Vec<f32>;
}

/// Simple deterministic embedder:
/// - Seed = base_seed ^ fnv1a64(text)
/// - Values = SplitMix64::next_f64() mapped to [-0.5, 0.5]
/// - Optional L2 normalize (good for cosine/IP correctness)
pub struct RandomEmbedder {
    dim_: usize,
    base_seed: u64,
    normalize: bool,
}

impl RandomEmbedder {
    pub fn new(dim: usize, base_seed: u64, normalize: bool) -> Self {
        Self { dim_: dim, base_seed, normalize }
    }

    #[inline]
    fn seed_for(&self, text: &str) -> u64 {
        self.base_seed ^ fnv1a64(text.as_bytes())
    }

    #[inline]
    fn maybe_normalize(&self, v: &mut [f32]) {
        if !self.normalize { return; }
        let mut n = 0.0f32;
        for &x in v.iter() { n += x * x; }
        if n > 0.0 {
            let inv = 1.0f32 / n.sqrt();
            for x in v.iter_mut() { *x *= inv; }
        }
    }
}

impl Embedder for RandomEmbedder {
    fn dim(&self) -> usize { self.dim_ }

    fn embed(&mut self, text: &str) -> Vec<f32> {
        // Per-text RNG so results are independent of call order.
        let mut rng = SplitMix64::new(self.seed_for(text));
        let mut v = vec![0.0f32; self.dim_];
        for i in 0..self.dim_ {
            // next_f64 in [0,1). Map to [-0.5, 0.5] for a zero-mean-ish toy embedding.
            let u = rng.next_f64() as f32;
            v[i] = u - 0.5;
        }
        self.maybe_normalize(&mut v);
        v
    }
}

#[inline]
fn fnv1a64(bytes: &[u8]) -> u64 {
    // FNV-1a 64-bit
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
