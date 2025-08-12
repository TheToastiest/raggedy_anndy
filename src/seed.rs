/// SplitMix64 for deterministic RNG (tiny, portable, reproducible).
#[derive(Clone, Copy, Debug)]
pub struct SplitMix64 { state: u64 }

impl SplitMix64 {
    pub fn new(seed: u64) -> Self { Self { state: seed } }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut z = { self.state = self.state.wrapping_add(0x9E3779B97F4A7C15); self.state };
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        let x = self.next_u64() >> 11; // 53 bits
        (x as f64) * (1.0 / ((1u64 << 53) as f64))
    }
    #[inline]
    pub fn gen_range(&mut self, end: usize) -> usize {
        // simple, deterministic bounded int
        (self.next_u64() % (end as u64)) as usize
    }
}