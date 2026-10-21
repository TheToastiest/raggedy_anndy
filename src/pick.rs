//! Seeded pick of `pick` hits out of a ranked top-k (default 2 of 5).
//!
//! URIEL asks one k=5 lookup and gets 2 of those 5 back, chosen uniformly.
//! The choice is a pure function of the caller's (tick, npc_id, query_id):
//! no OS randomness, so replays and tests see the same pair every time.
//! The picked hits keep their input order (score desc, id asc) and their
//! scores untouched, so URIEL can still tell which ranked higher.

use crate::types::Hit;

pub const TOP_K_DEFAULT: usize = 5;
pub const PICK_DEFAULT: usize = 2;

/// SplitMix64 finaliser (same constants as `seed::SplitMix64`).
#[inline]
fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Seed for one NPC query. Order-sensitive: (1, 2, 3) != (3, 2, 1).
#[inline]
pub fn pick_seed(tick: u64, npc_id: u64, query_id: u64) -> u64 {
    mix64(mix64(mix64(tick) ^ npc_id) ^ query_id)
}

/// Uniform in [0, n) by multiply-high (Lemire). Bias <= n / 2^64.
#[inline]
fn bounded(r: u64, n: u64) -> u64 { ((r as u128 * n as u128) >> 64) as u64 }

/// `min(pick, hits.len())` hits from `hits`, every subset equally likely,
/// in input order. Selection sampling (Knuth, Algorithm S): walk the hits
/// once and keep hit i with probability needed / remaining. One draw per
/// hit visited, no scratch buffer, output already in rank order.
pub fn pick_ranked(hits: &[Hit], pick: usize, seed: u64) -> Vec<Hit> {
    let n = hits.len();
    let want = pick.min(n);
    let mut out = Vec::with_capacity(want);
    if want == n { out.extend_from_slice(hits); return out; }
    let mut state = seed;
    for (i, h) in hits.iter().enumerate() {
        let need = want - out.len();
        if need == 0 { break; }
        let left = (n - i) as u64;
        state = mix64(state);
        if bounded(state, left) < need as u64 { out.push(*h); }
    }
    out
}
