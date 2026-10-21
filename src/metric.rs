// src/metric.rs

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric { L2, Cosine }

#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() { let d = a[i] - b[i]; s += d * d; }
    s
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() { s += a[i] * b[i]; }
    s
}

/// `dot(q, rows[r])` for 4 rows at once. Each row keeps its own sequential
/// sum in the same order as `dot`, so results are bit-identical to 4 calls;
/// interleaving only removes the add-latency stall of one serial chain.
#[inline]
pub fn dot_x4(q: &[f32], rows: [&[f32]; 4]) -> [f32; 4] {
    let n = q.len();
    let (r0, r1, r2, r3) = (&rows[0][..n], &rows[1][..n], &rows[2][..n], &rows[3][..n]);
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for i in 0..n {
        let x = q[i];
        s0 += x * r0[i]; s1 += x * r1[i]; s2 += x * r2[i]; s3 += x * r3[i];
    }
    [s0, s1, s2, s3]
}

/// `l2_distance(q, rows[r])` for 4 rows at once; bit-identical to 4 calls.
#[inline]
pub fn l2_distance_x4(q: &[f32], rows: [&[f32]; 4]) -> [f32; 4] {
    let n = q.len();
    let (r0, r1, r2, r3) = (&rows[0][..n], &rows[1][..n], &rows[2][..n], &rows[3][..n]);
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for i in 0..n {
        let x = q[i];
        let (d0, d1, d2, d3) = (x - r0[i], x - r1[i], x - r2[i], x - r3[i]);
        s0 += d0 * d0; s1 += d1 * d1; s2 += d2 * d2; s3 += d3 * d3;
    }
    [s0, s1, s2, s3]
}

#[inline]
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let num = dot(a, b);
    let na = (dot(a, a)).sqrt();
    let nb = (dot(b, b)).sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { num / (na * nb) }
}

// ================= Canonicalization & helpers =================
#[inline]
pub fn is_cosine(metric: Metric) -> bool { matches!(metric, Metric::Cosine) }

/// Normalize v in-place to unit length (no-op for zero vector).
#[inline]
pub fn normalize_inplace(v: &mut [f32]) {
    let mut n = 0.0f32; for &x in v.iter() { n += x*x; }
    if n > 0.0 { let inv = n.sqrt().recip(); for x in v { *x *= inv; } }
}

/// Return a normalized copy of v (unit length); zero vector stays zero.
#[inline]
pub fn normalized(v: &[f32]) -> Vec<f32> {
    let mut out = v.to_vec();
    normalize_inplace(&mut out);
    out
}

/// Convert raw distance/sim into a unified score (higher is better).
#[inline]
pub fn score(metric: Metric, q: &[f32], v: &[f32]) -> f32 {
    match metric {
        Metric::L2 => -l2_distance(q, v),
        Metric::Cosine => cosine_sim(q, v),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seed::SplitMix64;

    fn rows(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut r = SplitMix64::new(seed);
        (0..n).map(|_| (0..dim).map(|_| r.next_f64() as f32 * 2.0 - 1.0).collect()).collect()
    }

    #[test]
    fn x4_kernels_are_bit_identical_to_scalar() {
        for dim in [1usize, 3, 4, 30, 64, 128, 257] {
            let q = &rows(1, dim, 1)[0];
            let rs = rows(4, dim, dim as u64);
            let rr = [&rs[0][..], &rs[1][..], &rs[2][..], &rs[3][..]];
            let d = dot_x4(q, rr);
            let l = l2_distance_x4(q, rr);
            for i in 0..4 {
                assert_eq!(d[i].to_bits(), dot(q, &rs[i]).to_bits(), "dot dim={dim} row={i}");
                assert_eq!(l[i].to_bits(), l2_distance(q, &rs[i]).to_bits(), "l2 dim={dim} row={i}");
            }
        }
    }
}
