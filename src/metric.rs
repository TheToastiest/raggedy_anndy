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
