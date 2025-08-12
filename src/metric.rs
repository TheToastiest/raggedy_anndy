#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric { L2, Cosine }

#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
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

/// Convert a raw distance/sim into a unified **score** (higher is better).
#[inline]
pub fn score(metric: Metric, q: &[f32], v: &[f32]) -> f32 {
    match metric {
        Metric::L2 => -l2_distance(q, v),
        Metric::Cosine => cosine_sim(q, v),
    }
}