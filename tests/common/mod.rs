//! Shared fixtures for the recall and determinism integration tests.
//!
//! Everything here is plain data + free functions. Datasets come from
//! SplitMix64 (crate::seed) so they do not depend on the `rand` crate's
//! algorithm choices, and are bit-identical on every machine.
#![allow(dead_code)]

use raggedy_anndy::seed::SplitMix64;
use raggedy_anndy::{FlatIndex, Hit, IvfPqIndex, IvfPqParams, Metric};
use raggedy_anndy::ivfpq::OpqMode;

// ---------------------------------------------------------------- datasets

/// Row-major dataset: `ids[i]` owns `vecs[i*dim .. (i+1)*dim]`.
pub struct Dataset {
    pub dim: usize,
    pub ids: Vec<u64>,
    pub vecs: Vec<f32>,
}

#[inline]
pub fn row(ds: &Dataset, i: usize) -> &[f32] {
    &ds.vecs[i * ds.dim..(i + 1) * ds.dim]
}

/// Standard normal via Box-Muller on SplitMix64. Deterministic.
#[inline]
pub fn gauss(rng: &mut SplitMix64) -> f32 {
    let u1 = rng.next_f64().max(f64::MIN_POSITIVE);
    let u2 = rng.next_f64();
    ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
}

pub fn normalize(v: &mut [f32]) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in v.iter_mut() { *x /= n; }
    }
}

/// Points spread uniformly on the unit sphere. The hard case for IVF:
/// no cluster structure, so neighbours straddle list boundaries.
pub fn uniform_sphere(n: usize, dim: usize, seed: u64) -> Dataset {
    let mut rng = SplitMix64::new(seed);
    let mut vecs = Vec::with_capacity(n * dim);
    for _ in 0..n {
        let start = vecs.len();
        for _ in 0..dim { vecs.push(gauss(&mut rng)); }
        normalize(&mut vecs[start..]);
    }
    Dataset { dim, ids: (0..n as u64).collect(), vecs }
}

/// Cluster centres for `clustered`. Kept separate so queries can be drawn
/// from the same centres without being copies of stored points.
pub fn cluster_centres(nclusters: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut c = Vec::with_capacity(nclusters * dim);
    for _ in 0..nclusters {
        let start = c.len();
        for _ in 0..dim { c.push(gauss(&mut rng)); }
        normalize(&mut c[start..]);
    }
    c
}

/// Gaussian blobs around `centres`, then unit-normalised. Stand-in for
/// game events that cluster by zone / entity / action type.
pub fn clustered(n: usize, dim: usize, centres: &[f32], spread: f32, seed: u64) -> Dataset {
    let nclusters = centres.len() / dim;
    let mut rng = SplitMix64::new(seed);
    let mut vecs = Vec::with_capacity(n * dim);
    for _ in 0..n {
        let c = rng.gen_range(nclusters);
        let start = vecs.len();
        for d in 0..dim { vecs.push(centres[c * dim + d] + spread * gauss(&mut rng)); }
        normalize(&mut vecs[start..]);
    }
    Dataset { dim, ids: (0..n as u64).collect(), vecs }
}

// ---------------------------------------------------------------- indexes

pub fn flat_from(ds: &Dataset, metric: Metric) -> FlatIndex {
    let mut f = FlatIndex::new(ds.dim, metric);
    for i in 0..ds.ids.len() { f.add(ds.ids[i], row(ds, i)); }
    f
}

/// IVF-PQ build input. Timestamps are all 0 so temporal decay plays no part.
pub fn ivfpq_rows(ds: &Dataset) -> Vec<(u64, Vec<f32>, u64)> {
    (0..ds.ids.len()).map(|i| (ds.ids[i], row(ds, i).to_vec(), 0u64)).collect()
}

pub fn ivfpq_params(nlist: usize, nprobe: usize, refine: usize, m: usize, seed: u64) -> IvfPqParams {
    IvfPqParams {
        nlist, nprobe, refine, seed, m,
        nbits: 8, iters: 25, use_opq: false,
        opq_mode: OpqMode::Perm, opq_sweeps: 0, store_vecs: true,
    }
}

/// Copy of `idx` with different search-time knobs. Build state is shared,
/// so a sweep measures search behaviour only, not k-means luck.
pub fn with_search_params(idx: &IvfPqIndex, nprobe: usize, refine: usize) -> IvfPqIndex {
    let mut out = idx.clone();
    out.params.nprobe = nprobe;
    out.params.refine = refine;
    out
}

// ---------------------------------------------------------------- recall

/// Hits / possible for recall@k summed over all queries (micro-average).
#[derive(Clone, Copy, Debug, Default)]
pub struct RecallTally { pub hits: usize, pub possible: usize }

pub fn tally_one(truth: &[Hit], ann: &[Hit], k: usize, t: &mut RecallTally) {
    let kk = k.min(truth.len());
    for h in &truth[..kk] {
        if ann.iter().take(k).any(|a| a.id == h.id) { t.hits += 1; }
    }
    t.possible += kk;
}

pub fn recall(t: RecallTally) -> f64 {
    if t.possible == 0 { 0.0 } else { t.hits as f64 / t.possible as f64 }
}

/// Exact top-k for every query, computed once and reused across sweeps.
pub fn truth_sets(flat: &FlatIndex, queries: &Dataset, k: usize) -> Vec<Vec<Hit>> {
    (0..queries.ids.len()).map(|i| flat.search(row(queries, i), k)).collect()
}

pub fn ivfpq_recall(idx: &IvfPqIndex, queries: &Dataset, truth: &[Vec<Hit>], k: usize) -> RecallTally {
    let mut t = RecallTally::default();
    for i in 0..queries.ids.len() {
        let ann = idx.search(row(queries, i), k);
        tally_one(&truth[i], &ann, k, &mut t);
    }
    t
}

// ---------------------------------------------------------------- digests

/// FNV-1a over every bit of build state that search reads. Two builds are
/// "bit-identical" iff their digests match (and the test that uses this
/// also compares the raw arrays on mismatch for a readable failure).
pub fn ivfpq_build_digest(idx: &IvfPqIndex) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    #[inline] fn mix(h: &mut u64, x: u64) { *h ^= x; *h = h.wrapping_mul(0x0000_0100_0000_01b3); }
    mix(&mut h, idx.dim as u64);
    mix(&mut h, idx.centroids.len() as u64);
    for c in &idx.centroids { for &x in c { mix(&mut h, x.to_bits() as u64); } }
    if let Some(cu) = &idx.centroids_u { for c in cu { for &x in c { mix(&mut h, x.to_bits() as u64); } } }
    mix(&mut h, idx.pq.m as u64);
    mix(&mut h, idx.pq.k as u64);
    for cb in &idx.pq.codebooks { for &x in cb { mix(&mut h, x.to_bits() as u64); } }
    for l in &idx.lists {
        mix(&mut h, l.ids.len() as u64);
        for &id in &l.ids { mix(&mut h, id); }
        for &b in &l.codes { mix(&mut h, b as u64); }
        for &t in &l.timestamps { mix(&mut h, t); }
        for &t in &l.tomb { mix(&mut h, t as u64); }
        if let Some(v) = &l.vecs { for &x in v { mix(&mut h, x.to_bits() as u64); } }
        if let Some(tg) = &l.tags { for &x in tg { mix(&mut h, x); } }
    }
    h
}
