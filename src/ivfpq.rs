use crate::metric::{self, Metric};
use crate::types::{Hit, stable_top_k};
#[cfg(test)]
use crate::flat::FlatIndex;
use crate::pq::ProductQuantizer;
use crate::opq::Opq;
use crate::par::parallel_map_indexed;
use crate::seed::SplitMix64;
use std::cmp::Ordering;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum OpqMode { Perm, Pca, PcaPerm }

#[derive(Clone, Copy)]
struct Candidate { id: u64, score: f32, li: usize, pos: usize }

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct IvfPqParams {
    pub nlist: usize, pub nprobe: usize, pub refine: usize, pub seed: u64,
    pub m: usize, pub nbits: u8, pub iters: usize, pub use_opq: bool,
    pub opq_mode: OpqMode, pub opq_sweeps: usize, pub store_vecs: bool,
}

/// One inverted list, columns side by side (SoA). Rows are in arrival
/// order, so `timestamps` never decreases within a list. Rows before
/// `head` were expired and are always dead (`tomb == 1`); `dead` counts
/// every dead row, before or after `head`. See `live` for the write path.
#[derive(Clone, Debug)]
pub struct IvfPqList {
    pub ids: Vec<u64>,
    pub codes: Vec<u8>,
    pub tomb: Vec<u8>,
    pub timestamps: Vec<u64>,
    pub tags: Vec<u64>, // 0 = untagged
    pub vecs: Option<Vec<f32>>,
    pub head: usize,
    pub dead: usize,
}

/// Where a live id is stored: `lists[list]`, row `row`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowLoc { pub list: u32, pub row: u32 }

impl IvfPqList {
    #[inline] pub fn len(&self) -> usize { self.ids.len() }
    #[inline] pub fn code_row<'a>(&'a self, m: usize, i: usize) -> &'a [u8] { &self.codes[i*m .. (i+1)*m] }
    #[inline] pub fn vec_row<'a>(&'a self, dim: usize, i: usize) -> &'a [f32] { &self.vecs.as_ref().unwrap()[i*dim .. (i+1)*dim] }
}

/// Codes store PQ(OPQ(v - c_list)). Scores are reconstructed per probed list:
/// - Cosine/IP: q·c_list + Σ_j lut_ip(OPQ(q))[j][code_j]
/// - L2:        -Σ_j lut_l2(OPQ(q - c_list))[j][code_j]
/// so rows from different lists are on one comparable scale.
#[derive(Clone, Debug)]
pub struct IvfPqIndex {
    pub metric: Metric, pub dim: usize, pub params: IvfPqParams,
    pub centroids: Vec<Vec<f32>>, pub centroids_u: Option<Vec<Vec<f32>>>,
    pub pq: ProductQuantizer, pub opq: Option<Opq>, pub lists: Vec<IvfPqList>,
    /// id -> location of its live row. Lookup only: never iterated, so its
    /// hash order cannot leak into results or fingerprints.
    pub dir: HashMap<u64, RowLoc>,
    /// Newest timestamp written. Live inserts may not go below it.
    pub max_ts: u64,
}

/// Rows per work unit. Fixed (not derived from core count) so every reduction runs in the same order.
const CHUNK_ROWS: usize = 1024;
/// Coarse k-means trains on at most this many rows per list (deterministic stride sample).
const COARSE_ROWS_PER_LIST: usize = 256;
/// Residual rows used to train OPQ and PQ (deterministic stride sample).
const PQ_TRAIN_ROWS: usize = 5000;

#[inline] pub(crate) fn norm_inplace(v: &mut [f32]) {
    let mut n = 0.0f32; for &x in v.iter() { n += x * x; }
    if n > 0.0 { let inv = n.sqrt().recip(); for x in v { *x *= inv; } }
}

#[inline] fn normalized(v: &[f32]) -> Vec<f32> { let mut o = v.to_vec(); norm_inplace(&mut o); o }

/// Coarse score, higher is better. Cosine inputs are unit length, so q·c ranks like -||q-c||².
#[inline] pub(crate) fn coarse_score(metric: Metric, q: &[f32], c: &[f32]) -> f32 {
    match metric { Metric::Cosine => metric::dot(q, c), Metric::L2 => -metric::l2_distance(q, c) }
}

/// `coarse_score` of 4 centroids at once; bit-identical to 4 calls.
#[inline] fn coarse_score_x4(metric: Metric, q: &[f32], rows: [&[f32]; 4]) -> [f32; 4] {
    match metric {
        Metric::Cosine => metric::dot_x4(q, rows),
        Metric::L2 => { let d = metric::l2_distance_x4(q, rows); [-d[0], -d[1], -d[2], -d[3]] }
    }
}

/// Coarse score of every centroid as (list, score), 4 centroids at a time.
/// Bit-identical to `coarse_score` per centroid.
fn coarse_scores(metric: Metric, q: &[f32], centroids: &[Vec<f32>]) -> Vec<(usize, f32)> {
    let n = centroids.len();
    let mut out = Vec::with_capacity(n);
    let blocks = n / 4;
    for b in 0..blocks {
        let j = b * 4;
        let rows = [&centroids[j][..], &centroids[j + 1][..], &centroids[j + 2][..], &centroids[j + 3][..]];
        let s = coarse_score_x4(metric, q, rows);
        for t in 0..4 { out.push((j + t, s[t])); }
    }
    for j in blocks * 4..n { out.push((j, coarse_score(metric, q, &centroids[j]))); }
    out
}

/// The list a row is assigned to, shared by `build` and `live::insert`: the
/// highest `coarse_score` over `centroid(0..nlist)`, ties to the lower list
/// index. Scores 4 centroids at a time but compares them in list order with
/// strictly-greater-wins, so the choice equals a one-at-a-time scan. No allocation.
#[inline] pub(crate) fn nearest_centroid<'a>(metric: Metric, v: &[f32], nlist: usize, centroid: impl Fn(usize) -> &'a [f32]) -> usize {
    let mut best = 0usize; let mut bs = f32::NEG_INFINITY;
    let blocks = nlist / 4;
    for b in 0..blocks {
        let j = b * 4;
        let s = coarse_score_x4(metric, v, [centroid(j), centroid(j + 1), centroid(j + 2), centroid(j + 3)]);
        for t in 0..4 { if s[t] > bs { bs = s[t]; best = j + t; } }
    }
    for j in blocks * 4..nlist {
        let sc = coarse_score(metric, v, centroid(j));
        if sc > bs { bs = sc; best = j; }
    }
    best
}

/// Total order: score desc, then id asc. NaN-safe.
#[inline] fn by_score_then_id(a_score: f32, a_id: u64, b_score: f32, b_id: u64) -> Ordering {
    b_score.total_cmp(&a_score).then(a_id.cmp(&b_id))
}

#[inline] fn apply_decay(score: f32, ts: u64, current_time: u64, lambda: f32) -> f32 {
    if lambda <= 0.0 { return score; }
    let dt = current_time.saturating_sub(ts) as f32;
    let decay = (-lambda * dt).exp().max(1e-9);
    if score > 0.0 { score * decay } else { score / decay }
}

fn chunk_ranges(n: usize) -> Vec<(usize, usize)> {
    (0..n).step_by(CHUNK_ROWS).map(|s| (s, (s + CHUNK_ROWS).min(n))).collect()
}

/// Deterministic stride sample of row indices, at most `cap` of them.
fn stride_sample(n: usize, cap: usize) -> Vec<usize> {
    if n <= cap { return (0..n).collect(); }
    let stride = n / cap;
    (0..cap).map(|i| i * stride).collect()
}

/// Nearest centroid per row of `rows` (flat, row-major). Ties go to the lower list index.
fn assign_rows(metric: Metric, rows: &[f32], dim: usize, centroids: &[f32], nlist: usize, threads: usize) -> Vec<usize> {
    let n = rows.len() / dim;
    let ranges = chunk_ranges(n);
    parallel_map_indexed(&ranges, threads, |&(s, e), _| {
        (s..e).map(|i| {
            nearest_centroid(metric, &rows[i * dim..(i + 1) * dim], nlist, |j| &centroids[j * dim..(j + 1) * dim])
        }).collect::<Vec<usize>>()
    }).into_iter().flatten().collect()
}

/// Lloyd k-means over a flat row-major sample. Seeded random init (partial Fisher-Yates),
/// parallel assignment in fixed chunks, serial accumulation in row order: bit-identical
/// output for any thread count. Empty lists keep their previous centroid.
fn train_coarse(metric: Metric, rows: &[f32], dim: usize, nlist: usize, iters: usize, seed: u64, threads: usize) -> Vec<f32> {
    let n = rows.len() / dim;
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = SplitMix64::new(seed);
    for i in 0..nlist {
        let j = i + rng.gen_range(n - i);
        order.swap(i, j);
    }
    let mut centroids = vec![0.0f32; nlist * dim];
    for (j, &r) in order[..nlist].iter().enumerate() {
        centroids[j * dim..(j + 1) * dim].copy_from_slice(&rows[r * dim..(r + 1) * dim]);
    }

    let mut assign = vec![usize::MAX; n];
    let mut sums = vec![0.0f64; nlist * dim];
    let mut counts = vec![0usize; nlist];
    for _ in 0..iters {
        let next = assign_rows(metric, rows, dim, &centroids, nlist, threads);
        if next == assign { break; }
        assign = next;

        sums.fill(0.0); counts.fill(0);
        for i in 0..n {
            let j = assign[i];
            counts[j] += 1;
            let acc = &mut sums[j * dim..(j + 1) * dim];
            for (a, &x) in acc.iter_mut().zip(&rows[i * dim..(i + 1) * dim]) { *a += x as f64; }
        }
        for j in 0..nlist {
            if counts[j] == 0 { continue; }
            let inv = 1.0 / counts[j] as f64;
            let c = &mut centroids[j * dim..(j + 1) * dim];
            for (cd, &s) in c.iter_mut().zip(&sums[j * dim..(j + 1) * dim]) { *cd = (s * inv) as f32; }
            if matches!(metric, Metric::Cosine) { norm_inplace(c); }
        }
    }
    centroids
}

fn train_opq(mode: OpqMode, dim: usize, m: usize, residuals: &[Vec<f32>], sweeps: usize) -> Opq {
    match mode {
        OpqMode::Perm => Opq::train_perm(dim, m, residuals),
        OpqMode::Pca => Opq::train_pca(dim, m, residuals, sweeps),
        OpqMode::PcaPerm => Opq::train_pca_then_perm(dim, m, residuals, sweeps),
    }
}

#[inline] pub(crate) fn rotate(opq: &Option<Opq>, x: &[f32]) -> Vec<f32> {
    match opq { Some(o) => o.apply(x), None => x.to_vec() }
}

/// FNV-1a fold.
#[inline] fn fnv(h: u64, x: u64) -> u64 { (h ^ x).wrapping_mul(0x100000001b3) }

impl IvfPqIndex {
    // Signature updated to require (ID, Vector, Timestamp)
    pub fn build(metric: Metric, dim: usize, data: &[(u64, Vec<f32>, u64)], params: IvfPqParams) -> Self {
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        build_with_threads(metric, dim, data, params, threads)
    }

    /// Primary search API injected with native temporal decay math.
    /// A query with any NaN or infinite component has no usable score and returns no hits.
    pub fn search_temporal(&self, q: &[f32], k: usize, required_tag: Option<u64>, current_time: u64, lambda: f32) -> Vec<Hit> {
        if q.iter().any(|x| !x.is_finite()) { return Vec::new(); }
        let dim = self.dim;
        let m = self.params.m;
        let cosine = matches!(self.metric, Metric::Cosine);
        let q_eff = if cosine { normalized(q) } else { q.to_vec() };

        // 1. Coarse: score every centroid, keep the top nprobe by (score desc, list asc).
        let mut probe = coarse_scores(self.metric, &q_eff, &self.centroids);
        let np = self.params.nprobe.max(1).min(probe.len());
        let by_probe = |a: &(usize, f32), b: &(usize, f32)| by_score_then_id(a.1, a.0 as u64, b.1, b.0 as u64);
        if np > 0 && np < probe.len() { probe.select_nth_unstable_by(np - 1, by_probe); }
        probe.truncate(np);
        probe.sort_unstable_by(by_probe);

        // 2. ADC scan: LUT sums for whole lists at a time, then filter.
        // IP LUT is list-independent; the list enters through q·c.
        let lut_ip = if cosine { Some(self.pq.adc_lut_ip(&rotate(&self.opq, &q_eff))) } else { None };
        let mut qres = vec![0.0f32; dim];
        let total: usize = probe.iter().map(|&(li, _)| self.lists[li].len()).sum();
        let longest = probe.iter().map(|&(li, _)| self.lists[li].len() - self.lists[li].head).max().unwrap_or(0);
        let mut sums = vec![0.0f32; longest];
        let mut approx: Vec<Candidate> = Vec::with_capacity(total);
        for &(li, cscore) in &probe {
            let list = &self.lists[li];
            let head = list.head;
            let rows = list.len() - head;
            if rows == 0 { continue; }
            let lut_l2;
            let lut = match &lut_ip {
                Some(l) => l,
                None => {
                    for ((r, &qd), &cd) in qres.iter_mut().zip(&q_eff).zip(&self.centroids[li]) { *r = qd - cd; }
                    lut_l2 = self.pq.adc_lut_l2(&rotate(&self.opq, &qres));
                    &lut_l2
                }
            };
            let sums = &mut sums[..rows];
            self.pq.adc_sum_rows(lut, &list.codes[head * m..list.len() * m], sums);
            for (j, &sum) in sums.iter().enumerate() {
                let i = head + j;
                if list.tomb[i] == 1 { continue; }
                if let Some(mask) = required_tag {
                    if (list.tags[i] & mask) == 0 { continue; }
                }
                let raw = if cosine { cscore + sum } else { -sum };
                let score = apply_decay(raw, list.timestamps[i], current_time, lambda);
                approx.push(Candidate { id: list.ids[i], score, li, pos: i });
            }
        }

        let keep = self.params.refine.max(k);
        let cmp = |a: &Candidate, b: &Candidate| by_score_then_id(a.score, a.id, b.score, b.id);
        if approx.len() > keep {
            approx.select_nth_unstable_by(keep - 1, cmp);
            approx.truncate(keep);
        }

        // 3. Exact re-rank of the shortlist, 4 rows at a time.
        let hits: Vec<Hit> = if self.params.store_vecs {
            let exact = |c: &Candidate, s: f32| Hit { id: c.id, score: apply_decay(s, self.lists[c.li].timestamps[c.pos], current_time, lambda) };
            let row = |c: &Candidate| self.lists[c.li].vec_row(dim, c.pos);
            let mut out = Vec::with_capacity(approx.len());
            let mut chunks = approx.chunks_exact(4);
            for c in &mut chunks {
                let rows = [row(&c[0]), row(&c[1]), row(&c[2]), row(&c[3])];
                let s = if cosine { metric::dot_x4(&q_eff, rows) } else {
                    let d = metric::l2_distance_x4(&q_eff, rows); [-d[0], -d[1], -d[2], -d[3]]
                };
                for j in 0..4 { out.push(exact(&c[j], s[j])); }
            }
            for c in chunks.remainder() {
                let v = row(c);
                out.push(exact(c, if cosine { metric::dot(&q_eff, v) } else { -metric::l2_distance(&q_eff, v) }));
            }
            out
        } else {
            approx.iter().map(|c| Hit { id: c.id, score: c.score }).collect()
        };
        stable_top_k(hits, k)
    }

    /// One top-k lookup, then a seeded uniform pick of `pick` of those hits
    /// (default 5 and 2; see `pick`). Seed with `pick::pick_seed`.
    pub fn search_pick(&self, q: &[f32], k: usize, pick: usize, required_tag: Option<u64>, current_time: u64, lambda: f32, seed: u64) -> Vec<Hit> {
        crate::pick::pick_ranked(&self.search_temporal(q, k, required_tag, current_time, lambda), pick, seed)
    }

    /// Legacy wrapper for the standard trait layout (zero decay)
    pub fn search(&self, q: &[f32], k: usize) -> Vec<Hit> {
        self.search_temporal(q, k, None, 0, 0.0)
    }

    pub fn search_with_filter(&self, q: &[f32], k: usize, required_tag: Option<u64>) -> Vec<Hit> {
        self.search_temporal(q, k, required_tag, 0, 0.0)
    }

    /// FNV-1a over everything search reads: shape, centroids, OPQ, codebooks, and every list column.
    pub fn fingerprint(&self) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        h = fnv(h, self.dim as u64);
        h = fnv(h, self.params.m as u64);
        h = fnv(h, self.pq.nbits as u64);
        h = fnv(h, self.centroids.len() as u64);
        h = fnv(h, self.max_ts);
        for c in &self.centroids { for &x in c { h = fnv(h, x.to_bits() as u64); } }
        if let Some(o) = &self.opq { for x in o.fingerprint_bits() { h = fnv(h, x); } }
        for cb in &self.pq.codebooks { for &x in cb { h = fnv(h, x.to_bits() as u64); } }
        for l in &self.lists {
            h = fnv(h, l.len() as u64);
            h = fnv(h, l.head as u64);
            h = fnv(h, l.dead as u64);
            for &id in &l.ids { h = fnv(h, id); }
            for &ts in &l.timestamps { h = fnv(h, ts); }
            for &t in &l.tomb { h = fnv(h, t as u64); }
            for &c in &l.codes { h = fnv(h, c as u64); }
            for &t in &l.tags { h = fnv(h, t); }
            if let Some(vecs) = &l.vecs { for &x in vecs { h = fnv(h, x.to_bits() as u64); } }
        }
        h
    }
}

/// Build with an explicit worker count. Output is bit-identical for every `threads` value.
pub(crate) fn build_with_threads(metric: Metric, dim: usize, data: &[(u64, Vec<f32>, u64)], params: IvfPqParams, threads: usize) -> IvfPqIndex {
    let n = data.len();
    let nlist = params.nlist;
    assert!(nlist > 0 && nlist <= n, "nlist ({nlist}) must be in 1..=n ({n})");
    assert!(params.m > 0 && dim % params.m == 0, "dim ({dim}) must be divisible by m ({})", params.m);
    assert!((1..=8).contains(&params.nbits), "nbits must be in 1..=8 (codes are u8), got {}", params.nbits);
    let cosine = matches!(metric, Metric::Cosine);

    // 1. Row-major views (unit length for cosine).
    let mut views = Vec::with_capacity(n * dim);
    for (_, v, _) in data {
        assert_eq!(v.len(), dim);
        let start = views.len();
        views.extend_from_slice(v);
        if cosine { norm_inplace(&mut views[start..]); }
    }

    // 2. Coarse quantizer on a stride sample, then assign every row.
    let coarse_idx = stride_sample(n, nlist * COARSE_ROWS_PER_LIST);
    let mut coarse_rows = Vec::with_capacity(coarse_idx.len() * dim);
    for &i in &coarse_idx { coarse_rows.extend_from_slice(&views[i * dim..(i + 1) * dim]); }
    let centroids = train_coarse(metric, &coarse_rows, dim, nlist, params.iters.max(1), params.seed, threads);
    let assign = assign_rows(metric, &views, dim, &centroids, nlist, threads);

    // 3. Residuals r = v - c_assign.
    let mut residuals = views.clone();
    for i in 0..n {
        let c = &centroids[assign[i] * dim..(assign[i] + 1) * dim];
        for (r, &cd) in residuals[i * dim..(i + 1) * dim].iter_mut().zip(c) { *r -= cd; }
    }

    // 4. OPQ (optional) and PQ, both trained on the same residual sample.
    let train: Vec<Vec<f32>> = stride_sample(n, PQ_TRAIN_ROWS).into_iter()
        .map(|i| residuals[i * dim..(i + 1) * dim].to_vec()).collect();
    let opq = if params.use_opq {
        Some(train_opq(params.opq_mode, dim, params.m, &train, params.opq_sweeps.max(1)))
    } else { None };
    let train_rot: Vec<Vec<f32>> = train.iter().map(|r| rotate(&opq, r)).collect();
    let mut pq = ProductQuantizer::new(dim, params.m, params.nbits, params.iters.max(1), params.seed);
    pq.train(&train_rot);

    // 5. Encode all residuals (ordered chunks).
    let ranges = chunk_ranges(n);
    let codes: Vec<u8> = parallel_map_indexed(&ranges, threads, |&(s, e), _| {
        let mut out = Vec::with_capacity((e - s) * params.m);
        for i in s..e { out.extend_from_slice(&pq.encode(&rotate(&opq, &residuals[i * dim..(i + 1) * dim]))); }
        out
    }).into_iter().flatten().collect();

    // 6. Scatter into per-list SoA columns in (timestamp, input) order, the
    //    order live inserts would have produced: lists stay sorted by time,
    //    which is what lets `live::expire` stop at the first young row.
    let mut lists: Vec<IvfPqList> = (0..nlist).map(|_| empty_list(params.store_vecs)).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| data[i].2); // stable: equal timestamps keep input order
    let mut dir = HashMap::with_capacity(n);
    for i in order {
        let (id, _, ts) = &data[i];
        let li = assign[i];
        let list = &mut lists[li];
        let prev = dir.insert(*id, RowLoc { list: li as u32, row: list.ids.len() as u32 });
        assert!(prev.is_none(), "duplicate id {id} in build input");
        list.ids.push(*id);
        list.timestamps.push(*ts);
        list.tomb.push(0);
        list.tags.push(0);
        list.codes.extend_from_slice(&codes[i * params.m..(i + 1) * params.m]);
        if let Some(vecs) = &mut list.vecs { vecs.extend_from_slice(&views[i * dim..(i + 1) * dim]); }
    }
    let max_ts = data.iter().map(|d| d.2).max().unwrap_or(0);

    let centroids = centroids.chunks(dim).map(|c| c.to_vec()).collect();
    IvfPqIndex { metric, dim, params, centroids, centroids_u: None, pq, opq, lists, dir, max_ts }
}

pub(crate) fn empty_list(store_vecs: bool) -> IvfPqList {
    IvfPqList {
        ids: Vec::new(), codes: Vec::new(), tomb: Vec::new(), timestamps: Vec::new(), tags: Vec::new(),
        vecs: if store_vecs { Some(Vec::new()) } else { None }, head: 0, dead: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seed::SplitMix64;
    use std::collections::HashSet;

    // ---------- deterministic test data (no rand crate: SplitMix64 + Box-Muller) ----------
    fn gaussian(rng: &mut SplitMix64) -> f32 {
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }

    fn sphere_rows(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = SplitMix64::new(seed);
        (0..n).map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| gaussian(&mut rng)).collect();
            norm_inplace(&mut v);
            v
        }).collect()
    }

    fn as_data(rows: &[Vec<f32>]) -> Vec<(u64, Vec<f32>, u64)> {
        rows.iter().enumerate().map(|(i, v)| (i as u64, v.clone(), i as u64)).collect()
    }

    fn params(nlist: usize, nprobe: usize, refine: usize, m: usize) -> IvfPqParams {
        IvfPqParams {
            nlist, nprobe, refine, seed: 7, m, nbits: 8, iters: 10, use_opq: false,
            opq_mode: OpqMode::PcaPerm, opq_sweeps: 4, store_vecs: true,
        }
    }

    fn exact_top_k(metric: Metric, rows: &[Vec<f32>], q: &[f32], k: usize) -> Vec<u64> {
        let mut flat = FlatIndex::new(q.len(), metric);
        for (i, v) in rows.iter().enumerate() { flat.add(i as u64, v); }
        flat.search(q, k).into_iter().map(|h| h.id).collect()
    }

    fn mean_recall(idx: &IvfPqIndex, rows: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> f32 {
        let mut total = 0usize;
        for q in queries {
            let truth: HashSet<u64> = exact_top_k(idx.metric, rows, q, k).into_iter().collect();
            total += idx.search(q, k).iter().filter(|h| truth.contains(&h.id)).count();
        }
        total as f32 / (k * queries.len()) as f32
    }

    /// Approximate (PQ-only) scores must estimate the true score of the same row.
    /// Without the query-to-centroid term the cosine path returns q·(v - c), which is off by q·c.
    fn assert_adc_tracks_exact(metric: Metric) {
        let dim = 32;
        let rows = sphere_rows(3000, dim, 11);
        let data = as_data(&rows);
        let mut p = params(16, 16, 10, 16);
        p.store_vecs = false; // return raw ADC scores
        let idx = IvfPqIndex::build(metric, dim, &data, p);
        let queries = sphere_rows(20, dim, 99);
        let mut err_sum = 0.0f32; let mut cnt = 0usize;
        for q in &queries {
            for h in idx.search(q, 10) {
                let exact = metric::score(metric, q, &rows[h.id as usize]);
                err_sum += (h.score - exact).abs(); cnt += 1;
            }
        }
        let mae = err_sum / cnt as f32;
        assert!(mae < 0.05, "{:?}: mean |adc - exact| = {mae}", metric);
    }

    #[test]
    fn cosine_adc_score_includes_centroid_term() { assert_adc_tracks_exact(Metric::Cosine); }

    #[test]
    fn l2_adc_score_uses_query_residual() { assert_adc_tracks_exact(Metric::L2); }

    /// With exact re-rank, probing every list scans a superset of rows, so recall must not drop.
    #[test]
    fn probing_all_lists_beats_probing_fewer() {
        let dim = 64; let nlist = 64; let k = 10;
        let rows = sphere_rows(6000, dim, 1);
        let data = as_data(&rows);
        let queries = sphere_rows(50, dim, 2);
        let mut idx = IvfPqIndex::build(Metric::Cosine, dim, &data, params(nlist, nlist, 100, 16));
        let mut recalls = Vec::new();
        for nprobe in [4, 16, nlist] {
            idx.params.nprobe = nprobe;
            recalls.push(mean_recall(&idx, &rows, &queries, k));
        }
        assert!(recalls[2] >= recalls[1] && recalls[1] >= recalls[0], "recall by nprobe [4,16,all] = {recalls:?}");
        assert!(recalls[2] >= 0.90, "recall with all lists probed = {}", recalls[2]);
    }

    #[test]
    fn nbits_and_iters_are_honoured() {
        let dim = 16;
        let rows = sphere_rows(1000, dim, 3);
        let mut p = params(8, 8, 50, 4);
        p.nbits = 4; p.iters = 3;
        let idx = IvfPqIndex::build(Metric::Cosine, dim, &as_data(&rows), p);
        assert_eq!(idx.pq.nbits, 4);
        assert_eq!(idx.pq.k, 16);
        assert_eq!(idx.pq.iters, 3);
        assert!(idx.lists.iter().all(|l| l.codes.iter().all(|&c| c < 16)));
    }

    #[test]
    fn opq_is_trained_when_requested_and_keeps_recall() {
        let dim = 32; let k = 10;
        let rows = sphere_rows(3000, dim, 5);
        let data = as_data(&rows);
        let queries = sphere_rows(30, dim, 6);
        let base = IvfPqIndex::build(Metric::Cosine, dim, &data, params(16, 16, 100, 8));
        assert!(base.opq.is_none());
        let base_r = mean_recall(&base, &rows, &queries, k);
        for mode in [OpqMode::Perm, OpqMode::Pca, OpqMode::PcaPerm] {
            let mut p = params(16, 16, 100, 8);
            p.use_opq = true; p.opq_mode = mode;
            let idx = IvfPqIndex::build(Metric::Cosine, dim, &data, p);
            assert!(idx.opq.as_ref().map_or(false, |o| !o.is_identity()), "{mode:?}: OPQ not trained");
            let r = mean_recall(&idx, &rows, &queries, k);
            assert!(r >= base_r - 0.05, "{mode:?}: recall {r} vs no-OPQ {base_r}");
        }
    }

    #[test]
    fn build_is_identical_across_thread_counts() {
        let dim = 16;
        let rows = sphere_rows(2000, dim, 8);
        let data = as_data(&rows);
        let mut p = params(16, 4, 50, 4);
        p.use_opq = true;
        let a = build_with_threads(Metric::Cosine, dim, &data, p, 1);
        let b = build_with_threads(Metric::Cosine, dim, &data, p, 7);
        assert_eq!(a.centroids, b.centroids);
        assert_eq!(a.pq.codebooks, b.pq.codebooks);
        for (la, lb) in a.lists.iter().zip(&b.lists) {
            assert_eq!(la.ids, lb.ids);
            assert_eq!(la.codes, lb.codes);
        }
        assert_eq!(a.fingerprint(), b.fingerprint());
    }

    /// The scan `nearest_centroid` replaced: one centroid at a time, strictly greater wins.
    fn nearest_scalar(metric: Metric, v: &[f32], centroids: &[Vec<f32>]) -> usize {
        let mut best = 0usize; let mut bs = f32::NEG_INFINITY;
        for (j, c) in centroids.iter().enumerate() {
            let sc = coarse_score(metric, v, c);
            if sc > bs { bs = sc; best = j; }
        }
        best
    }

    /// 4-wide assignment picks the same list as the scalar scan, ties included:
    /// duplicate centroids inside one block of 4, across blocks, and in the tail.
    #[test]
    fn nearest_centroid_matches_scalar_scan_including_ties() {
        for metric in [Metric::Cosine, Metric::L2] {
            for (dim, nlist) in [(8usize, 1usize), (8, 3), (16, 4), (16, 13), (32, 64), (30, 67)] {
                let mut cents = sphere_rows(nlist, dim, (dim * 1000 + nlist) as u64);
                // Exact ties: copy earlier centroids forward, within and across blocks of 4.
                for (src, dst) in [(0usize, 1usize), (0, 5), (2, 3), (6, 11), (4, 12), (9, nlist - 1)] {
                    if dst < nlist && src < dst { cents[dst] = cents[src].clone(); }
                }
                let flat: Vec<f32> = cents.concat();
                let mut queries = sphere_rows(500, dim, 31 + nlist as u64);
                queries.extend(cents.iter().cloned()); // a query on a duplicated centroid is a tie at the top
                for q in &queries {
                    let want = nearest_scalar(metric, q, &cents);
                    assert_eq!(nearest_centroid(metric, q, nlist, |j| &cents[j][..]), want, "{metric:?} dim={dim} nlist={nlist}");
                    assert_eq!(nearest_centroid(metric, q, nlist, |j| &flat[j * dim..(j + 1) * dim]), want);
                }
            }
        }
    }

    #[test]
    fn fingerprint_covers_codes_and_centroids() {
        let dim = 16;
        let rows = sphere_rows(500, dim, 9);
        let idx = IvfPqIndex::build(Metric::Cosine, dim, &as_data(&rows), params(4, 4, 50, 4));
        let fp = idx.fingerprint();
        let mut c = idx.clone();
        let li = c.lists.iter().position(|l| !l.codes.is_empty()).unwrap();
        c.lists[li].codes[0] ^= 1;
        assert_ne!(fp, c.fingerprint());
        let mut d = idx.clone();
        d.centroids[0][0] += 1e-3;
        assert_ne!(fp, d.fingerprint());
    }

    #[test]
    fn non_finite_query_returns_nothing_and_ties_break_by_id() {
        let dim = 8;
        let mut rows = sphere_rows(200, dim, 10);
        rows[150] = rows[3].clone(); // exact duplicate of id 3
        let idx = IvfPqIndex::build(Metric::Cosine, dim, &as_data(&rows), params(4, 4, 200, 2));
        let hits = idx.search(&rows[3], 2);
        assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![3, 150]);
        assert!(idx.search(&vec![f32::NAN; dim], 5).is_empty());
        let mut one_nan = rows[3].clone(); one_nan[0] = f32::NAN;
        assert!(idx.search(&one_nan, 5).is_empty());
        let mut inf = rows[3].clone(); inf[1] = f32::INFINITY;
        assert!(idx.search(&inf, 5).is_empty());
    }
}
