// src/ivfpq.rs
use crate::metric::{self, Metric};
use crate::types::{Hit, stable_top_k};
use crate::flat::FlatIndex;
use crate::kmeans::kmeans_seeded;
use crate::pq::ProductQuantizer;
use crate::opq::Opq;
use std::thread;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OpqMode { Perm, Pca, PcaPerm }

#[derive(Clone, Copy, Debug)]
pub struct IvfPqParams {
    pub nlist: usize,
    pub nprobe: usize,
    pub refine: usize,     // candidates to exact re-rank
    pub seed: u64,         // coarse k-means seed
    pub m: usize,          // PQ sub-quantizers (dim % m == 0)
    pub nbits: u8,         // 8 only for now
    pub iters: usize,      // k-means iters for PQ
    pub use_opq: bool,     // master switch
    pub opq_mode: OpqMode, // which OPQ flavor to train
    pub opq_sweeps: usize, // Jacobi sweeps for PCA (if used)
    pub store_vecs: bool,  // if false, return ADC result (no exact re-rank)
}

#[derive(Clone, Debug)]
pub struct IvfPqList {
    pub ids: Vec<u64>,
    pub codes: Vec<u8>,          // len = len(ids) * m
    pub tomb: Vec<u8>,           // 0 alive, 1 deleted
    pub tags: Option<Vec<u64>>,  // optional tag mask per id
    pub vecs: Option<Vec<f32>>,  // stored floats for exact re-rank (Cosine: unit-norm)
}
impl IvfPqList {
    #[inline] fn len(&self) -> usize { self.ids.len() }
    #[inline] fn code_row<'a>(&'a self, m: usize, i: usize) -> &'a [u8] {
        let s = i * m; &self.codes[s..s+m]
    }
    #[inline] fn vec_row<'a>(&'a self, dim: usize, i: usize) -> &'a [f32] {
        let vecs = self.vecs.as_ref().expect("vecs missing (store_vecs=false)");
        let s = i * dim; &vecs[s..s+dim]
    }
}

pub struct IvfPqIndex {
    pub metric: Metric,
    pub dim: usize,
    pub params: IvfPqParams,
    pub centroids: Vec<Vec<f32>>, // if Cosine, unit-norm
    pub pq: ProductQuantizer,     // trained on L2 residuals (Cosine: after unit-norm)
    pub opq: Option<Opq>,         // permutation / rotation (+ perm) if enabled
    pub lists: Vec<IvfPqList>,
}

// ---- helpers ----
#[inline]
fn l2_normalize_in_place(v: &mut [f32]) {
    let mut n = 0.0f32; for &x in v.iter() { n += x*x; }
    if n > 0.0 { n = n.sqrt(); for x in v.iter_mut() { *x /= n; } }
}
#[inline]
fn l2_normalized(v: &[f32]) -> Vec<f32> {
    let mut out = v.to_vec();
    l2_normalize_in_place(&mut out);
    out
}

/// Unified centroid “score” (higher is better).
#[inline]
fn centroid_score(metric_t: Metric, q: &[f32], c: &[f32]) -> f32 {
    match metric_t {
        Metric::L2 => -metric::l2_distance(q, c),
        Metric::Cosine => metric::cosine_sim(q, c),
    }
}
// For Cosine with unit-norm queries/centroids, -L2 and cosine are monotone.
#[inline]
fn centroid_score_l2(q: &[f32], c: &[f32]) -> f32 { -metric::l2_distance(q, c) }

impl IvfPqIndex {
    pub fn build(metric: Metric, dim: usize, data: &[(u64, Vec<f32>)], params: IvfPqParams) -> Self {
        assert!(dim > 0);
        assert!(!data.is_empty());
        assert!(params.nlist > 0 && params.nlist <= data.len());
        assert!(params.m > 0 && dim % params.m == 0, "dim must be divisible by m");
        assert!(params.nbits == 8, "only nbits=8 supported");
        for (_, v) in data { assert_eq!(v.len(), dim); }

        let use_cosine = matches!(metric, Metric::Cosine);

        // 1) Training views: for Cosine, normalize and operate in L2 space.
        let views: Vec<Vec<f32>> = if use_cosine {
            data.iter().map(|(_, v)| l2_normalized(v)).collect()
        } else {
            data.iter().map(|(_, v)| v.clone()).collect()
        };

        // 2) Coarse k-means on views using L2 (if Cosine) or given metric (if L2).
        let km_metric = if use_cosine { Metric::L2 } else { metric };
        let mut centroids = kmeans_seeded(&views, params.nlist, km_metric, params.seed, 50);
        if use_cosine {
            for c in &mut centroids { l2_normalize_in_place(c); }
        }

        // 3) Assign to lists + compute residuals in effective training space.
        let mut assign: Vec<usize> = vec![0; data.len()];
        let mut residuals: Vec<Vec<f32>> = Vec::with_capacity(data.len());
        for (i, view) in views.iter().enumerate() {
            // rank centroids by -L2 on normalized space when Cosine
            let (li, _) = {
                let mut best = 0usize; let mut bests = f32::NEG_INFINITY;
                for (j, c) in centroids.iter().enumerate() {
                    let s = centroid_score_l2(view, c);
                    if s > bests { bests = s; best = j; }
                }
                (best, bests)
            };
            assign[i] = li;

            let mut r = vec![0.0f32; dim];
            for d in 0..dim { r[d] = view[d] - centroids[li][d]; }
            residuals.push(r);
        }

        // 4) Train OPQ in residual space (if enabled).
        let opq = if params.use_opq {
            Some(match params.opq_mode {
                OpqMode::Perm    => Opq::train_perm(dim, params.m, &residuals),
                OpqMode::Pca     => Opq::train_pca(dim, params.m, &residuals, params.opq_sweeps),
                OpqMode::PcaPerm => Opq::train_pca_then_perm(dim, params.m, &residuals, params.opq_sweeps),
            })
        } else { None };

        let pq_train_in: Vec<Vec<f32>> = if let Some(opq_ref) = opq.as_ref() {
            residuals.iter().map(|r| opq_ref.apply(r)).collect()
        } else {
            residuals.clone()
        };

        // 5) Train PQ (deterministic seed) on residuals in effective L2 space.
        let mut pq = ProductQuantizer::new(dim, params.m, params.nbits, params.iters.max(25), params.seed ^ 0xA5A5A5A5);
        pq.train(&pq_train_in);

        // 6) Create empty lists.
        let mut lists: Vec<IvfPqList> = (0..params.nlist).map(|_| IvfPqList{
            ids: Vec::new(), codes: Vec::new(), tomb: Vec::new(), tags: None,
            vecs: if params.store_vecs { Some(Vec::new()) } else { None },
        }).collect();

        // 7) Encode + insert (lists are id-sorted). For Cosine, store unit-norm rows.
        for (i, (id, v_orig)) in data.iter().enumerate() {
            let li = assign[i];

            let mut r = residuals[i].clone();
            if let Some(opq_ref) = opq.as_ref() { r = opq_ref.apply(&r); }
            let codes = pq.encode(&r);

            let list = &mut lists[li];
            match list.ids.binary_search(id) {
                Ok(pos) => {
                    let s = pos * params.m; list.codes[s..s+params.m].copy_from_slice(&codes);
                    list.tomb[pos] = 0;
                    if let Some(vecs) = &mut list.vecs {
                        let s = pos * dim;
                        if use_cosine {
                            let mut vn = v_orig.clone(); l2_normalize_in_place(&mut vn);
                            vecs[s..s+dim].copy_from_slice(&vn);
                        } else {
                            vecs[s..s+dim].copy_from_slice(v_orig);
                        }
                    }
                }
                Err(pos) => {
                    list.ids.insert(pos, *id);
                    list.tomb.insert(pos, 0);
                    let s_codes = pos * params.m;
                    list.codes.splice(s_codes..s_codes, codes.iter().copied());
                    if let Some(vecs) = &mut list.vecs {
                        let s = pos * dim;
                        if use_cosine {
                            let mut vn = v_orig.clone(); l2_normalize_in_place(&mut vn);
                            vecs.splice(s..s, vn.into_iter());
                        } else {
                            vecs.splice(s..s, v_orig.iter().copied());
                        }
                    }
                }
            }
        }

        Self { metric, dim, params, centroids, pq, opq, lists }
    }

    /// Search using ADC on residuals, then exact re-rank (if store_vecs=true).
    /// Cosine: normalize query and operate in L2 for coarse + PQ.
    pub fn search_with_filter(&self, q: &[f32], k: usize, required_tag: Option<u64>) -> Vec<Hit> {
        assert_eq!(q.len(), self.dim);
        let use_cosine = matches!(self.metric, Metric::Cosine);

        // Effective query for coarse+PQ
        let q_eff_vec;
        let q_eff = if use_cosine {
            q_eff_vec = l2_normalized(q);
            &q_eff_vec
        } else { q };

        let nprobe = self.params.nprobe.min(self.centroids.len());
        let refine = self.params.refine.max(k);

        // 1) Rank centroids
        let mut probe: Vec<(usize, f32)> = (0..self.centroids.len()).map(|i| {
            let s = if use_cosine {
                centroid_score_l2(q_eff, &self.centroids[i])
            } else {
                centroid_score(self.metric, q_eff, &self.centroids[i])
            };
            (i, s)
        }).collect();
        probe.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
        let probe: Vec<usize> = probe.into_iter().take(nprobe).map(|(i,_)| i).collect();

        // 2) Gather ADC candidates
        let mut approx: Vec<Hit> = Vec::new();
        for li in probe {
            // residual for this list
            let mut qres = vec![0.0f32; self.dim];
            for d in 0..self.dim { qres[d] = q_eff[d] - self.centroids[li][d]; }
            let qres = if let Some(opq_ref) = self.opq.as_ref() { opq_ref.apply(&qres) } else { qres };

            let lut = self.pq.adc_lut(&qres);

            let list = &self.lists[li];
            for i in 0..list.len() {
                if list.tomb[i] != 0 { continue; }
                if let Some(mask) = required_tag {
                    if let Some(tags) = &list.tags { if (tags[i] & mask) != mask { continue; } } else { continue; }
                }
                let codes = list.code_row(self.params.m, i);
                let dist = self.pq.adc_distance(&lut, codes);
                approx.push(Hit { id: list.ids[i], score: -dist }); // higher is better
            }
        }
        if approx.is_empty() { return Vec::new(); }

        // Keep top R by approximate score, stable ties
        let mut top_r = stable_top_k(approx, refine);

        // 3) Exact re-rank with stored floats (if available)
        if self.params.store_vecs {
            let mut refine_flat = FlatIndex::new(self.dim, self.metric);
            for h in &top_r {
                'outer: for list in &self.lists {
                    if let Ok(pos) = list.ids.binary_search(&h.id) {
                        if list.tomb[pos] == 0 { refine_flat.add(h.id, &list.vec_row(self.dim, pos)); }
                        break 'outer;
                    }
                }
            }
            return refine_flat.search(q, k);
        }
        // else, return approximate top-k
        stable_top_k(top_r, k)
    }

    #[inline]
    pub fn search(&self, q: &[f32], k: usize) -> Vec<Hit> { self.search_with_filter(q, k, None) }

    /// Deterministic build fingerprint: params, centroids, PQ codebooks, OPQ transform, list ids.
    pub fn fingerprint(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        #[inline] fn h64(h: &mut u64, x: u64) { *h ^= x; *h = h.wrapping_mul(0x100000001b3); }
        #[inline] fn hf(h: &mut u64, x: f32) { h64(h, x.to_bits() as u64); }

        h64(&mut h, self.dim as u64);
        h64(&mut h, match self.metric { Metric::L2 => 0, Metric::Cosine => 1 });
        let p = &self.params;
        h64(&mut h, p.nlist as u64); h64(&mut h, p.nprobe as u64); h64(&mut h, p.refine as u64);
        h64(&mut h, p.seed as u64);  h64(&mut h, p.m as u64);      h64(&mut h, p.nbits as u64);
        h64(&mut h, p.iters as u64); h64(&mut h, p.use_opq as u64);
        h64(&mut h, match p.opq_mode { OpqMode::Perm=>1, OpqMode::Pca=>2, OpqMode::PcaPerm=>3 });
        h64(&mut h, p.opq_sweeps as u64);
        h64(&mut h, p.store_vecs as u64);

        for c in &self.centroids { for &f in c { hf(&mut h, f); } }
        for j in 0..self.pq.m { for &f in &self.pq.codebooks[j] { hf(&mut h, f); } }
        if let Some(opq) = &self.opq { for x in opq.fingerprint_bits() { h64(&mut h, x); } }
        for list in &self.lists {
            h64(&mut h, list.ids.len() as u64);
            for id in &list.ids { h64(&mut h, *id); }
        }
        h
    }

    /// Parallel ADC gather with per-thread partial top-R, then exact refine (if enabled).
    /// Mirrors search_with_filter logic (Cosine→normalize + L2 residual + OPQ).
    pub fn search_parallel(&self, q: &[f32], k: usize, threads: usize) -> Vec<Hit> {
        assert!(threads >= 1);
        assert_eq!(q.len(), self.dim);
        let use_cosine = matches!(self.metric, Metric::Cosine);

        // Effective query
        let q_eff_vec;
        let q_eff = if use_cosine {
            q_eff_vec = l2_normalized(q);
            &q_eff_vec
        } else { q };

        let nprobe = self.params.nprobe.min(self.centroids.len());
        let refine_total = self.params.refine.max(k);

        // Probe lists by centroid score in the effective space
        let mut probe: Vec<(usize, f32)> = (0..self.centroids.len()).map(|i| {
            let s = if use_cosine {
                centroid_score_l2(q_eff, &self.centroids[i])
            } else {
                centroid_score(self.metric, q_eff, &self.centroids[i])
            };
            (i, s)
        }).collect();
        probe.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
        let probe: Vec<usize> = probe.into_iter().take(nprobe).map(|(i,_)| i).collect();

        if threads == 1 || probe.is_empty() {
            return self.search(q, k);
        }

        let t = threads.min(probe.len());
        let chunk = (probe.len() + t - 1) / t;
        let refine_each = ((refine_total + t - 1) / t).max(k);

        let mut partials: Vec<Vec<Hit>> = vec![Vec::new(); t];
        let self_dim = self.dim;
        let self_m = self.params.m;
        let this = self; // capture &self for the scoped threads

        thread::scope(|scope| {
            let mut handles = Vec::with_capacity(t);
            for ti in 0..t {
                let start = ti * chunk;
                let end = ((ti + 1) * chunk).min(probe.len());
                let lists: Vec<usize> = probe[start..end].to_vec();
                let q_eff_ref: &[f32] = q_eff;

                handles.push(scope.spawn(move || -> (usize, Vec<Hit>) {
                    let mut local: Vec<Hit> = Vec::new();
                    for li in lists {
                        // residual for this list
                        let mut qres = vec![0.0f32; self_dim];
                        for d in 0..self_dim { qres[d] = q_eff_ref[d] - this.centroids[li][d]; }
                        let qres = if let Some(opq_ref) = this.opq.as_ref() { opq_ref.apply(&qres) } else { qres };

                        let lut = this.pq.adc_lut(&qres);
                        let list = &this.lists[li];
                        for i in 0..list.len() {
                            if list.tomb[i] != 0 { continue; }
                            let codes = list.code_row(self_m, i);
                            let dist = this.pq.adc_distance(&lut, codes);
                            local.push(Hit { id: list.ids[i], score: -dist });
                        }
                    }
                    local.sort_by(|a,b| b.score.partial_cmp(&a.score).unwrap().then(a.id.cmp(&b.id)));
                    if local.len() > refine_each { local.truncate(refine_each); }
                    (ti, local)
                }));
            }
            for h in handles {
                let (ti, local) = h.join().unwrap();
                partials[ti] = local;
            }
        });

        let mut approx: Vec<Hit> = partials.into_iter().flatten().collect();
        approx.sort_by(|a,b| b.score.partial_cmp(&a.score).unwrap().then(a.id.cmp(&b.id)));
        if approx.len() > refine_total { approx.truncate(refine_total); }

        if self.params.store_vecs {
            let mut refine_flat = FlatIndex::new(self.dim, self.metric);
            for h in &approx {
                'outer: for list in &self.lists {
                    if let Ok(pos) = list.ids.binary_search(&h.id) {
                        if list.tomb[pos] == 0 { refine_flat.add(h.id, &list.vec_row(self_dim, pos)); }
                        break 'outer;
                    }
                }
            }
            return refine_flat.search(q, k);
        }
        stable_top_k(approx, k)
    }
}
