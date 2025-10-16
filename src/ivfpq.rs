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
    pub refine: usize,
    pub seed: u64,
    pub m: usize,
    pub nbits: u8,
    pub iters: usize,
    pub use_opq: bool,
    pub opq_mode: OpqMode,
    pub opq_sweeps: usize,
    pub store_vecs: bool,
}

#[derive(Clone, Debug)]
pub struct IvfPqList {
    pub ids: Vec<u64>,
    pub codes: Vec<u8>,
    pub tomb: Vec<u8>,
    pub tags: Option<Vec<u64>>,
    pub vecs: Option<Vec<f32>>, // unit-norm if Cosine
}
impl IvfPqList {
    #[inline] fn len(&self) -> usize { self.ids.len() }
    #[inline] fn code_row<'a>(&'a self, m: usize, i: usize) -> &'a [u8] { let s = i*m; &self.codes[s..s+m] }
    #[inline] fn vec_row<'a>(&'a self, dim: usize, i: usize) -> &'a [f32] { let s = i*dim; &self.vecs.as_ref().unwrap()[s..s+dim] }
}

pub struct IvfPqIndex {
    pub metric: Metric,
    pub dim: usize,
    pub params: IvfPqParams,
    pub centroids: Vec<Vec<f32>>,            // normalized if Cosine
    pub centroids_u: Option<Vec<Vec<f32>>>,  // OPQ-transformed centroids (for IP per-list const)
    pub pq: ProductQuantizer,
    pub opq: Option<Opq>,
    pub lists: Vec<IvfPqList>,
}

#[inline]
fn norm_inplace(v: &mut [f32]) {
    let mut n = 0.0f32; for &x in v.iter() { n += x*x; }
    if n > 0.0 { let inv = n.sqrt().recip(); for x in v { *x *= inv; } }
}
#[inline]
fn normalized(v: &[f32]) -> Vec<f32> { let mut o = v.to_vec(); norm_inplace(&mut o); o }
#[inline]
fn centroid_score_l2(q: &[f32], c: &[f32]) -> f32 { -metric::l2_distance(q, c) }

impl IvfPqIndex {
    pub fn build(metric: Metric, dim: usize, data: &[(u64, Vec<f32>)], params: IvfPqParams) -> Self {
        assert!(dim > 0 && !data.is_empty());
        assert!(params.nlist > 0 && params.nlist <= data.len());
        assert!(params.m > 0 && dim % params.m == 0);
        assert!(params.nbits == 8);
        for (_, v) in data { assert_eq!(v.len(), dim); }

        let use_cosine = matches!(metric, Metric::Cosine);

        // Training views: Cosine → normalize then operate in L2 space
        let views: Vec<Vec<f32>> = if use_cosine {
            data.iter().map(|(_, v)| normalized(v)).collect()
        } else {
            data.iter().map(|(_, v)| v.clone()).collect()
        };

        // Coarse k-means (L2 for Cosine, metric otherwise)
        let km_metric = if use_cosine { Metric::L2 } else { metric };
        let mut centroids = kmeans_seeded(&views, params.nlist, km_metric, params.seed, 50);
        if use_cosine { for c in &mut centroids { norm_inplace(c); } }

        // Assign lists, build residuals in training space
        let mut assign = vec![0usize; data.len()];
        let mut residuals: Vec<Vec<f32>> = Vec::with_capacity(data.len());
        for (i, view) in views.iter().enumerate() {
            let mut best = 0usize; let mut bests = f32::NEG_INFINITY;
            for (j, c) in centroids.iter().enumerate() {
                let s = centroid_score_l2(view, c);
                if s > bests { bests = s; best = j; }
            }
            assign[i] = best;
            let mut r = vec![0.0f32; dim];
            for d in 0..dim { r[d] = view[d] - centroids[best][d]; }
            residuals.push(r);
        }

        // Train OPQ in residual space (optional)
        let opq = if params.use_opq {
            Some(match params.opq_mode {
                OpqMode::Perm    => Opq::train_perm(dim, params.m, &residuals),
                OpqMode::Pca     => Opq::train_pca(dim, params.m, &residuals, params.opq_sweeps),
                OpqMode::PcaPerm => Opq::train_pca_then_perm(dim, params.m, &residuals, params.opq_sweeps),
            })
        } else { None };

        let pq_train: Vec<Vec<f32>> = if let Some(U) = opq.as_ref() {
            residuals.iter().map(|r| U.apply(r)).collect()
        } else { residuals.clone() };

        let mut pq = ProductQuantizer::new(dim, params.m, params.nbits, params.iters.max(25), params.seed ^ 0xA5A5A5A5);
        pq.train(&pq_train);

        // Precompute OPQ-transformed centroids for IP path
        let centroids_u = if let Some(U) = opq.as_ref() {
            Some(centroids.iter().map(|c| U.apply(c)).collect())
        } else { None };

        // Create lists
        let mut lists: Vec<IvfPqList> = (0..params.nlist).map(|_| IvfPqList{
            ids: Vec::new(), codes: Vec::new(), tomb: Vec::new(), tags: None,
            vecs: if params.store_vecs { Some(Vec::new()) } else { None },
        }).collect();

        // Encode and insert
        for (i, (id, v_orig)) in data.iter().enumerate() {
            let li = assign[i];
            let mut r = residuals[i].clone();
            if let Some(U) = opq.as_ref() { r = U.apply(&r); }
            let codes = pq.encode(&r);

            let list = &mut lists[li];
            match list.ids.binary_search(id) {
                Ok(pos) => {
                    let s = pos * params.m; list.codes[s..s+params.m].copy_from_slice(&codes);
                    list.tomb[pos] = 0;
                    if let Some(vecs) = &mut list.vecs {
                        let s = pos * dim;
                        if use_cosine { let mut vn = v_orig.clone(); norm_inplace(&mut vn); vecs[s..s+dim].copy_from_slice(&vn); }
                        else { vecs[s..s+dim].copy_from_slice(v_orig); }
                    }
                }
                Err(pos) => {
                    list.ids.insert(pos, *id);
                    list.tomb.insert(pos, 0);
                    let s = pos * params.m; list.codes.splice(s..s, codes.iter().copied());
                    if let Some(vecs) = &mut list.vecs {
                        let s = pos * dim;
                        if use_cosine { let mut vn = v_orig.clone(); norm_inplace(&mut vn); vecs.splice(s..s, vn.into_iter()); }
                        else { vecs.splice(s..s, v_orig.iter().copied()); }
                    }
                }
            }
        }

        Self { metric, dim, params, centroids, centroids_u, pq, opq, lists }
    }

    pub fn search_with_filter(&self, q: &[f32], k: usize, required_tag: Option<u64>) -> Vec<Hit> {
        assert_eq!(q.len(), self.dim);
        let use_cosine = matches!(self.metric, Metric::Cosine);

        // Effective query for coarse probing (Cosine → normalized)
        let q_eff_buf; let q_eff: &[f32] = if use_cosine { q_eff_buf = normalized(q); &q_eff_buf } else { q };

        let nprobe = self.params.nprobe.min(self.centroids.len());
        let refine = self.params.refine.max(k);

        // Rank centroids (Cosine: -L2 on normalized space)
        let mut probe: Vec<(usize, f32)> = (0..self.centroids.len())
            .map(|i| (i, centroid_score_l2(q_eff, &self.centroids[i])))
            .collect();
        probe.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
        let probe: Vec<usize> = probe.into_iter().take(nprobe).map(|(i,_)| i).collect();

        // For IP ADC we need the query in the same space as codebooks
        let q_ip_buf; let q_ip: &[f32] = if use_cosine {
            if let Some(U) = self.opq.as_ref() { q_ip_buf = U.apply(q_eff); &q_ip_buf } else { q_eff }
        } else { &[] };

        let mut approx: Vec<Hit> = Vec::new();
        for li in probe {
            if use_cosine {
                // IP-style ADC: score = q·c + Σ_j q·codeword_j
                let per_list = if let Some(ref cu) = self.centroids_u { metric::cosine_sim(q_ip, &cu[li]) } else { metric::cosine_sim(q_eff, &self.centroids[li]) };
                let lut = if let Some(_) = self.opq { self.pq.adc_lut_ip(q_ip) } else { self.pq.adc_lut_ip(q_eff) }; // requires pq::adc_lut_ip
                let list = &self.lists[li];
                for i in 0..list.len() {
                    if list.tomb[i] != 0 { continue; }
                    if let Some(mask) = required_tag { if let Some(tags) = &list.tags { if (tags[i] & mask) != mask { continue; } } else { continue; } }
                    let codes = list.code_row(self.params.m, i);
                    let contrib = self.pq.adc_score_ip(&lut, codes); // Σ_j dot(q_slice, codeword)
                    approx.push(Hit { id: list.ids[i], score: per_list + contrib });
                }
            } else {
                // L2 residual ADC
                let mut qres = vec![0.0f32; self.dim];
                for d in 0..self.dim { qres[d] = q_eff[d] - self.centroids[li][d]; }
                let qres = if let Some(U) = self.opq.as_ref() { U.apply(&qres) } else { qres };
                let lut = self.pq.adc_lut_l2(&qres); // rename of adc_lut
                let list = &self.lists[li];
                for i in 0..list.len() {
                    if list.tomb[i] != 0 { continue; }
                    if let Some(mask) = required_tag { if let Some(tags) = &list.tags { if (tags[i] & mask) != mask { continue; } } else { continue; } }
                    let codes = list.code_row(self.params.m, i);
                    let dist = self.pq.adc_distance_l2(&lut, codes);
                    approx.push(Hit { id: list.ids[i], score: -dist });
                }
            }
        }
        if approx.is_empty() { return Vec::new(); }

        let top_r = stable_top_k(approx, refine);

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
        stable_top_k(top_r, k)
    }

    #[inline]
    pub fn search(&self, q: &[f32], k: usize) -> Vec<Hit> { self.search_with_filter(q, k, None) }

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
        h64(&mut h, p.opq_sweeps as u64); h64(&mut h, p.store_vecs as u64);
        for c in &self.centroids { for &f in c { hf(&mut h, f); } }
        if let Some(cu) = &self.centroids_u { for c in cu { for &f in c { hf(&mut h, f); } } }
        for j in 0..self.pq.m { for &f in &self.pq.codebooks[j] { hf(&mut h, f); } }
        if let Some(opq) = &self.opq { for x in opq.fingerprint_bits() { h64(&mut h, x); } }
        for list in &self.lists { h64(&mut h, list.ids.len() as u64); for id in &list.ids { h64(&mut h, *id); } }
        h
    }

    pub fn search_parallel(&self, q: &[f32], k: usize, threads: usize) -> Vec<Hit> {
        assert!(threads >= 1);
        assert_eq!(q.len(), self.dim);
        let use_cosine = matches!(self.metric, Metric::Cosine);

        let q_eff_buf; let q_eff: &[f32] = if use_cosine { q_eff_buf = normalized(q); &q_eff_buf } else { q };
        let nprobe = self.params.nprobe.min(self.centroids.len());
        let refine_total = self.params.refine.max(k);

        let mut probe: Vec<(usize, f32)> = (0..self.centroids.len())
            .map(|i| (i, centroid_score_l2(q_eff, &self.centroids[i])))
            .collect();
        probe.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
        let probe: Vec<usize> = probe.into_iter().take(nprobe).map(|(i,_)| i).collect();

        if threads == 1 || probe.is_empty() { return self.search(q, k); }

        let q_ip_buf; let q_ip: &[f32] = if use_cosine { if let Some(U) = self.opq.as_ref() { q_ip_buf = U.apply(q_eff); &q_ip_buf } else { q_eff } } else { &[] };

        let t = threads.min(probe.len());
        let chunk = (probe.len() + t - 1) / t;
        let refine_each = ((refine_total + t - 1) / t).max(k);

        let mut partials: Vec<Vec<Hit>> = vec![Vec::new(); t];
        let self_dim = self.dim; let self_m = self.params.m; let this = self;

        thread::scope(|scope| {
            let mut handles = Vec::with_capacity(t);
            for ti in 0..t {
                let start = ti * chunk; let end = ((ti + 1) * chunk).min(probe.len());
                let lists: Vec<usize> = probe[start..end].to_vec();
                handles.push(scope.spawn(move || -> (usize, Vec<Hit>) {
                    let mut local: Vec<Hit> = Vec::new();
                    for li in lists {
                        if use_cosine {
                            let per_list = if let Some(ref cu) = this.centroids_u { metric::cosine_sim(q_ip, &cu[li]) } else { metric::cosine_sim(q_eff, &this.centroids[li]) };
                            let lut = if let Some(_) = this.opq { this.pq.adc_lut_ip(q_ip) } else { this.pq.adc_lut_ip(q_eff) };
                            let list = &this.lists[li];
                            for i in 0..list.len() {
                                if list.tomb[i] != 0 { continue; }
                                let codes = list.code_row(self_m, i);
                                let contrib = this.pq.adc_score_ip(&lut, codes);
                                local.push(Hit { id: list.ids[i], score: per_list + contrib });
                            }
                        } else {
                            let mut qres = vec![0.0f32; self_dim];
                            for d in 0..self_dim { qres[d] = q_eff[d] - this.centroids[li][d]; }
                            let qres = if let Some(U) = this.opq.as_ref() { U.apply(&qres) } else { qres };
                            let lut = this.pq.adc_lut_l2(&qres);
                            let list = &this.lists[li];
                            for i in 0..list.len() {
                                if list.tomb[i] != 0 { continue; }
                                let codes = list.code_row(self_m, i);
                                let dist = this.pq.adc_distance_l2(&lut, codes);
                                local.push(Hit { id: list.ids[i], score: -dist });
                            }
                        }
                    }
                    local.sort_by(|a,b| b.score.partial_cmp(&a.score).unwrap().then(a.id.cmp(&b.id)));
                    if local.len() > refine_each { local.truncate(refine_each); }
                    (ti, local)
                }));
            }
            for h in handles { let (ti, local) = h.join().unwrap(); partials[ti] = local; }
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
