use crate::metric::{self, Metric};
use crate::types::{Hit, stable_top_k};
use crate::flat::FlatIndex;
use crate::pq::ProductQuantizer;
use crate::opq::Opq;
use std::thread;
use serde::{Serialize, Deserialize};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::StdRng;

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

#[derive(Clone, Debug)]
pub struct IvfPqList {
    pub ids: Vec<u64>,
    pub codes: Vec<u8>,
    pub tomb: Vec<u8>,
    pub timestamps: Vec<u64>, // NEW: Temporal tracking array
    pub tags: Option<Vec<u64>>,
    pub vecs: Option<Vec<f32>>,
}

impl IvfPqList {
    #[inline] pub fn len(&self) -> usize { self.ids.len() }
    #[inline] pub fn code_row<'a>(&'a self, m: usize, i: usize) -> &'a [u8] { &self.codes[i*m .. (i+1)*m] }
    #[inline] pub fn vec_row<'a>(&'a self, dim: usize, i: usize) -> &'a [f32] { &self.vecs.as_ref().unwrap()[i*dim .. (i+1)*dim] }
}

#[derive(Clone, Debug)]
pub struct IvfPqIndex {
    pub metric: Metric, pub dim: usize, pub params: IvfPqParams,
    pub centroids: Vec<Vec<f32>>, pub centroids_u: Option<Vec<Vec<f32>>>,
    pub pq: ProductQuantizer, pub opq: Option<Opq>, pub lists: Vec<IvfPqList>,
}

#[inline] fn norm_inplace(v: &mut [f32]) {
    let mut n = 0.0f32; for &x in v.iter() { n += x * x; }
    if n > 0.0 { let inv = n.sqrt().recip(); for x in v { *x *= inv; } }
}

#[inline] fn normalized(v: &[f32]) -> Vec<f32> { let mut o = v.to_vec(); norm_inplace(&mut o); o }
#[inline] fn centroid_score_l2(q: &[f32], c: &[f32]) -> f32 { -metric::l2_distance(q, c) }

impl IvfPqIndex {
    // Signature updated to require (ID, Vector, Timestamp)
    pub fn build(metric: Metric, dim: usize, data: &[(u64, Vec<f32>, u64)], params: IvfPqParams) -> Self {
        let use_cosine = matches!(metric, Metric::Cosine);
        let views: Vec<Vec<f32>> = data.iter().map(|(_, v, _)| if use_cosine { normalized(v) } else { v.clone() }).collect();

        let mut rng = StdRng::seed_from_u64(params.seed);
        let mut centroids: Vec<Vec<f32>> = views.choose_multiple(&mut rng, params.nlist).cloned().collect();

        let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
        let batch_size = (params.nlist * 20).min(views.len());

        for _ in 0..15 {
            let batch: Vec<&Vec<f32>> = views.choose_multiple(&mut rng, batch_size).collect();
            let centroids_ref = &centroids;

            let results: Vec<(Vec<usize>, Vec<Vec<f32>>)> = thread::scope(|s| {
                let mut handles = Vec::new();
                for chunk in batch.chunks((batch.len() / num_threads).max(1)) {
                    handles.push(s.spawn(move || {
                        let mut local_sums = vec![vec![0.0f32; dim]; params.nlist];
                        let mut local_counts = vec![0usize; params.nlist];
                        for &v in chunk {
                            let mut best = 0; let mut bs = f32::NEG_INFINITY;
                            for (j, c) in centroids_ref.iter().enumerate() {
                                let s = centroid_score_l2(v, c);
                                if s > bs { bs = s; best = j; }
                            }
                            local_counts[best] += 1;
                            for d in 0..dim { local_sums[best][d] += v[d]; }
                        }
                        (local_counts, local_sums)
                    }));
                }
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });

            let mut global_counts = vec![0usize; params.nlist];
            let mut global_sums = vec![vec![0.0f32; dim]; params.nlist];
            for (l_counts, l_sums) in results {
                for j in 0..params.nlist {
                    global_counts[j] += l_counts[j];
                    for d in 0..dim { global_sums[j][d] += l_sums[j][d]; }
                }
            }

            for j in 0..params.nlist {
                if global_counts[j] > 0 {
                    let inv = 1.0 / global_counts[j] as f32;
                    for d in 0..dim { centroids[j][d] = global_sums[j][d] * inv; }
                    if use_cosine { norm_inplace(&mut centroids[j]); }
                }
            }
        }

        let centroids_ref = &centroids;
        let assign: Vec<usize> = thread::scope(|s| {
            let mut handles = Vec::new();
            for chunk in views.chunks((views.len() / num_threads).max(1)) {
                handles.push(s.spawn(move || {
                    chunk.iter().map(|v| {
                        let mut b = 0; let mut bs = f32::NEG_INFINITY;
                        for (j, c) in centroids_ref.iter().enumerate() {
                            let s = centroid_score_l2(v, c);
                            if s > bs { bs = s; b = j; }
                        }
                        b
                    }).collect::<Vec<usize>>()
                }));
            }
            handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });

        let pq_train_data: Vec<Vec<f32>> = views.iter().step_by((views.len()/4000).max(1)).enumerate().map(|(i, v)| {
            let li = assign[i * (views.len()/4000).max(1)];
            v.iter().zip(&centroids[li]).map(|(vi, ci)| vi - ci).collect()
        }).collect();
        let mut pq = ProductQuantizer::new(dim, params.m, 8, 15, params.seed);
        pq.train(&pq_train_data);

        let mut lists: Vec<IvfPqList> = (0..params.nlist).map(|_| IvfPqList {
            ids: Vec::new(), codes: Vec::new(), tomb: Vec::new(), timestamps: Vec::new(), tags: None,
            vecs: if params.store_vecs { Some(Vec::new()) } else { None },
        }).collect();

        // Populate the lists with IDs, vectors, and timestamps
        for (i, (id, _, ts)) in data.iter().enumerate() {
            let li = assign[i];
            let res: Vec<f32> = views[i].iter().zip(&centroids[li]).map(|(vi, ci)| vi - ci).collect();
            let list = &mut lists[li];
            list.ids.push(*id);
            list.timestamps.push(*ts); // Commit the temporal data
            list.tomb.push(0);
            list.codes.extend_from_slice(&pq.encode(&res));
            if let Some(vecs) = &mut list.vecs { vecs.extend_from_slice(&views[i]); }
        }

        Self { metric, dim, params, centroids, centroids_u: None, pq, opq: None, lists }
    }

    /// Primary search API injected with native temporal decay math.
    pub fn search_temporal(&self, q: &[f32], k: usize, required_tag: Option<u64>, current_time: u64, lambda: f32) -> Vec<Hit> {
        let q_eff = if matches!(self.metric, Metric::Cosine) { normalized(q) } else { q.to_vec() };
        let lut = if matches!(self.metric, Metric::Cosine) { self.pq.adc_lut_ip(&q_eff) } else { self.pq.adc_lut_l2(&q_eff) };
        let mut probe: Vec<(usize, f32)> = (0..self.centroids.len()).map(|i| (i, centroid_score_l2(&q_eff, &self.centroids[i]))).collect();
        probe.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut approx = Vec::new();
        for (li, _) in probe.into_iter().take(self.params.nprobe) {
            let list = &self.lists[li];
            for i in 0..list.len() {
                if list.tomb[i] == 1 { continue; }
                if let Some(mask) = required_tag {
                    if let Some(tags) = &list.tags {
                        if (tags[i] & mask) == 0 { continue; }
                    } else { continue; }
                }

                let mut score = if matches!(self.metric, Metric::Cosine) {
                    self.pq.adc_score_ip(&lut, list.code_row(self.params.m, i))
                } else {
                    -self.pq.adc_distance_l2(&lut, list.code_row(self.params.m, i))
                };

                // Temporal Decay Math (O(1) scalar multiplication)
                if lambda > 0.0 {
                    let dt = current_time.saturating_sub(list.timestamps[i]) as f32;
                    let decay = (-lambda * dt).exp().max(1e-9);
                    if score > 0.0 { score *= decay; } else { score /= decay; }
                }

                approx.push(Candidate { id: list.ids[i], score, li, pos: i });
            }
        }
        approx.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        approx.truncate(self.params.refine.max(k));

        if self.params.store_vecs {
            let mut final_hits = Vec::with_capacity(approx.len());
            for c in approx {
                let v = self.lists[c.li].vec_row(self.dim, c.pos);
                let mut s = if matches!(self.metric, Metric::Cosine) { metric::cosine_sim(&q_eff, v) } else { -metric::l2_distance(&q_eff, v) };

                // Re-apply exact temporal decay during the high-fidelity refinement
                if lambda > 0.0 {
                    let dt = current_time.saturating_sub(self.lists[c.li].timestamps[c.pos]) as f32;
                    let decay = (-lambda * dt).exp().max(1e-9);
                    if s > 0.0 { s *= decay; } else { s /= decay; }
                }

                final_hits.push(Hit { id: c.id, score: s });
            }
            final_hits.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            final_hits.truncate(k);
            final_hits
        } else {
            approx.into_iter().take(k).map(|c| Hit { id: c.id, score: c.score }).collect()
        }
    }

    /// Legacy wrapper for the standard trait layout (zero decay)
    pub fn search(&self, q: &[f32], k: usize) -> Vec<Hit> {
        self.search_temporal(q, k, None, 0, 0.0)
    }

    pub fn search_with_filter(&self, q: &[f32], k: usize, required_tag: Option<u64>) -> Vec<Hit> {
        self.search_temporal(q, k, required_tag, 0, 0.0)
    }

    pub fn fingerprint(&self) -> u64 { self.lists.iter().fold(0, |acc, l| acc ^ (l.ids.len() as u64)) }
}