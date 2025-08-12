use crate::metric::{self, Metric};
use crate::types::{Hit, stable_top_k};
use crate::flat::FlatIndex;
use crate::kmeans::kmeans_seeded;

#[derive(Clone, Copy, Debug)]
pub struct IvfParams {
    pub nlist: usize,
    pub nprobe: usize, // at search time
    pub refine: usize, // top‑R refine exact
    pub seed: u64,
}

pub struct IvfIndex {
    metric: Metric,
    dim: usize,
    params: IvfParams,
    centroids: Vec<Vec<f32>>,              // [nlist][dim]
    lists: Vec<Vec<(u64, Vec<f32>)>>,      // per‑list storage (IVF‑Flat)
}

impl IvfIndex {
    pub fn build(
        metric: Metric,
        dim: usize,
        data: &[(u64, Vec<f32>)],
        params: IvfParams,
    ) -> Self {
        assert!(params.nlist > 0 && params.nlist <= data.len());
        for (_, v) in data { assert_eq!(v.len(), dim); }

        // Prepare views for kmeans
        let views: Vec<&[f32]> = data.iter().map(|(_, v)| v.as_slice()).collect();
        let views2: Vec<Vec<f32>> = views.iter().map(|x| x.to_vec()).collect();
        let centers = kmeans_seeded(&views2, params.nlist, metric, params.seed, 50);

        let mut lists: Vec<Vec<(u64, Vec<f32>)>> = vec![Vec::new(); params.nlist];
        // Assign by nearest centroid (tie‑break by smaller id)
        for (id, v) in data.iter() {
            let mut best = 0usize; let mut bestd = f32::INFINITY; let mut bestid = *id;
            for (c, ctr) in centers.iter().enumerate() {
                let d = match metric {
                    Metric::L2 => metric::l2_distance(v, ctr),
                    Metric::Cosine => 1.0 - metric::cosine_sim(v, ctr),
                };
                if d < bestd || (d == bestd && *id < bestid) {
                    bestd = d; best = c; bestid = *id;
                }
            }
            lists[best].push((*id, v.clone()));
        }
        // Optional: sort each list by id for stable iteration
        for list in lists.iter_mut() { list.sort_by_key(|(id, _)| *id); }

        Self { metric, dim, params, centroids: centers, lists }
    }

    pub fn search(&self, q: &[f32], k: usize) -> Vec<Hit> {
        assert_eq!(q.len(), self.dim);
        let nprobe = self.params.nprobe.min(self.centroids.len());
        let refine = self.params.refine.max(k);

        // 1) choose probe lists by centroid score
        let mut cands: Vec<(usize, f32)> = Vec::with_capacity(self.centroids.len());
        for (i, c) in self.centroids.iter().enumerate() {
            let s = match self.metric {
                Metric::L2 => -metric::l2_distance(q, c),
                Metric::Cosine => metric::cosine_sim(q, c),
            };
            cands.push((i, s));
        }
        cands.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
        let probe: Vec<usize> = cands.into_iter().take(nprobe).map(|(i,_)| i).collect();

        // 2) gather candidates from probed lists
        let mut pool: Vec<(u64, Vec<f32>)> = Vec::new();
        for li in probe { pool.extend(self.lists[li].iter().cloned()); }
        if pool.is_empty() { return Vec::new(); }

        // 3) coarse exact scores on pool (IVF‑Flat search)
        let mut scored: Vec<Hit> = Vec::with_capacity(pool.len());
        for (id, v) in pool.iter() {
            let s = metric::score(self.metric, q, v);
            scored.push(Hit { id: *id, score: s });
        }
        // take top‑R for refine
        scored.sort_by(|a,b| b.score.partial_cmp(&a.score).unwrap().then(a.id.cmp(&b.id)) );
        if scored.len() > refine { scored.truncate(refine); }

        // 4) refine with exact Flat
        let mut refine_flat = FlatIndex::new(self.dim, self.metric);
        for h in &scored {
            // we already have vectors in `pool`, re‑lookup by id (pool is small)
            let v = pool.iter().find(|(id, _)| *id == h.id).unwrap().1.clone();
            refine_flat.add(h.id, &v);
        }
        refine_flat.search(q, k)
    }
    pub fn fingerprint(&self) -> u64 {
        let mut h:u64 = 0xcbf29ce484222325; // FNV-1a 64
        #[inline] fn h64(h:&mut u64, x:u64){ *h ^= x; *h = h.wrapping_mul(0x100000001b3); }
        #[inline] fn hf(h:&mut u64, x:f32){ h64(h, x.to_bits() as u64); }
        for c in &self.centroids { for &f in c { hf(&mut h, f); } }
        for list in &self.lists {
            h64(&mut h, list.len() as u64);
            for (id, _) in list { h64(&mut h, *id); }
        }
        h
    }
}