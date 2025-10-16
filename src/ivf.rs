use crate::metric::{self, Metric};
use crate::types::Hit;
use crate::flat::FlatIndex;
use crate::kmeans::kmeans_seeded;
use crate::header::{IndexHeader, Seeds};
use std::thread;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug)]
pub struct IvfParams {
    pub nlist: usize,
    pub nprobe: usize,
    pub refine: usize,
    pub seed: u64,
}

#[derive(Clone, Debug)]
pub struct ListFlat {
    pub ids: Vec<u64>,
    pub vecs: Vec<f32>,     // row-major [count * dim]
    pub tomb: Vec<u8>,      // 0 alive, 1 deleted
    pub tags: Option<Vec<u64>>, // optional per-id bitmask
}
impl ListFlat {
    #[inline] fn len(&self) -> usize { self.ids.len() }
    #[inline] fn row<'a>(&'a self, dim: usize, i: usize) -> &'a [f32] {
        let s = i * dim; &self.vecs[s..s+dim]
    }
}

pub struct IvfIndex {
    pub header: IndexHeader,
    pub params: IvfParams,
    pub centroids: Vec<Vec<f32>>, // [nlist][dim] (normalized if Cosine)
    pub lists: Vec<ListFlat>,     // IVF-Flat storage (vectors normalized if Cosine)
}

#[inline]
fn normalize_inplace(v: &mut [f32]) {
    let mut n = 0.0f32; for &x in v.iter() { n += x*x; }
    if n > 0.0 { let inv = n.sqrt().recip(); for x in v { *x *= inv; } }
}

impl IvfIndex {
    pub fn build(metric: Metric, dim: usize, data: &[(u64, Vec<f32>)], mut params: IvfParams) -> Self {
        let n = data.len();
        assert!(n > 0, "IvfIndex::build requires at least one vector (N=0)");

        // Clamp nlist → [1, N]
        if params.nlist == 0 { params.nlist = 1; }
        if params.nlist > n { params.nlist = n; }

        // For Cosine, normalize once up-front so search can use pure dot.
        let mut train_vecs: Vec<Vec<f32>> = data.iter().map(|(_, v)| v.clone()).collect();
        if matches!(metric, Metric::Cosine) {
            for v in &mut train_vecs { normalize_inplace(v); }
        }

        // K-means in the SAME metric space we will search.
        let mut centers = kmeans_seeded(&train_vecs, params.nlist, metric, params.seed, 50);
        if matches!(metric, Metric::Cosine) {
            for c in &mut centers { normalize_inplace(c); }
        }

        // Create empty lists
        let mut lists: Vec<ListFlat> = (0..params.nlist)
            .map(|_| ListFlat{ ids:Vec::new(), vecs:Vec::new(), tomb:Vec::new(), tags:None })
            .collect();

        // Assign by nearest centroid (id tiebreak). Use normalized vectors for Cosine.
        for (i, (id, _)) in data.iter().enumerate() {
            let v = &train_vecs[i];
            let mut best = 0usize; let mut bestd = f32::INFINITY; let mut bestid = *id;
            for (c, ctr) in centers.iter().enumerate() {
                let d = match metric {
                    Metric::L2 => metric::l2_distance(v, ctr),
                    Metric::Cosine => 1.0 - metric::cosine_sim(v, ctr),
                };
                if d < bestd || (d == bestd && *id < bestid) { bestd = d; best = c; bestid = *id; }
            }
            let list = &mut lists[best];
            list.ids.push(*id);
            list.vecs.extend_from_slice(v); // already normalized if Cosine
            list.tomb.push(0);
        }

        // Stable id order per list (reorder vecs/tomb/tags accordingly)
        for list in lists.iter_mut() {
            let mut idx: Vec<usize> = (0..list.ids.len()).collect();
            idx.sort_by_key(|&i| list.ids[i]);
            let mut new_ids = Vec::with_capacity(list.ids.len());
            let mut new_vecs = Vec::with_capacity(list.vecs.len());
            let mut new_tomb = Vec::with_capacity(list.tomb.len());
            let mut new_tags = list.tags.as_ref().map(|t| Vec::with_capacity(t.len()));
            for i in idx {
                new_ids.push(list.ids[i]);
                new_tomb.push(list.tomb[i]);
                new_vecs.extend_from_slice(&list.row(dim, i));
                if let (Some(tags), Some(nt)) = (&list.tags, new_tags.as_mut()) { nt.push(tags[i]); }
            }
            list.ids = new_ids; list.vecs = new_vecs; list.tomb = new_tomb; list.tags = new_tags;
        }

        let header = IndexHeader::new(dim, metric, Seeds{ data:0, queries:0, kmeans: params.seed, hnsw: None }, params, false);
        Self { header, params, centroids: centers, lists }
    }

    /// Mark an id as deleted if present.
    pub fn delete(&mut self, id: u64) -> bool {
        for list in &mut self.lists {
            if let Ok(pos) = list.ids.binary_search(&id) { list.tomb[pos] = 1; return true; }
        }
        false
    }

    /// Upsert vector and optional tag. Keeps id-sort; replaces in-place if found.
    pub fn upsert(&mut self, id: u64, v: &[f32], tag: Option<u64>) {
        assert_eq!(v.len(), self.header.dim);
        // Normalize for cosine so storage is consistent.
        let mut vstore: Vec<f32> = v.to_vec();
        if matches!(self.header.metric, Metric::Cosine) { normalize_inplace(&mut vstore); }

        // pick list by highest centroid score
        let mut best = 0usize; let mut bests = f32::NEG_INFINITY;
        for (i, c) in self.centroids.iter().enumerate() {
            let s = match self.header.metric {
                Metric::L2 => -metric::l2_distance(&vstore, c),
                Metric::Cosine => metric::cosine_sim(&vstore, c),
            };
            if s > bests { bests = s; best = i; }
        }
        let list = &mut self.lists[best];
        match list.ids.binary_search(&id) {
            Ok(pos) => {
                let s = pos * self.header.dim;
                for d in 0..self.header.dim { list.vecs[s+d] = vstore[d]; }
                list.tomb[pos] = 0;
                if let Some(mask) = tag {
                    if let Some(tags) = &mut list.tags { tags[pos] = mask; }
                    else { let mut t = vec![0u64; list.ids.len()]; t[pos]=mask; list.tags = Some(t); }
                }
            }
            Err(pos) => {
                list.ids.insert(pos, id);
                list.tomb.insert(pos, 0);
                let s = pos * self.header.dim;
                list.vecs.splice(s..s, vstore.into_iter());
                if let Some(mask) = tag {
                    if let Some(tags) = &mut list.tags { tags.insert(pos, mask); }
                    else { let mut t = vec![0u64; list.ids.len()-1]; t.insert(pos, mask); list.tags = Some(t); }
                } else if let Some(tags) = &mut list.tags { tags.insert(pos, 0); }
            }
        }
    }

    /// Remove tombstoned entries and re-pack storage (deterministic).
    pub fn compact(&mut self) {
        let dim = self.header.dim;
        for list in &mut self.lists {
            let mut new_ids = Vec::with_capacity(list.ids.len());
            let mut new_vecs = Vec::with_capacity(list.vecs.len());
            let mut new_tomb = Vec::new();
            let mut new_tags: Option<Vec<u64>> = list.tags.as_ref().map(|_| Vec::new());
            for i in 0..list.ids.len() {
                if list.tomb[i] == 0 {
                    new_ids.push(list.ids[i]);
                    new_tomb.push(0);
                    new_vecs.extend_from_slice(&list.row(dim, i));
                    if let Some(tags) = &list.tags { new_tags.as_mut().unwrap().push(tags[i]); }
                }
            }
            list.ids = new_ids; list.vecs = new_vecs; list.tomb = new_tomb; list.tags = new_tags;
        }
    }

    #[inline]
    fn normalized_query<'a>(&self, q: &'a [f32]) -> Vec<f32> {
        if matches!(self.header.metric, Metric::Cosine) {
            let mut qq = q.to_vec();
            normalize_inplace(&mut qq); qq
        } else { Vec::new() } // sentinel: empty means "use q as-is"
    }

    /// Rank centroids and return the top nprobe list indices.
    fn select_probe_lists(&self, q: &[f32]) -> Vec<usize> {
        let metric = self.header.metric;
        let qbuf = self.normalized_query(q);
        let qref: &[f32] = if qbuf.is_empty() { q } else { &qbuf };

        let nprobe = self.params.nprobe.min(self.centroids.len());
        let mut scores: Vec<(usize, f32)> = self.centroids.iter().enumerate().map(|(i,c)| {
            let s = match metric { Metric::L2 => -metric::l2_distance(qref, c), Metric::Cosine => metric::cosine_sim(qref, c) };
            (i, s)
        }).collect();
        // partial select would be faster, but nlist is small; full sort keeps determinism
        scores.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(nprobe);
        scores.into_iter().map(|(i,_)| i).collect()
    }

    /// Search with optional required tag mask. A hit must satisfy (tags[id] & mask) == mask.
    pub fn search_with_filter(&self, q: &[f32], k: usize, required_tag: Option<u64>) -> Vec<Hit> {
        assert_eq!(q.len(), self.header.dim);
        let dim = self.header.dim;
        let metric = self.header.metric;

        // Use normalized query for Cosine
        let qbuf = self.normalized_query(q);
        let qref: &[f32] = if qbuf.is_empty() { q } else { &qbuf };

        // choose lists
        let probe = self.select_probe_lists(qref);
        let refine = self.params.refine.max(k);

        // gather candidates
        let mut scored: Vec<Hit> = Vec::new();
        for li in probe {
            let list = &self.lists[li];
            for i in 0..list.len() {
                if list.tomb[i] != 0 { continue; }
                if let Some(mask) = required_tag {
                    if let Some(tags) = &list.tags {
                        if (tags[i] & mask) != mask { continue; }
                    } else { continue; }
                }
                let s = metric::score(metric, qref, list.row(dim, i));
                scored.push(Hit{ id: list.ids[i], score: s });
            }
        }
        if scored.is_empty() { return Vec::new(); }

        scored.sort_by(|a,b| b.score.partial_cmp(&a.score).unwrap().then(a.id.cmp(&b.id)));
        if scored.len() > refine { scored.truncate(refine); }

        // exact refine
        let mut refine_flat = FlatIndex::new(dim, metric);
        for h in &scored {
            for list in &self.lists {
                if let Ok(pos) = list.ids.binary_search(&h.id) {
                    if list.tomb[pos] == 0 { refine_flat.add(h.id, list.row(dim, pos)); }
                    break;
                }
            }
        }
        refine_flat.search(qref, k)
    }

    #[inline]
    pub fn search(&self, q: &[f32], k: usize) -> Vec<Hit> { self.search_with_filter(q, k, None) }

    /// Deterministic build fingerprint (FNV-1a over header+centroids+ids).
    pub fn fingerprint(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        #[inline] fn h64(h: &mut u64, x: u64) { *h ^= x; *h = h.wrapping_mul(0x100000001b3); }
        #[inline] fn hf(h: &mut u64, x: f32) { h64(h, x.to_bits() as u64); }
        h64(&mut h, self.header.dim as u64);
        h64(&mut h, match self.header.metric { Metric::L2 => 0, Metric::Cosine => 1 });
        h64(&mut h, self.params.nlist as u64);
        h64(&mut h, self.params.nprobe as u64);
        h64(&mut h, self.params.refine as u64);
        h64(&mut h, self.params.seed as u64);
        for c in &self.centroids { for &f in c { hf(&mut h, f); } }
        for list in &self.lists { h64(&mut h, list.ids.len() as u64); for id in &list.ids { h64(&mut h, *id); } }
        h
    }

    pub fn search_parallel(&self, q: &[f32], k: usize, threads: usize) -> Vec<Hit> {
        assert!(threads >= 1);
        assert_eq!(q.len(), self.header.dim);
        let dim = self.header.dim;
        let metric = self.header.metric;

        // normalized query for Cosine
        let qbuf = self.normalized_query(q);
        let qref: &[f32] = if qbuf.is_empty() { q } else { &qbuf };

        // choose probe lists
        let probe = self.select_probe_lists(qref);
        if threads == 1 || probe.is_empty() { return self.search(qref, k); }

        let t = threads.min(probe.len());
        let chunk = (probe.len() + t - 1) / t;
        let refine_total = self.params.refine.max(k);
        let refine_each = ((refine_total + t - 1) / t).max(k);

        // collect partials deterministically by thread index
        let mut partials: Vec<Vec<Hit>> = vec![Vec::new(); t];

        thread::scope(|scope| {
            let mut handles = Vec::with_capacity(t);
            for ti in 0..t {
                let start = ti * chunk;
                let end = ((ti + 1) * chunk).min(probe.len());
                let lists: Vec<usize> = probe[start..end].to_vec();

                handles.push(scope.spawn(move || -> (usize, Vec<Hit>) {
                    let mut local: Vec<Hit> = Vec::new();
                    for li in lists {
                        let list = &self.lists[li];
                        for i in 0..list.len() {
                            if list.tomb[i] != 0 { continue; }
                            let s = metric::score(metric, qref, list.row(dim, i));
                            local.push(Hit { id: list.ids[i], score: s });
                        }
                    }
                    local.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap().then(a.id.cmp(&b.id)));
                    if local.len() > refine_each { local.truncate(refine_each); }
                    (ti, local)
                }));
            }

            for h in handles { let (ti, local) = h.join().unwrap(); partials[ti] = local; }
        });

        let mut approx: Vec<Hit> = partials.into_iter().flatten().collect();
        approx.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap().then(a.id.cmp(&b.id)));
        if approx.len() > refine_total { approx.truncate(refine_total); }

        // exact re-rank
        let mut refine_flat = FlatIndex::new(dim, metric);
        for h in &approx {
            for list in &self.lists {
                if let Ok(pos) = list.ids.binary_search(&h.id) {
                    if list.tomb[pos] == 0 { refine_flat.add(h.id, list.row(dim, pos)); }
                    break;
                }
            }
        }
        refine_flat.search(qref, k)
    }
}
