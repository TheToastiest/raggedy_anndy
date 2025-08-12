use crate::metric::{self, Metric};
use crate::types::Hit;
use crate::flat::FlatIndex;
use crate::kmeans::kmeans_seeded;
use crate::header::{IndexHeader, Seeds};

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
    pub vecs: Vec<f32>, // row-major [count * dim]
    pub tomb: Vec<u8>,  // 0 alive, 1 deleted
    pub tags: Option<Vec<u64>>, // optional tag mask per id
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
    pub centroids: Vec<Vec<f32>>,
    pub lists: Vec<ListFlat>,
}

impl IvfIndex {
    pub fn build(metric: Metric, dim: usize, data: &[(u64, Vec<f32>)], params: IvfParams) -> Self {
        assert!(params.nlist > 0 && params.nlist <= data.len());
        for (_, v) in data { assert_eq!(v.len(), dim); }

        let centers = kmeans_seeded(
            &data.iter().map(|(_, v)| v.clone()).collect::<Vec<_>>(),
            params.nlist, metric, params.seed, 50);

        let mut lists: Vec<ListFlat> = (0..params.nlist)
            .map(|_| ListFlat { ids: Vec::new(), vecs: Vec::new(), tomb: Vec::new(), tags: None })
            .collect();

        // assign by nearest centroid (id tiebreak)
        for (id, v) in data.iter() {
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
            list.vecs.extend_from_slice(v);
            list.tomb.push(0);
        }

        // stable id order per list
        for list in lists.iter_mut() {
            let mut idx: Vec<usize> = (0..list.ids.len()).collect();
            idx.sort_by_key(|&i| list.ids[i]);
            let mut new_ids = Vec::with_capacity(list.ids.len());
            let mut new_vecs = Vec::with_capacity(list.vecs.len());
            let mut new_tomb = Vec::with_capacity(list.tomb.len());
            for i in idx {
                new_ids.push(list.ids[i]);
                new_tomb.push(list.tomb[i]);
                new_vecs.extend_from_slice(list.row(dim, i));
            }
            list.ids = new_ids; list.vecs = new_vecs; list.tomb = new_tomb;
            if let Some(tags) = &list.tags {
                let mut nt = Vec::with_capacity(tags.len());
                for id in &list.ids {
                    let pos = list.ids.binary_search(id).unwrap();
                    nt.push(tags[pos]);
                }
                list.tags = Some(nt);
            }
        }

        let header = IndexHeader::new(dim, metric, Seeds { data: 0, queries: 0, kmeans: params.seed, hnsw: None }, params, false);
        Self { header, params, centroids: centers, lists }
    }

    pub fn delete(&mut self, id: u64) -> bool {
        for list in &mut self.lists {
            if let Ok(pos) = list.ids.binary_search(&id) { list.tomb[pos] = 1; return true; }
        }
        false
    }

    pub fn upsert(&mut self, id: u64, v: &[f32], tag: Option<u64>) {
        assert_eq!(v.len(), self.header.dim);
        // nearest centroid by score
        let mut best = 0usize; let mut bests = f32::NEG_INFINITY;
        for (i, c) in self.centroids.iter().enumerate() {
            let s = match self.header.metric {
                Metric::L2 => -metric::l2_distance(v, c),
                Metric::Cosine => metric::cosine_sim(v, c),
            };
            if s > bests { bests = s; best = i; }
        }
        let list = &mut self.lists[best];
        match list.ids.binary_search(&id) {
            Ok(pos) => {
                let s = pos * self.header.dim;
                for d in 0..self.header.dim { list.vecs[s + d] = v[d]; }
                list.tomb[pos] = 0;
                if let Some(mask) = tag {
                    if let Some(tags) = &mut list.tags { tags[pos] = mask; }
                    else { let mut t = vec![0u64; list.ids.len()]; t[pos] = mask; list.tags = Some(t); }
                }
            }
            Err(pos) => {
                list.ids.insert(pos, id);
                list.tomb.insert(pos, 0);
                let s = pos * self.header.dim;
                list.vecs.splice(s..s, v.iter().cloned());
                if let Some(mask) = tag {
                    if let Some(tags) = &mut list.tags { tags.insert(pos, mask); }
                    else { let mut t = vec![0u64; list.ids.len()-1]; t.insert(pos, mask); list.tags = Some(t); }
                } else if let Some(tags) = &mut list.tags { tags.insert(pos, 0); }
            }
        }
    }

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
                    new_vecs.extend_from_slice(list.row(dim, i));
                    if let Some(tags) = &list.tags { new_tags.as_mut().unwrap().push(tags[i]); }
                }
            }
            list.ids = new_ids; list.vecs = new_vecs; list.tomb = new_tomb; list.tags = new_tags;
        }
    }

    pub fn search_with_filter(&self, q: &[f32], k: usize, required_tag: Option<u64>) -> Vec<Hit> {
        assert_eq!(q.len(), self.header.dim);
        let dim = self.header.dim;
        let nprobe = self.params.nprobe.min(self.centroids.len());
        let refine = self.params.refine.max(k);

        // probe lists
        let mut cands: Vec<(usize, f32)> = Vec::with_capacity(self.centroids.len());
        for (i, c) in self.centroids.iter().enumerate() {
            let s = match self.header.metric { Metric::L2 => -metric::l2_distance(q, c), Metric::Cosine => metric::cosine_sim(q, c) };
            cands.push((i, s));
        }
        cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let probe: Vec<usize> = cands.into_iter().take(nprobe).map(|(i, _)| i).collect();

        // gather
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
                let s = metric::score(self.header.metric, q, list.row(dim, i));
                scored.push(Hit { id: list.ids[i], score: s });
            }
        }
        if scored.is_empty() { return Vec::new(); }
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap().then(a.id.cmp(&b.id)));
        if scored.len() > refine { scored.truncate(refine); }

        // refine
        let mut refine_flat = FlatIndex::new(dim, self.header.metric);
        for h in &scored {
            for list in &self.lists {
                if let Ok(pos) = list.ids.binary_search(&h.id) {
                    if list.tomb[pos] == 0 { refine_flat.add(h.id, list.row(dim, pos)); }
                    break;
                }
            }
        }
        refine_flat.search(q, k)
    }

    #[inline]
    pub fn search(&self, q: &[f32], k: usize) -> Vec<Hit> { self.search_with_filter(q, k, None) }

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
}
