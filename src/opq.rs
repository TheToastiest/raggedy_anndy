// in src/opq.rs
#[derive(Clone, Debug)]
pub struct Opq {
    pub dim: usize,
    pub m: usize,
    perm: Option<Vec<usize>>,   // permutation mapping old->new
    r: Option<Vec<f32>>,        // optional dense rotation (row-major dim x dim)
}

impl Opq {
    pub fn identity(dim: usize, m: usize) -> Self {
        Self { dim, m, perm: None, r: None }
    }
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.dim);
        if let Some(p) = &self.perm {
            let mut y = vec![0.0f32; self.dim];
            for (new_i, &old_i) in p.iter().enumerate() { y[new_i] = x[old_i]; }
            return y;
        }
        if let Some(r) = &self.r {
            let mut y = vec![0.0f32; self.dim];
            for i in 0..self.dim {
                let row = &r[i*self.dim..(i+1)*self.dim];
                let mut acc = 0.0f32;
                for d in 0..self.dim { acc += row[d] * x[d]; }
                y[i] = acc;
            }
            return y;
        }
        x.to_vec()
    }

    /// Train *permutation-only* OPQ on residuals to balance variance per subspace.
    pub fn train_perm(dim: usize, m: usize, residuals: &[Vec<f32>]) -> Self {
        assert!(dim % m == 0);
        let dsub = dim / m;
        // 1) per-dim variance
        let mut mean = vec![0.0f64; dim];
        let n = residuals.len().max(1);
        for v in residuals { for d in 0..dim { mean[d] += v[d] as f64; } }
        for d in 0..dim { mean[d] /= n as f64; }

        let mut var = vec![0.0f64; dim];
        for v in residuals {
            for d in 0..dim { let t = (v[d] as f64) - mean[d]; var[d] += t * t; }
        }
        for d in 0..dim { var[d] /= n as f64; }

        // 2) dims sorted by variance desc
        let mut dims: Vec<usize> = (0..dim).collect();
        dims.sort_by(|&a,&b| var[b].partial_cmp(&var[a]).unwrap());

        // 3) greedy bin-packing into m buckets
        let mut bucket_sum = vec![0.0f64; m];
        let mut buckets: Vec<Vec<usize>> = vec![Vec::with_capacity(dsub); m];
        for d in dims {
            // pick bucket with min current sum (ties -> lower index)
            let mut best = 0usize;
            for b in 1..m {
                if bucket_sum[b] < bucket_sum[best] { best = b; }
            }
            if buckets[best].len() < dsub {
                buckets[best].push(d);
                bucket_sum[best] += var[d];
            } else {
                // fallback if bucket full: put in next with space
                for b in 0..m {
                    if buckets[b].len() < dsub { buckets[b].push(d); break; }
                }
            }
        }

        // 4) build permutation: new index i <- old dim buckets[b][j]
        let mut perm = vec![0usize; dim];
        let mut idx = 0usize;
        for b in 0..m {
            for &old_d in &buckets[b] { perm[idx] = old_d; idx += 1; }
        }
        Self { dim, m, perm: Some(perm), r: None }
    }
    pub fn fingerprint_bits(&self) -> Vec<u64> {
        let mut out = Vec::with_capacity(2 + self.dim);
        out.push(self.dim as u64);
        out.push(self.m as u64);
        if let Some(p) = &self.perm {
            for &u in p { out.push(u as u64); }
        } else if let Some(r) = &self.r {
            for &f in r { out.push(f.to_bits() as u64); }
        }
        out
    }
}
