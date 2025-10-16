// src/opq.rs
// Optimized Product Quantization helpers:
// - OPQ-P: variance-balancing permutation across subspaces (deterministic).
// - Optional PCA rotation via a few Jacobi sweeps (deterministic).
//
// Notes:
// * apply(): y = P_after * (R * x). If R is None -> identity; if P is None -> no permutation.
// * train_perm(dim, m, residuals): computes per-dimension variance on residuals and assigns
//   dims to m subspaces to balance total variance per subspace (bin-packing style).
// * train_pca(dim, m, residuals, sweeps): builds a rotation R from covariance via Jacobi.
//   You can combine with a permutation if you want (see train_pca_then_perm).
//
// All math uses f64 for accumulation to keep training deterministic and stable.

#[derive(Clone, Debug)]
pub struct Opq {
    pub dim: usize,
    pub m: usize,
    // Optional rotation R in row-major (dim x dim). If None, R = I.
    r: Option<Vec<f32>>,
    // Optional permutation P: output_index -> input_index mapping.
    // If None, P = identity. We store mapping such that y[i] = x_perm[i] = x[P[i]] after rotation.
    perm: Option<Vec<usize>>,
}

impl Opq {
    /// Identity transform (no rotation, no permutation).
    pub fn identity(dim: usize, m: usize) -> Self {
        Self { dim, m, r: None, perm: None }
    }

    pub fn is_identity(&self) -> bool { self.r.is_none() && self.perm.is_none() }

    /// Set an explicit rotation matrix (dim x dim, row-major).
    pub fn with_matrix(mut self, r: Vec<f32>) -> Self {
        assert_eq!(r.len(), self.dim * self.dim); self.r = Some(r); self
    }

    /// Set an explicit permutation mapping (len = dim), y[i] = x[perm[i]] (after rotation).
    pub fn with_perm(mut self, perm: Vec<usize>) -> Self {
        assert_eq!(perm.len(), self.dim);
        let mut seen = vec![false; self.dim];
        for &p in &perm { assert!(p < self.dim, "perm index out of range"); assert!(!seen[p], "perm has duplicates"); seen[p] = true; }
        self.perm = Some(perm); self
    }

    /// Apply y = P * (R * x). If R is None -> identity; if P is None -> identity.
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.dim);
        let y0: Vec<f32> = if let Some(rm) = &self.r {
            let d = self.dim; let mut out = vec![0.0f32; d];
            for i in 0..d { let row = &rm[i*d..(i+1)*d]; let mut acc = 0.0f32; for j in 0..d { acc += row[j] * x[j]; } out[i] = acc; }
            out
        } else { x.to_vec() };
        if let Some(p) = &self.perm { let mut out = vec![0.0f32; self.dim]; for i in 0..self.dim { out[i] = y0[p[i]]; } out } else { y0 }
    }

    /// In-place variant to avoid reallocs in hot paths.
    pub fn apply_inplace(&self, x: &mut [f32]) {
        debug_assert_eq!(x.len(), self.dim);
        if self.r.is_none() && self.perm.is_none() { return; }
        let d = self.dim;
        let mut buf = vec![0.0f32; d];
        // R * x -> buf
        if let Some(rm) = &self.r {
            for i in 0..d { let row = &rm[i*d..(i+1)*d]; let mut acc = 0.0f32; for j in 0..d { acc += row[j] * x[j]; } buf[i] = acc; }
        } else {
            buf.copy_from_slice(x);
        }
        // P * buf -> x
        if let Some(p) = &self.perm { for i in 0..d { x[i] = buf[p[i]]; } } else { x.copy_from_slice(&buf); }
    }

    /// Fingerprint state: emit a few 64-bit words representing rotation/permutation.
    pub fn fingerprint_bits(&self) -> Vec<u64> {
        fn h64(mut h: u64, x: u64) -> u64 { h ^= x; h = h.wrapping_mul(0x100000001b3); h }
        let mut out = Vec::with_capacity(4);
        let mut a = 0xcbf29ce484222325u64; a = h64(a, self.dim as u64); a = h64(a, self.m as u64);
        a = h64(a, if self.r.is_some() { 1 } else { 0 }); a = h64(a, if self.perm.is_some() { 1 } else { 0 }); out.push(a);
        let mut p = 0xcbf29ce484222325u64; if let Some(perm) = &self.perm { for &ix in perm { p = h64(p, ix as u64); } } out.push(p);
        let mut r = 0xcbf29ce484222325u64; if let Some(mat) = &self.r {
            let d = self.dim; let step = (d / 7).max(1); let mut diag_e = 0.0f64;
            for i in (0..d).step_by(step) { for j in (0..d).step_by(step) { r = h64(r, mat[i*d + j].to_bits() as u64); }
                let v = mat[i*d + i] as f64; diag_e += v*v; }
            r = h64(r, diag_e.to_bits());
        } out.push(r);
        out
    }

    // ---------- Training APIs ----------

    /// Permutation-only OPQ: balance per-sub variance across m blocks (block = dim/m).
    pub fn train_perm(dim: usize, m: usize, residuals: &[Vec<f32>]) -> Self {
        assert!(dim > 0 && m > 0 && dim % m == 0 && !residuals.is_empty());
        for v in residuals { assert_eq!(v.len(), dim); }
        let n = residuals.len() as f64; let block = dim / m;
        let mut mean = vec![0.0f64; dim]; let mut m2 = vec![0.0f64; dim];
        for v in residuals { for d in 0..dim { let x = v[d] as f64; mean[d]+=x; m2[d]+=x*x; } }
        for d in 0..dim { mean[d] /= n; }
        let mut var: Vec<(usize, f64)> = (0..dim).map(|d| { let ex2 = m2[d]/n; let mu = mean[d]; let v = (ex2 - mu*mu).max(0.0); (d,v) }).collect();
        var.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap().then_with(|| a.0.cmp(&b.0)));
        let mut group_sum = vec![0.0f64; m]; let mut group_dims: Vec<Vec<usize>> = (0..m).map(|_| Vec::with_capacity(block)).collect();
        for &(dix, v) in &var { let mut best_g=0usize; let mut best_s=f64::INFINITY; for g in 0..m { if group_dims[g].len()<block { let s=group_sum[g]; if s<best_s { best_s=s; best_g=g; } } }
            group_dims[best_g].push(dix); group_sum[best_g]+=v; }
        let mut perm = Vec::with_capacity(dim); for g in 0..m { perm.extend(group_dims[g].iter().copied()); }
        Self { dim, m, r: None, perm: Some(perm) }
    }

    /// PCA rotation via deterministic Jacobi sweeps. No permutation.
    pub fn train_pca(dim: usize, m: usize, residuals: &[Vec<f32>], sweeps: usize) -> Self {
        assert!(dim > 0 && m > 0 && dim % m == 0 && !residuals.is_empty());
        for v in residuals { assert_eq!(v.len(), dim); }
        let n = residuals.len() as f64;
        let mut mean = vec![0.0f64; dim]; for v in residuals { for d in 0..dim { mean[d]+=v[d] as f64; } }
        for d in 0..dim { mean[d] /= n; }
        let mut c = vec![0.0f64; dim*dim];
        for v in residuals {
            for i in 0..dim { let xi = v[i] as f64 - mean[i]; let row = i*dim; for j in i..dim { let xj = v[j] as f64 - mean[j]; c[row+j] += xi*xj; } }
        }
        let inv_n = 1.0/n; for i in 0..dim { let row=i*dim; for j in i..dim { c[row+j]*=inv_n; c[j*dim+i]=c[row+j]; } }
        let (u, _lambda) = jacobi_eigen_symmetric(&c, dim, sweeps);
        let mut r = vec![0.0f32; dim*dim]; for i in 0..dim { for j in 0..dim { r[i*dim + j] = u[j*dim + i] as f32; } }
        Self { dim, m, r: Some(r), perm: None }
    }

    /// PCA rotation followed by permutation in the rotated space.
    pub fn train_pca_then_perm(dim: usize, m: usize, residuals: &[Vec<f32>], sweeps: usize) -> Self {
        let opq_r = Self::train_pca(dim, m, residuals, sweeps);
        let rotated: Vec<Vec<f32>> = residuals.iter().map(|r| opq_r.apply(r)).collect();
        let mut opq = Self::train_perm(dim, m, &rotated);
        opq.r = opq_r.r; opq
    }
}

// =============== internal helpers ===============
fn jacobi_eigen_symmetric(a_in: &[f64], dim: usize, sweeps: usize) -> (Vec<f64>, Vec<f64>) {
    let n = dim; let mut a = a_in.to_vec(); let mut u = vec![0.0f64; n*n]; let mut lambda = vec![0.0f64; n];
    for i in 0..n { u[i*n + i] = 1.0; }
    for _ in 0..sweeps {
        for p in 0..n { for q in (p+1)..n {
            let apq = a[p*n + q]; if apq.abs() < 1e-12 { continue; }
            let app = a[p*n + p]; let aqq = a[q*n + q];
            let tau = (aqq - app) / (2.0 * apq);
            let t = if tau >= 0.0 { 1.0 / (tau + (1.0 + tau*tau).sqrt()) } else { -1.0 / (-tau + (1.0 + tau*tau).sqrt()) };
            let c = 1.0 / (1.0 + t*t).sqrt(); let s = t * c;
            for k in 0..n { let aik = a[p*n + k]; let aqk = a[q*n + k]; a[p*n + k] = c*aik - s*aqk; a[q*n + k] = s*aik + c*aqk; }
            for k in 0..n { let akp = a[k*n + p]; let akq = a[k*n + q]; a[k*n + p] = c*akp - s*akq; a[k*n + q] = s*akp + c*akq; }
            for k in 0..n { let ukp = u[k*n + p]; let ukq = u[k*n + q]; u[k*n + p] = c*ukp - s*ukq; u[k*n + q] = s*ukp + c*ukq; }
        }}
    }
    for i in 0..n { lambda[i] = a[i*n + i]; }
    (u, lambda)
}
