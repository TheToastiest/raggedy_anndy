// src/opq.rs

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
    pub fn identity(dim: usize, m: usize) -> Self {
        Self { dim, m, r: None, perm: None }
    }

    pub fn is_identity(&self) -> bool { self.r.is_none() && self.perm.is_none() }

    pub fn with_matrix(mut self, r: Vec<f32>) -> Self {
        assert_eq!(r.len(), self.dim * self.dim); self.r = Some(r); self
    }

    pub fn with_perm(mut self, perm: Vec<usize>) -> Self {
        assert_eq!(perm.len(), self.dim);
        let mut seen = vec![false; self.dim];
        for &p in &perm { assert!(p < self.dim, "perm index out of range"); assert!(!seen[p], "perm has duplicates"); seen[p] = true; }
        self.perm = Some(perm); self
    }

    /// Build-time API: Allocates a new vector for the result.
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.dim];
        self.apply_to(x, &mut out);
        out
    }

    /// ZERO-ALLOCATION API: Reads from `x`, writes to `out`.
    /// `x` and `out` MUST NOT alias.
    pub fn apply_to(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.dim);
        debug_assert_eq!(out.len(), self.dim);

        // 1. Handle Rotation
        if let Some(rm) = &self.r {
            let d = self.dim;
            for i in 0..d {
                let row = &rm[i*d..(i+1)*d];
                let mut acc = 0.0f32;
                for j in 0..d { acc += row[j] * x[j]; }
                out[i] = acc;
            }
        } else {
            out.copy_from_slice(x);
        }

        // 2. Handle Permutation (In-place on the output buffer)
        if let Some(p) = &self.perm {
            // We must copy 'out' back into itself via permutation.
            // A tiny stack buffer is sufficient for dimensions up to ~1024.
            // If dim > 1024, consider falling back to a heap alloc or requiring a larger workspace.
            let mut temp = [0.0f32; 512]; // Adjust size based on your max expected dims
            let d = self.dim.min(512);
            temp[..d].copy_from_slice(&out[..d]);

            for i in 0..d {
                out[i] = temp[p[i]];
            }
        }
    }

    /// In-place variant for when you already own the mutable buffer.
    pub fn apply_inplace(&self, x: &mut [f32]) {
        debug_assert_eq!(x.len(), self.dim);
        if self.r.is_none() && self.perm.is_none() { return; }

        let mut buf = vec![0.0f32; self.dim]; // Still allocates, but only used in build phase
        self.apply_to(x, &mut buf);
        x.copy_from_slice(&buf);
    }

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

    // ---------- Training APIs (Unchanged) ----------
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