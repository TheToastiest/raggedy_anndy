// src/opq.rs
// OPQ scaffold: identity transform for now (deterministic, no-op).
// API is designed so we can later train an orthonormal rotation.

#[derive(Clone, Debug)]
pub struct Opq {
    pub dim: usize,
    pub m: usize,
    // Row-major dim x dim matrix. For identity we keep None to avoid costs.
    r: Option<Vec<f32>>,
}

impl Opq {
    pub fn identity(dim: usize, m: usize) -> Self { Self { dim, m, r: None } }

    pub fn is_identity(&self) -> bool { self.r.is_none() }

    /// Apply R * x. If identity, just clone input to avoid overhead.
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.dim);
        if self.r.is_none() { return x.to_vec(); }
        let r = self.r.as_ref().unwrap();
        let mut y = vec![0.0f32; self.dim];
        // y = R x, R row-major
        for i in 0..self.dim {
            let mut acc = 0.0f32;
            let row = &r[i*self.dim..(i+1)*self.dim];
            for d in 0..self.dim { acc += row[d] * x[d]; }
            y[i] = acc;
        }
        y
    }

    /// Set an explicit matrix (must be dim x dim). Caller ensures near-orthonormal.
    pub fn with_matrix(mut self, r: Vec<f32>) -> Self {
        assert_eq!(r.len(), self.dim * self.dim);
        self.r = Some(r); self
    }
}
