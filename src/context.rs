// src/context.rs
//! Context + time geometry for RAG / ANN.
//!
//! This replaces the old "entropy" concept. Nothing here mutates based on usage;
//! vectors are purely a function of (semantic, context, absolute time).

use std::hash::{Hash, Hasher};

/// Dimensionality of the context subspace.
pub const CTX_DIM: usize = 32;
/// Dimensionality of the time subspace.
pub const TIME_DIM: usize = 8;

/// Symbolic/contextual tags for a memory or query.
#[derive(Clone, Debug, Default)]
pub struct ContextKey {
    /// Free-form tags like ["domain:rust", "topic:router", "source:log"].
    pub tags: Vec<String>,

    /// Optional: file, stream, or corpus identity.
    pub source: Option<String>,
    /// Optional: session / conversation identity.
    pub session: Option<String>,
}

impl ContextKey {
    pub fn new<T: Into<String>>(tags: Vec<T>) -> Self {
        Self {
            tags: tags.into_iter().map(Into::into).collect(),
            source: None,
            session: None,
        }
    }

    pub fn with_source<T: Into<String>>(mut self, src: T) -> Self {
        self.source = Some(src.into());
        self
    }

    pub fn with_session<T: Into<String>>(mut self, sess: T) -> Self {
        self.session = Some(sess.into());
        self
    }
}

/// Absolute time for a memory or query, stored as unix seconds.
#[derive(Clone, Copy, Debug)]
pub struct TimeKey {
    pub ts_s: i64,
}

impl TimeKey {
    pub fn from_unix(ts_s: i64) -> Self {
        Self { ts_s }
    }
}


/// How a query anchors itself in time.
/// Context bundle for a query.
/// Time is already resolved by the caller (URIEL or your host app).
#[derive(Clone, Debug)]
pub struct QueryCtx {
    pub ctx_key: ContextKey,
    pub time: TimeKey,
}

impl QueryCtx {
    /// Build a query context anchored at a specific unix timestamp.
    pub fn at(ctx_key: ContextKey, ts_s: i64) -> Self {
        Self {
            ctx_key,
            time: TimeKey::from_unix(ts_s),
        }
    }

    /// If you already have a TimeKey, use this.
    pub fn with_time(ctx_key: ContextKey, time: TimeKey) -> Self {
        Self { ctx_key, time }
    }
}


/// Tunable scales for how much each subspace (semantic / context / time)
/// bends the final geometry.
#[derive(Clone, Debug)]
pub struct ContextCfg {
    /// Scale factor applied to the base semantic vector.
    pub sem_scale: f32,
    /// Scale factor applied to the context subspace.
    pub ctx_scale: f32,
    /// Scale factor applied to the time subspace.
    pub time_scale: f32,
}

impl Default for ContextCfg {
    fn default() -> Self {
        Self {
            sem_scale: 0.95,
            ctx_scale: 0.55,
            time_scale: 0.6,
        }
    }
}

/// Encodes (semantic, context, time) into a single ANN-ready vector.
///
/// IMPORTANT: This does *not* update with usage. Geometry is fixed at write time,
/// and queries use the same mapping so past-context retrieval is symmetric.
#[derive(Clone, Debug)]
pub struct ContextEncoder {
    cfg: ContextCfg,
}

impl ContextEncoder {
    pub fn new(cfg: ContextCfg) -> Self {
        Self { cfg }
    }

    pub fn cfg(&self) -> &ContextCfg {
        &self.cfg
    }

    /// For *memory write*: semantic + context + absolute time.
    pub fn encode_memory(
        &self,
        semantic: &[f32],
        ctx: &ContextKey,
        time: &TimeKey,
    ) -> Vec<f32> {
        let ctx_vec = encode_ctx(tags_for(ctx));
        let time_vec = encode_time(time.ts_s);
        combine(&self.cfg, semantic, &ctx_vec, &time_vec)
    }

    /// For *query*: same mapping; time can be "now" or an explicit anchor.
    pub fn encode_query(
        &self,
        semantic_q: &[f32],
        qctx: &QueryCtx,
    ) -> Vec<f32> {
        let ctx_vec = encode_ctx(tags_for(&qctx.ctx_key));
        let time_vec = encode_time(qctx.time.ts_s);
        combine(&self.cfg, semantic_q, &ctx_vec, &time_vec)
    }

}

// --- internal helpers -------------------------------------------------------

fn tags_for(ctx: &ContextKey) -> Vec<String> {
    let mut tags = ctx.tags.clone();
    if let Some(src) = &ctx.source {
        tags.push(format!("source:{}", src));
    }
    if let Some(sess) = &ctx.session {
        tags.push(format!("session:{}", sess));
    }
    tags
}

/// Very cheap hashed bag-of-tags -> fixed-size context vector.
fn encode_ctx(tags: Vec<String>) -> [f32; CTX_DIM] {
    let mut out = [0.0f32; CTX_DIM];
    if tags.is_empty() {
        return out;
    }

    for t in tags.iter() {
        let h = hash_str(t);
        let idx = (h % CTX_DIM as u64) as usize;
        out[idx] += 1.0;
    }

    l2_normalize_array(&mut out);
    out
}

/// Absolute-time positional-style encoding.
/// Uses days since epoch with several periodic frequencies.
fn encode_time(ts_s: i64) -> [f32; TIME_DIM] {
    let t_days = ts_s as f32 / 86_400.0;

    // Frequencies: ~1 day, 1 week, 1 month, 1 year (in days^-1).
    let freqs = [1.0, 1.0 / 7.0, 1.0 / 30.0, 1.0 / 365.0];

    let mut out = [0.0f32; TIME_DIM];
    for (i, f) in freqs.iter().enumerate() {
        let v = t_days * f;
        out[2 * i] = v.sin();
        out[2 * i + 1] = v.cos();
    }

    // Make it unit-norm so `time_scale` in ContextCfg is meaningful.
    l2_normalize_array(&mut out);
    out
}


/// Concatenate subspaces, apply scales, then L2-normalize for ANN.
///
/// Layout: [ semantic * sem_scale | ctx * ctx_scale | time * time_scale ]
fn combine(
    cfg: &ContextCfg,
    semantic: &[f32],
    ctx: &[f32; CTX_DIM],
    time: &[f32; TIME_DIM],
) -> Vec<f32> {
    let mut out = Vec::with_capacity(semantic.len() + CTX_DIM + TIME_DIM);

    out.extend(semantic.iter().map(|x| x * cfg.sem_scale));
    out.extend(ctx.iter().map(|x| x * cfg.ctx_scale));
    out.extend(time.iter().map(|x| x * cfg.time_scale));

    l2_normalize_slice(&mut out);
    out
}

fn hash_str(s: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME:  u64 = 0x00000100000001b3;

    let mut hash = FNV_OFFSET;
    for b in s.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}


fn l2_normalize_slice(v: &mut [f32]) {
    let mut sum_sq = 0.0f32;
    for x in v.iter() {
        sum_sq += x * x;
    }
    if sum_sq > 0.0 {
        let inv = 1.0 / sum_sq.sqrt();
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

fn l2_normalize_array<const N: usize>(v: &mut [f32; N]) {
    l2_normalize_slice(&mut v[..]);
}

// --- optional sanity tests --------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_and_time_vectors_are_normalized() {
        let ctx = encode_ctx(vec!["domain:rust".into(), "topic:rag".into()]);
        let time = encode_time(1_700_000_000);

        let ctx_norm: f32 = ctx.iter().map(|x| x * x).sum::<f32>().sqrt();
        let time_norm: f32 = time.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!((ctx_norm - 1.0).abs() < 1e-3 || ctx_norm == 0.0);
        assert!((time_norm - 1.0).abs() < 1e-3 || time_norm == 0.0);
    }

    #[test]
    fn combined_vector_is_normalized() {
        let sem = vec![0.1f32; 128];
        let ctx_key = ContextKey::new(vec!["domain:rust"]);
        let t = TimeKey::from_unix(1_700_000_000);
        let enc = ContextEncoder::new(ContextCfg::default());

        let v = enc.encode_memory(&sem, &ctx_key, &t);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3);
    }
}
