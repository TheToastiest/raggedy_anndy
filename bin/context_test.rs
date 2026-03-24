use raggedy_anndy::context::{ContextCfg, ContextEncoder, ContextKey, QueryCtx, TimeKey};
use raggedy_anndy::metric::{self, Metric};
use raggedy_anndy::flat::FlatIndex;
use raggedy_anndy::ivf::{IvfIndex, IvfParams};
use raggedy_anndy::ivfpq::{IvfPqIndex, IvfPqParams, OpqMode};

use rand::{rngs::StdRng, Rng, SeedableRng};
use std::cmp::Ordering;
use std::time::Instant;

/// Dimensionality of your semantic vectors.
const SEM_DIM: usize = 128;

/// Synthetic time span configuration.
const BASE_TS: i64 = 1_700_000_000;
const DAYS: i32 = 730;

/// One synthetic memory item.
#[derive(Clone, Debug)]
struct Memory {
    id: usize,
    day_idx: i32,
    ctx: ContextKey,
    time: TimeKey,
    semantic: Vec<f32>,
    encoded: Vec<f32>,
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn base_semantic_for(domain: &str, topic: &str) -> Vec<f32> {
    let key = format!("{}::{}", domain, topic);
    let h = fnv_hash(&key);
    let mut rng = StdRng::seed_from_u64(h);
    let mut v = vec![0.0f32; SEM_DIM];
    for x in v.iter_mut() {
        *x = rng.gen_range(-1.0..1.0);
    }
    l2_normalize(&mut v);
    v
}

fn jitter_semantic(base: &[f32], rng: &mut StdRng, noise_scale: f32) -> Vec<f32> {
    let mut v = base.to_vec();
    for x in v.iter_mut() {
        *x += rng.gen_range(-noise_scale..noise_scale);
    }
    l2_normalize(&mut v);
    v
}

fn fnv_hash(s: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001b3;
    let mut hash = FNV_OFFSET;
    for b in s.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn l2_normalize(v: &mut [f32]) {
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

fn top_k<'a>(mems: &'a [Memory], q_vec: &[f32], k: usize) -> Vec<(&'a Memory, f32)> {
    let mut scored: Vec<(&Memory, f32)> = mems
        .iter()
        .map(|m| (m, cosine(&m.encoded, q_vec)))
        .collect();

    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.id.cmp(&b.0.id))
    });

    scored.truncate(k);
    scored
}

fn build_corpus(enc: &ContextEncoder, mems_per_ctx: usize) -> Vec<Memory> {
    let domains = ["rust", "sql", "python"];
    let topics = ["ann", "rag", "router"];
    let mut rng = StdRng::seed_from_u64(42);
    let mut memories = Vec::new();
    let mut id = 0usize;

    for d in domains.iter() {
        for t in topics.iter() {
            let base_sem = base_semantic_for(d, t);
            let ctx_key = ContextKey::new(vec![
                format!("domain:{}", d),
                format!("topic:{}", t),
            ])
                .with_source("synthetic_corpus")
                .with_session(format!("sess_{}_{}", d, t));

            for _ in 0..mems_per_ctx {
                let day_offset = rng.gen_range(0..DAYS);
                let ts_s = BASE_TS + (day_offset as i64) * 86_400;
                let time_key = TimeKey::from_unix(ts_s);
                let semantic = jitter_semantic(&base_sem, &mut rng, 0.05);
                let encoded = enc.encode_memory(&semantic, &ctx_key, &time_key);

                memories.push(Memory {
                    id,
                    day_idx: day_offset,
                    ctx: ctx_key.clone(),
                    time: time_key,
                    semantic,
                    encoded,
                });
                id += 1;
            }
        }
    }
    memories
}

fn print_latency_stats(label: &str, samples: &mut [u128]) {
    if samples.is_empty() { return; }
    samples.sort_unstable();
    let len = samples.len();
    let idx_for = |p: f64| ((p * (len as f64 - 1.0)).round() as usize).min(len - 1);
    println!("[latency] {} (ns): p50={} p95={} p99={}", label, samples[idx_for(0.50)], samples[idx_for(0.95)], samples[idx_for(0.99)]);
}

fn ann_error_suite(memories: &[Memory], enc: &ContextEncoder, flat: &FlatIndex, ivf: &IvfIndex, ivfpq: &IvfPqIndex) {
    let domains = ["rust", "sql", "python"];
    let topics = ["ann", "rag", "router"];
    let mut id_to_idx = vec![usize::MAX; memories.len()];
    for (idx, m) in memories.iter().enumerate() { id_to_idx[m.id] = idx; }

    let mut rng = StdRng::seed_from_u64(999);
    let trials = 1_000;
    let (mut flat_fail, mut flat_miss) = (0, 0);
    let (mut ivf_fail, mut ivf_miss) = (0, 0);
    let (mut pq_fail, mut pq_miss) = (0, 0);

    for _ in 0..trials {
        let d = domains[rng.gen_range(0..domains.len())];
        let t = topics[rng.gen_range(0..topics.len())];
        let base_sem = base_semantic_for(d, t);
        let ts_s = BASE_TS + (rng.gen_range(0..DAYS) as i64) * 86_400;
        let t_key = TimeKey::from_unix(ts_s);
        let ctx_key = ContextKey::new(vec![format!("domain:{}", d), format!("topic:{}", t)])
            .with_source("synthetic_corpus").with_session(format!("sess_{}_{}", d, t));

        let qctx = QueryCtx::with_time(ctx_key, t_key);
        let q_vec = enc.encode_query(&base_sem, &qctx);
        let gt = top_k(memories, &q_vec, 1);
        if gt.is_empty() { continue; }
        let gt_id = gt[0].0.id as u64;

        let expected_domain = format!("domain:{}", d);
        let expected_topic = format!("topic:{}", t);

        // Score logic...
        for (idx_type, hits) in [("flat", flat.search(&q_vec, 1)), ("ivf", ivf.search(&q_vec, 1)), ("pq", ivfpq.search(&q_vec, 1))] {
            let (fail, miss) = match idx_type {
                "flat" => (&mut flat_fail, &mut flat_miss),
                "ivf" => (&mut ivf_fail, &mut ivf_miss),
                _ => (&mut pq_fail, &mut pq_miss),
            };
            if hits.is_empty() { *fail += 1; } else {
                let m = &memories[id_to_idx[hits[0].id as usize]];
                if !m.ctx.tags.contains(&expected_domain) || !m.ctx.tags.contains(&expected_topic) { *fail += 1; }
                if hits[0].id != gt_id { *miss += 1; }
            }
        }
    }
    println!("[ANN error] FlatIndex: ctx_err_rate={:.4} gt_mismatch_rate={:.4}", flat_fail as f32/trials as f32, flat_miss as f32/trials as f32);
    println!("[ANN error] IVF-Flat:  ctx_err_rate={:.4} gt_mismatch_rate={:.4}", ivf_fail as f32/trials as f32, ivf_miss as f32/trials as f32);
    println!("[ANN error] IVF-PQ:    ctx_err_rate={:.4} gt_mismatch_rate={:.4}", pq_fail as f32/trials as f32, pq_miss as f32/trials as f32);
}

fn time_geometry_suite(memories: &[Memory], enc: &ContextEncoder) {
    let domains = ["rust", "sql", "python"];
    let topics = ["ann", "rag", "router"];
    let (early_ts, mid_ts) = (BASE_TS + 10 * 86_400, BASE_TS + 180 * 86_400);
    let (mut trials, mut failures) = (0, 0);

    for d in domains.iter() {
        for t in topics.iter() {
            let base_sem = base_semantic_for(d, t);
            let ctx = ContextKey::new(vec![format!("domain:{}", d), format!("topic:{}", t)]);

            let mut avg_day = |ts| {
                let qctx = QueryCtx::with_time(ctx.clone(), TimeKey::from_unix(ts));
                let q = enc.encode_query(&base_sem, &qctx);
                let top = top_k(memories, &q, 10);
                top.iter().map(|(m, _)| m.day_idx as f32).sum::<f32>() / top.len().max(1) as f32
            };

            trials += 1;
            if !(avg_day(early_ts) < avg_day(mid_ts)) { failures += 1; }
        }
    }
    println!("[error] time-geometry (oracle): trials={} failures={} error_rate={:.4}", trials, failures, failures as f32/trials as f32);
}

fn main() {
    let enc = ContextEncoder::new(ContextCfg::default());
    let memories = build_corpus(&enc, 10000);
    let dim = memories[0].encoded.len();

    // UPDATED: Standardizing data for temporal signatures (ID, Vector, Timestamp)
    let mut data_temporal: Vec<(u64, Vec<f32>, u64)> = memories.iter()
        .map(|m| (m.id as u64, m.encoded.clone(), (BASE_TS + (m.day_idx as i64 * 86400)) as u64))
        .collect();

    let mut data_flat: Vec<(u64, Vec<f32>)> = memories.iter()
        .map(|m| (m.id as u64, m.encoded.clone()))
        .collect();

    let mut flat_index = FlatIndex::new(dim, Metric::Cosine);
    for (id, v) in &data_flat { flat_index.add(*id, v); }

    let nlist = 256.min(data_temporal.len());
    let ivf_index = IvfIndex::build(Metric::Cosine, dim, &data_flat, IvfParams { nlist, nprobe: 8, refine: 32, seed: 0xDEADBEEF });

    let ivfpq_index = IvfPqIndex::build(Metric::Cosine, dim, &data_temporal, IvfPqParams {
        nlist, nprobe: 16, refine: 64, seed: 0xCAFEBABE, m: 21, nbits: 8, iters: 20,
        use_opq: true, opq_mode: OpqMode::PcaPerm, opq_sweeps: 4, store_vecs: true,
    });

    println!("Context Test: Corpus Scaled to {} Memories", memories.len());
    ann_error_suite(&memories, &enc, &flat_index, &ivf_index, &ivfpq_index);
    time_geometry_suite(&memories, &enc);
}