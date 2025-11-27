// bin/context_test.rs
//
// Synthetic sandbox to probe context + time geometry over ~1 year of data.

use raggedy_anndy::context::{ContextCfg, ContextEncoder, ContextKey, QueryCtx, TimeKey};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::cmp::Ordering;

/// Dimensionality of your semantic vectors.
/// Adjust if raggedy_anndy uses a different size.
const SEM_DIM: usize = 128;

/// One synthetic memory item.
#[derive(Clone, Debug)]
struct Memory {
    id: usize,
    day_idx: i32, // day offset from base_ts
    ctx: ContextKey,
    time: TimeKey,
    semantic: Vec<f32>,
    encoded: Vec<f32>, // semantic+context+time
}

/// Simple cosine, assuming both sides are L2-normalized,
/// which ContextEncoder guarantees for `encoded`.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Deterministic "base semantic" for (domain, topic) using hashing.
fn base_semantic_for(domain: &str, topic: &str) -> Vec<f32> {
    let key = format!("{}::{}", domain, topic);
    let h = fnv_hash(&key);

    // Seed an RNG from the hash and make a random unit vector.
    let mut rng = StdRng::seed_from_u64(h);
    let mut v = vec![0.0f32; SEM_DIM];
    for x in v.iter_mut() {
        *x = rng.gen_range(-1.0..1.0);
    }
    l2_normalize(&mut v);
    v
}

/// Add a bit of deterministic noise to semantic, so we simulate "nearby" content.
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


/// Brute-force nearest neighbors in encoded space.
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
    let base_ts: i64 = 1_700_000_000; // arbitrary epoch anchor
    let days: i32 = 365;

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
                let day_offset = rng.gen_range(0..days);
                let ts_s = base_ts + (day_offset as i64) * 86_400;

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

fn main() {
    let enc = ContextEncoder::new(ContextCfg::default());

    // 3 domains * 3 topics * 2500 ≈ 22,500 memories
    let mems_per_ctx = 2500;
    let memories = build_corpus(&enc, mems_per_ctx);
    println!(
        "Built synthetic corpus: {} memories ({} per (domain,topic))",
        memories.len(),
        mems_per_ctx
    );

    // Pick one context to probe.
    let domain = "rust";
    let topic = "ann";
    let base_sem = base_semantic_for(domain, topic);

    let ctx_key = ContextKey::new(vec![
        format!("domain:{}", domain),
        format!("topic:{}", topic),
    ])
        .with_source("synthetic_corpus")
        .with_session("probe-session");

    let base_ts: i64 = 1_700_000_000;
    let mid_year_day: i32 = 180;
    let mid_year_ts: i64 = base_ts + (mid_year_day as i64) * 86_400;

    // --- Scenario 1: exact match (same sem, ctx, time) ---
    let t_exact = TimeKey::from_unix(mid_year_ts);
    let mem_exact = enc.encode_memory(&base_sem, &ctx_key, &t_exact);
    let qctx_exact = QueryCtx::with_time(ctx_key.clone(), t_exact);
    let q_exact = enc.encode_query(&base_sem, &qctx_exact);
    let cos_exact = cosine(&mem_exact, &q_exact);
    println!("\n[Scenario 1] Same sem + ctx + time");
    println!("cos(mem_exact, q_exact) = {:.6}", cos_exact);

    // --- Scenario 2: mid-year query over full corpus ---
    let qctx_mid = QueryCtx::with_time(ctx_key.clone(), t_exact);
    let q_mid = enc.encode_query(&base_sem, &qctx_mid);
    let top = top_k(&memories, &q_mid, 10);

    println!("\n[Scenario 2] Same sem+ctx, mid-year query over full corpus");
    let mut avg_day = 0.0f32;
    for (m, c) in top.iter() {
        println!(
            "  id={} day={} domain/topic ~ {:?} cos={:.4}",
            m.id, m.day_idx, m.ctx.tags, c
        );
        avg_day += m.day_idx as f32;
    }
    avg_day /= top.len() as f32;
    println!("  mean day_idx in top-10 = {:.2} (query day_idx = {})", avg_day, mid_year_day);

    // --- Scenario 3: early-year anchored query ---
    let early_day: i32 = 10;
    let early_ts: i64 = base_ts + (early_day as i64) * 86_400;
    let t_early = TimeKey::from_unix(early_ts);

    let qctx_early = QueryCtx::with_time(ctx_key.clone(), t_early);
    let q_early = enc.encode_query(&base_sem, &qctx_early);
    let top_early = top_k(&memories, &q_early, 10);

    println!("\n[Scenario 3] Same sem+ctx, early-year anchored query");
    let mut avg_day_early = 0.0f32;
    for (m, c) in top_early.iter() {
        println!(
            "  id={} day={} domain/topic ~ {:?} cos={:.4}",
            m.id, m.day_idx, m.ctx.tags, c
        );
        avg_day_early += m.day_idx as f32;
    }
    avg_day_early /= top_early.len() as f32;
    println!(
        "  mean day_idx in top-10 = {:.2} (query day_idx = {})",
        avg_day_early, early_day
    );

    if avg_day_early < avg_day {
        println!("\n[OK] Time geometry is behaving: early-year query → earlier neighbors.");
    } else {
        println!("\n[WARN] Time geometry looks off: early-year query not earlier than mid-year.");
    }
}