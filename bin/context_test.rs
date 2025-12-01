// bin/context_test.rs
//
// Synthetic sandbox to probe context + time geometry over ~1 year of data,
// and to benchmark p50/p95/p99 latencies + ANN error rates
// for FlatIndex, IVF-Flat, and IVF-PQ over ContextEncoder vectors.

use raggedy_anndy::context::{ContextCfg, ContextEncoder, ContextKey, QueryCtx, TimeKey};
use raggedy_anndy::metric::Metric;
use raggedy_anndy::flat::FlatIndex;
use raggedy_anndy::ivf::{IvfIndex, IvfParams};
use raggedy_anndy::ivfpq::{IvfPqIndex, IvfPqParams, OpqMode};

use rand::{rngs::StdRng, Rng, SeedableRng};
use std::cmp::Ordering;
use std::time::Instant;

/// Dimensionality of your semantic vectors.
/// Adjust if raggedy_anndy uses a different size.
const SEM_DIM: usize = 128;

/// Synthetic time span configuration.
const BASE_TS: i64 = 1_700_000_000; // arbitrary epoch anchor
const DAYS: i32 = 730;

/// One synthetic memory item.
#[derive(Clone, Debug)]
struct Memory {
    id: usize,
    day_idx: i32, // day offset from BASE_TS
    ctx: ContextKey,
    time: TimeKey,
    semantic: Vec<f32>,
    encoded: Vec<f32>, // semantic+context+time (L2-normalized)
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

/// Brute-force nearest neighbors in encoded space (oracle).
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

/// Compute p50/p95/p99 from a slice of nanosecond samples.
fn print_latency_stats(label: &str, samples: &mut [u128]) {
    if samples.is_empty() {
        println!("[latency] {}: no samples", label);
        return;
    }

    samples.sort_unstable();
    let len = samples.len();

    let idx_for = |p: f64| -> usize {
        let pos = p * (len as f64 - 1.0);
        let idx = pos.round() as usize;
        idx.min(len - 1)
    };

    let p50 = samples[idx_for(0.50)];
    let p95 = samples[idx_for(0.95)];
    let p99 = samples[idx_for(0.99)];

    println!(
        "[latency] {} (ns): p50={} p95={} p99={}",
        label, p50, p95, p99
    );
}

/// Error + quality suite for ANN vs brute-force.
///
/// - Checks that top-1 stays in the right (domain,topic) cell.
/// - Checks mismatch rate vs brute-force oracle top-1.
fn ann_error_suite(
    memories: &[Memory],
    enc: &ContextEncoder,
    flat: &FlatIndex,
    ivf: &IvfIndex,
    ivfpq: &IvfPqIndex,
) {
    let domains = ["rust", "sql", "python"];
    let topics = ["ann", "rag", "router"];

    // id → index map (ids are 0..N-1)
    let mut id_to_idx = vec![usize::MAX; memories.len()];
    for (idx, m) in memories.iter().enumerate() {
        id_to_idx[m.id] = idx;
    }

    let mut rng = StdRng::seed_from_u64(999);
    let trials = 1_000;

    let mut flat_fail_ctx = 0usize;
    let mut flat_mismatch_gt = 0usize;

    let mut ivf_fail_ctx = 0usize;
    let mut ivf_mismatch_gt = 0usize;

    let mut pq_fail_ctx = 0usize;
    let mut pq_mismatch_gt = 0usize;

    for _ in 0..trials {
        let d = domains[rng.gen_range(0..domains.len())];
        let t = topics[rng.gen_range(0..topics.len())];

        let base_sem = base_semantic_for(d, t);
        let day_offset = rng.gen_range(0..DAYS);
        let ts_s = BASE_TS + (day_offset as i64) * 86_400;
        let t_key = TimeKey::from_unix(ts_s);

        let ctx_key = ContextKey::new(vec![
            format!("domain:{}", d),
            format!("topic:{}", t),
        ])
            .with_source("synthetic_corpus")
            .with_session(format!("sess_{}_{}", d, t));

        let qctx = QueryCtx::with_time(ctx_key, t_key);
        let q_vec = enc.encode_query(&base_sem, &qctx);

        // Ground-truth top-1 via brute-force
        let gt = top_k(memories, &q_vec, 1);
        if gt.is_empty() {
            continue;
        }
        let gt_id = gt[0].0.id as u64;

        // Helper: check result for one index
        let expected_domain = format!("domain:{}", d);
        let expected_topic = format!("topic:{}", t);

        // FlatIndex (exact)
        let flat_hits = flat.search(&q_vec, 1);
        if flat_hits.is_empty() {
            flat_fail_ctx += 1;
        } else {
            let hid = flat_hits[0].id as usize;
            let midx = id_to_idx[hid];
            let m = &memories[midx];
            let tags = &m.ctx.tags;
            let ok_ctx = tags.contains(&expected_domain) && tags.contains(&expected_topic);
            if !ok_ctx {
                flat_fail_ctx += 1;
            }
            if flat_hits[0].id != gt_id {
                flat_mismatch_gt += 1;
            }
        }

        // IVF-Flat
        let ivf_hits = ivf.search(&q_vec, 1);
        if ivf_hits.is_empty() {
            ivf_fail_ctx += 1;
        } else {
            let hid = ivf_hits[0].id as usize;
            let midx = id_to_idx[hid];
            let m = &memories[midx];
            let tags = &m.ctx.tags;
            let ok_ctx = tags.contains(&expected_domain) && tags.contains(&expected_topic);
            if !ok_ctx {
                ivf_fail_ctx += 1;
            }
            if ivf_hits[0].id != gt_id {
                ivf_mismatch_gt += 1;
            }
        }

        // IVF-PQ
        let pq_hits = ivfpq.search(&q_vec, 1);
        if pq_hits.is_empty() {
            pq_fail_ctx += 1;
        } else {
            let hid = pq_hits[0].id as usize;
            let midx = id_to_idx[hid];
            let m = &memories[midx];
            let tags = &m.ctx.tags;
            let ok_ctx = tags.contains(&expected_domain) && tags.contains(&expected_topic);
            if !ok_ctx {
                pq_fail_ctx += 1;
            }
            if pq_hits[0].id != gt_id {
                pq_mismatch_gt += 1;
            }
        }
    }

    println!(
        "[ANN error] FlatIndex: ctx_err_rate={:.4} gt_mismatch_rate={:.4}",
        flat_fail_ctx as f32 / trials as f32,
        flat_mismatch_gt as f32 / trials as f32
    );
    println!(
        "[ANN error] IVF-Flat:  ctx_err_rate={:.4} gt_mismatch_rate={:.4}",
        ivf_fail_ctx as f32 / trials as f32,
        ivf_mismatch_gt as f32 / trials as f32
    );
    println!(
        "[ANN error] IVF-PQ:    ctx_err_rate={:.4} gt_mismatch_rate={:.4}",
        pq_fail_ctx as f32 / trials as f32,
        pq_mismatch_gt as f32 / trials as f32
    );

    // Time-geometry still tested via brute-force oracle; that’s a property of ContextEncoder,
    // not the index. If ANN respects gt reasonably, geometry is preserved up to approximation.
}

/// Time-geometry error rate across all (domain,topic) cells (oracle).
fn time_geometry_suite(memories: &[Memory], enc: &ContextEncoder) {
    let domains = ["rust", "sql", "python"];
    let topics = ["ann", "rag", "router"];

    let early_day: i32 = 10;
    let mid_year_day: i32 = 180;
    let early_ts: i64 = BASE_TS + (early_day as i64) * 86_400;
    let mid_year_ts: i64 = BASE_TS + (mid_year_day as i64) * 86_400;

    let mut time_trials = 0usize;
    let mut time_failures = 0usize;

    for d in domains.iter() {
        for t in topics.iter() {
            let base_sem = base_semantic_for(d, t);
            let ctx_key = ContextKey::new(vec![
                format!("domain:{}", d),
                format!("topic:{}", t),
            ])
                .with_source("synthetic_corpus")
                .with_session(format!("sess_{}_{}", d, t));

            // mid-year query
            let t_mid = TimeKey::from_unix(mid_year_ts);
            let qctx_mid = QueryCtx::with_time(ctx_key.clone(), t_mid);
            let q_mid = enc.encode_query(&base_sem, &qctx_mid);
            let top_mid = top_k(memories, &q_mid, 10);
            let mut avg_mid = 0.0f32;
            for (m, _) in top_mid.iter() {
                avg_mid += m.day_idx as f32;
            }
            avg_mid /= top_mid.len().max(1) as f32;

            // early-year query
            let t_early = TimeKey::from_unix(early_ts);
            let qctx_early = QueryCtx::with_time(ctx_key.clone(), t_early);
            let q_early = enc.encode_query(&base_sem, &qctx_early);
            let top_early = top_k(memories, &q_early, 10);
            let mut avg_early = 0.0f32;
            for (m, _) in top_early.iter() {
                avg_early += m.day_idx as f32;
            }
            avg_early /= top_early.len().max(1) as f32;

            time_trials += 1;
            if !(avg_early < avg_mid) {
                time_failures += 1;
            }
        }
    }

    let time_err_rate = time_failures as f32 / time_trials as f32;
    println!(
        "[error] time-geometry (oracle): trials={} failures={} error_rate={:.4}",
        time_trials, time_failures, time_err_rate
    );
}

fn main() {
    let enc = ContextEncoder::new(ContextCfg::default());

    // 3 domains * 3 topics * 2500 ≈ 22,500 memories
    let mems_per_ctx = 10000;
    let memories = build_corpus(&enc, mems_per_ctx);
    println!(
        "Built synthetic corpus: {} memories ({} per (domain,topic))",
        memories.len(),
        mems_per_ctx
    );

    let dim = memories[0].encoded.len();

    // Training data for ANN: operate in ContextEncoder space.
    let data: Vec<(u64, Vec<f32>)> = memories
        .iter()
        .map(|m| (m.id as u64, m.encoded.clone()))
        .collect();

    // --- Build FlatIndex (exact ANN) ---------------------------------------
    let mut flat_index = FlatIndex::new(dim, Metric::Cosine);
    for (id, v) in &data {
        flat_index.add(*id, v);
    }

    // --- Build IVF-Flat ----------------------------------------------------
    let nlist = 256.min(data.len()); // clamp to N
    let ivf_params = IvfParams {
        nlist,
        nprobe: 8,
        refine: 32,
        seed: 0xDEADBEEF,
    };
    let ivf_index = IvfIndex::build(Metric::Cosine, dim, &data, ivf_params);

    // --- Build IVF-PQ ------------------------------------------------------
    // Encoded dim = 128 (sem) + 32 (ctx) + 8 (time) = 168 → choose m that divides 168.
    let ivfpq_params = IvfPqParams {
        nlist,
        nprobe: 16,
        refine: 64,
        seed: 0xCAFEBABE,
        m: 21,            // 168 / 8 = 21 dims per subvector
        nbits: 8,
        iters: 20,
        use_opq: true,
        opq_mode: OpqMode::PcaPerm,
        opq_sweeps: 4,
        store_vecs: true, // enable exact refine
    };
    let ivfpq_index = IvfPqIndex::build(Metric::Cosine, dim, &data, ivfpq_params);

    // Pick one context to probe for the printed scenarios.
    let domain = "rust";
    let topic = "ann";
    let base_sem = base_semantic_for(domain, topic);

    let ctx_key = ContextKey::new(vec![
        format!("domain:{}", domain),
        format!("topic:{}", topic),
    ])
        .with_source("synthetic_corpus")
        .with_session(format!("sess_{}_{}", domain, topic));

    let mid_year_day: i32 = 180;
    let mid_year_ts: i64 = BASE_TS + (mid_year_day as i64) * 86_400;

    // --- Scenario 1: exact match (same sem, ctx, time) ---
    let t_exact = TimeKey::from_unix(mid_year_ts);
    let mem_exact = enc.encode_memory(&base_sem, &ctx_key, &t_exact);
    let qctx_exact = QueryCtx::with_time(ctx_key.clone(), t_exact);
    let q_exact = enc.encode_query(&base_sem, &qctx_exact);
    let cos_exact = cosine(&mem_exact, &q_exact);
    println!("\n[Scenario 1] Same sem + ctx + time");
    println!("cos(mem_exact, q_exact) = {:.6}", cos_exact);

    // --- Scenario 2: mid-year query over full corpus (oracle) ---
    let qctx_mid = QueryCtx::with_time(ctx_key.clone(), t_exact);
    let q_mid = enc.encode_query(&base_sem, &qctx_mid);
    let top = top_k(&memories, &q_mid, 10);

    println!("\n[Scenario 2] Same sem+ctx, mid-year query over full corpus (oracle)");
    let mut avg_day = 0.0f32;
    for (m, c) in top.iter() {
        println!(
            "  id={} day={} domain/topic ~ {:?} cos={:.4}",
            m.id, m.day_idx, m.ctx.tags, c
        );
        avg_day += m.day_idx as f32;
    }
    avg_day /= top.len() as f32;
    println!(
        "  mean day_idx in top-10 = {:.2} (query day_idx = {})",
        avg_day, mid_year_day
    );

    // --- Scenario 3: early-year anchored query (oracle) ---
    let early_day: i32 = 10;
    let early_ts: i64 = BASE_TS + (early_day as i64) * 86_400;
    let t_early = TimeKey::from_unix(early_ts);

    let qctx_early = QueryCtx::with_time(ctx_key.clone(), t_early);
    let q_early = enc.encode_query(&base_sem, &qctx_early);
    let top_early = top_k(&memories, &q_early, 10);

    println!("\n[Scenario 3] Same sem+ctx, early-year anchored query (oracle)");
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

    // --- Latency suite ------------------------------------------------------
    println!("\n--- Latency suite ---");

    // 1) encode_memory latency over the whole corpus (re-encoding semantics).
    let mut lat_mem = Vec::<u128>::with_capacity(memories.len());
    for m in &memories {
        let t0 = Instant::now();
        let _ = enc.encode_memory(&m.semantic, &m.ctx, &m.time);
        let dt = t0.elapsed().as_nanos() as u128;
        lat_mem.push(dt);
    }
    print_latency_stats("encode_memory", &mut lat_mem);

    // 2) encode_query latency over one query per memory.
    let mut lat_query = Vec::<u128>::with_capacity(memories.len());
    for m in &memories {
        let qctx = QueryCtx::with_time(m.ctx.clone(), m.time);
        let t0 = Instant::now();
        let _ = enc.encode_query(&m.semantic, &qctx);
        let dt = t0.elapsed().as_nanos() as u128;
        lat_query.push(dt);
    }
    print_latency_stats("encode_query", &mut lat_query);

    // 3) Full brute-force retrieval: encode_query + oracle top_k.
    let full_trials = 2_000.min(memories.len());
    let mut lat_full = Vec::<u128>::with_capacity(full_trials);
    let mut rng = StdRng::seed_from_u64(12345);
    for _ in 0..full_trials {
        let idx = rng.gen_range(0..memories.len());
        let m = &memories[idx];

        let qctx = QueryCtx::with_time(m.ctx.clone(), m.time);
        let t0 = Instant::now();
        let q_vec = enc.encode_query(&m.semantic, &qctx);
        let _ = top_k(&memories, &q_vec, 10);
        let dt = t0.elapsed().as_nanos() as u128;
        lat_full.push(dt);
    }
    print_latency_stats("encode_query+oracle_top_k", &mut lat_full);

    // 4) FlatIndex ANN latency (exact over same vectors).
    let mut lat_flat_ann = Vec::<u128>::with_capacity(full_trials);
    for _ in 0..full_trials {
        let idx = rng.gen_range(0..memories.len());
        let m = &memories[idx];

        let qctx = QueryCtx::with_time(m.ctx.clone(), m.time);
        let t0 = Instant::now();
        let q_vec = enc.encode_query(&m.semantic, &qctx);
        let _ = flat_index.search(&q_vec, 10);
        let dt = t0.elapsed().as_nanos() as u128;
        lat_flat_ann.push(dt);
    }
    print_latency_stats("FlatIndex(search)", &mut lat_flat_ann);

    // 5) IVF-Flat latency.
    let mut lat_ivf = Vec::<u128>::with_capacity(full_trials);
    for _ in 0..full_trials {
        let idx = rng.gen_range(0..memories.len());
        let m = &memories[idx];

        let qctx = QueryCtx::with_time(m.ctx.clone(), m.time);
        let t0 = Instant::now();
        let q_vec = enc.encode_query(&m.semantic, &qctx);
        let _ = ivf_index.search(&q_vec, 10);
        let dt = t0.elapsed().as_nanos() as u128;
        lat_ivf.push(dt);
    }
    print_latency_stats("IVF-Flat(search)", &mut lat_ivf);

    // 6) IVF-PQ latency.
    let mut lat_pq = Vec::<u128>::with_capacity(full_trials);
    for _ in 0..full_trials {
        let idx = rng.gen_range(0..memories.len());
        let m = &memories[idx];

        let qctx = QueryCtx::with_time(m.ctx.clone(), m.time);
        let t0 = Instant::now();
        let q_vec = enc.encode_query(&m.semantic, &qctx);
        let _ = ivfpq_index.search(&q_vec, 10);
        let dt = t0.elapsed().as_nanos() as u128;
        lat_pq.push(dt);
    }
    print_latency_stats("IVF-PQ(search)", &mut lat_pq);

    // --- Error / quality suite ---------------------------------------------
    println!("\n--- Error / quality suite ---");
    ann_error_suite(&memories, &enc, &flat_index, &ivf_index, &ivfpq_index);

    // Keep the geometry validation explicitly visible.
    time_geometry_suite(&memories, &enc);
}
