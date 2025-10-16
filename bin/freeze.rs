use std::sync::Arc;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use raggedy_anndy::{Metric, FlatIndex, IvfIndex, IvfParams};
use raggedy_anndy::ivfpq::{IvfPqIndex, IvfPqParams, OpqMode};
use raggedy_anndy::eval::wilson_lower_bound;
use raggedy_anndy::par::parallel_map_indexed;

#[derive(Clone, Copy, ValueEnum, Debug)]
enum MetricArg { Cosine, L2 }
impl From<MetricArg> for Metric {
    fn from(m: MetricArg) -> Self { match m { MetricArg::Cosine => Metric::Cosine, MetricArg::L2 => Metric::L2 } }
}

#[derive(Clone, Copy, ValueEnum, Debug)]
enum Backend { IvfFlat, IvfPq }

#[derive(Clone, Copy, ValueEnum, Debug)]
enum OpqModeArg { Perm, Pca, PcaPerm }
impl From<OpqModeArg> for OpqMode {
    fn from(m: OpqModeArg) -> Self { match m { OpqModeArg::Perm => OpqMode::Perm, OpqModeArg::Pca => OpqMode::Pca, OpqModeArg::PcaPerm => OpqMode::PcaPerm } }
}

/// Freeze-test: run a single fixed configuration and fail fast on regressions.
#[derive(Parser, Debug)]
#[command(name="freeze", about="Deterministic, CI-friendly regression gate for recall & latency")]
struct Args {
    // dataset
    #[arg(long, default_value_t=25000)] n: usize,
    #[arg(long, default_value_t=256)] dim: usize,
    #[arg(long, value_enum, default_value_t=MetricArg::Cosine)] metric: MetricArg,

    // evaluation
    #[arg(long, default_value_t=10)] k: usize,
    #[arg(long, default_value_t=200)] queries: usize,
    #[arg(long, default_value_t=10)] warmup: usize,

    // seeds
    #[arg(long, default_value_t=42)] seed_data: u64,
    #[arg(long, default_value_t=999)] seed_queries: u64,
    #[arg(long, default_value_t=7)] seed_kmeans: u64,

    // backend
    #[arg(long, value_enum, default_value_t=Backend::IvfPq)] backend: Backend,

    // IVF coarse
    #[arg(long, default_value_t=2048)] nlist: usize,
    #[arg(long, default_value_t=386)] nprobe: usize,
    #[arg(long, default_value_t=200)] refine: usize,

    // PQ params (ivf-pq only)
    #[arg(long, default_value_t=64)] m: usize,
    #[arg(long, default_value_t=8)] nbits: u8,
    #[arg(long, default_value_t=80)] iters: usize,
    #[arg(long, default_value_t=true, action=clap::ArgAction::Set)]opq: bool,
    #[arg(long, value_enum, default_value_t=OpqModeArg::PcaPerm)] opq_mode: OpqModeArg,
    #[arg(long, default_value_t=6)] opq_sweeps: usize,
    #[arg(long, default_value_t=true)] store_vecs: bool,

    // thresholds (freeze gates)
    #[arg(long, default_value_t=0.83)] min_lb95: f32,
    #[arg(long, default_value_t=0.85)] min_recall: f32,
    #[arg(long, default_value_t=250.0)] max_p95_ms: f64,

    // threading
    #[arg(long, default_value_t=1)] threads: usize, // concurrency across queries
    #[arg(long, default_value_t=1)] intra: usize,   // intra-query parallelism (ivf-pq)
}

fn random_unit_vec(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
    let mut n = 0.0f32; for x in &v { n += *x * *x; } n = n.sqrt();
    if n > 0.0 { for x in v.iter_mut() { *x /= n; } }
    v
}

fn percentile_us(mut v: Vec<u128>, p: f64) -> u128 {
    if v.is_empty() { return 0; }
    v.sort_unstable();
    let idx = ((p * (v.len() as f64 - 1.0)).round() as usize).min(v.len() - 1);
    v[idx]
}

#[derive(Clone, Copy)]
struct PerQ { hits: usize, poss: usize, dt_us: u128, det_ok: bool }

fn main() {
    let args = Args::parse();
    let metric: Metric = args.metric.into();

    // Build dataset & exact baseline
    let t_build0 = Instant::now();
    let mut rng = StdRng::seed_from_u64(args.seed_data);
    let mut flat_exact = FlatIndex::new(args.dim, metric);
    let mut data: Vec<(u64, Vec<f32>)> = Vec::with_capacity(args.n);
    for i in 0..args.n {
        let v = match metric { Metric::Cosine => random_unit_vec(&mut rng, args.dim), Metric::L2 => (0..args.dim).map(|_| rng.gen::<f32>() - 0.5).collect(), };
        flat_exact.add(i as u64, &v); data.push((i as u64, v));
    }
    let dataset_ms = t_build0.elapsed().as_millis();

    // Queries (warm + measured)
    let mut q_rng = StdRng::seed_from_u64(args.seed_queries);
    let queries: Vec<Vec<f32>> = (0..(args.warmup + args.queries)).map(|_| match metric {
        Metric::Cosine => random_unit_vec(&mut q_rng, args.dim),
        Metric::L2 => (0..args.dim).map(|_| q_rng.gen::<f32>() - 0.5).collect(),
    }).collect();

    println!(
        "Freeze profile: {:?} backend={:?} N={} dim={} k={} nlist={} nprobe={} refine={} m={} nbits={} iters={} opq={} opq_mode={:?} sweeps={} store_vecs={} threads={} intra={}",
        metric, args.backend, args.n, args.dim, args.k, args.nlist, args.nprobe, args.refine,
        args.m, args.nbits, args.iters, args.opq, args.opq_mode, args.opq_sweeps, args.store_vecs, args.threads, args.intra
    );
    println!("Dataset built in {} ms", dataset_ms);

    // Build index + determinism check, and wrap a thread-safe callable
    let build_t0 = Instant::now();
    let (build_det, search_fn): (bool, Arc<dyn Fn(&[f32]) -> Vec<raggedy_anndy::Hit> + Send + Sync>) = match args.backend {
        Backend::IvfFlat => {
            let params = IvfParams { nlist: args.nlist, nprobe: args.nprobe, refine: args.refine, seed: args.seed_kmeans };
            let idx = IvfIndex::build(metric, args.dim, &data, params);
            let fp1 = idx.fingerprint();
            let idx2 = IvfIndex::build(metric, args.dim, &data, params);
            let det = fp1 == idx2.fingerprint();
            let idx = Arc::new(idx); let k = args.k;
            (det, Arc::new(move |q: &[f32]| idx.search(q, k)))
        }
        Backend::IvfPq => {
            let params = IvfPqParams {
                nlist: args.nlist, nprobe: args.nprobe, refine: args.refine, seed: args.seed_kmeans,
                m: args.m, nbits: args.nbits, iters: args.iters,
                use_opq: args.opq, opq_mode: args.opq_mode.into(), opq_sweeps: args.opq_sweeps,
                store_vecs: args.store_vecs,
            };
            let idx = IvfPqIndex::build(metric, args.dim, &data, params);
            let fp1 = idx.fingerprint();
            let idx2 = IvfPqIndex::build(metric, args.dim, &data, params);
            let det = fp1 == idx2.fingerprint();
            let idx = Arc::new(idx); let k = args.k; let intra = args.intra.max(1);
            let f = move |q: &[f32]| { if intra > 1 { idx.search_parallel(q, k, intra) } else { idx.search(q, k) } };
            (det, Arc::new(f))
        }
    };
    let build_ms = build_t0.elapsed().as_millis();

    // Warmup (do not time)
    for qi in 0..args.warmup { let _ = search_fn.as_ref()(&queries[qi]); }

    // Evaluate (optionally parallel at the query level only)
    let flat_exact = Arc::new(flat_exact);
    let eval_slice = &queries[args.warmup..];

    let perq: Vec<PerQ> = if args.threads <= 1 {
        let mut out = Vec::with_capacity(eval_slice.len());
        for q in eval_slice {
            let t = Instant::now();
            let ann1 = search_fn.as_ref()(q);
            let dt = t.elapsed().as_micros() as u128;
            let ann2 = search_fn.as_ref()(q);
            let det_ok = ann1 == ann2;
            let truth = flat_exact.search(q, args.k);
            let truth_ids: Vec<u64> = truth.iter().map(|h| h.id).collect();
            let ann_ids: Vec<u64> = ann1.iter().map(|h| h.id).collect();
            let mut hits = 0usize; let mut poss = 0usize;
            for id in truth_ids.iter().take(args.k) { if ann_ids.contains(id) { hits += 1; } poss += 1; }
            out.push(PerQ { hits, poss, dt_us: dt, det_ok });
        }
        out
    } else {
        parallel_map_indexed(eval_slice, args.threads, {
            let search_fn = Arc::clone(&search_fn);
            let flat = Arc::clone(&flat_exact);
            let k = args.k;
            move |q, _i| {
                let t = Instant::now();
                let ann1 = search_fn.as_ref()(q);
                let dt = t.elapsed().as_micros() as u128;
                let ann2 = search_fn.as_ref()(q);
                let det_ok = ann1 == ann2;
                let truth = flat.search(q, k);
                let truth_ids: Vec<u64> = truth.iter().map(|h| h.id).collect();
                let ann_ids: Vec<u64> = ann1.iter().map(|h| h.id).collect();
                let mut hits = 0usize; let mut poss = 0usize;
                for id in truth_ids.iter().take(k) { if ann_ids.contains(id) { hits += 1; } poss += 1; }
                PerQ { hits, poss, dt_us: dt, det_ok }
            }
        })
    };

    // Reduce metrics
    let hits: usize = perq.iter().map(|r| r.hits).sum();
    let poss: usize = perq.iter().map(|r| r.poss).sum();
    let det = build_det && perq.iter().all(|r| r.det_ok);

    let recall = if poss > 0 { (hits as f32) / (poss as f32) } else { 0.0 };
    let lb = if poss > 0 { wilson_lower_bound(hits, poss, 1.96) as f32 } else { 0.0 };

    let lat_us: Vec<u128> = perq.iter().map(|r| r.dt_us).collect();
    let p50 = percentile_us(lat_us.clone(), 0.50) as f64 / 1000.0;
    let p90 = percentile_us(lat_us.clone(), 0.90) as f64 / 1000.0;
    let p95 = percentile_us(lat_us.clone(), 0.95) as f64 / 1000.0;
    let p99 = percentile_us(lat_us, 0.99) as f64 / 1000.0;

    let sum_us: u128 = perq.iter().map(|r| r.dt_us).sum();
    let qps = if sum_us > 0 { (perq.len() as f64) / (sum_us as f64 / 1_000_000.0) } else { 0.0 };

    println!(
        "\nrecall={:.3}  lb95={:.3}  p50={:.3}ms  p90={:.3}ms  p95={:.3}ms  p99={:.3}ms  QPS={:.1}{}  (build {} ms)",
        recall, lb, p50, p90, p95, p99, qps, if det { "" } else { "  (non-det)" }, build_ms
    );

    // Freeze gates
    let mut ok = true;
    if lb < args.min_lb95    { eprintln!("FAIL: lb95 {:.3} < target {:.3}", lb, args.min_lb95); ok = false; }
    if recall < args.min_recall { eprintln!("FAIL: recall {:.3} < target {:.3}", recall, args.min_recall); ok = false; }
    if p95 > args.max_p95_ms { eprintln!("FAIL: p95 {:.3} ms > budget {:.3} ms", p95, args.max_p95_ms); ok = false; }
    if !det { eprintln!("FAIL: non-deterministic build/search"); ok = false; }

    if !ok { std::process::exit(1); }
}
