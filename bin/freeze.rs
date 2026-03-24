use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

use raggedy_anndy::eval::wilson_lower_bound;
use raggedy_anndy::ivfpq::{IvfPqIndex, IvfPqParams, OpqMode};
use raggedy_anndy::par::parallel_map_indexed;
use raggedy_anndy::{FlatIndex, Hit, Metric};
use raggedy_anndy::curve;
use raggedy_anndy::recall::calculate_recall;

/// The "Brain" that manages the two tiers of memory.
pub struct HybridIndex {
    pub base: Arc<dyn Fn(&[f32]) -> Vec<Hit> + Send + Sync>,
    pub buffer: FlatIndex,
    pub tombstones: HashSet<u64>,
}

impl HybridIndex {
    pub fn search(&self, q: &[f32], k: usize) -> Vec<Hit> {
        let base_hits = (self.base)(q);
        let buffer_hits = self.buffer.search(q, k);

        let mut combined: Vec<Hit> = base_hits.into_iter()
            .chain(buffer_hits.into_iter())
            .filter(|h| !self.tombstones.contains(&h.id))
            .collect();

        combined.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal).reverse());
        combined.truncate(k);
        combined
    }
}

#[derive(Clone, Copy, ValueEnum, Debug)]
enum MetricArg { Cosine, L2 }
impl From<MetricArg> for Metric {
    fn from(m: MetricArg) -> Self {
        match m { MetricArg::Cosine => Metric::Cosine, MetricArg::L2 => Metric::L2 }
    }
}

#[derive(Clone, Copy, ValueEnum, Debug)]
enum Backend { IvfFlat, IvfPq }

#[derive(Clone, Copy, ValueEnum, Debug)]
enum OpqModeArg { Perm, Pca, PcaPerm }
impl From<OpqModeArg> for OpqMode {
    fn from(m: OpqModeArg) -> Self {
        match m {
            OpqModeArg::Perm => OpqMode::Perm,
            OpqModeArg::Pca => OpqMode::Pca,
            OpqModeArg::PcaPerm => OpqMode::PcaPerm,
        }
    }
}

#[derive(Clone, Copy, ValueEnum, Debug)]
enum DistMode {
    Hierarchical, Overlapping, PowerLaw, NoiseHeavy, LowSeparation, UniformSphere, OpenAiLike,
}

#[derive(Parser, Debug)]
#[command(name = "freeze", about = "Adversarial Stress Test with Hybrid Mutation")]
struct Args {
    #[arg(long, default_value_t = 20000)] n: usize,
    #[arg(long, default_value_t = 256)] dim: usize,
    #[arg(long, value_enum, default_value_t = MetricArg::Cosine)] metric: MetricArg,
    #[arg(long, value_enum, default_value_t = DistMode::Hierarchical)] dist: DistMode,
    #[arg(long, default_value_t = 10)] k: usize,
    #[arg(long, default_value_t = 200)] queries: usize,
    #[arg(long, default_value_t = 10)] warmup: usize,
    #[arg(long, default_value_t = 42)] seed_data: u64,
    #[arg(long, default_value_t = 999)] seed_queries: u64,
    #[arg(long, default_value_t = 7)] seed_kmeans: u64,
    #[arg(long, value_enum, default_value_t = Backend::IvfPq)] backend: Backend,
    #[arg(long, default_value_t = 256)] nlist: usize,
    #[arg(long, default_value_t = 32)] nprobe: usize,
    #[arg(long, default_value_t = 500)] refine: usize,
    #[arg(long, default_value_t = 64)] m: usize,
    #[arg(long, default_value_t = 8)] nbits: u8,
    #[arg(long, default_value_t = 25)] iters: usize,
    #[arg(long, default_value_t = false)] opq: bool,
    #[arg(long, value_enum, default_value_t = OpqModeArg::PcaPerm)] opq_mode: OpqModeArg,
    #[arg(long, default_value_t = 6)] opq_sweeps: usize,
    #[arg(long, default_value_t = true)] store_vecs: bool,
    #[arg(long, default_value_t = 0.75)] min_lb95: f32,
    #[arg(long, default_value_t = 0.80)] min_recall: f32,
    #[arg(long, default_value_t = 8.33)] max_p95_ms: f64,
    #[arg(long, default_value_t = 0.0)] max_p99_ms: f64,
    #[arg(long, default_value_t = 8)] threads: usize,
    #[arg(long, default_value_t = 1)] intra: usize,
    #[arg(long, default_value_t = false)] stress_test: bool,
    #[arg(long, default_value_t = false)] plot: bool,
}

#[inline]
fn normalize_slice(v: &mut [f32]) {
    let mut n = 0.0f32;
    for &x in v.iter() { n += x * x; }
    if n > 0.0 {
        let inv = 1.0 / n.sqrt();
        for x in v { *x *= inv; }
    }
}

fn generate_adversarial_data(mode: DistMode, n: usize, dim: usize, is_cosine: bool, mut rng: StdRng) -> Vec<Vec<f32>> {
    let mut data = Vec::with_capacity(n);
    match mode {
        DistMode::UniformSphere => {
            let normal = Normal::new(0.0, 1.0).unwrap();
            for _ in 0..n {
                let mut v: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
                if is_cosine { normalize_slice(&mut v); }
                data.push(v);
            }
        }
        DistMode::Hierarchical => {
            let num_topics = 10;
            let subtopics = 5;
            let seeds: Vec<Vec<f32>> = (0..num_topics).map(|_| {
                let mut v: Vec<f32> = (0..dim).map(|_| rng.sample(Normal::new(0.0, 1.0).unwrap())).collect();
                if is_cosine { normalize_slice(&mut v); }
                v
            }).collect();
            let jitter_t = Normal::new(0.0, 0.15).unwrap();
            let jitter_s = Normal::new(0.0, 0.02).unwrap();
            for i in 0..n {
                let mut v = seeds[i % num_topics].clone();
                let sub_idx = (i / num_topics) % subtopics;
                for d in 0..dim { v[d] += jitter_t.sample(&mut rng) * (sub_idx as f32) + jitter_s.sample(&mut rng); }
                if is_cosine { normalize_slice(&mut v); }
                data.push(v);
            }
        }
        DistMode::OpenAiLike => {
            let mut variance_profile = vec![0.0f32; dim];
            for d in 0..dim { variance_profile[d] = std::f32::consts::E.powf(-(d as f32) / 20.0).max(0.001); }
            for _ in 0..n {
                let mut v = vec![0.0f32; dim];
                for d in 0..dim { v[d] = rng.sample(Normal::new(0.0, variance_profile[d]).unwrap()); }
                if is_cosine { normalize_slice(&mut v); }
                data.push(v);
            }
        }
        _ => {
            let normal = Normal::new(0.0, 1.0).unwrap();
            for _ in 0..n {
                let mut v: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
                if is_cosine { normalize_slice(&mut v); }
                data.push(v);
            }
        }
    }
    data
}

fn percentile_us(mut v: Vec<u128>, p: f64) -> u128 {
    if v.is_empty() { return 0; }
    v.sort_unstable();
    let idx = ((p * (v.len() as f64 - 1.0)).round() as usize).min(v.len() - 1);
    v[idx]
}

#[inline]
fn ids_sorted(hits: &[Hit]) -> Vec<u64> {
    let mut ids: Vec<u64> = hits.iter().map(|h| h.id).collect();
    ids.sort_unstable();
    ids
}

#[derive(Clone, Copy)]
struct PerQ { hits: usize, poss: usize, dt_us: u128, det_ok: bool }

fn main() {
    let args = Args::parse();
    let metric: Metric = args.metric.into();
    let is_cosine = matches!(metric, Metric::Cosine);

    let t_gen = Instant::now();
    let mut rng = StdRng::seed_from_u64(args.seed_data);
    let mut flat_exact = FlatIndex::new(args.dim, metric);
    let data_vecs = generate_adversarial_data(args.dist, args.n, args.dim, is_cosine, rng);

    // UPDATED: Now initialized as a 3-tuple (u64, Vec<f32>, u64) to support timestamps.
    let mut data: Vec<(u64, Vec<f32>, u64)> = Vec::with_capacity(args.n);

    for (i, v) in data_vecs.into_iter().enumerate() {
        flat_exact.add(i as u64, &v);
        // We inject `i as u64` as a synthetic timestamp sequence for the stress test.
        data.push((i as u64, v, i as u64));
    }
    println!("Freeze: Adversarial Mode ({:?}) | Dataset Gen: {}ms", args.dist, t_gen.elapsed().as_millis());

    let build_t0 = Instant::now();

    let mut ivf_pq_index = None;

    let (build_det, search_fn): (bool, Arc<dyn Fn(&[f32]) -> Vec<Hit> + Send + Sync>) = match args.backend {
        Backend::IvfPq => {
            let params = IvfPqParams {
                nlist: args.nlist, nprobe: args.nprobe, refine: args.refine, seed: args.seed_kmeans,
                m: args.m, nbits: args.nbits, iters: args.iters, use_opq: args.opq,
                opq_mode: args.opq_mode.into(), opq_sweeps: args.opq_sweeps, store_vecs: args.store_vecs,
            };
            let idx = IvfPqIndex::build(metric, args.dim, &data, params);
            let fp = idx.fingerprint();
            let det = fp == IvfPqIndex::build(metric, args.dim, &data, params).fingerprint();

            let idx_arc = Arc::new(idx.clone());
            ivf_pq_index = Some(idx);

            let k = args.k;
            (det, Arc::new(move |q| idx_arc.search(q, k)))
        }
        Backend::IvfFlat => {
            (true, Arc::new(move |_| vec![]))
        }
    };
    println!("Index build complete: {}ms", build_t0.elapsed().as_millis());

    if args.plot {
        if let Some(ref idx) = ivf_pq_index {
            let plot_queries = &generate_adversarial_data(args.dist, 100, args.dim, is_cosine, StdRng::seed_from_u64(args.seed_queries));
            curve::generate_curve(idx, &flat_exact, plot_queries, args.k);
        } else {
            println!("Error: Pareto sweep only supported for IvfPq backend.");
        }
    }

    // --- HYBRID STRESS TEST ---
    let mut hybrid = HybridIndex {
        base: search_fn.clone(),
        buffer: FlatIndex::new(args.dim, metric),
        tombstones: HashSet::new(),
    };

    if args.stress_test {
        let to_delete: Vec<u64> = (0..args.n).step_by(10).map(|i| i as u64).collect();
        for id in &to_delete { hybrid.tombstones.insert(*id); }

        let mut mutation_rng = StdRng::seed_from_u64(1337);
        for i in 0..100 {
            let mut v = vec![0.0f32; args.dim];
            for d in 0..args.dim { v[d] = mutation_rng.gen_range(-1.0..1.0); }
            if is_cosine { normalize_slice(&mut v); }
            hybrid.buffer.add(999_000 + i, &v);
            flat_exact.add(999_000 + i, &v);
        }
        println!("Stress Test: Deleted {} | Added 100 to Delta Buffer", to_delete.len());
    }

    let mut q_rng = StdRng::seed_from_u64(args.seed_queries);
    let queries: Vec<Vec<f32>> = (0..(args.warmup + args.queries)).map(|_| {
        let source_idx = q_rng.gen_range(0..args.n);
        // Note: data[source_idx].1 safely targets the vector payload of the new 3-tuple
        let mut q = data[source_idx].1.clone();
        for d in 0..args.dim { q[d] += Normal::new(0.0, 0.01).unwrap().sample(&mut q_rng); }
        if is_cosine { normalize_slice(&mut q); }
        q
    }).collect();

    for qi in 0..args.warmup { let _ = hybrid.search(&queries[qi], args.k); }

    let eval_slice = &queries[args.warmup..];
    let perq: Vec<PerQ> = parallel_map_indexed(eval_slice, args.threads, |q, _| {
        let t = Instant::now();
        let ann = hybrid.search(q, args.k);
        let dt = t.elapsed().as_micros() as u128;
        let det_ok = ids_sorted(&ann) == ids_sorted(&hybrid.search(q, args.k));
        let truth = flat_exact.search(q, args.k);
        let ann_set: HashSet<u64> = ann.iter().map(|h| h.id).collect();
        let hits = truth.iter().take(args.k).filter(|h| ann_set.contains(&h.id)).count();
        PerQ { hits, poss: args.k, dt_us: dt, det_ok }
    });

    let hits: usize = perq.iter().map(|r| r.hits).sum();
    let poss: usize = perq.iter().map(|r| r.poss).sum();
    let det = build_det && perq.iter().all(|r| r.det_ok);
    let recall = hits as f32 / poss as f32;
    let lb = wilson_lower_bound(hits, poss, 1.96) as f32;
    let p95 = percentile_us(perq.iter().map(|r| r.dt_us).collect(), 0.95) as f64 / 1000.0;

    println!("\nrecall={:.3} lb95={:.3} p95={:.3}ms det={}", recall, lb, p95, det);
    if lb < args.min_lb95 || p95 > args.max_p95_ms { std::process::exit(1); }
}