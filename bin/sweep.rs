mod build_index;

use std::{time::Instant, fs::File, io::Write};
use clap::{Parser, ValueEnum};
use rand::{SeedableRng, Rng};
use rand::rngs::StdRng;

use raggedy_anndy::{Metric, FlatIndex, IvfIndex, IvfParams};
use raggedy_anndy::eval::wilson_lower_bound;

#[derive(Clone, Copy, ValueEnum, Debug)]
enum MetricArg { Cosine, L2 }
impl From<MetricArg> for Metric {
    fn from(m: MetricArg) -> Self {
        match m { MetricArg::Cosine => Metric::Cosine, MetricArg::L2 => Metric::L2 }
    }
}

#[derive(Parser, Debug)]
#[command(name="raggedy-anndy sweep", about="Recall and latency sweeps for IVF-Flat (deterministic)")]
struct Args {
    #[arg(long, default_value_t=4000)] n: usize,
    #[arg(long, default_value_t=32)] dim: usize,
    #[arg(long, value_enum, default_value_t=MetricArg::Cosine)] metric: MetricArg,
    #[arg(long, default_value_t=10)] k: usize,
    #[arg(long, default_value_t=256)] nlist: usize,
    #[arg(long, default_value="64,96,128")] nprobe: String,
    #[arg(long, default_value="200,300,400")] refine: String,
    #[arg(long, default_value_t=200)] queries: usize,
    #[arg(long, default_value_t=0)] warmup: usize,
    #[arg(long, default_value_t=42)] seed_data: u64,
    #[arg(long, default_value_t=999)] seed_queries: u64,
    #[arg(long, default_value_t=7)] seed_kmeans: u64,
    #[arg(long)] csv: Option<String>,
    #[arg(long, default_value_t=0.90)] target_lb: f32,
    #[arg(long, default_value_t=false)] enforce: bool,
    #[arg(long, default_value_t=false)] require_all: bool,
}

fn parse_list(s: &str) -> Vec<usize> {
    s.split(',').filter_map(|t| t.trim().parse::<usize>().ok()).collect()
}

fn random_unit_vec(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
    let mut n = 0.0f32;
    for x in &v { n += *x * *x; }
    n = n.sqrt();
    if n > 0.0 { for x in v.iter_mut() { *x /= n; } }
    v
}

fn percentile_us(v: &mut [u128], p: f64) -> u128 {
    if v.is_empty() { return 0; }
    v.sort_unstable();
    let idx = ((p * (v.len() as f64 - 1.0)).round() as usize).min(v.len() - 1);
    v[idx]
}

fn main() {
    let args = Args::parse();
    let metric: Metric = args.metric.into();

    // Build dataset
    let t_build0 = Instant::now();
    let mut rng = StdRng::seed_from_u64(args.seed_data);
    let mut flat = FlatIndex::new(args.dim, metric);
    let mut data: Vec<(u64, Vec<f32>)> = Vec::with_capacity(args.n);
    for i in 0..args.n {
        let v = match metric {
            Metric::Cosine => random_unit_vec(&mut rng, args.dim),
            Metric::L2 => (0..args.dim).map(|_| rng.gen::<f32>() - 0.5).collect(),
        };
        flat.add(i as u64, &v);
        data.push((i as u64, v));
    }
    let dataset_ms = t_build0.elapsed().as_millis();

    // Prepare queries
    let mut q_rng = StdRng::seed_from_u64(args.seed_queries);
    let queries: Vec<Vec<f32>> = (0..(args.warmup + args.queries)).map(|_| match metric {
        Metric::Cosine => random_unit_vec(&mut q_rng, args.dim),
        Metric::L2 => (0..args.dim).map(|_| q_rng.gen::<f32>() - 0.5).collect(),
    }).collect();

    let nprobe_list = parse_list(&args.nprobe);
    let refine_list = parse_list(&args.refine);

    // CSV
    let mut csv: Option<Box<dyn Write>> = if let Some(path) = args.csv.as_ref() {
        match File::create(path) {
            Ok(mut f) => {
                writeln!(f, "metric,n,dim,k,nlist,nprobe,refine,recall,lb95,p50_us,p90_us,p95_us,p99_us,qps,build_det,build_ms,eval_ms").ok();
                Some(Box::new(f))
            }
            Err(e) => {
                #[cfg(windows)]
                {
                    if e.raw_os_error() == Some(32) {
                        use std::path::Path;
                        let p = Path::new(path);
                        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("results");
                        let parent = p.parent().unwrap_or(Path::new("."));
                        let ts = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
                        let alt = parent.join(format!("{}-{}.csv", stem, ts));
                        eprintln!("CSV locked, writing to {}", alt.display());
                        let mut f = File::create(&alt).expect("csv open alt");
                        writeln!(f, "metric,n,dim,k,nlist,nprobe,refine,recall,lb95,p50_us,p90_us,p95_us,p99_us,qps,build_det,build_ms,eval_ms").ok();
                        Some(Box::new(f))
                    } else {
                        panic!("csv open: {}", e);
                    }
                }
                #[cfg(not(windows))]
                { panic!("csv open: {}", e); }
            }
        }
    } else { None };

    println!("Dataset built in {} ms (N={}, dim={}, metric={:?})", dataset_ms, args.n, args.dim, metric);
    println!("{:>8} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10}",
             "nprobe", "refine", "recall", "lb95", "p50(ms)", "p95(ms)", "QPS");

    #[derive(Clone)]
    struct Row {
        nprobe: usize,
        refine: usize,
        recall: f32,
        lb: f32,
        p50_ms: f64,
        p95_ms: f64,
        qps: f64,
        build_det: bool,
    }
    let mut rows: Vec<Row> = Vec::new();

    for &nprobe in &nprobe_list {
        for &refine in &refine_list {
            let params = IvfParams { nlist: args.nlist, nprobe, refine, seed: args.seed_kmeans };

            // Build index and check build determinism
            let t0 = Instant::now();
            let ivf = IvfIndex::build(metric, args.dim, &data, params);
            let fp1 = ivf.fingerprint();
            let ivf2 = IvfIndex::build(metric, args.dim, &data, params);
            let build_det = fp1 == ivf2.fingerprint();
            let build_ms = t0.elapsed().as_millis();

            // Evaluate recall and latency
            let t_eval0 = Instant::now();
            let mut hits = 0usize;
            let mut poss = 0usize;
            let mut det = true;
            let mut lat_us: Vec<u128> = Vec::with_capacity(args.queries);

            for (qi, q) in queries.iter().enumerate() {
                if qi < args.warmup {
                    let _ = ivf.search(q, args.k);
                    continue;
                }
                let t = Instant::now();
                let ann1 = ivf.search(q, args.k);
                let dt = t.elapsed().as_micros();
                lat_us.push(dt);

                let ann2 = ivf.search(q, args.k);
                if ann1 != ann2 { det = false; }

                let truth = flat.search(q, args.k);
                let truth_ids: Vec<u64> = truth.iter().map(|h| h.id).collect();
                let ann_ids: Vec<u64> = ann1.iter().map(|h| h.id).collect();
                for id in truth_ids.iter().take(args.k) {
                    if ann_ids.contains(id) { hits += 1; }
                    poss += 1;
                }
            }
            let eval_ms = t_eval0.elapsed().as_millis();

            let recall = (hits as f32) / (poss as f32);
            let lb = wilson_lower_bound(hits, poss, 1.96) as f32;

            let mut lat_sorted = lat_us.clone();
            let p50 = percentile_us(&mut lat_sorted, 0.50) as f64 / 1000.0;
            let p90 = percentile_us(&mut lat_sorted, 0.90) as f64 / 1000.0;
            let p95 = percentile_us(&mut lat_sorted, 0.95) as f64 / 1000.0;
            let p99 = percentile_us(&mut lat_sorted, 0.99) as f64 / 1000.0;

            let sum_us: u128 = lat_us.iter().sum();
            let qps = if sum_us > 0 {
                (args.queries as f64) / (sum_us as f64 / 1_000_000.0)
            } else { 0.0 };

            println!("{:>8} {:>8} {:>8.3} {:>8.3} {:>10.3} {:>10.3} {:>10.1}{}",
                     nprobe, refine, recall, lb, p50, p95, qps,
                     if det && build_det { "" } else { "  (non-det)" });

            if let Some(w) = csv.as_mut() {
                writeln!(
                    w,
                    "{:?},{},{},{},{},{},{},{:.6},{:.6},{:.0},{:.0},{:.0},{:.0},{:.2},{},{},{}",
                    metric, args.n, args.dim, args.k, args.nlist, nprobe, refine,
                    recall, lb, p50*1000.0, p90*1000.0, p95*1000.0, p99*1000.0, qps,
                    build_det, build_ms, eval_ms
                ).ok();
            }

            rows.push(Row { nprobe, refine, recall, lb, p50_ms: p50, p95_ms: p95, qps, build_det: det && build_det });
        }
    }

    if args.enforce {
        if args.require_all {
            if rows.iter().any(|r| r.lb < args.target_lb) {
                eprintln!("FAIL: at least one config has lb95 < target {:.3}", args.target_lb);
                std::process::exit(1);
            }
        } else {
            rows.sort_by(|a, b|
                b.lb.partial_cmp(&a.lb).unwrap()
                    .then(b.recall.partial_cmp(&a.recall).unwrap())
                    .then(a.p95_ms.partial_cmp(&b.p95_ms).unwrap())
            );
            let best = &rows[0];
            println!("BEST -> nprobe={} refine={} recall={:.3} lb95={:.3} p95={:.3} ms",
                     best.nprobe, best.refine, best.recall, best.lb, best.p95_ms);
            if best.lb < args.target_lb {
                eprintln!("FAIL: best lb95 {:.3} < target {:.3}", best.lb, args.target_lb);
                std::process::exit(1);
            }
        }
    }
}
