use std::sync::Arc;
use std::time::{Instant, Duration};
use rand::{rngs::StdRng, Rng, SeedableRng};
use raggedy_anndy::ivfpq::{IvfPqIndex, IvfPqParams, OpqMode};
use raggedy_anndy::{Metric, Hit};
use crossbeam::channel;

const AGENT_COUNT: usize = 1000;
const DURATION_SECS: u64 = 30;

fn main() {
    // 1. Initial Build (Smaller 100k scale for rapid stress iteration)
    let dim = 128;
    let n = 100_000;
    let mut rng = StdRng::seed_from_u64(42);
    let mut data = Vec::new();
    for i in 0..n {
        let v = (0..dim).map(|_| rng.gen::<f32>()).collect();
        data.push((i as u64, v, i as u64)); // ID, Vec, Timestamp
    }

    let params = IvfPqParams {
        nlist: 1024, nprobe: 64, refine: 2000, seed: 7,
        m: 16, nbits: 8, iters: 15, use_opq: true,
        opq_mode: OpqMode::PcaPerm, opq_sweeps: 4, store_vecs: true,
    };

    println!("Building initial Cognitive Index...");
    let index = Arc::new(IvfPqIndex::build(Metric::Cosine, dim, &data, params));

    // 2. Stress Simulation
    println!("Starting Combat Load: {} agents, {}s duration", AGENT_COUNT, DURATION_SECS);
    let start_time = Instant::now();
    let (tx, rx) = channel::unbounded();

    std::thread::scope(|s| {
        for agent_id in 0..AGENT_COUNT {
            let idx = index.clone();
            let sender = tx.clone();
            s.spawn(move || {
                let mut agent_rng = StdRng::seed_from_u64(agent_id as u64);
                let mut local_latencies = Vec::new();

                while start_time.elapsed() < Duration::from_secs(DURATION_SECS) {
                    let q: Vec<f32> = (0..dim).map(|_| agent_rng.gen::<f32>()).collect();
                    let tick = start_time.elapsed().as_millis() as u64;

                    let q_start = Instant::now();
                    // Lambda 0.01 for temporal decay simulation
                    let _hits = idx.search_temporal(&q, 10, None, tick, 0.01);
                    local_latencies.push(q_start.elapsed().as_micros());
                }
                sender.send(local_latencies).unwrap();
            });
        }
    });

    // 3. Telemetry Consolidation
    let mut all_latencies: Vec<u128> = rx.into_iter().flatten().collect();
    all_latencies.sort_unstable();

    let p50 = all_latencies[all_latencies.len() / 2] as f64 / 1000.0;
    let p99 = all_latencies[(all_latencies.len() as f64 * 0.99) as usize] as f64 / 1000.0;
    let total_queries = all_latencies.len();

    println!("\n--- Cognitive Stress Results ---");
    println!("Total Queries: {}", total_queries);
    println!("Throughput:    {:.2} QPS", total_queries as f64 / DURATION_SECS as f64);
    println!("P50 Latency:   {:.3} ms", p50);
    println!("P99 Latency:   {:.3} ms", p99);
}