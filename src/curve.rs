// src/curve.rs
use std::time::Instant;
use crate::ivfpq::IvfPqIndex;
use crate::recall::{calculate_recall, RecallStats};
use crate::{Hit, FlatIndex};

pub fn generate_curve(
    index: &IvfPqIndex,
    flat: &FlatIndex,
    queries: &[Vec<f32>],
    k: usize
) -> Vec<(usize, RecallStats)> {
    let mut curve = Vec::new();
    // Sweep nprobe from 1 to 256 in powers of 2
    let probes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

    println!("\n--- Generating Recall vs P95 Curve ---");
    println!("nprobe\trecall\tp95_ms");

    for &p in &probes {
        let mut local_index = index.clone();
        local_index.params.nprobe = p;

        let mut latencies = Vec::new();
        let mut total_recall = 0.0;

        for q in queries {
            let start = Instant::now();
            let ann = local_index.search(q, k);
            latencies.push(start.elapsed().as_micros() as u128);

            let truth = flat.search(q, k);
            total_recall += calculate_recall(&truth, &ann, k);
        }

        latencies.sort_unstable();
        let p95 = latencies[(latencies.len() as f64 * 0.95) as usize] as f64 / 1000.0;
        let avg_recall = total_recall / queries.len() as f32;

        println!("{}\t{:.3}\t{:.3}ms", p, avg_recall, p95);
        curve.push((p, RecallStats { recall: avg_recall, p95_ms: p95 }));
    }
    curve
}