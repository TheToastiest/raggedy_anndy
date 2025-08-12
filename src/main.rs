use raggedy_anndy::{Metric, FlatIndex, IvfIndex, IvfParams};

fn main() {
    // Build exact index
    let dim = 64; let metric = Metric::Cosine;
    let mut flat = FlatIndex::new(dim, metric);
    // ... add your (id, vector)

    // Build IVF‑Flat
    let data: Vec<(u64, Vec<f32>)> = vec![ /* fill with your dataset */ ];
    let ivf = IvfIndex::build(metric, dim, &data, IvfParams { nlist: 256, nprobe: 32, refine: 200, seed: 1337 });

    // Search
    let q: Vec<f32> = vec![0.0; dim]; // your query vector
    let hits = ivf.search(&q, 10);
    for h in hits { println!("{}\t{}", h.id, h.score); }
}