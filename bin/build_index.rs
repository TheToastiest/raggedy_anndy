use std::{fs::File, io::{BufRead, BufReader}};
use clap::Parser;
use raggedy_anndy::{Metric, IvfIndex, IvfParams};

#[derive(Parser, Debug)]
#[command(name="build-index", about="Build an IVF-Flat index from a CSV of id,vector...")]
struct Args {
    #[arg(long)] input: String, // CSV: id,v1,...,vd
    #[arg(long)] out: String,   // output directory
    #[arg(long, default_value_t=32)] dim: usize,
    #[arg(long, default_value_t=256)] nlist: usize,
    #[arg(long, default_value_t=64)] nprobe: usize,
    #[arg(long, default_value_t=200)] refine: usize,
    #[arg(long, default_value_t=7)] seed_kmeans: u64,
    #[arg(long, default_value="cosine")] metric: String,
}

fn main() {
    let args = Args::parse();
    let metric = match args.metric.to_lowercase().as_str() { "l2" => Metric::L2, _ => Metric::Cosine };
    let f = File::open(&args.input).expect("open input");
    let r = BufReader::new(f);
    let mut data: Vec<(u64, Vec<f32>)> = Vec::new();
    for line in r.lines() {
        let line = line.expect("read line");
        let mut it = line.split(',');
        let id: u64 = it.next().unwrap().parse().expect("id");
        let mut v = Vec::with_capacity(args.dim);
        for _ in 0..args.dim { v.push(it.next().unwrap().parse::<f32>().expect("f32")); }
        data.push((id, v));
    }
    let params = IvfParams { nlist: args.nlist, nprobe: args.nprobe, refine: args.refine, seed: args.seed_kmeans };
    let idx = IvfIndex::build(metric, args.dim, &data, params);
    idx.save_dir(std::path::Path::new(&args.out)).expect("save");
    eprintln!("saved {}", &args.out);
}
