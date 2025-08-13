use std::{fs::File, io::{BufRead, BufReader, Write}};
use clap::Parser;
use raggedy_anndy::embed::{Embedder, RandomEmbedder};

#[derive(Parser, Debug)]
#[command(name="ingest", about="Read JSONL with {id, text}, emit CSV of id and vector")]
struct Args {
    #[arg(long)] input: String,
    #[arg(long, default_value_t=768)] dim: usize,
    #[arg(long, default_value_t=12345)] seed: u64,
    #[arg(long, default_value_t=true)] normalize: bool,
    #[arg(long, default_value="vectors.csv")] out: String,
}

fn main() {
    let args = Args::parse();
    let f = File::open(&args.input).expect("open input");
    let reader = BufReader::new(f);
    let mut emb = RandomEmbedder::new(args.dim, args.seed, args.normalize);
    let mut w = File::create(&args.out).expect("open out");
    for line in reader.lines() {
        let line = line.expect("read line");
        let v: serde_json::Value = serde_json::from_str(&line).expect("jsonl");
        let id = v["id"].as_u64().expect("id");
        let text = v["text"].as_str().unwrap_or("");
        let vec = emb.embed(text);
        write!(w, "{}", id).ok();
        for x in vec { write!(w, ",{}", x).ok(); }
        writeln!(w).ok();
    }
    eprintln!("wrote {}", args.out);
}
