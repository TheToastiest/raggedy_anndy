# raggedy\_anndy

Deterministic ANN for RAG in Rust. Flat exact search and IVF Flat with seeded k means. Reproducible builds and searches, recall targets with Wilson 95 percent lower bounds, and a latency sweep harness.

## Features

* Exact Flat index for L2 or cosine
* IVF Flat with deterministic k means plus plus and stable assignment
* Stable top k ordering by score then id
* Build fingerprint to catch non deterministic builds
* Sweep CLI that reports recall, Wilson 95 percent lower bound, p50 and p95 and p99 latency, and QPS
* Optional pass or fail gating on a target lower bound

## Status

* Shipping: Flat, IVF Flat, seeded k means, deterministic search, sweep CLI
* Planned: IVF PQ, OPQ, HNSW with deterministic insertion, sparse fusion for RAG

## Install

Clone and build with stable Rust.

```bash
git clone https://github.com/yourname/raggedy_anndy.git
cd raggedy_anndy
cargo build
```

## Quick test

```bash
cargo test -q
```

## Quick sweep

Run in release mode for realistic timing. This evaluates a grid of nprobe and refine values on synthetic data and writes a CSV.

```bash
cargo run --release --bin sweep -- \
  --n 4000 --dim 32 --metric cosine --k 10 \
  --nlist 256 --nprobe 64,96,128 --refine 200,300,400 \
  --queries 200 --warmup 25 --seed-data 42 --seed-queries 999 --seed-kmeans 7 \
  --csv results.csv
```

Typical results on a laptop for cosine with N 4000 and dim 32 and k 10:

* nprobe 96 refine 200 gives recall about 0.963 and lb95 about 0.954
* nprobe 128 refine 200 gives recall about 0.987 and lb95 about 0.980 with p95 near 0.42 ms

## Enforce a target lower bound

Fail the run if the Wilson 95 percent lower bound falls below a target.

```bash
cargo run --release --bin sweep -- \
  --n 4000 --dim 32 --metric cosine --k 10 \
  --nlist 256 --nprobe 96 --refine 200 \
  --queries 200 --warmup 25 --target-lb 0.90 --enforce
```

## Output columns

The sweep prints and writes these fields per configuration:

* nprobe
* refine
* recall
* lb95
* p50 and p90 and p95 and p99 in microseconds in the CSV and p50 and p95 in milliseconds in the console
* QPS
* build determinism flag
* build and eval time in milliseconds

## Library usage

Minimal example that builds IVF Flat and searches.

```rust
use raggedy_anndy::{Metric, IvfIndex, IvfParams, FlatIndex};

fn main() {
    let dim = 64;
    let metric = Metric::Cosine;

    // build data
    let mut data: Vec<(u64, Vec<f32>)> = Vec::new();
    for i in 0..1000u64 {
        data.push((i, vec![0.0; dim])); // replace with real vectors
    }

    // build index
    let params = IvfParams { nlist: 256, nprobe: 96, refine: 200, seed: 7 };
    let ivf = IvfIndex::build(metric, dim, &data, params);

    // search
    let q = vec![0.0f32; dim];
    let hits = ivf.search(&q, 10);
    for h in hits { println!("{}\t{}", h.id, h.score); }
}
```

## Reproducibility model

* Global seeds are carried through k means, dataset generation in tests, and query generation
* Exact math path with fixed f32 or f64 per operation
* Stable sorting of candidates by score then id
* Deterministic assignment to centroids with id tiebreakers
* Build fingerprint covers centroid bits and per list ids

## Profiles

Use the sweep to select a profile that meets your recall and latency requirements. For the synthetic setup above a good default is nprobe 96 and refine 200. Persist the choice in your app configuration and pass it into IvfParams.

## CI example

Run the sweep in CI and fail if the best configuration does not meet your target lower bound.

```yaml
jobs:
  sweep:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo run --release --bin sweep -- --n 4000 --dim 32 --metric cosine --k 10 --nlist 256 --nprobe 96 --refine 200 --queries 200 --warmup 25 --target-lb 0.90 --enforce --csv results.csv
      - uses: actions/upload-artifact@v4
        with:
          name: sweep-results
          path: results.csv
```

## License

Dual licensed under MIT or Apache 2.0 at your option.

* See LICENSE MIT and LICENSE APACHE for details
* You may not use any file except in compliance with one of the licenses

## Acknowledgments

This crate was inspired by FAISS style indexing and common RAG retrieval patterns. All code here is original to this repository.
