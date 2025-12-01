# raggedy_anndy

Deterministic approximate nearest neighbor (ANN) search for RAG workloads, built in Rust. The project ships reproducible Flat, IVF-Flat, and IVF-PQ (with OPQ-P) indexes plus CLI tooling for sweeps, gating, and demo ingestion.

<!-- Badges -->
- License: Dual MIT/Apache-2.0
- crates.io: TODO (publish crate and link)

---

## Quick links
- [Source layout](#source-layout)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Highlights
- Deterministic training and search: seeded k-means++, stable top-k tie-breaks, and build fingerprints.
- Indexes: Flat (exact), IVF-Flat, and IVF-PQ with OPQ-P permutations/PCA.
- Quality gates and profiling: Wilson lb95 recall, latency percentiles, and CSV outputs for sweeps.
- CLI utilities: `freeze` for fixed-profile evaluation, `sweep` for grid searches, and `ingest` for demo embeddings.
- Optional tags, tombstones, and persist/load helpers to support RAG-ish lifecycles.

## Architecture (high level)
```
[embedder/demo data] --> [index builders: flat / ivf-flat / ivf-pq + opq-p] --> [searchers: deterministic top-k] --> [metrics + fingerprints]
```
> Replace with a real diagram (TODO).

## Supported environments
- Rust stable toolchain (tested with `cargo` via Makefile).
- Linux x86_64 (primary CI target); other Unix-like systems may work but are untested here.

## Installation
1. Install the Rust toolchain (stable) and `cargo`.
2. Clone the repository:
   ```bash
   git clone https://github.com/TheToastiest/raggedy_anndy.git
   cd raggedy_anndy
   ```
3. Build release binaries with Cargo:
   ```bash
   cargo build --release
   ```
4. Or use the Makefile to build the `freeze` binary:
   ```bash
   make freeze-bin
   ```

## Quickstart
### Freeze (gated single run)
Use the baked-in defaults from the Makefile for a CI-friendly profile:
```bash
make default-freeze
```
Override parameters on the command line (example tweaks `nprobe` and `refine`):
```bash
make default-freeze NPROBE=840 REFINE=1200
```

### Sweep (grid search to CSV)
```bash
./target/release/sweep \
  --backend ivf-pq --n 50000 --dim 256 --metric cosine --k 10 \
  --nlist 1536 --nprobe 704,840,906 --refine 1200,1600,2000 \
  --m 128 --nbits 8 --iters 60 --opq --opq-mode pca-perm --opq-sweeps 8 \
  --queries 200 --warmup 25 --seed-data 42 --seed-queries 999 --seed-kmeans 7 \
  --csv results.csv
```

### Ingest (demo embedder)
```bash
./target/release/ingest --input docs.jsonl --dim 768 --seed 12345 --normalize --out vectors.csv
```

## Source layout
```
src/
├─ lib.rs
├─ metric.rs       # L2/Cosine + unified scoring
├─ types.rs        # Hit, stable_top_k, etc.
├─ seed.rs         # SplitMix64
├─ flat.rs         # FlatIndex (exact)
├─ kmeans.rs       # Seeded k-means++
├─ ivf.rs          # IVF-Flat
├─ pq.rs           # ProductQuantizer
├─ opq.rs          # OPQ-P (perm / pca / pca-perm)
├─ ivfpq.rs        # IVF-PQ (ADC + OPQ-P + refine)
├─ par.rs          # parallel_map_indexed helper
├─ eval.rs         # recall, Wilson bound, helpers
├─ persist.rs      # (feature) save/load index
├─ header.rs       # index header + fingerprint material
├─ embed.rs        # toy embedder for ingest demo
bin/
├─ sweep.rs
├─ freeze.rs
└─ ingest.rs
profiles/
└─ default.toml    # commented template for freeze/sweep
```

## Testing
- Unit and integration tests:
  ```bash
  cargo test -q
  ```
- Deterministic performance gate with defaults (build + run `freeze`):
  ```bash
  make default-freeze
  ```

## Troubleshooting
- Rebuild in release mode if binaries are missing: `cargo build --release` or `make freeze-bin`.
- Ensure the Makefile defaults match your hardware; lower `NPROBE`/`REFINE` if hitting timeouts.
- Determinism issues? Check seeds in `profiles/default.toml` or pass `--seed-*` flags explicitly.
- Large datasets may require more memory when `--store-vecs` is enabled; disable it to trade speed for RAM.

## Contributing
- Start with the provided profile template at `profiles/default.toml` and document any new presets.
- Prefer reproducible runs: keep seeds fixed in examples and CI gates.
- Use `cargo fmt`/`cargo clippy` locally (if available) and run `cargo test -q` before sending a PR.
- For performance regressions, capture `make default-freeze` output and relevant `sweep` CSVs.

## License
Dual-licensed under MIT or Apache-2.0. See `MIT_LICENSE.md` and `APACHE_LICENSE.md`.