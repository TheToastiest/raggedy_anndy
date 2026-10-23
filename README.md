# raggedy\_anndy

Deterministic ANN for RAG in Rust. Exact Flat, IVF‑Flat, and **IVF‑PQ with OPQ‑P**, all with seeded training, stable tie‑breaks, and build fingerprints so results are reproducible across runs and machines.

> Goal: ship profiles that hit **recall\@k ≥ 0.90** (by Wilson 95% lower bound) under a tight p95 budget.

---

## What’s in the box

**Indexes**

* **Flat (Exact)** for Cosine or L2
* **IVF‑Flat** (coarse k‑means + exact refine)
* **IVF‑PQ** (ADC on residuals + optional exact re‑rank)

  * **OPQ‑P** (learned permutation / PCA‑perm) to balance subspace variance
  * Modes: `perm`, `pca`, `pca-perm`

**Determinism**

* Seeded **k‑means++** and PQ training
* **Stable top‑k** by (score desc, id asc)
* **Build fingerprint** over params/codebooks/centroids/list ids

**CLI tools**

* `sweep` – grid search over `nprobe`/`refine` (and PQ knobs) → CSV + summary
* `freeze` – single fixed profile; hard gates on lb95/recall/p95; CI‑friendly
* `ingest` – toy JSONL→CSV embedder (for demos/tests)

**Quality metrics**

* Recall\@k vs. Flat baseline
* Wilson 95% lower bound
* p50/p90/p95/p99 latency (per‑query), QPS

**Extras**

* Optional per‑id **tags** with `search_with_filter` masks
* Tombstones + deterministic compaction
* (Opt‑in) index **persist/load** helpers

---

## File tree (high level)

```
raggedy_anndy/
├─ Cargo.toml
├─ src/
|   ├─ lib.rs
|   ├─ metric.rs        # L2/Cosine + unified scoring
|   ├─ types.rs         # Hit, stable_top_k, etc.
|   ├─ seed.rs          # SplitMix64
|   ├─ flat.rs          # FlatIndex (exact)
|   ├─ kmeans.rs        # Seeded k‑means++
|   ├─ ivf.rs           # IVF‑Flat
|   ├─ pq.rs            # ProductQuantizer
|   ├─ opq.rs           # OPQ‑P (perm / pca / pca‑perm)
|   ├─ ivfpq.rs         # IVF‑PQ (ADC + OPQ‑P + refine)
|   ├─ par.rs           # parallel_map_indexed helper
|   ├─ eval.rs          # recall, Wilson bound, helpers
|   ├─ persist.rs       # (feature) save/load index
|   ├─ header.rs        # index header + fingerprint material
|   ├─ embed.rs         # toy embedder for ingest demo
└─ bin/
    ├─ sweep.rs
    ├─ freeze.rs
    └─ ingest.rs
```

---

## Install

Requires stable Rust.

```bash
git clone https://github.com/TheToastiest/raggedy_anndy.git
cd raggedy_anndy
cargo build --release
```

Run tests:

```bash
cargo test -q
```

---

## Binaries & usage

### 1) `freeze` – fixed profile, gates, and p95

Single‑run evaluator for CI and reproducible bake‑offs.

**Key flags (subset):**

* Dataset: `--n`, `--dim`, `--metric {cosine|l2}`, `--k`, `--queries`, `--warmup`
* Seeds: `--seed-data`, `--seed-queries`, `--seed-kmeans`
* Backend: `--backend {ivf-flat|ivf-pq}`
* IVF: `--nlist`, `--nprobe`, `--refine`
* PQ: `--m`, `--nbits`, `--iters`, `--opq`, `--opq-mode {perm|pca|pca-perm}`, `--opq-sweeps`, `--store-vecs`
* Gates: `--min-lb95`, `--min-recall`, `--max-p95-ms`
* Threads: `--threads` (query‑level), `--intra` (intra‑search, if supported)

**Example (a passing config on a laptop):**

```bash
./target/release/freeze --backend ivf-pq --n 50000 --dim 256 --metric cosine --k 5 --nlist 1536 --nprobe 906 --refine 2000 --m 128 --nbits 8 --iters 60 --opq --opq-mode pca-perm --opq-sweeps 8 --queries 200 --warmup 5 --seed-data 42 --seed-queries 999 --seed-kmeans 7 --min-lb95 0.90 --min-recall 0.90 --max-p95-ms 40 --threads 2 --intra 8
```

This prints recall, lb95, p50/p90/p95/p99, QPS, and fails non‑zero if any gate is violated. It also checks build+search determinism.

---

### 2) `sweep` – grid search → CSV

Grid over `nprobe` × `refine` (and PQ knobs when using IVF‑PQ). Writes a CSV and prints a compact table.

**Example:**

```bash
./target/release/sweep --backend ivf-pq --n 50000 --dim 256 --metric cosine --k 10 --nlist 1536 --nprobe 704,840,906 --refine 1200,1600,2000 --m 128 --nbits 8 --iters 60 --opq --opq-mode pca-perm --opq-sweeps 8 --queries 200 --warmup 25 --seed-data 42 --seed-queries 999 --seed-kmeans 7 --csv results.csv
```

**CSV columns:** `metric,n,dim,k,nlist,nprobe,refine,recall,lb95,p50_us,p90_us,p95_us,p99_us,qps,build_det,build_ms,eval_ms`

To enforce quality in sweep, use `--target-lb 0.90 --enforce` (and optionally `--require-all`).

---

### 3) `ingest` – demo embedder for JSONL

Reads `{id, text}` JSONL and writes `id,vec...` CSV with a seeded random embedder (for demos/tests only).

```bash
./target/release/ingest --input docs.jsonl --dim 768 --seed 12345 --normalize --out vectors.csv
```

---

## Tuning cheat‑sheet

* **Cosine + unit‑norm** is the default for text embeddings; enables stable OPQ/PQ behavior.
* **nprobe** ↑ → recall ↑ (latency ↑). Good first knob.
* **refine** ↑ → recall ↑ via exact re‑rank size. Keep `refine ≳ k`.
* **m (PQ sub‑quantizers)**: more subspaces → higher ADC accuracy (slower encode, slightly bigger codes).
* **OPQ‑P**: set `--opq --opq-mode pca-perm --opq-sweeps 6..12` for balanced subspace variance.
* **store‑vecs**: if true, exact re‑rank uses in‑index floats (more RAM, faster refine). If false, you can plug a remote store later.

Typical good starting point (Cosine, dim=256, N≈50k, k∈{5,10}):

* `nlist=1536`, `nprobe≈800–900`, `refine≈1500–2000`, `m=128`, `nbits=8`, `iters≈60–80`, `opq-mode=pca-perm`, `opq-sweeps≈6–10`.

---

## Reproducibility model

* All trainers and data/query generators are seeded.
* Fixed f32 math path; stable sort by (score desc, id asc).
* Deterministic tie‑breaks and list insertion order.
* `fingerprint()` covers params, centroids, PQ codebooks, and per‑list ids.

### Build target

`.cargo/config.toml` pins x86_64 Linux builds to `-C target-cpu=x86-64-v3`
(AVX, AVX2, BMI1/2, F16C, FMA, LZCNT, MOVBE). Other targets are unaffected.

* **Why a fixed target:** determinism is promised for the *same binary and
  CPU features*. A fixed baseline means the binary built on the dev laptop
  (WSL Debian) and the live server (Xeon E‑2286G, Coffee Lake: AVX2 + FMA,
  no AVX‑512) is the same code on both.
* **Why not `target-cpu=native`:** it compiles for whatever the build host
  has. A binary built on an AVX‑512 machine can crash (illegal instruction)
  or take different code paths on the server, and two hosts would produce
  two different binaries.
* **FMA:** Rust never contracts `a*b+c` into a fused multiply‑add on its
  own, so enabling the FMA feature does not change float results. Fingerprints
  and top‑k id/score bits were checked identical against the default
  `x86-64` target.

Check a machine supports it before running a v3 binary there:

```bash
grep -m1 -o avx2 /proc/cpuinfo && grep -m1 -o fma /proc/cpuinfo   # both must print
rustc --print cfg -C target-cpu=x86-64-v3 | grep target_feature  # what v3 enables
```

The server supports x86‑64‑v3. The dev laptop's support is **not yet
confirmed**; run the check above on it. To build for an older CPU, override
with `RUSTFLAGS="-C target-cpu=x86-64" cargo build --release` (env
`RUSTFLAGS` takes precedence over the config file).

---

## Status

**Shipping**

* Flat, IVF‑Flat, IVF‑PQ (ADC) with optional exact refine
* OPQ‑P (`perm`, `pca`, `pca‑perm`)
* Sweep + Freeze CLIs with Wilson lb95 gates
* Deterministic build/search + fingerprints
* Tags, tombstones, compact; (feature) persist/load

**Planned**

* HNSW with deterministic insertion
* Sparse fusion for RAG
* Optional remote refine (e.g., RAM KV cache)

---

## License

Dual‑licensed under MIT or Apache‑2.0.

* See `LICENSE-MIT` and `LICENSE-APACHE` for details.
* You may not use any file except in compliance with one of the licenses.

---

## Acknowledgments

Inspired by FAISS‑style indexing and common RAG retrieval patterns. Implementation is original to this repository.
