# Live writes: staged plan

Design: "IVF-PQ live insert and delete design" (project doc). This file tracks
what is built and what comes next. Each stage is one PR with its tests first.

Agreed constraints (Brinn, 2026-10-21):
- At most 10 memories per server tick; server tick 60 Hz, 80 Hz ceiling.
- Short-term window of 1 hour in RAM, index budget 8 GB.
- Query p95 under 2 ms on the query threads; writes run on a separate writer thread.
- Same binary and CPU for determinism; live box is a Xeon E-2286G, build target `x86-64-v3`.
- Long-term ("real") memories live in PostgreSQL and are read with SQL at query time,
  not indexed here. They are committed every 12 hours.

## Stage 1: write path on the current layout (this PR)

`src/live.rs`, plain functions over `IvfPqIndex`:

| Op | Effect | Cost |
| --- | --- | --- |
| `Insert` | assign, residual, encode exactly as `build`, append | nlist x dim + 2^nbits x dim multiply-adds |
| `Upsert` | delete if live, then append a new row | as insert |
| `Delete` | tombstone via the id directory | O(1) |
| `Expire` | kill rows with `ts < cutoff` from each list's head, capped per call | O(rows walked) |
| `Compact` | rewrite one list without dead rows, repoint directory | O(list) |

Supporting changes in `src/ivfpq.rs`: `tags` is always present, lists carry
`head` and `dead`, the index carries the id directory (`dir`) and `max_ts`,
`build` scatters rows in (timestamp, input) order so lists are time-sorted,
and search starts each list at `head`.

Tests: `tests/live_writes.rs` (16) and `tests/live_alloc.rs` (no allocation on
insert/delete/upsert after `reserve`).

Measured insert cost on the dev container, one thread, release build
(`insert_latency`, ignored by default):

| N | dim | nlist | m | scalar scan p50 / p99 | 4-wide scan p50 / p99 |
| --- | --- | --- | --- | --- | --- |
| 100k | 128 | 256 | 32 | 52 us / 95 us | 26 us / 51 us |
| 100k | 256 | 1024 | 32 | 192 us / 468 us | 101 us / 269 us |

The centroid scan now uses the 4-wide kernel (`ivfpq::nearest_centroid`,
shared with `build`'s assignment; same scores bit for bit, same tie rule).
At 10 inserts per tick, dim 256 / nlist 1024 costs ~1 ms of the writer
thread per tick.

## Stage 2: separate query threads

Append-only row arena in fixed-capacity segments (rows never move), plus a
per-tick published view (`Arc`) of the small mutable metadata: tombstones,
row counts, list heads. Query threads read the view from the end of the
previous tick; the writer never blocks them. Blocks freed by compaction or
expiry are reused only after every view that can see them is dropped.
Tests: view isolation under loom, arena under Miri, no early reuse.

## Stage 3: drift and background rebuild

Built in new modules, `src/drift.rs` and `src/rebuild.rs`; the only change
to stage 1 code is the `live::last_encoded` stats hook.

The model (centroids, OPQ, PQ codebooks) is frozen per epoch; centroids
never move live.

`drift`: per-list f64 sums accumulated in op order by `drift::record`:
quantisation error ||r - decode(code)||^2 (r = OPQ(v - c), read from the
writer's scratch, so no row is encoded twice), residual norm ||r||^2, insert
counts, and removed rows (deletes, upsert replacements, expiries).
`drift::baseline` measures the same sums on the training rows right after a
build. `drift::check` (O(nlist)) returns the values and a `fired` bitmask:

| Signal | Measure | Default threshold |
| --- | --- | --- |
| `QERR` | mean qerr per insert this epoch / baseline mean | 1.5 |
| `RNORM` | mean residual norm per insert / baseline mean | 1.5 |
| `IMBALANCE` | largest live list / mean live list | 4.0 |
| `CHURN` | (inserts + removed this epoch) / live rows at epoch start | 0.5 |

Thresholds are a `Thresholds` value (`default_thresholds()` gives the above).
It also has `min_inserts` (default 256): the two ratios are not checked
before that many inserts in the epoch, because a handful of rows moves the
mean too much. On the test data the in-distribution ratios peak at 1.08 and
1.02 after 256 inserts; shifted content reaches 12.3 and 3.8.

`rebuild`, plain data plus free functions:

| Function | Thread | Does |
| --- | --- | --- |
| `start(idx)` | writer | epoch 0 state (`Live`): index, scratch, `seq`, epoch, base seed, baseline, stats |
| `write(st, op)` | writer | `live::apply` + `drift::record` (+ carry while a rebuild is pending); caller logs `LogOp::Write(op)` |
| `snapshot(st)` | writer | live rows in ascending id order (raw vectors, ts, tags), `max_ts`; starts the carry |
| `rebuild(snap, base_seed, threads)` | worker | seed `epoch_seed(base_seed, epoch + 1)`, existing deterministic build, tags and `max_ts` restored, new baseline |
| `catch_up(r, carry)` | writer | applies the carried ops not yet applied; any split gives the same result |
| `swap(st, r)` | writer | refuses unless caught up; returns `LogOp::SwapModel { epoch, snapshot_seq, model_hash }` for the log |
| `replay(base, log, threads)` | any | snapshots where `SwapModel` entries say, rebuilds, checks the hash, swaps |

The carry: a capped `Expire` walks lists in layout order, so re-running it
on the new layout kills different rows than the live index killed and
reported. Catch-up therefore applies effects, not ops: an expiry carries as
a `Delete` of each id it killed, `Compact` (storage only) and refused ops
carry nothing, inserts, upserts and deletes carry as they are. Refusals agree
on both indexes because the live ids and `max_ts` agree.

The new model depends only on the snapshot (the log up to `snapshot_seq`),
never on when the worker finishes. Where the swap lands is the writer's
choice and is recorded in the log; `replay` of that log reproduces the
state byte for byte. Lists stay time-sorted: build sorts rows by
(ts, input order).

Tests (`tests/live_drift.rs`, 11 + 1 timing aid): signals fire at the same op
(319) on every run, including on another thread, for shifted content and
never fire for 40% more training-distribution rows; thresholds are
parameters; the baseline equals the stats of inserting the training rows,
with and without OPQ;
rebuilds from one snapshot match (model hash and fingerprint) for 1 and 5
threads and on a worker vs inline, with a capped expiry during the rebuild
leaving exactly the live index's ids; replaying a log with one or two
`SwapModel` entries equals the live run for three swap timings; a wrong
logged hash and an early swap are refused.

Recall@10 against an exact `FlatIndex` (20k rows, dim 32, nlist 32, nprobe 4,
refine 64): epoch 0 0.7075; after the oldest half expired and the same
number of shifted rows arrived, 0.5817 (0.2845 on shifted queries); after
the rebuild and swap, 0.8295. Higher than epoch 0 because each distribution
now holds half the rows.

Snapshot + rebuild of 100k rows on the dev container (4 threads, release,
`rebuild_time`, ignored by default):

| N | dim | nlist | m | snapshot (writer) | rebuild (worker) |
| --- | --- | --- | --- | --- | --- |
| 100k | 128 | 256 | 32 | 36 ms | 9.0 s |
| 100k | 256 | 1024 | 32 | 82 ms | 102 s |

The rebuild is dominated by the scalar coarse k-means (25 iterations over
up to nlist x 256 rows) and the encode; same first speed-up as the insert path.

Open:
- Rebuild thrash: while a shift is still arriving, each new model is
  trained on part of it, so the signals fire again soon after every swap
  (every ~320 ops in the test). Needs a policy: a cooldown, hysteresis, or
  a minimum number of ops per epoch. The tests cap rebuilds at 2.
- Snapshot cost is O(live rows) on the writer thread. With the stage 2
  arena (rows never move) the snapshot can be a view handed to the worker.
- While a rebuild is pending, `write` clones each carried op (allocates).
- Swap position: now "when the writer sees the worker is done", recorded
  in the log. A fixed lag after the snapshot would make the live run itself
  independent of timing, at the cost of blocking the writer if the worker is late.

## Stage 4: durability

Op log on disk, fsync once a second (up to 1 s of loss accepted), periodic
snapshots, recovery = snapshot + log tail, full rebuild as fallback.

## Stage 5: real memories to PostgreSQL

`Insert` gains a server-set `real` flag. Real memories stay searchable in a
small pending buffer until the 12-hour commit to SQL, then leave RAM.
LEGENDARY is a separate flag that never expires in SQL.

## Test data

Tests use seeded synthetic vectors. A crawl4ai script will write a versioned
fixture of realistic memory text plus embeddings, committed to the repo, so
tests never touch the network. Waiting on: which sources to crawl and which
embedding model to use.

## Open

- Event ids (pending RiftForged orchestrator).
- Embedding model, dim and metric.
- Real average memories per tick.
- Tag matching: any-bit vs all-bits (recommend a per-query choice).
