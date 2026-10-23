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

Built in new modules, `src/drift.rs`, `src/rebuild.rs` and `src/gate.rs`;
the only change to stage 1 code is the `live::last_encoded` stats hook.

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

Cadence (Brinn, 2026-10-23: "start with ops, move to hysteresis"):
`Thresholds::min_ops_per_epoch` (default 4096). `rebuild::due(st, sig, th)`
starts a rebuild only when a signal fired, no rebuild is pending, and at
least that many writes were applied since the last model decision (swap or
reject; epoch 0 counts from the start). Why 4096: 16 x `min_inserts`, so the
next epoch's ratios are means over ~4k rows; a shift still arriving puts
4k ops into the new model before it can ask again; and at the 10 writes per
tick ceiling it is >= 6.8 s at 60 Hz, about one 100k-row rebuild, so the
worker does not fall behind. Counted in ops, not time, so a trigger lands at
the same op on every run. In the test, a continuing shift that re-triggered
at ops 319, 640, 652 without the floor now triggers at 4096 and 8203
(= previous decision + 1 + 4096).

Triggers are not logged: the decision entries name their `snapshot_seq`, and
replay snapshots there. So the trigger policy can change without touching
the log format. Hysteresis later = a second `Thresholds` for "clear" and an
`armed` flag in `Live`, set when a signal fires and cleared when all values
drop under the clear thresholds; `due` then reads `armed` instead of `fired`.

`rebuild`, plain data plus free functions:

| Function | Thread | Does |
| --- | --- | --- |
| `start(idx)` | writer | epoch 0 state (`Live`): index, scratch, `seq`, epoch, base seed, baseline, stats |
| `write(st, op)` | writer | `live::apply` + `drift::record` (+ carry while a rebuild is pending); caller logs `LogOp::Write(op)` |
| `snapshot(st)` | writer | live rows in ascending id order (raw vectors, ts, tags), `max_ts`; starts the carry |
| `rebuild(snap, base_seed, threads)` | worker | seed `epoch_seed(base_seed, epoch + 1)`, existing deterministic build, tags and `max_ts` restored, new baseline |
| `catch_up(r, carry)` | writer | applies the carried ops not yet applied; any split gives the same result |
| `gate::evaluate(snap, r, params, game, threads)` | worker | swap gate on the rebuild as built (before catch-up); returns a `Gate` |
| `swap(st, r, gate)` | writer | refuses unless the gate accepted and caught up; returns `LogOp::SwapModel { epoch, snapshot_seq, model_hash, gate }` |
| `reject(st, r, gate)` | writer | refuses unless the gate rejected; ends the carry, keeps the old model; returns `LogOp::RejectModel { .. }` (same fields) |
| `replay(base, log, threads)` | any | snapshots where decision entries say, rebuilds, checks the hash, re-runs the gate and checks it equals the logged one, then swaps or rejects |

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

Swap gate (Brinn: "model finish -> test against baseline -> deploy if
genuinely new or better"). `src/gate.rs`, a pure function of the snapshot
and the candidate, run on the worker:

- Current model: `snap.model` (the live model with no rows) with every
  snapshot row inserted in (ts, id) order: same codes, same results as the
  live index at the snapshot. Candidate: the rebuild as built.
- Eval set: the `recent` (4096) newest snapshot rows; `queries` (256) of them,
  picked with a seed derived from the candidate's epoch seed, are the
  queries. Truth: exact `FlatIndex` over all snapshot rows; recall@`k` (10).
- Drift: the recent rows inserted into an empty copy of each model, measured
  against that model's own baseline (`drift::check`); churn ignored.
- Accept if candidate recall >= current recall + `min_gain` (0.02: the
  standard error of one recall over 2560 trials is <= 0.01, and both models
  answer the same queries), or the current model's QERR/RNORM/IMBALANCE fire
  on the recent rows and the candidate's do not. Otherwise reject.
- `Gate` (plain data) holds the parameters, hit counts, both `Signals`, the
  decision and its reason (`RecallGain`, `DriftCleared`, `NoGain`,
  `GameCheckFailed`). It goes in the `SwapModel` or `RejectModel` entry.
  Replay recomputes it from the logged parameters and checks equality.
- Game-level hook: `GameCheck { NotRun, Pass, Fail }`, an argument to
  `evaluate`. A caller-side test (e.g. replay a recorded "Best Time" run on
  both models and compare) runs on the worker and passes its verdict; `Fail`
  rejects, the verdict is logged, and replay takes it from the log. No game
  metric is defined here.
- After a reject the drift stats carry on and `due` waits another
  `min_ops_per_epoch` writes before the next try.

Gate numbers on the test data (20k rows, dim 32, nlist 32, recall@10 on 256
recent rows):

| Case | current | candidate | drift fired (cur -> cand) | decision |
| --- | --- | --- | --- | --- |
| half the rows shifted | 0.3723 | 0.8590 | QERR, RNORM -> none | accept, RecallGain |
| 3000 more training-distribution rows | 0.7078 | 0.7223 | none -> none | reject, NoGain |
| quarter shifted (6000 of 24500 rows) | 0.3395 | 0.3242 | QERR, RNORM -> QERR, RNORM | reject, NoGain |
| continuing shift, 1st try (op 4096) | 0.4055 | 0.3949 | QERR, RNORM -> QERR, RNORM | reject, NoGain |
| continuing shift, 2nd try (op 8203) | 0.3422 | 0.9121 | QERR, RNORM -> none | accept, RecallGain |

A model trained on a small share of new content is not better on that
content: k-means and the codebooks still spend themselves on the old
distribution. The gate catches this and the next try, 4096 ops later,
wins.

Tests (`tests/live_drift.rs`, 17 + 1 timing aid): signals fire at the same op
(319) on every run, including on another thread, for shifted content and
never fire for 40% more training-distribution rows; thresholds are
parameters; the baseline equals the stats of inserting the training rows,
with and without OPQ;
rebuilds from one snapshot match (model hash and fingerprint) for 1 and 5
threads and on a worker vs inline, with a capped expiry during the rebuild
leaving exactly the live index's ids; replaying a log with one or two
decision entries equals the live run for three swap timings; a wrong
logged hash and an early swap are refused. Cadence: a continuing shift waits
`min_ops_per_epoch` between rebuilds, same positions on every run. Gate: a
half-shifted candidate is accepted, the same `Gate` on a worker and for 1 or
5 threads; no-better candidates (more training data, a quarter shift) are
rejected and the old model, stats and fingerprint stay; `swap` refuses a
rejecting gate; replay of a log with a reject and an accept reproduces the
fingerprint and model hash, and a tampered logged gate is refused; the gate
runs on the worker thread (checked by thread id); `decide` rules incl. the
game veto.

Recall@10 against an exact `FlatIndex` (20k rows, dim 32, nlist 32, nprobe 4,
refine 64): epoch 0 0.7075; after the oldest half expired and the same
number of shifted rows arrived, 0.5817 (0.2845 on shifted queries); after
the rebuild and swap, 0.8295. Higher than epoch 0 because each distribution
now holds half the rows.

Snapshot + rebuild + gate of 100k rows on the dev container (4 threads,
release, `rebuild_time`, ignored by default):

| N | dim | nlist | m | snapshot (writer) | rebuild (worker) | gate (worker) |
| --- | --- | --- | --- | --- | --- | --- |
| 100k | 128 | 256 | 32 | 54 ms | 15.7 s | 13.0 s |
| 100k | 256 | 1024 | 32 | 108 ms | 98 s | 31.8 s |

The rebuild is dominated by the scalar coarse k-means (25 iterations over
up to nlist x 256 rows) and the encode; same first speed-up as the insert path.
The gate is the serial re-encode of every snapshot row under the current
model (one insert each) plus 256 exact scans over the snapshot (threaded).

Open:
- Hysteresis on top of `min_ops_per_epoch` (see Cadence).
- The gate's re-encode of the current model is serial; it could encode rows
  in parallel and scatter in (ts, id) order. With the stage 2 arena the
  worker could also read the live codes instead of re-encoding.
- The game-level check (`GameCheck`) has no implementation yet.
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
