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

## Stage 2: separate query threads (built)

One writer thread applies the ops of tick T while query threads read the
view published at the end of tick T-1. `IvfPqIndex` stays the
single-threaded type; the writer/view pair sits beside it and both search
through the same scoring code (`ivfpq::search_rows`).

- `src/arena.rs`: `Block`, a fixed-capacity run of rows of one list (ids,
  timestamps, tags, PQ codes, raw vectors as SoA columns), allocated whole
  so rows never move. Rows are appended in slot order; a written slot is
  never written again until `reset`, which needs `&mut` (no other holder).
  Readers only form references below an Acquire load of the block's
  `filled` count. All `unsafe` of this stage is in this file.
- `src/views.rs`: `Writer` and `View`. Row `r` of a list is in
  `blocks[r / block_rows]`; `block_rows` defaults to
  `views::DEFAULT_BLOCK_ROWS` = 256 (still a parameter). The small mutable metadata per list (block
  table, row count, head, dead count, tombstones) is `ListMeta`: the writer
  owns one set, each `View` owns a copy, and rows past a view's count are
  invisible to it. Deletes and expiry change only the writer's copy. The id
  directory is writer-only.
- Publish (`views::publish`, once per tick): the writer owns a fixed pool
  of views, `views::DEFAULT_VIEWS` (8) at start and grown by
  `views::reserve(w, rows_per_list, views)` (views = max query threads
  holding a view at once + 2). It takes the newest pooled view no query
  holds (`Arc::get_mut`), copies in only the lists changed since that
  view's own tick, gives it the new seq and `store`s it in an
  `arc_swap::ArcSwap` (crate `arc-swap`, pinned `=1.7.1`). Queries get the
  newest view with `views::current`, one lock-free `load_full`: no mutex
  anywhere on the query path. `ArcSwap::store` pays every reader's
  outstanding debt on the old value before it returns, so `Arc::get_mut`
  on a view that is not in the slot sees an exact count; the view in the
  slot is never unique (the pool holds it too).
- Publish does not allocate after `reserve`: `reserve` sizes the pool and
  fits every free view's block tables and tombstones to the writer's
  capacity, and copies reuse those buffers (`clone_from`). If every view
  is held, publish is deferred: nothing is written or allocated, the
  previous view stays current, and the returned `Published { seq,
  current, deferred }` says so (`ArenaStats::deferred` counts them). The
  tick counter still advances, so the next publish that goes through
  shows every tick up to its own, and a view's seq is always "state after
  tick seq". Writer results are unaffected. A view that was held while a
  `reserve` grew the writer is fitted by a later `reserve` (call it every
  tick; it is cheap when nothing grows); until then publish prefers views
  that fit and grows one only if none does (`ArenaStats::grown`, 0 in all
  tests).
- Reclamation: compaction retires a list's old blocks. A retired block is
  reset and reused only when its `Arc` is unique, i.e. no view, held by a
  query or pooled by the writer, can see it. A compaction marks every
  pooled view dirty; at the next publish where a dirty view is free, it
  drops its handles to the blocks of the lists changed since its seq
  (which it would recopy on reuse anyway), so idle views do not pin
  retired blocks. Compaction keeps the free room `reserve` made in the
  list, so inserts after it stay allocation-free.
- Compaction policy (`views::compactions_due`, Brinn: "rolling dead-row
  window"). Once per tick the writer calls it and applies the returned
  `Op::Compact`s through `views::apply` like any other op, so they are in
  the log and replay through `live::apply` to the same bytes. A list with
  dead rows (deleted, or expired before `head`) is due when its oldest
  dead row died `window_ticks` or more ticks ago (the rolling window), or
  its dead rows exceed `max_dead_share` of its rows, or reach
  `max_dead_rows`. At most `max_per_tick` lists per tick, oldest dead row
  first, then lower list index. Parameters are the plain struct
  `CompactPolicy`; defaults `COMPACT_DEFAULT` = 3600 ticks (1 min at
  60 Hz), 0.25, 512 rows, 2 per tick. Its only state is
  `Writer::dead_since` (per list, the tick its oldest dead row died), set
  by delete/expire and cleared by compaction, so the op log determines
  it. This is what frees the blocks of expired rows: expiry only advances
  `head` (as in stage 1). Note: a capped `Expire` counts the dead rows it
  walks, so after a compaction the same capped Expire reaches further;
  both paths do the same, and the log replays exactly.
- Same ops, same results: `views::apply` returns the same `Applied` as
  `live::apply`, and `views::to_index(view)` has the same `fingerprint` as
  the `IvfPqIndex` after the same ops. Inserts stay allocation-free after
  `views::reserve`.

Tests: `tests/live_views.rs` (13): view 0 equals the index for block sizes
1 to 4096; writer equals `live::apply` tick by tick (results, search bits,
fingerprint), with and without raw vectors; an old view returns identical
bits after every op of the next tick (deletes of its top hits, upserts,
inserts, expiry of half the rows, compaction of every list) and while a
writer thread runs 6000 ops; retired blocks are not reused while an old
view lives and are reused after; a writer thread plus 4 query threads
(some holding old views) match the single-threaded reference for every
view seq (with a pool smaller than the readers can hold, so some
publishes are deferred); publish defers while every view is held, held
views keep their bits, and the next free publish shows the newest tick;
default block size 256; the compaction policy picks the same lists for the
same log, exactly the first `max_per_tick` due lists by (oldest dead row,
list), never changes search results (against an index that never
compacts), and its logged compactions replay to the same fingerprint; the
per-tick cap and the window (a lone dead row is collected exactly
`window_ticks` later); expiring 90% of rows plus policy compaction cuts the
lists' blocks more than 4x, all retired blocks are reclaimed, and new
inserts reuse them. `tests/live_alloc.rs` adds the writer's
insert/delete/upsert, and ticks with publish (zero allocations, including
the exhausted pool and inserts after compaction).
Miri (data races, weak memory) passes on the arena unit tests and on a
tiny writer-plus-readers run:
`cargo +nightly miri test --lib arena` and
`MIRIFLAGS=-Zmiri-deterministic-floats cargo +nightly miri test --test live_views tiny_`
(Miri otherwise perturbs `exp`, which decay uses). Loom is not used: it
needs its own sync types through the crate, and `--cfg loom` also switches
tokio into loom mode.

Measured on the dev container (4 cores, shared, noisy), release, N=100k,
dim 128, nlist 512, nprobe 16, refine 200, m 32, k=10 (first version, mutex
slot):

| | query p50 | query p95 |
| --- | --- | --- |
| `IvfPqIndex::search` | 123-167 us | 185-244 us |
| view, writer idle | 143-150 us | 190-214 us |
| view, writer thread inserting 10 rows per 60 Hz tick | 137-159 us | 186-248 us |

Writer tick (10 inserts + publish): p50 ~530 us, p95 ~820 us; publish
alone p95 25-33 us. `latency_n100k` gates unchanged within noise (query
p95 150-180 us at k=10, 233 us at k=5; before 215 / 263 us).
`cargo test --release --test live_views concurrent_query_latency -- --ignored --nocapture`
reprints these (`RA_BLOCK_ROWS` sets the block size, default 256).

After arc-swap, the fixed pool and the policy, A/B against the mutex
version in the same session (load average 6-17 on 4 cores, so absolute
numbers are higher than above; three interleaved runs each):

| | query p50 | query p95 | tick busy p50 | publish p95 |
| --- | --- | --- | --- | --- |
| mutex slot (before) | 190-201 us | 319-344 us | 838-891 us | 44-58 us |
| arc-swap, fixed pool | 203-206 us | 277-352 us | 845-876 us | 30-50 us |

Taking the oldest free view instead of the newest (to cycle the pool) cost
~100 us per publish (its metadata is cold), hence newest-first plus the
dirty-view sweep.

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

Index side built in `src/real_memory.rs` (branch `claude/live-real-memory`);
the SQL side is not started. The game server decides what is real
(endgame-only: meteors, black holes, specialization relics, legendary
achievements) and flags it at insert time. No automatic importance or
touch thresholds.

The gap it closes: a real memory leaves the 1-hour window but is not
committed for up to 12 hours. Each real memory goes into the short-term
index like any row and also into a pending buffer (`Pending`: SoA columns,
raw f32 vector, exact Flat scoring) until `commit` takes it.

| Item | Shape |
| --- | --- |
| Flags | `flags: u8`, own field, never a tag bit. `REAL = 1` required; `LEGENDARY = 2` only with `REAL`; other bits rejected (`BadFlags`). |
| Insert | `insert_real(idx, s, buf, id, ts, tags, flags, v)`, or `RealOp::InsertReal` through `real_memory::apply`. Same checks as `live::insert`, plus the id must not be pending or handed. `Op::Insert` is unchanged, so stage-1 callers and logs keep working. |
| Ops | `RealOp { Live(Op), InsertReal, UpsertReal, Commit, Ack }`. `Live` is `live::apply` on the index; for a pending or handed id, `Delete`/`Upsert` also update the buffer (below) and `Insert` is refused. |
| Delete | Pending id: its pending row is dropped and is never committed. Works after the 1-hour expiry too (index row already gone). Handed id: a `Delete` record is appended to the outbox at once, after the id's `Put`. |
| Upsert | Pending id: the pending row is replaced (removed, the new one appended; its ts is the newest), so the commit carries the latest version. Handed id: the new version becomes pending again and the next commit appends a newer `Put`. `Op::Upsert` carries no flags, so the row keeps its flags: LEGENDARY is never dropped by accident. `RealOp::UpsertReal { flags }` sets the flags explicitly (can add or drop LEGENDARY, or promote a normal row to real; `REAL` still required). |
| Search | `search(idx, buf, q, k, tag, now, lambda)`: index top `k + pending` minus pending ids, plus every pending row scored exactly with the same score and decay as the index re-rank; top-k by (score desc, id asc), each id once. Handed rows are not in RAM. |
| Commit | `RealOp::Commit` / `commit(buf, outbox)`: drains the buffer into the outbox as `Put` records in (ts, id) order, marks each id handed, returns how many it appended. Does not touch the short-term index. |
| Ack | `RealOp::Ack { through_seq }`: the SQL side has every record up to `through_seq`. Drops them from the outbox and forgets those handed ids (unless a later `Put` is unacked). Ack of a `seq` not yet emitted is `AckAhead`. |
| Outbox | `Outbox { records: Vec<RealRecord>, next_seq }`, `RealRecord { seq, kind: Put \| Delete, id, ts, tags, flags, vec }`, plain data. `seq` comes from one counter in `Pending` (`next_seq`), so a replay from a snapshot emits the same `seq`s and the outbox skips any it already took: a commit re-run after a crash appends nothing. |

The server owns the clock (Brinn, 2026-10-23): the 12-hour commit and the
ack are ops the server sends, applied in log order. Nothing in
`real_memory.rs` or `live.rs` reads a clock or runs a timer; tests pin both
the behaviour (48 h of inserts and expiry without `Commit` commit nothing)
and the source (no `SystemTime`, `Instant`, `sleep`, `spawn`).

"Handed" means committed and not yet acked, and it is log state, not a
peek at the outbox: the ack is an op so that replay decides delete/upsert
of a committed id the same way every time. After the ack this side forgets
the id; deleting or changing an acked long-term memory is the SQL side's
job.

Why a separate op, not a flag on `Op::Insert`: adding a field would change
every stage-1 caller and the op shape stages 2-4 are building on; the real
path also needs the pending buffer, which `live::apply` does not take.

Tests: `tests/live_real_memory.rs` (22): survives expiry until commit; gone
from RAM and once in the outbox after; deterministic commit order and
contents; a commit re-run after a crash appends nothing; LEGENDARY
independent of tag bits; bad flags and index errors change nothing; merged
search deterministic, no repeated ids, tag filter on pending rows, pending
scores bit-equal to the index re-rank; delete of a pending id (before and
after expiry) never commits it; upsert of a pending id commits the latest
version and keeps LEGENDARY; `UpsertReal` sets flags; delete of a handed id
appends a `Delete` after its `Put`; upsert of a handed id commits a newer
`Put`; ack ahead refused; a seeded churn log (inserts, deletes, upserts,
expiry, commits, acks) replays identically and a tail replayed from a
snapshot appends nothing; no commit without the op; no clock in the write
path.

What the SQL side needs to decide and build (no Postgres dependency here;
pgvector not yet confirmed):
- Schema and client. A record holds `seq u64`, `kind`, `id u64` (Postgres
  has no unsigned bigint: store as `bigint` bit-cast or `numeric`), `ts u64`
  (unit is the server's), `tags u64`, `flags u8`, `vec f32 x dim`.
- Apply records in `seq` order: `Put` = insert or update the row
  (`ON CONFLICT (id) DO UPDATE ... WHERE excluded.seq > row.seq`), `Delete`
  = delete by id. Skip any `seq` at or below the last applied one; then send
  `Ack { through_seq }` through the server's log.
- Until acked, the outbox must survive a crash (or be rebuilt by replaying
  the log, which stage 4 must do for `RealOp`).
- Expiry of non-LEGENDARY rows in SQL (how long?).

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
- Stage 5: stage 4's log must record `RealOp` (or `Op` gains its variants at
  merge time) so recovery rebuilds the pending buffer.
