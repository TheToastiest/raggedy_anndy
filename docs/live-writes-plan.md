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

Per-list stats (quantisation error, residual norm, list imbalance, churn) on
every insert. Crossing a threshold starts a rebuild on the shared worker from
a snapshot, which then catches up from the op log and swaps in with a logged
`SwapModel` op. Tests: rebuild reproducible, swap replays.

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
