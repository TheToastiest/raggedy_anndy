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

Per-list stats (quantisation error, residual norm, list imbalance, churn) on
every insert. Crossing a threshold starts a rebuild on the shared worker from
a snapshot, which then catches up from the op log and swaps in with a logged
`SwapModel` op. Tests: rebuild reproducible, swap replays.

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
