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

| N | dim | nlist | m | p50 | p99 |
| --- | --- | --- | --- | --- | --- |
| 100k | 128 | 256 | 32 | 50 us | 107 us |
| 100k | 256 | 1024 | 32 | 319 us | 525 us |

At 10 inserts per tick, dim 256 / nlist 1024 costs ~3.2 ms of the writer
thread per tick. The centroid scan is scalar; using the existing 4-wide dot
kernel there is the first speed-up to try.

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

Built on branch `claude/live-oplog` in three new modules (no new
dependencies; CRC32 is in-repo). `src/live.rs` and `src/ivfpq.rs` are
unchanged; `src/opq.rs` gained two read accessors for the snapshot.

| Module | What |
| --- | --- |
| `oplog` | record and segment-header format, CRC32, encode/decode |
| `snapshot` | whole-index snapshot file, atomic write, verified read |
| `durable` | writer (`create`, `append`, `apply_logged`, `end_tick`, `sync`, `snapshot`), `recover`, `resume` |

Directory layout: `log-{start_seq:020}.log` segments and
`snap-{seq:020}.snap` snapshots. Seq 1 is the first logged op; "state at
S" is the index after ops 1..=S.

Log segment: 36-byte header (magic `RAOPLOG\0`, version, `start_seq`, the
index fingerprint at `start_seq`, CRC32), then records `len u32 | crc32
u32 | seq u64, tick u64, kind u8, body`, little-endian. A record counts
only if its length fits, its CRC matches, its body parses to exactly
`len` bytes and its seq is the previous + 1; the first one that fails ends
the log, so a torn or corrupt tail is dropped, never mis-parsed. Every
segment boundary is checked against the header fingerprint.

Writer, per tick: `apply_logged` (logs only ops the index accepted), then
`end_tick(now)`: one `write` of the tick's records, and `fdatasync` once
`sync_every` (default 1 s) has passed. A process crash loses at most the
unflushed tick; a power loss at most `sync_every`.

Snapshot: 56-byte header (magic `RASNAP\0\0`, version, seq, tick,
fingerprint, payload length, payload CRC32, header CRC32), then model +
lists + `max_ts`. The id directory is rebuilt from the lists on load
(every row with `tomb == 0`), not stored. Written to `.tmp`, fsynced,
renamed, directory fsynced. Load refuses it unless both CRCs, the length,
`live::check_invariants` and the fingerprint check out. Each snapshot
starts a new log segment; the newest `keep_snapshots` (default 2)
snapshots are kept, and the log from the oldest of them on, so a corrupt
newest snapshot still recovers from the one before. Until two snapshots
exist the whole log is kept.

Recovery (`recover`, read-only): newest snapshot that verifies + the log
from its seq; else the whole log on the caller's base model (the index the
log was created from, checked by fingerprint); else `NoState` and the
fallback is a full rebuild from source data. `resume` then cuts the bad
tail off, sets bad snapshots and unreachable segments aside as
`.corrupt`, and reopens the log at the recovered seq.

**Stage 2 note:** `snapshot::put_list` / `get_list` are the only code that
knows the list layout. The segmented arena changes them (and bumps
`SNAPSHOT_VERSION`); the header, atomic write, checks and the op log stay.
The snapshot currently blocks the writer thread for the whole write; with
stage 2 it should be written from a published view on another thread.

Tests: `tests/live_durability.rs` (12): every op variant round-trips (bytes
and file); every truncation and every flipped byte of the last record is
dropped; seq gaps stop replay; kill after tick T recovers the last flushed
record (process crash) or the last fsynced one (power loss, file cut at the
fsync point plus a torn fragment) with the live fingerprint; fsync
interval; snapshot round-trip incl. OPQ and no raw vectors, flipped bytes
refused; snapshot + tail equals full replay; corrupt snapshot falls back to
full replay, wrong base refused; corrupt newest snapshot falls back to the
older one, pruning; crash between snapshot and new segment.

Measured on the dev container, release build (`durability_timing`, ignored
by default), 100k built rows + 100k logged inserts, 10 per tick:

| dim | nlist | m | append | log | full replay of 100k ops | snapshot write / read |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 256 | 32 | 195k ops/s | 56.5 MB | 6.0 s | 1.3 s / 0.6 s (114 MB) |
| 256 | 1024 | 32 | 154k ops/s | 108 MB | 19.4 s | 1.0 s / 0.8 s (218 MB) |

Replay is dominated by the inserts themselves (same cost as live, see
stage 1); decoding the log alone is 0.5 to 1.0 s. Recovery time is bounded
by the snapshot interval: at 600 ops/s, a snapshot every 5 minutes leaves at
most ~180k ops to replay.

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
