//! Durability for the live system: op log, background snapshots, recovery.
//!
//! These tests pin the guarantees from docs/live-writes-plan.md, stage 4:
//! - every record kind (plain writes incl. policy compactions, real-memory
//!   ops, commit, ack, model swap and reject, end of tick) round-trips bit
//!   for bit, in bytes and through a log file;
//! - a tick counts only as a whole: any cut or flipped byte inside the last
//!   tick drops that tick entirely, never part of it;
//! - a writer killed after tick T (process crash, or power loss with the
//!   file cut at the last fsync) recovers the exact live state of the last
//!   durable tick: rows and views, compaction state, model epoch and drift
//!   stats, real-memory buffer and outbox;
//! - snapshot + log tail gives the same system as replaying the whole log,
//!   across a `SwapModel` and a `RejectModel`;
//! - a corrupt snapshot is skipped and recovery falls back to an older
//!   snapshot or to the full log on the base model;
//! - snapshots are written on another thread from a published view and
//!   never change what the writer does;
//! - the snapshot cadence defaults to 60 s of 60 Hz ticks, counted in
//!   ticks, not read from a clock.

mod common;
use common::*;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use raggedy_anndy::drift::{Signals, Thresholds};
use raggedy_anndy::durable::{self, DurableParams, LogWriter, RecoverError};
use raggedy_anndy::gate::{self, GameCheck, Gate, GateParams, GateReason};
use raggedy_anndy::live::Op;
use raggedy_anndy::oplog::{self, Body, Record, SEGMENT_HEADER_LEN};
use raggedy_anndy::real_memory::{RealOp, LEGENDARY, REAL};
use raggedy_anndy::rebuild::{self, LogOp};
use raggedy_anndy::seed::SplitMix64;
use raggedy_anndy::snapshot;
use raggedy_anndy::system::{self, State, System};
use raggedy_anndy::views::{self, CompactPolicy};
use raggedy_anndy::{IvfPqIndex, Metric};

const DIM: usize = 32;
const NLIST: usize = 16;
const M: usize = 8;
const BLOCK: usize = 64;
const TICK: Duration = Duration::from_micros(16_667); // 60 Hz

/// Compacts often enough that every run logs policy-issued `Compact`s.
const POLICY: CompactPolicy = CompactPolicy { window_ticks: 30, max_dead_share: 0.25, max_dead_rows: 64, max_per_tick: 2 };

fn temp_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("raggedy_dur_{name}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn base_index(n: usize, seed: u64, opq: bool) -> IvfPqIndex {
    let ds = uniform_sphere(n, DIM, seed);
    let rows: Vec<(u64, Vec<f32>, u64)> = (0..n).map(|i| (ds.ids[i], row(&ds, i).to_vec(), i as u64)).collect();
    let mut p = ivfpq_params(NLIST, 4, 64, M, 7);
    p.use_opq = opq;
    IvfPqIndex::build(Metric::Cosine, DIM, &rows, p)
}

fn random_unit(rng: &mut SplitMix64) -> Vec<f32> {
    let mut v: Vec<f32> = (0..DIM).map(|_| gauss(rng)).collect();
    normalize(&mut v);
    v
}

fn params(every: u64) -> DurableParams { DurableParams { sync_every: Duration::from_secs(1), keep_snapshots: 2, snapshot_every_ticks: every } }

/// Up to 10 ops for one tick: mostly inserts, plus upserts, deletes,
/// capped expiry, real memories (insert, upsert with flags, delete),
/// and now and then a commit or an ack of the oldest outbox records.
fn tick_ops(rng: &mut SplitMix64, sys: &System, tick: u64, next_id: &mut u64) -> Vec<RealOp> {
    let ts = 10_000 + tick;
    let n = 1 + rng.gen_range(10);
    let mut ops = Vec::with_capacity(n);
    for _ in 0..n {
        let any = |rng: &mut SplitMix64, next: u64| rng.gen_range(next as usize + 1) as u64;
        ops.push(match rng.gen_range(24) {
            0..=9 => { *next_id += 1; RealOp::Live(Op::Insert { id: *next_id, ts, tags: rng.next_u64() & 0xff, vec: random_unit(rng) }) }
            10..=11 => RealOp::Live(Op::Upsert { id: any(rng, *next_id), ts, tags: 1, vec: random_unit(rng) }),
            12..=14 => RealOp::Live(Op::Delete { id: any(rng, *next_id) }),
            15 => RealOp::Live(Op::Expire { before_ts: (tick * 9).min(ts), max_rows: 40 }),
            16..=17 => { *next_id += 1; RealOp::InsertReal { id: *next_id, ts, tags: 2, flags: if rng.gen_range(3) == 0 { REAL | LEGENDARY } else { REAL }, vec: random_unit(rng) } }
            18 => RealOp::UpsertReal { id: any(rng, *next_id), ts, tags: 4, flags: REAL, vec: random_unit(rng) },
            19 => RealOp::Live(Op::Compact { list: rng.gen_range(NLIST) as u32 }),
            20 if tick % 7 == 0 => RealOp::Commit,
            21 => match sys.st.outbox.records.get(rng.gen_range(3)) {
                Some(r) => RealOp::Ack { through_seq: r.seq },
                None => RealOp::Ack { through_seq: u64::MAX },
            },
            _ => RealOp::Live(Op::Delete { id: any(rng, *next_id) }),
        });
    }
    ops
}

/// Live state at the end of one tick.
#[derive(Clone)]
struct At { fp: u64, view_fp: u64, st: State, epoch_hash: u64 }

fn at(sys: &System) -> At {
    let view = views::current(&views::latest(&sys.w));
    At { fp: system::fingerprint(sys), view_fp: views::fingerprint(&view), st: sys.st.clone(), epoch_hash: rebuild::model_hash(&sys.w.model) }
}

/// The writer thread of a server: a system, its log, the clock it is handed.
struct Run {
    sys: System, w: LogWriter, now: Instant, rng: SplitMix64, next_id: u64,
    /// Live state after every tick, and the tick each fsync made durable.
    at: BTreeMap<u64, At>, synced_tick: u64,
    snaps: Vec<durable::SnapDone>,
}

fn start(dir: &Path, base: &IvfPqIndex, seed: u64, every: u64) -> Run {
    let sys = system::start(base, BLOCK);
    let now = Instant::now();
    let w = durable::create(dir, &sys, params(every), now).unwrap();
    let mut at_map = BTreeMap::new();
    at_map.insert(0, at(&sys));
    Run { sys, w, now, rng: SplitMix64::new(seed ^ 0xd00d), next_id: base.dir.len() as u64, at: at_map, synced_tick: 0, snaps: Vec::new() }
}

fn end_tick(r: &mut Run) {
    r.now += TICK;
    let te = durable::end_tick(&mut r.w, &mut r.sys, &POLICY, r.now).unwrap();
    if te.synced { r.synced_tick = te.tick; }
    if let Some(d) = te.snapshot_done { r.snaps.push(d.unwrap()); }
    r.at.insert(te.tick, at(&r.sys));
}

fn run_ticks(r: &mut Run, ticks: u64) {
    for _ in 0..ticks {
        let tick = r.sys.w.seq;
        for op in tick_ops(&mut r.rng, &r.sys, tick, &mut r.next_id) { let _ = durable::write(&mut r.w, &mut r.sys, &op); }
        end_tick(r);
    }
}

fn assert_ok(sys: &System) {
    if let Err(e) = system::check_invariants(sys) { panic!("invariant broken: {e}"); }
}

/// The recovered system is the live one at `tick`: fingerprint, the view
/// queries get, model, epoch, drift stats, buffer and outbox.
fn assert_is(sys: &System, want: &At, what: &str) {
    let got = at(sys);
    assert_eq!(got.st, want.st, "{what}: epoch / buffer / outbox state");
    assert_eq!(got.view_fp, want.view_fp, "{what}: view");
    assert_eq!(got.epoch_hash, want.epoch_hash, "{what}: model");
    assert_eq!(got.fp, want.fp, "{what}: system fingerprint");
    assert_ok(sys);
}

fn files_with_ext(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = fs::read_dir(dir).unwrap().map(|e| e.unwrap().path())
        .filter(|p| p.extension().map_or(false, |e| e == ext)).collect();
    v.sort();
    v
}

fn flip_byte(path: &Path, at_from_end: u64) {
    let mut b = fs::read(path).unwrap();
    let i = b.len() - at_from_end as usize;
    b[i] ^= 0x5a;
    fs::write(path, b).unwrap();
}

/// Offset a few bytes into the next EndTick at or after `from`: every
/// entry of that tick is in the file, its end marker is torn.
fn inside_next_end_tick(bytes: &[u8], from: usize) -> usize {
    let mut at = from;
    loop {
        let (rec, used) = oplog::decode_record(&bytes[at..]).expect("a whole tick follows");
        if matches!(rec.body, Body::EndTick { .. }) { return at + 3; }
        at += used;
    }
}

/// Rewrite every segment in `dir` with `f` applied to each entry, ticks
/// re-closed with matching EndTicks.
fn rewrite_log(dir: &Path, f: impl Fn(&Body) -> Body) {
    for seg in files_with_ext(dir, "log") {
        let bytes = fs::read(&seg).unwrap();
        let (seq0, tick0, _) = oplog::decode_segment_header(&bytes).unwrap();
        let (ticks, _) = oplog::decode_ticks(&bytes[SEGMENT_HEADER_LEN..], seq0, tick0);
        let mut out = bytes[..SEGMENT_HEADER_LEN].to_vec();
        let mut seq = seq0;
        for t in &ticks {
            let start = out.len();
            for e in &t.entries { oplog::encode_record(&mut out, e.seq, e.tick, &f(&e.body)); seq = e.seq; }
            let crc = oplog::crc32(&out[start..]);
            oplog::encode_record(&mut out, seq, t.tick, &Body::EndTick { count: t.entries.len() as u32, crc });
        }
        fs::write(&seg, out).unwrap();
    }
}

fn recover(dir: &Path, base: &IvfPqIndex) -> durable::Recovered {
    durable::recover(dir, Some(base), BLOCK, 2).unwrap()
}

// ------------------------------------------------------------ record format

fn signals(x: f64) -> Signals { Signals { qerr_ratio: x, rnorm_ratio: -0.0, imbalance: f64::MAX, churn: f64::MIN_POSITIVE, inserts: u64::MAX, fired: 0b1011 } }

fn odd_gate(accept: bool) -> Gate {
    Gate {
        params: GateParams { queries: 7, recent: usize::MAX, k: 10, min_gain: 0.02, th: Thresholds { qerr_ratio: 1.5, rnorm_ratio: -0.0, imbalance: 4.0, churn: 1e-300, min_inserts: 256, min_ops_per_epoch: u64::MAX } },
        game: if accept { GameCheck::Pass } else { GameCheck::Fail }, queries: 7, trials: 70, current_hits: 1, candidate_hits: u32::MAX,
        current: signals(1.25), candidate: signals(f64::EPSILON), accept,
        reason: if accept { GateReason::DriftCleared } else { GateReason::GameCheckFailed },
    }
}

fn every_body() -> Vec<Body> {
    let mut rng = SplitMix64::new(1);
    vec![
        Body::Write(RealOp::Live(Op::Insert { id: 7, ts: 100, tags: 0b1010, vec: random_unit(&mut rng) })),
        Body::Write(RealOp::Live(Op::Upsert { id: u64::MAX, ts: u64::MAX, tags: u64::MAX, vec: vec![-0.0, f32::MIN_POSITIVE, 1.5, f32::MAX] })),
        Body::Write(RealOp::Live(Op::Delete { id: 0 })),
        Body::Write(RealOp::Live(Op::Expire { before_ts: 99, max_rows: 3 })),
        Body::Write(RealOp::Live(Op::Expire { before_ts: u64::MAX, max_rows: usize::MAX })),
        Body::Write(RealOp::Live(Op::Compact { list: u32::MAX })),
        Body::Write(RealOp::Live(Op::Insert { id: 1, ts: 0, tags: 0, vec: Vec::new() })),
        Body::Write(RealOp::InsertReal { id: 9, ts: 5, tags: 3, flags: REAL | LEGENDARY, vec: random_unit(&mut rng) }),
        Body::Write(RealOp::UpsertReal { id: 9, ts: 6, tags: 0, flags: 0xff, vec: vec![-0.0; 3] }),
        Body::Write(RealOp::Commit),
        Body::Write(RealOp::Ack { through_seq: u64::MAX }),
        Body::Model(LogOp::SwapModel { epoch: 3, snapshot_seq: 41, model_hash: 0xdead_beef_0123_4567, gate: odd_gate(true) }),
        Body::Model(LogOp::RejectModel { epoch: u64::MAX, snapshot_seq: 0, model_hash: 1, gate: odd_gate(false) }),
        Body::EndTick { count: 12, crc: 0xffff_0000 },
    ]
}

#[test]
fn every_record_kind_round_trips_bit_for_bit() {
    let bodies = every_body();
    let mut buf = Vec::new();
    for (i, b) in bodies.iter().enumerate() { oplog::encode_record(&mut buf, i as u64 + 1, 1000 + i as u64, b); }
    let mut at = 0usize;
    for (i, b) in bodies.iter().enumerate() {
        let (rec, used) = oplog::decode_record(&buf[at..]).expect("record decodes");
        assert_eq!(rec, Record { seq: i as u64 + 1, tick: 1000 + i as u64, body: b.clone() });
        // Bits, not just values: re-encoding gives the same bytes (-0.0 ==
        // 0.0 would hide a sign loss in a vector or a gate's f64s).
        let mut again = Vec::new();
        oplog::encode_record(&mut again, rec.seq, rec.tick, &rec.body);
        assert_eq!(again, buf[at..at + used], "kind {i}");
        at += used;
    }
    assert_eq!(at, buf.len());
    // A NaN in a logged gate survives too (bytes; NaN != NaN as a value).
    let mut g = odd_gate(true);
    g.current.qerr_ratio = f64::from_bits(0x7ff8_dead_beef_0001);
    let mut a = Vec::new();
    oplog::encode_record(&mut a, 1, 1, &Body::Model(LogOp::SwapModel { epoch: 1, snapshot_seq: 0, model_hash: 2, gate: g }));
    let (rec, _) = oplog::decode_record(&a).unwrap();
    let mut b = Vec::new();
    oplog::encode_record(&mut b, rec.seq, rec.tick, &rec.body);
    assert_eq!(a, b);
    // `encode_write` is `encode_record` of `Body::Write` without the clone.
    for body in &bodies {
        if let Body::Write(op) = body {
            let (mut x, mut y) = (Vec::new(), Vec::new());
            oplog::encode_write(&mut x, 5, 6, op);
            oplog::encode_record(&mut y, 5, 6, body);
            assert_eq!(x, y);
        }
    }
}

#[test]
#[should_panic(expected = "plain write is Body::Write")]
fn a_plain_write_is_never_logged_as_a_model_entry() {
    oplog::encode_record(&mut Vec::new(), 1, 1, &Body::Model(LogOp::Write(Op::Delete { id: 1 })));
}

/// Encode `ticks` (lists of entry bodies) as whole ticks after (seq 0, tick 0).
fn encode_ticks(ticks: &[Vec<Body>]) -> (Vec<u8>, Vec<usize>) {
    let (mut buf, mut ends, mut seq) = (Vec::new(), Vec::new(), 0u64);
    for (t, entries) in ticks.iter().enumerate() {
        let start = buf.len();
        for b in entries { seq += 1; oplog::encode_record(&mut buf, seq, t as u64 + 1, b); }
        let crc = oplog::crc32(&buf[start..]);
        oplog::encode_record(&mut buf, seq, t as u64 + 1, &Body::EndTick { count: entries.len() as u32, crc });
        ends.push(buf.len());
    }
    (buf, ends)
}

#[test]
fn every_cut_or_flip_inside_the_last_tick_drops_the_whole_tick() {
    let bodies: Vec<Body> = every_body().into_iter().filter(|b| !matches!(b, Body::EndTick { .. })).collect();
    let (buf, ends) = encode_ticks(&[bodies[..3].to_vec(), Vec::new(), bodies[3..].to_vec()]);
    let (ticks, valid) = oplog::decode_ticks(&buf, 0, 0);
    assert_eq!((ticks.len(), valid), (3, buf.len()));
    assert_eq!(ticks[1].entries.len(), 0);
    assert_eq!(ticks[2].entries.len(), bodies.len() - 3);
    let keep = ends[1];
    for cut in keep..buf.len() {
        let (ticks, valid) = oplog::decode_ticks(&buf[..cut], 0, 0);
        assert_eq!((ticks.len(), valid), (2, keep), "cut at {cut}");
    }
    for i in keep..buf.len() {
        let mut b = buf.clone();
        b[i] ^= 0x01;
        let (ticks, valid) = oplog::decode_ticks(&b, 0, 0);
        assert_eq!((ticks.len(), valid), (2, keep), "flipped byte {i}");
    }
    // A wild length prefix must not allocate or read past the end.
    let mut b = buf[..keep].to_vec();
    b.extend_from_slice(&u32::MAX.to_le_bytes());
    b.extend_from_slice(&[0; 12]);
    assert_eq!(oplog::decode_ticks(&b, 0, 0).1, keep);
}

#[test]
fn ticks_must_follow_on_and_end_markers_must_match() {
    let d = |id| Body::Write(RealOp::Live(Op::Delete { id }));
    let good = |entries: &[Body]| encode_ticks(&[vec![d(1)], entries.to_vec()]);
    let (_, ends) = good(&[d(2)]);
    let one = ends[0];
    // Each tampered second tick: well-formed records, wrong as a tick.
    let mut cases: Vec<(&str, Vec<u8>)> = Vec::new();
    let (b, _) = good(&[]);
    let mut wrong_count = b[..one].to_vec();
    oplog::encode_record(&mut wrong_count, 1, 2, &Body::EndTick { count: 1, crc: 0 });
    cases.push(("count", wrong_count));
    let mut wrong_crc = b[..one].to_vec();
    let s = wrong_crc.len();
    oplog::encode_record(&mut wrong_crc, 2, 2, &d(2));
    let crc = oplog::crc32(&wrong_crc[s..]) ^ 1;
    oplog::encode_record(&mut wrong_crc, 2, 2, &Body::EndTick { count: 1, crc });
    cases.push(("crc", wrong_crc));
    let mut gap = b[..one].to_vec();
    let s = gap.len();
    oplog::encode_record(&mut gap, 3, 2, &d(2)); // seq gap
    let crc = oplog::crc32(&gap[s..]);
    oplog::encode_record(&mut gap, 3, 2, &Body::EndTick { count: 1, crc });
    cases.push(("seq gap", gap));
    let mut skip = b[..one].to_vec();
    oplog::encode_record(&mut skip, 1, 3, &Body::EndTick { count: 0, crc: 0 }); // tick 2 missing
    cases.push(("tick gap", skip));
    let mut mixed = b[..one].to_vec();
    let s = mixed.len();
    oplog::encode_record(&mut mixed, 2, 3, &d(2)); // entry of another tick
    let crc = oplog::crc32(&mixed[s..]);
    oplog::encode_record(&mut mixed, 2, 2, &Body::EndTick { count: 1, crc });
    cases.push(("entry tick", mixed));
    for (what, b) in cases {
        let (ticks, valid) = oplog::decode_ticks(&b, 0, 0);
        assert_eq!((ticks.len(), valid), (1, one), "{what}");
    }
}

#[test]
fn every_record_kind_round_trips_through_a_log_file() {
    let dir = temp_dir("kinds");
    let base = base_index(1000, 2, false);
    let mut r = start(&dir, &base, 2, 0);
    let mut rng = SplitMix64::new(3);
    let ops = vec![
        RealOp::Live(Op::Insert { id: 50_000, ts: 5000, tags: 3, vec: random_unit(&mut rng) }),
        RealOp::Live(Op::Upsert { id: 50_000, ts: 5001, tags: 4, vec: random_unit(&mut rng) }),
        RealOp::Live(Op::Upsert { id: 50_001, ts: 5002, tags: 0, vec: random_unit(&mut rng) }),
        RealOp::Live(Op::Delete { id: 10 }),
        RealOp::Live(Op::Expire { before_ts: 300, max_rows: 40 }),
        RealOp::Live(Op::Expire { before_ts: 300, max_rows: usize::MAX }),
        RealOp::Live(Op::Compact { list: 3 }),
        RealOp::InsertReal { id: 60_000, ts: 5003, tags: 1, flags: REAL | LEGENDARY, vec: random_unit(&mut rng) },
        RealOp::InsertReal { id: 60_001, ts: 5003, tags: 1, flags: REAL, vec: random_unit(&mut rng) },
        RealOp::UpsertReal { id: 50_001, ts: 5004, tags: 2, flags: REAL, vec: random_unit(&mut rng) },
        RealOp::Commit,
        RealOp::Ack { through_seq: 0 },
        RealOp::Live(Op::Delete { id: 60_001 }), // handed: a Delete record in the outbox
    ];
    for op in &ops { durable::write(&mut r.w, &mut r.sys, op).unwrap(); }
    end_tick(&mut r); // plus the policy's compactions of the expired lists
    let policy_compacts = r.w.last_seq - ops.len() as u64;
    assert!(policy_compacts > 0, "policy issued no compaction");
    assert!(!r.sys.st.outbox.records.is_empty() && r.sys.st.buf.handed.len() == 1);
    durable::sync(&mut r.w, r.now).unwrap();
    let last = r.w.last_seq;
    drop(r.w);
    let rec = recover(&dir, &base);
    assert_eq!((rec.seq, rec.tick, rec.replayed, rec.ticks), (last, 1, last, 1));
    assert_is(&rec.sys, &r.at[&1], "kinds");
    let _ = fs::remove_dir_all(&dir);
}

// ------------------------------------------------------------ crash simulation

/// Kill the writer after tick T: the ops of tick T+1 were applied but never
/// reached the OS. A process crash keeps every tick handed to the OS; a
/// power loss also drops whatever was written after the last fsync.
#[test]
fn kill_after_tick_t_recovers_the_exact_live_state() {
    for (case, kill_tick) in [37u64, 61, 150].into_iter().enumerate() {
        let dir = temp_dir(&format!("kill_{case}"));
        let base = base_index(1500, 6, case == 1);
        let mut r = start(&dir, &base, 6 + case as u64, 0);
        run_ticks(&mut r, kill_tick);
        assert!(r.sys.st.outbox.next_seq > 0 && !r.sys.st.buf.ids.is_empty(), "case {case}: real memories exercised");
        // Half a tick T+1, never flushed.
        let tick = r.sys.w.seq;
        for op in tick_ops(&mut r.rng, &r.sys, tick, &mut r.next_id) { let _ = durable::write(&mut r.w, &mut r.sys, &op); }
        let (synced_len, synced_tick) = (r.w.synced_len, r.synced_tick);
        assert!(synced_tick < kill_tick && r.w.written_seq < r.w.last_seq);
        let seg = r.w.seg_path.clone();
        drop(r.w); // no final flush, no fsync

        // Process crash: everything handed to the OS survives.
        let rec = recover(&dir, &base);
        assert_eq!(rec.tick, kill_tick, "case {case}");
        assert_is(&rec.sys, &r.at[&kill_tick], &format!("case {case} crash"));

        // Power loss: the file is cut at the last fsync, plus a torn
        // fragment of the next tick (all its entries, part of its EndTick).
        let bytes = fs::read(&seg).unwrap();
        fs::write(&seg, &bytes[..inside_next_end_tick(&bytes, synced_len as usize)]).unwrap();
        let rec = recover(&dir, &base);
        assert_eq!(rec.tick, synced_tick, "case {case}");
        assert!(rec.dropped_bytes > 0);
        assert_is(&rec.sys, &r.at[&synced_tick], &format!("case {case} power loss"));
        let _ = fs::remove_dir_all(&dir);
    }
}

#[test]
fn a_torn_tick_is_dropped_whole_and_resume_continues() {
    let dir = temp_dir("torn_tick");
    let base = base_index(1500, 5, false);
    let mut r = start(&dir, &base, 5, 0);
    run_ticks(&mut r, 40);
    durable::sync(&mut r.w, r.now).unwrap();
    let before = r.w.written_len as usize;
    // One more tick with many entries.
    let tick = r.sys.w.seq;
    let mut n = 0;
    while n < 8 {
        for op in tick_ops(&mut r.rng, &r.sys, tick, &mut r.next_id) { n += durable::write(&mut r.w, &mut r.sys, &op).is_ok() as usize; }
    }
    end_tick(&mut r);
    durable::sync(&mut r.w, r.now).unwrap();
    let seg = r.w.seg_path.clone();
    drop(r.w);
    let full = fs::read(&seg).unwrap();
    assert_eq!(recover(&dir, &base).tick, tick);

    // Every record boundary inside the tick (whole entries, no EndTick),
    // and cuts inside records: the tick is dropped as a unit.
    let (ticks, _) = oplog::decode_ticks(&full[before..], r.at[&(tick - 1)].st.seq, tick - 1);
    assert_eq!(ticks.len(), 1);
    let mut cuts = Vec::new();
    let mut at = before;
    while let Some((_, used)) = oplog::decode_record(&full[at..]) { at += used; cuts.push(at); cuts.push(at - used / 2); }
    cuts.retain(|&c| c < full.len()); // all but the whole tick
    assert!(cuts.len() >= 16);
    for &cut in &cuts {
        fs::write(&seg, &full[..cut]).unwrap();
        let rec = recover(&dir, &base);
        assert_eq!(rec.tick, tick - 1, "cut at {cut}");
        assert_eq!(rec.dropped_bytes, (cut - before) as u64);
        assert_is(&rec.sys, &r.at[&(tick - 1)], &format!("cut at {cut}"));
    }
    // A flipped byte in the EndTick drops the tick too.
    fs::write(&seg, &full).unwrap();
    flip_byte(&seg, 3);
    let rec = recover(&dir, &base);
    assert_eq!(rec.tick, tick - 1);

    // Resume cuts the bad tick off and appends after the good prefix; the
    // re-run tick replays to the same state as before.
    let mut w = durable::resume(&dir, &rec, params(0), r.now).unwrap();
    let mut sys = rec.sys;
    assert_eq!((w.last_seq, w.last_tick), (r.at[&(tick - 1)].st.seq, tick - 1));
    durable::write(&mut w, &mut sys, &RealOp::Live(Op::Insert { id: 9_000_000, ts: 20_000, tags: 0, vec: random_unit(&mut r.rng) })).unwrap();
    durable::end_tick(&mut w, &mut sys, &POLICY, r.now).unwrap();
    durable::sync(&mut w, r.now).unwrap();
    drop(w);
    let again = recover(&dir, &base);
    assert_eq!((again.tick, again.dropped_bytes), (tick, 0));
    assert_eq!(system::fingerprint(&again.sys), system::fingerprint(&sys));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn fsync_runs_at_the_configured_interval() {
    let dir = temp_dir("interval");
    let base = base_index(1000, 7, false);
    let mut sys = system::start(&base, BLOCK);
    let t0 = Instant::now();
    let p = DurableParams { sync_every: Duration::from_millis(250), ..params(0) };
    let mut w = durable::create(&dir, &sys, p, t0).unwrap();
    let mut syncs = 0;
    for t in 1..=60u64 {
        let _ = durable::write(&mut w, &mut sys, &RealOp::Live(Op::Delete { id: t }));
        if durable::end_tick(&mut w, &mut sys, &POLICY, t0 + TICK * t as u32).unwrap().synced { syncs += 1; }
    }
    assert_eq!(syncs, 4); // 60 ticks at 60 Hz = 1 s: at ticks 15, 30, 45, 60
    assert_eq!(w.written_seq, w.last_seq);
    let _ = fs::remove_dir_all(&dir);
}

// ------------------------------------------------------------ snapshots

#[test]
fn snapshot_file_round_trips_the_whole_system() {
    let dir = temp_dir("snap_rt");
    for (case, opq) in [false, true].into_iter().enumerate() {
        let base = base_index(2000, 8, opq);
        let mut r = start(&dir.join(format!("log{case}")), &base, 8, 0);
        run_ticks(&mut r, 80);
        let cap = system::capture(&r.sys).expect("view is current after end_tick");
        let path = dir.join(format!("case{case}.snap"));
        let out = snapshot::write(&path, &cap, r.w.last_seq, r.w.last_tick, 0xabc).unwrap();
        assert_eq!(out.fp, r.at[&80].fp);
        assert_eq!(out.bytes, fs::metadata(&path).unwrap().len());
        for block_rows in [1, BLOCK, 4096] {
            let snap = snapshot::read(&path, block_rows).unwrap();
            assert_eq!((snap.seq, snap.tick, snap.log_id), (r.w.last_seq, 80, 0xabc));
            assert_is(&snap.sys, &r.at[&80], &format!("case {case} block_rows {block_rows}"));
            let q = random_unit(&mut SplitMix64::new(42));
            let (a, b) = (views::current(&views::latest(&snap.sys.w)), views::current(&views::latest(&r.sys.w)));
            assert_eq!(views::search(&a, &q, 10), views::search(&b, &q, 10));
        }
        assert!(!dir.join(format!("case{case}.snap.tmp")).exists());
        // Any flipped byte is refused, not loaded.
        for back in [1u64, 30, 100, 5000, out.bytes - 70, out.bytes - 20] {
            let mut b = fs::read(&path).unwrap();
            let i = b.len() - back as usize; b[i] ^= 0x10;
            let bad = dir.join("bad.snap");
            fs::write(&bad, b).unwrap();
            assert!(snapshot::read(&bad, BLOCK).is_err(), "case {case}: flip {back} from end accepted");
        }
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn background_snapshots_never_change_writer_results() {
    let dir = temp_dir("bg");
    let base = base_index(3000, 14, false);
    // A: durable, a snapshot every 6 ticks on its own thread, a query
    // thread holding views. B: the same ops on a plain system.
    let mut r = start(&dir, &base, 14, 6);
    let mut b = system::start(&base, BLOCK);
    let stop = Arc::new(AtomicBool::new(false));
    let reader = {
        let (latest, stop) = (views::latest(&r.sys.w), stop.clone());
        std::thread::spawn(move || {
            let mut rng = SplitMix64::new(77);
            let mut n = 0u64;
            while !stop.load(Ordering::Relaxed) {
                let v = views::current(&latest);
                let q = random_unit(&mut rng);
                n += views::search(&v, &q, 5).len() as u64;
            }
            n
        })
    };
    let mut due = Vec::new();
    for _ in 0..240 {
        let tick = r.sys.w.seq;
        for op in tick_ops(&mut r.rng, &r.sys, tick, &mut r.next_id) {
            let ra = durable::write(&mut r.w, &mut r.sys, &op);
            assert_eq!(ra, system::write(&mut b, &op), "tick {tick} op {op:?}");
        }
        views::compactions_due(&POLICY, &b.w, &mut due);
        for op in due.drain(..) { system::write(&mut b, &RealOp::Live(op)).unwrap(); }
        views::publish(&mut b.w);
        end_tick(&mut r);
        assert_eq!(r.at[&tick].fp, system::fingerprint(&b), "tick {tick}");
    }
    stop.store(true, Ordering::Relaxed);
    assert!(reader.join().unwrap() > 0);
    if let Some(d) = durable::wait_snapshot(&mut r.w) { r.snaps.push(d.unwrap()); }
    assert!(r.snaps.len() >= 10, "only {} snapshots", r.snaps.len());
    // Each snapshot holds the state of the tick it was taken at.
    for s in &r.snaps { assert_eq!(s.fp, r.at[&s.tick].fp, "snapshot of tick {}", s.tick); }
    let kept = files_with_ext(&dir, "snap");
    assert_eq!(kept.len(), 2);
    durable::sync(&mut r.w, r.now).unwrap();
    drop(r.w);
    let rec = recover(&dir, &base);
    assert_eq!(rec.snapshot_tick, Some(r.snaps.last().unwrap().tick));
    assert_is(&rec.sys, &r.at[&240], "after background snapshots");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn snapshot_cadence_defaults_to_60_seconds_of_ticks() {
    assert_eq!(durable::SNAPSHOT_EVERY_TICKS, 60 * 60);
    assert_eq!(durable::default_params().snapshot_every_ticks, durable::SNAPSHOT_EVERY_TICKS);
    assert_eq!(durable::default_params().sync_every, Duration::from_secs(1));
    let dir = temp_dir("cadence");
    let base = base_index(600, 15, false);
    let mut sys = system::start(&base, BLOCK);
    let mut now = Instant::now();
    let mut w = durable::create(&dir, &sys, durable::default_params(), now).unwrap();
    let mut started = Vec::new();
    let mut rng = SplitMix64::new(15);
    for t in 1..=7300u64 {
        if t % 100 == 0 { durable::write(&mut w, &mut sys, &RealOp::Live(Op::Insert { id: 1_000_000 + t, ts: 1_000 + t, tags: 0, vec: random_unit(&mut rng) })).unwrap(); }
        now += TICK;
        let te = durable::end_tick(&mut w, &mut sys, &POLICY, now).unwrap();
        assert_eq!(te.tick, t);
        if te.snapshot_started { started.push(t); }
    }
    assert_eq!(started, vec![3600, 7200]);
    durable::wait_snapshot(&mut w).unwrap().unwrap();
    assert_eq!(files_with_ext(&dir, "snap").len(), 2);
    // The cadence is counted in ticks: no clock is read in the durable path.
    for (name, src) in [("durable", include_str!("../src/durable.rs")), ("system", include_str!("../src/system.rs")),
                        ("snapshot", include_str!("../src/snapshot.rs")), ("oplog", include_str!("../src/oplog.rs"))] {
        for clock in ["Instant::now", "SystemTime", "elapsed()"] { assert!(!src.contains(clock), "{name}.rs reads a clock: {clock}"); }
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn corrupt_snapshot_falls_back_to_full_replay() {
    let dir = temp_dir("snap_corrupt");
    let base = base_index(1500, 10, false);
    let mut r = start(&dir, &base, 10, 0);
    run_ticks(&mut r, 30);
    assert!(durable::start_snapshot(&mut r.w, &r.sys).unwrap());
    durable::wait_snapshot(&mut r.w).unwrap().unwrap();
    run_ticks(&mut r, 30);
    durable::sync(&mut r.w, r.now).unwrap();
    drop(r.w);
    let snaps = files_with_ext(&dir, "snap");
    assert_eq!(snaps.len(), 1);
    flip_byte(&snaps[0], 200);

    let rec = recover(&dir, &base);
    assert_eq!(rec.snapshot_tick, None);
    assert_eq!(rec.bad_snapshots, snaps);
    assert_eq!(rec.tick, 60);
    assert_is(&rec.sys, &r.at[&60], "full replay");

    // Without a base model there is nothing to replay onto.
    assert!(matches!(durable::recover(&dir, None, BLOCK, 1), Err(RecoverError::NoState)));
    // A base with a different model is refused by the log id.
    assert!(matches!(durable::recover(&dir, Some(&base_index(1500, 11, false)), BLOCK, 1), Err(RecoverError::NoState)));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn corrupt_newest_snapshot_falls_back_to_older_snapshot_and_rotation_prunes() {
    let dir = temp_dir("snap_older");
    let base = base_index(1500, 12, false);
    let mut r = start(&dir, &base, 12, 0);
    let mut snap_ticks = Vec::new();
    for _ in 0..3 {
        run_ticks(&mut r, 20);
        snap_ticks.push(r.w.last_tick);
        assert!(durable::start_snapshot(&mut r.w, &r.sys).unwrap());
        run_ticks(&mut r, 1); // the writer goes on while it is written
        durable::wait_snapshot(&mut r.w).unwrap().unwrap();
    }
    run_ticks(&mut r, 20);
    durable::sync(&mut r.w, r.now).unwrap();
    let last = r.w.last_tick;
    drop(r.w);

    // keep_snapshots = 2: the first snapshot and the log before the
    // second are gone, the rest is kept.
    let snaps = files_with_ext(&dir, "snap");
    let logs = files_with_ext(&dir, "log");
    assert_eq!(snaps.len(), 2);
    assert_eq!(logs.len(), 2);
    assert_eq!(snaps[0], durable::snap_path(&dir, snap_ticks[1]));
    assert_eq!(logs[0], durable::log_path(&dir, snap_ticks[1]));

    flip_byte(&snaps[1], 50);
    let rec = durable::recover(&dir, None, BLOCK, 1).unwrap();
    assert_eq!(rec.snapshot_tick, Some(snap_ticks[1]));
    assert_eq!(rec.tick, last);
    assert_is(&rec.sys, &r.at[&last], "older snapshot + tail");

    // Resume sets the bad snapshot aside; the next recovery is clean.
    drop(durable::resume(&dir, &rec, params(0), r.now).unwrap());
    let again = recover(&dir, &base);
    assert!(again.bad_snapshots.is_empty());
    assert_is(&again.sys, &r.at[&last], "after resume");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn crash_before_the_snapshot_file_is_written_recovers_across_the_new_segment() {
    let dir = temp_dir("snap_crash");
    let base = base_index(1500, 13, false);
    let mut r = start(&dir, &base, 13, 0);
    run_ticks(&mut r, 25);
    let tick = r.w.last_tick;
    assert!(durable::start_snapshot(&mut r.w, &r.sys).unwrap());
    let done = durable::wait_snapshot(&mut r.w).unwrap().unwrap();
    run_ticks(&mut r, 10);
    durable::sync(&mut r.w, r.now).unwrap();
    drop(r.w);
    // The snapshot never made it; the segment it opened did.
    fs::remove_file(&done.path).unwrap();
    assert!(durable::log_path(&dir, tick).exists());
    let rec = recover(&dir, &base);
    assert_eq!((rec.snapshot_tick, rec.tick), (None, 35));
    assert_is(&rec.sys, &r.at[&35], "two segments, no snapshot");
    // And the other way round: the snapshot is there, its segment is not.
    let dir2 = temp_dir("snap_crash2");
    let mut r = start(&dir2, &base, 13, 0);
    run_ticks(&mut r, 25);
    assert!(durable::start_snapshot(&mut r.w, &r.sys).unwrap());
    durable::wait_snapshot(&mut r.w).unwrap().unwrap();
    drop(r.w);
    fs::remove_file(durable::log_path(&dir2, 25)).unwrap();
    let rec = recover(&dir2, &base);
    assert_eq!((rec.snapshot_tick, rec.tick, rec.replayed), (Some(25), 25, 0));
    assert_is(&rec.sys, &r.at[&25], "snapshot only");
    let mut w = durable::resume(&dir2, &rec, params(0), r.now).unwrap();
    let mut sys = rec.sys;
    durable::end_tick(&mut w, &mut sys, &POLICY, r.now).unwrap();
    durable::sync(&mut w, r.now).unwrap();
    assert!(durable::log_path(&dir2, 25).exists());
    assert_eq!(recover(&dir2, &base).tick, 26);
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::remove_dir_all(&dir2);
}

// ------------------------------------------------------------ model decisions

const D_DIM: usize = 32;
const D_HALF: usize = D_DIM / 2;
const D_N: usize = 20_000;

/// Stage 3's test data: clusters in one half of the dimensions; the
/// shifted distribution uses the other half.
fn half_data(half: usize, n: usize, seed: u64) -> Dataset {
    let c = cluster_centres(64, D_HALF, half as u64 + 1);
    let mut centres = vec![0.0f32; 64 * D_DIM];
    for k in 0..64 { centres[k * D_DIM + half * D_HALF..k * D_DIM + (half + 1) * D_HALF].copy_from_slice(&c[k * D_HALF..(k + 1) * D_HALF]); }
    clustered(n, D_DIM, &centres, 0.08, seed)
}

/// Ten ops per tick from `ops`, then `end_tick`.
fn feed(r: &mut Run, ops: &[RealOp]) {
    for chunk in ops.chunks(10) {
        for op in chunk { durable::write(&mut r.w, &mut r.sys, op).unwrap(); }
        end_tick(r);
    }
}

fn inserts(ds: &Dataset, id0: u64, ts0: u64, real_every: usize) -> Vec<RealOp> {
    (0..ds.ids.len()).map(|i| {
        let (id, ts, vec) = (id0 + i as u64, ts0 + i as u64, row(ds, i).to_vec());
        if real_every > 0 && i % real_every == 0 { RealOp::InsertReal { id, ts, tags: 1, flags: REAL, vec } } else { RealOp::Live(Op::Insert { id, ts, tags: 0, vec }) }
    }).collect()
}

#[test]
fn snapshot_plus_tail_equals_full_replay_across_swap_and_reject() {
    let dir = temp_dir("decisions");
    let ds = half_data(0, D_N, 10);
    let rows: Vec<(u64, Vec<f32>, u64)> = (0..D_N).map(|i| (ds.ids[i], row(&ds, i).to_vec(), i as u64)).collect();
    let base = IvfPqIndex::build(Metric::Cosine, D_DIM, &rows, ivfpq_params(32, 4, 64, M, 7));
    let mut r = start(&dir, &base, 1, 0);
    r.w.params.keep_snapshots = 3;
    let gp = gate::default_gate_params();
    let mut ts = D_N as u64;

    // Some training-distribution traffic, then snapshot S1 (no rebuild pending).
    feed(&mut r, &inserts(&half_data(0, 200, 11), 1_000_000, ts, 7));
    ts += 1000;
    assert!(durable::start_snapshot(&mut r.w, &r.sys).unwrap());
    let s1 = r.w.last_tick;
    durable::wait_snapshot(&mut r.w).unwrap().unwrap();

    // Rebuild 1, vetoed by the game-level check: RejectModel.
    let snap = system::begin_rebuild(&mut r.sys).unwrap();
    assert!(!durable::start_snapshot(&mut r.w, &r.sys).unwrap(), "no snapshot while a rebuild is pending");
    feed(&mut r, &inserts(&half_data(0, 50, 12), 1_100_000, ts, 5));
    ts += 1000;
    let rb = rebuild::rebuild(&snap, r.sys.st.base_seed, 2).unwrap();
    let g = gate::evaluate(&snap, &rb, &gp, GameCheck::Fail, 2).unwrap();
    assert_eq!(g.reason, GateReason::GameCheckFailed);
    durable::reject(&mut r.w, &mut r.sys, &rb, g).unwrap();
    end_tick(&mut r);

    // Half the content shifts: rebuild 2 wins the gate: SwapModel.
    durable::write(&mut r.w, &mut r.sys, &RealOp::Live(Op::Expire { before_ts: (D_N / 2) as u64, max_rows: usize::MAX })).unwrap();
    end_tick(&mut r);
    feed(&mut r, &inserts(&half_data(1, D_N / 2, 52), 2_000_000, ts, 97));
    ts += D_N as u64;
    let snap = system::begin_rebuild(&mut r.sys).unwrap();
    feed(&mut r, &inserts(&half_data(1, 40, 53), 3_000_000, ts, 9));
    ts += 1000;
    let mut rb = rebuild::rebuild(&snap, r.sys.st.base_seed, 2).unwrap();
    let g = gate::evaluate(&snap, &rb, &gp, GameCheck::NotRun, 2).unwrap();
    assert!(g.accept, "{g:?}");
    rebuild::catch_up(&mut rb, r.sys.carry.as_ref().unwrap()).unwrap();
    durable::swap(&mut r.w, &mut r.sys, rb, g).unwrap();
    end_tick(&mut r);
    assert_eq!(r.sys.st.epoch, 1);
    feed(&mut r, &inserts(&half_data(1, 60, 54), 4_000_000, ts, 6));
    feed(&mut r, &[RealOp::Commit]);
    durable::sync(&mut r.w, r.now).unwrap();
    let last = r.w.last_tick;
    let mut w = r.w;

    // S1 + tail: the tail holds both decisions, replayed by rebuilding.
    let rec = recover(&dir, &base);
    assert_eq!((rec.snapshot_tick, rec.tick, rec.decisions), (Some(s1), last, 2));
    assert_is(&rec.sys, &r.at[&last], "S1 + tail");
    assert_eq!(rec.sys.st.epoch, 1);

    // A snapshot after the swap: recovery needs no rebuild.
    run_ticks_with(&mut w, &mut r.sys, &mut r.now, &mut r.at, 3);
    assert!(durable::start_snapshot(&mut w, &r.sys).unwrap());
    durable::wait_snapshot(&mut w).unwrap().unwrap();
    run_ticks_with(&mut w, &mut r.sys, &mut r.now, &mut r.at, 3);
    durable::sync(&mut w, r.now).unwrap();
    let last = w.last_tick;
    drop(w);
    let rec2 = recover(&dir, &base);
    assert_eq!((rec2.decisions, rec2.tick), (0, last));
    assert_is(&rec2.sys, &r.at[&last], "S2 + tail");

    // No snapshot at all: the whole log on the base model agrees.
    for p in files_with_ext(&dir, "snap") { fs::remove_file(p).unwrap(); }
    let full = recover(&dir, &base);
    assert_eq!((full.snapshot_tick, full.decisions, full.tick), (None, 2, last));
    assert_is(&full.sys, &r.at[&last], "full replay");

    // A tampered gate in the log is refused on replay.
    rewrite_log(&dir, |b| match b {
        Body::Model(LogOp::RejectModel { epoch, snapshot_seq, model_hash, gate }) => {
            let mut g = *gate;
            g.current_hits += 1;
            Body::Model(LogOp::RejectModel { epoch: *epoch, snapshot_seq: *snapshot_seq, model_hash: *model_hash, gate: g })
        }
        b => b.clone(),
    });
    assert!(matches!(durable::recover(&dir, Some(&base), BLOCK, 2), Err(RecoverError::Diverged { .. })));
    let _ = fs::remove_dir_all(&dir);
}

/// Ticks with a few plain inserts, outside a `Run` (after its log moved).
fn run_ticks_with(w: &mut LogWriter, sys: &mut System, now: &mut Instant, at_map: &mut BTreeMap<u64, At>, ticks: u64) {
    let mut rng = SplitMix64::new(sys.w.seq);
    for _ in 0..ticks {
        let ts = sys.w.max_ts + 1;
        for i in 0..3 { durable::write(w, sys, &RealOp::Live(Op::Insert { id: 8_000_000 + sys.st.seq * 4 + i, ts, tags: 0, vec: { let mut v: Vec<f32> = (0..D_DIM).map(|_| gauss(&mut rng)).collect(); normalize(&mut v); v } })).unwrap(); }
        *now += TICK;
        let te = durable::end_tick(w, sys, &POLICY, *now).unwrap();
        at_map.insert(te.tick, at(sys));
    }
}

// ------------------------------------------------------------ timing

/// Writer-thread pause of a snapshot, background write time and size,
/// replay times. Not a gate: prints the numbers for the plan doc.
/// cargo test --release --test live_durability durability_timing -- --ignored --nocapture
#[test]
#[ignore = "timing aid"]
fn durability_timing() {
    for &(dim, nlist, m) in &[(128usize, 256usize, 32usize), (256, 1024, 32)] {
        let n = 100_000usize;
        let ops_n = 100_000usize;
        let dir = temp_dir(&format!("timing_{dim}"));
        let ds = uniform_sphere(n, dim, 60);
        let rows: Vec<(u64, Vec<f32>, u64)> = (0..n).map(|i| (ds.ids[i], row(&ds, i).to_vec(), 0)).collect();
        let base = IvfPqIndex::build(Metric::Cosine, dim, &rows, ivfpq_params(nlist, 8, 64, m, 7));
        let mut rng = SplitMix64::new(61);
        let mut op = |i: usize| {
            let mut v: Vec<f32> = (0..dim).map(|_| gauss(&mut rng)).collect(); normalize(&mut v);
            RealOp::Live(Op::Insert { id: 10_000_000 + i as u64, ts: 1 + (i / 10) as u64, tags: 0, vec: v })
        };
        let policy = views::COMPACT_DEFAULT;
        let mut sys = system::start(&base, views::DEFAULT_BLOCK_ROWS);
        // Keep every snapshot so the whole log stays for a full replay.
        let mut w = durable::create(&dir, &sys, DurableParams { keep_snapshots: 100, ..params(0) }, Instant::now()).unwrap();

        // 10 inserts per tick, real clock for fsync; a snapshot every 1000
        // ticks up to tick 6000 (its writer-side pause timed), then a tail
        // of 4000 ticks, longer than one 60 s cadence at 10 writes per tick.
        let (mut pauses, mut bg, mut tick_us) = (Vec::new(), Vec::new(), Vec::new());
        let mut snap_bytes = 0;
        let t = Instant::now();
        for i in 0..ops_n {
            durable::write(&mut w, &mut sys, &op(i)).unwrap();
            if i % 10 == 9 {
                let t0 = Instant::now();
                durable::end_tick(&mut w, &mut sys, &policy, Instant::now()).unwrap();
                tick_us.push(t0.elapsed().as_secs_f64() * 1e6);
                if (i / 10) % 1000 == 999 && i / 10 < 6000 {
                    let t0 = Instant::now();
                    assert!(durable::start_snapshot(&mut w, &sys).unwrap());
                    pauses.push(t0.elapsed().as_secs_f64() * 1e6);
                    let t1 = Instant::now();
                    snap_bytes = durable::wait_snapshot(&mut w).unwrap().unwrap().bytes;
                    bg.push(t1.elapsed().as_secs_f64());
                }
            }
        }
        durable::sync(&mut w, Instant::now()).unwrap();
        let append_s = t.elapsed().as_secs_f64();
        drop(w);
        pauses.sort_by(|a, b| a.partial_cmp(b).unwrap());
        tick_us.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let t = Instant::now();
        let rec = durable::recover(&dir, Some(&base), views::DEFAULT_BLOCK_ROWS, 4).unwrap();
        let from_snap_s = t.elapsed().as_secs_f64();
        let tail = rec.replayed;
        for p in files_with_ext(&dir, "snap") { fs::remove_file(p).unwrap(); }
        let t = Instant::now();
        let full = durable::recover(&dir, Some(&base), views::DEFAULT_BLOCK_ROWS, 4).unwrap();
        let full_s = t.elapsed().as_secs_f64();
        assert_eq!(system::fingerprint(&full.sys), system::fingerprint(&rec.sys));
        let log_mb = files_with_ext(&dir, "log").iter().map(|p| fs::metadata(p).unwrap().len()).sum::<u64>() as f64 / 1e6;

        println!("dim={dim} nlist={nlist} m={m}, {n} built rows + {ops_n} logged inserts, 10 per tick:");
        println!("  append + end_tick: {:.0} ops/s; end_tick p50 {:.0} us, p99 {:.0} us", ops_n as f64 / append_s, tick_us[tick_us.len() / 2], tick_us[tick_us.len() * 99 / 100]);
        println!("  snapshot writer pause: min {:.0} us, median {:.0} us, max {:.0} us ({} snapshots)", pauses[0], pauses[pauses.len() / 2], pauses[pauses.len() - 1], pauses.len());
        println!("  snapshot background write: {:.2}-{:.2} s, last {:.1} MB", bg.iter().cloned().fold(f64::MAX, f64::min), bg.iter().cloned().fold(0.0, f64::max), snap_bytes as f64 / 1e6);
        println!("  recover from newest snapshot + {tail} ops: {:.2} s; full replay of {} ops: {:.2} s; log {:.1} MB", from_snap_s, full.replayed, full_s, log_mb);
        let _ = fs::remove_dir_all(&dir);
    }
}
