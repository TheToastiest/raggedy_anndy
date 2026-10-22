//! Durability for the live write path: op log, snapshots, recovery.
//!
//! These tests pin the guarantees from docs/live-writes-plan.md, stage 4:
//! - every `Op` variant survives the log byte for byte;
//! - a torn or corrupt tail record is dropped, never mis-parsed, and
//!   recovery stops at the last good record;
//! - a writer killed after tick T (no final fsync) recovers to the last
//!   durable record with the fingerprint the live index had at that record;
//! - snapshot + log tail gives the same index as replaying the whole log;
//! - a corrupt snapshot is skipped and recovery falls back to an older
//!   snapshot or to the full log on the base model.

mod common;
use common::*;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use raggedy_anndy::durable::{self, DurableParams, LogWriter, RecoverError};
use raggedy_anndy::live::{self, Op};
use raggedy_anndy::oplog::{self, Record};
use raggedy_anndy::seed::SplitMix64;
use raggedy_anndy::snapshot;
use raggedy_anndy::{IvfPqIndex, Metric};

const DIM: usize = 32;
const NLIST: usize = 16;
const M: usize = 8;
const TICK: Duration = Duration::from_micros(16_667); // 60 Hz

fn temp_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("raggedy_dur_{name}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn base_index(n: usize, seed: u64) -> IvfPqIndex {
    let ds = uniform_sphere(n, DIM, seed);
    let rows: Vec<(u64, Vec<f32>, u64)> = (0..n).map(|i| (ds.ids[i], row(&ds, i).to_vec(), i as u64)).collect();
    IvfPqIndex::build(Metric::Cosine, DIM, &rows, ivfpq_params(NLIST, 4, 64, M, 7))
}

fn random_unit(rng: &mut SplitMix64) -> Vec<f32> {
    let mut v: Vec<f32> = (0..DIM).map(|_| gauss(rng)).collect();
    normalize(&mut v);
    v
}

fn params() -> DurableParams { DurableParams { sync_every: Duration::from_secs(1), keep_snapshots: 2 } }

/// Up to 10 ops for one tick: mostly inserts, plus upserts, deletes of
/// existing ids, a capped expiry now and then, and compactions.
fn tick_ops(rng: &mut SplitMix64, idx: &IvfPqIndex, tick: u64, next_id: &mut u64) -> Vec<Op> {
    let ts = 10_000 + tick;
    let n = 1 + rng.gen_range(10);
    let mut ops = Vec::with_capacity(n);
    for _ in 0..n {
        match rng.gen_range(20) {
            0..=11 => { *next_id += 1; ops.push(Op::Insert { id: *next_id, ts, tags: rng.next_u64() & 0xff, vec: random_unit(rng) }); }
            12..=14 => ops.push(Op::Upsert { id: rng.gen_range(*next_id as usize + 1) as u64, ts, tags: 1, vec: random_unit(rng) }),
            15..=17 => ops.push(Op::Delete { id: rng.gen_range(*next_id as usize + 1) as u64 }),
            18 => ops.push(Op::Expire { before_ts: (tick * 7).min(ts), max_rows: 50 }),
            _ => ops.push(Op::Compact { list: rng.gen_range(idx.lists.len()) as u32 }),
        }
    }
    ops
}

/// Writer-thread loop for `ticks` ticks. Only ops the index accepted are
/// logged. Returns the live fingerprint after every logged seq.
struct Run { idx: IvfPqIndex, w: LogWriter, fp_at: BTreeMap<u64, u64>, now: Instant, tick: u64, next_id: u64, rng: SplitMix64 }

fn start(dir: &Path, n: usize, seed: u64) -> (Run, IvfPqIndex) {
    let idx = base_index(n, seed);
    let base = idx.clone();
    let now = Instant::now();
    let w = durable::create(dir, &idx, 0, params(), now).unwrap();
    let mut fp_at = BTreeMap::new();
    fp_at.insert(0, idx.fingerprint());
    (Run { idx, w, fp_at, now, tick: 0, next_id: n as u64, rng: SplitMix64::new(seed ^ 0xd00d) }, base)
}

fn run_ticks(r: &mut Run, ticks: u64) {
    let mut s = live::scratch_for(&r.idx);
    for _ in 0..ticks {
        r.tick += 1;
        let ops = tick_ops(&mut r.rng, &r.idx, r.tick, &mut r.next_id);
        for op in &ops {
            if durable::apply_logged(&mut r.w, &mut r.idx, &mut s, r.tick, op).is_ok() {
                r.fp_at.insert(r.w.last_seq, r.idx.fingerprint());
            }
        }
        r.now += TICK;
        durable::end_tick(&mut r.w, r.now).unwrap();
    }
}

fn assert_ok(idx: &IvfPqIndex) {
    if let Err(e) = live::check_invariants(idx) { panic!("invariant broken: {e}"); }
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

// ------------------------------------------------------------ record format

#[test]
fn every_op_variant_round_trips_through_the_record_format() {
    let mut rng = SplitMix64::new(1);
    let ops = vec![
        Op::Insert { id: 7, ts: 100, tags: 0b1010, vec: random_unit(&mut rng) },
        Op::Upsert { id: u64::MAX, ts: u64::MAX, tags: u64::MAX, vec: vec![-0.0, f32::MIN_POSITIVE, 1.5, f32::MAX] },
        Op::Delete { id: 0 },
        Op::Expire { before_ts: 99, max_rows: 3 },
        Op::Expire { before_ts: u64::MAX, max_rows: usize::MAX },
        Op::Compact { list: u32::MAX },
        Op::Insert { id: 1, ts: 0, tags: 0, vec: Vec::new() },
    ];
    let mut buf = Vec::new();
    for (i, op) in ops.iter().enumerate() { oplog::encode_record(&mut buf, i as u64 + 1, 1000 + i as u64, op); }
    let mut at = 0usize;
    for (i, op) in ops.iter().enumerate() {
        let (rec, used) = oplog::decode_record(&buf[at..]).expect("record decodes");
        assert_eq!(rec, Record { seq: i as u64 + 1, tick: 1000 + i as u64, op: op.clone() });
        // f32 bits, not just values (-0.0 == 0.0 would hide a sign loss).
        if let (Op::Upsert { vec: a, .. } | Op::Insert { vec: a, .. }, Op::Upsert { vec: b, .. } | Op::Insert { vec: b, .. }) = (&rec.op, op) {
            assert_eq!(a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), b.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        }
        at += used;
    }
    assert_eq!(at, buf.len());
}

#[test]
fn every_op_variant_round_trips_through_a_log_file() {
    let dir = temp_dir("variants");
    let mut idx = base_index(1000, 2);
    let base = idx.clone();
    let mut s = live::scratch_for(&idx);
    let now = Instant::now();
    let mut w = durable::create(&dir, &idx, 0, params(), now).unwrap();
    let mut rng = SplitMix64::new(3);
    let ops = vec![
        Op::Insert { id: 50_000, ts: 5000, tags: 3, vec: random_unit(&mut rng) },
        Op::Upsert { id: 50_000, ts: 5001, tags: 4, vec: random_unit(&mut rng) },
        Op::Upsert { id: 50_001, ts: 5002, tags: 0, vec: random_unit(&mut rng) },
        Op::Delete { id: 10 },
        Op::Expire { before_ts: 300, max_rows: 40 },
        Op::Expire { before_ts: 300, max_rows: usize::MAX },
        Op::Compact { list: 3 },
    ];
    for op in &ops { durable::apply_logged(&mut w, &mut idx, &mut s, 1, op).unwrap(); }
    durable::sync(&mut w, now).unwrap();
    assert_eq!(w.synced_seq, ops.len() as u64);
    drop(w);
    let rec = durable::recover(&dir, Some(&base)).unwrap();
    assert_eq!(rec.seq, ops.len() as u64);
    assert_eq!(rec.replayed, ops.len() as u64);
    assert_eq!(rec.idx.fingerprint(), idx.fingerprint());
    assert_ok(&rec.idx);
    let _ = fs::remove_dir_all(&dir);
}

// ------------------------------------------------------------ torn / corrupt tails

#[test]
fn every_truncation_of_the_last_record_is_dropped() {
    let mut rng = SplitMix64::new(4);
    let mut buf = Vec::new();
    oplog::encode_record(&mut buf, 1, 1, &Op::Delete { id: 9 });
    let keep = buf.len();
    oplog::encode_record(&mut buf, 2, 1, &Op::Insert { id: 1, ts: 2, tags: 3, vec: random_unit(&mut rng) });
    for cut in keep..buf.len() {
        let (recs, valid) = oplog::decode_all(&buf[..cut], 0);
        assert_eq!(recs.len(), 1, "cut at {cut}");
        assert_eq!(valid, keep);
    }
    // Any single corrupted byte in the last record is caught too.
    for i in keep..buf.len() {
        let mut b = buf.clone();
        b[i] ^= 0x01;
        let (recs, valid) = oplog::decode_all(&b, 0);
        assert_eq!((recs.len(), valid), (1, keep), "flipped byte {i}");
    }
    // A wild length prefix must not allocate or read past the end.
    let mut b = buf[..keep].to_vec();
    b.extend_from_slice(&u32::MAX.to_le_bytes());
    b.extend_from_slice(&[0; 12]);
    assert_eq!(oplog::decode_all(&b, 0).1, keep);
}

#[test]
fn records_must_have_consecutive_seqs() {
    let mut buf = Vec::new();
    oplog::encode_record(&mut buf, 1, 1, &Op::Delete { id: 1 });
    oplog::encode_record(&mut buf, 2, 1, &Op::Delete { id: 2 });
    let two = buf.len();
    oplog::encode_record(&mut buf, 4, 1, &Op::Delete { id: 3 }); // gap: well-formed but out of order
    let (recs, valid) = oplog::decode_all(&buf, 0);
    assert_eq!((recs.len(), valid), (2, two));
}

#[test]
fn corrupt_tail_on_disk_stops_recovery_at_last_good_record_and_resume_continues() {
    let dir = temp_dir("corrupt_tail");
    let (mut r, base) = start(&dir, 1500, 5);
    run_ticks(&mut r, 40);
    durable::sync(&mut r.w, r.now).unwrap();
    let last = r.w.last_seq;
    let seg = r.w.seg_path.clone();
    drop(r.w);

    // Corrupt a byte inside the last record: everything before it survives.
    flip_byte(&seg, 3);
    let rec = durable::recover(&dir, Some(&base)).unwrap();
    assert_eq!(rec.seq, last - 1);
    assert_eq!(rec.idx.fingerprint(), r.fp_at[&(last - 1)]);
    assert!(rec.dropped_bytes > 0);
    assert_ok(&rec.idx);

    // Resume cuts the bad tail and appends after the good prefix.
    let mut w = durable::resume(&dir, &rec, params(), r.now).unwrap();
    let mut idx = rec.idx;
    assert_eq!(w.last_seq, last - 1);
    let mut s = live::scratch_for(&idx);
    let mut rng = SplitMix64::new(55);
    durable::apply_logged(&mut w, &mut idx, &mut s, 99, &Op::Insert { id: 9_000_000, ts: 20_000, tags: 0, vec: random_unit(&mut rng) }).unwrap();
    durable::sync(&mut w, r.now).unwrap();
    drop(w);
    let again = durable::recover(&dir, Some(&base)).unwrap();
    assert_eq!(again.seq, last);
    assert_eq!(again.tick, 99);
    assert_eq!(again.dropped_bytes, 0);
    assert_eq!(again.idx.fingerprint(), idx.fingerprint());
    let _ = fs::remove_dir_all(&dir);
}

// ------------------------------------------------------------ crash simulation

/// Kill the writer after tick T: ops appended but not flushed are gone, and
/// a power loss also drops whatever was written after the last fsync.
#[test]
fn kill_after_tick_t_recovers_last_durable_record() {
    for (case, kill_tick) in [37u64, 61, 150].into_iter().enumerate() {
        let dir = temp_dir(&format!("kill_{case}"));
        let (mut r, base) = start(&dir, 1500, 6);
        run_ticks(&mut r, kill_tick);
        // Ops of a half-done tick T+1, never flushed.
        let mut s = live::scratch_for(&r.idx);
        let ops = tick_ops(&mut r.rng, &r.idx, r.tick + 1, &mut r.next_id);
        for op in &ops { let _ = durable::apply_logged(&mut r.w, &mut r.idx, &mut s, r.tick + 1, op); }
        let flushed_seq = r.w.written_seq;
        let synced_seq = r.w.synced_seq;
        let synced_len = r.w.synced_len;
        assert!(synced_seq <= flushed_seq && flushed_seq < r.w.last_seq);
        let seg = r.w.seg_path.clone();
        drop(r.w); // no final flush, no fsync

        // Process crash only: everything handed to the OS survives.
        let rec = durable::recover(&dir, Some(&base)).unwrap();
        assert_eq!(rec.seq, flushed_seq, "case {case}");
        assert_eq!(rec.idx.fingerprint(), r.fp_at[&flushed_seq]);

        // Power loss: the file is cut at the last fsync, plus a torn
        // fragment of the next record.
        let bytes = fs::read(&seg).unwrap();
        let mut cut = bytes[..synced_len as usize].to_vec();
        cut.extend_from_slice(&bytes[synced_len as usize..(synced_len as usize + 11).min(bytes.len())]);
        fs::write(&seg, cut).unwrap();
        let rec = durable::recover(&dir, Some(&base)).unwrap();
        assert_eq!(rec.seq, synced_seq, "case {case}");
        assert_eq!(rec.idx.fingerprint(), r.fp_at[&synced_seq], "case {case}");
        assert_ok(&rec.idx);
        let _ = fs::remove_dir_all(&dir);
    }
}

#[test]
fn fsync_runs_at_the_configured_interval() {
    let dir = temp_dir("interval");
    let idx = base_index(1000, 7);
    let t0 = Instant::now();
    let mut w = durable::create(&dir, &idx, 0, DurableParams { sync_every: Duration::from_millis(250), keep_snapshots: 2 }, t0).unwrap();
    let mut syncs = 0;
    for t in 1..=60u64 {
        durable::append(&mut w, t, &Op::Delete { id: 1_000_000 + t });
        if durable::end_tick(&mut w, t0 + TICK * t as u32).unwrap() { syncs += 1; }
        assert!(w.last_seq - w.synced_seq <= 15, "tick {t}: {} records not durable", w.last_seq - w.synced_seq);
    }
    assert_eq!(syncs, 4); // 60 ticks at 60 Hz = 1 s: at ticks 15, 30, 45, 60
    assert_eq!(w.written_seq, 60);
    let _ = fs::remove_dir_all(&dir);
}

// ------------------------------------------------------------ snapshots

#[test]
fn snapshot_file_round_trips_the_whole_index() {
    let dir = temp_dir("snap_rt");
    for (case, (opq, store_vecs)) in [(false, true), (true, false), (true, true)].into_iter().enumerate() {
        let ds = uniform_sphere(2000, DIM, 8);
        let rows: Vec<(u64, Vec<f32>, u64)> = (0..2000).map(|i| (ds.ids[i], row(&ds, i).to_vec(), i as u64)).collect();
        let mut p = ivfpq_params(NLIST, 4, 64, M, 7);
        p.use_opq = opq; p.store_vecs = store_vecs;
        let mut idx = IvfPqIndex::build(Metric::Cosine, DIM, &rows, p);
        let mut s = live::scratch_for(&idx);
        live::apply(&mut idx, &mut s, &Op::Expire { before_ts: 500, max_rows: usize::MAX }).unwrap();
        live::apply(&mut idx, &mut s, &Op::Delete { id: 1500 }).unwrap();
        live::apply(&mut idx, &mut s, &Op::Compact { list: 0 }).unwrap();
        let path = dir.join(format!("case{case}.snap"));
        snapshot::write(&path, &idx, 1234, 77).unwrap();
        let snap = snapshot::read(&path).unwrap();
        assert_eq!((snap.seq, snap.tick), (1234, 77));
        assert_eq!(snap.idx.fingerprint(), idx.fingerprint(), "case {case}");
        assert_eq!(snap.idx.max_ts, idx.max_ts);
        assert_eq!(live::live_rows(&snap.idx), live::live_rows(&idx));
        assert_ok(&snap.idx);
        let q = row(&ds, 42);
        assert_eq!(snap.idx.search(q, 10), idx.search(q, 10));
        assert!(!dir.join(format!("case{case}.snap.tmp")).exists());
        // Any flipped byte is refused, not loaded.
        for back in [1u64, 100, 5000] {
            let mut b = fs::read(&path).unwrap();
            let i = b.len() - back as usize; b[i] ^= 0x10;
            let bad = dir.join("bad.snap");
            fs::write(&bad, b).unwrap();
            assert!(snapshot::read(&bad).is_err(), "case {case}: flip {back} from end accepted");
        }
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn snapshot_plus_tail_equals_full_replay() {
    let dir = temp_dir("snap_tail");
    let (mut r, base) = start(&dir, 1500, 9);
    run_ticks(&mut r, 50);
    let snap_seq = r.w.last_seq;
    durable::snapshot(&mut r.w, &r.idx, r.now).unwrap();
    run_ticks(&mut r, 50);
    durable::sync(&mut r.w, r.now).unwrap();
    let last = r.w.last_seq;
    drop(r.w);

    let from_snap = durable::recover(&dir, Some(&base)).unwrap();
    assert_eq!(from_snap.snapshot_seq, Some(snap_seq));
    assert_eq!(from_snap.seq, last);
    assert_eq!(from_snap.replayed, last - snap_seq);

    // Remove the snapshot: the full log on the base model must agree.
    for p in files_with_ext(&dir, "snap") { fs::remove_file(p).unwrap(); }
    let full = durable::recover(&dir, Some(&base)).unwrap();
    assert_eq!(full.snapshot_seq, None);
    assert_eq!(full.replayed, last);
    assert_eq!(full.idx.fingerprint(), from_snap.idx.fingerprint());
    assert_eq!(full.idx.fingerprint(), r.fp_at[&last]);
    assert_ok(&full.idx);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn corrupt_snapshot_falls_back_to_full_replay() {
    let dir = temp_dir("snap_corrupt");
    let (mut r, base) = start(&dir, 1500, 10);
    run_ticks(&mut r, 30);
    durable::snapshot(&mut r.w, &r.idx, r.now).unwrap();
    run_ticks(&mut r, 30);
    durable::sync(&mut r.w, r.now).unwrap();
    let last = r.w.last_seq;
    drop(r.w);
    let snaps = files_with_ext(&dir, "snap");
    assert_eq!(snaps.len(), 1);
    flip_byte(&snaps[0], 200);

    let rec = durable::recover(&dir, Some(&base)).unwrap();
    assert_eq!(rec.snapshot_seq, None);
    assert_eq!(rec.bad_snapshots, snaps);
    assert_eq!(rec.seq, last);
    assert_eq!(rec.idx.fingerprint(), r.fp_at[&last]);

    // Without a base model there is nothing to replay onto.
    assert!(matches!(durable::recover(&dir, None), Err(RecoverError::NoState)));
    // A base with a different model is refused by its fingerprint.
    assert!(matches!(durable::recover(&dir, Some(&base_index(1500, 11))), Err(RecoverError::NoState)));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn corrupt_newest_snapshot_falls_back_to_older_snapshot_and_rotation_prunes() {
    let dir = temp_dir("snap_older");
    let (mut r, base) = start(&dir, 1500, 12);
    let mut snap_seqs = Vec::new();
    for _ in 0..3 {
        run_ticks(&mut r, 20);
        snap_seqs.push(r.w.last_seq);
        durable::snapshot(&mut r.w, &r.idx, r.now).unwrap();
    }
    run_ticks(&mut r, 20);
    durable::sync(&mut r.w, r.now).unwrap();
    let last = r.w.last_seq;
    drop(r.w);

    // keep_snapshots = 2: the first snapshot and the log before the
    // second are gone, the rest is kept.
    let snaps = files_with_ext(&dir, "snap");
    let logs = files_with_ext(&dir, "log");
    assert_eq!(snaps.len(), 2);
    assert_eq!(logs.len(), 2);
    assert_eq!(snaps[0], dir.join(format!("snap-{:020}.snap", snap_seqs[1])));
    assert_eq!(logs[0], dir.join(format!("log-{:020}.log", snap_seqs[1])));

    flip_byte(&snaps[1], 50);
    let rec = durable::recover(&dir, None).unwrap();
    assert_eq!(rec.snapshot_seq, Some(snap_seqs[1]));
    assert_eq!(rec.seq, last);
    assert_eq!(rec.idx.fingerprint(), r.fp_at[&last]);

    // Resume sets the bad snapshot aside; the next recovery is clean.
    drop(durable::resume(&dir, &rec, params(), r.now).unwrap());
    let again = durable::recover(&dir, Some(&base)).unwrap();
    assert!(again.bad_snapshots.is_empty());
    assert_eq!(again.idx.fingerprint(), r.fp_at[&last]);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn crash_between_snapshot_and_new_log_recovers_from_snapshot() {
    let dir = temp_dir("snap_crash");
    let (mut r, base) = start(&dir, 1500, 13);
    run_ticks(&mut r, 25);
    let seq = r.w.last_seq;
    durable::snapshot(&mut r.w, &r.idx, r.now).unwrap();
    drop(r.w);
    // The new, still empty log never made it to disk.
    fs::remove_file(dir.join(format!("log-{seq:020}.log"))).unwrap();
    let rec = durable::recover(&dir, Some(&base)).unwrap();
    assert_eq!((rec.snapshot_seq, rec.seq, rec.replayed), (Some(seq), seq, 0));
    assert_eq!(rec.idx.fingerprint(), r.fp_at[&seq]);
    let mut w = durable::resume(&dir, &rec, params(), r.now).unwrap();
    assert_eq!(w.last_seq, seq);
    durable::append(&mut w, 1, &Op::Delete { id: 0 });
    durable::sync(&mut w, r.now).unwrap();
    assert!(dir.join(format!("log-{seq:020}.log")).exists());
    let _ = fs::remove_dir_all(&dir);
}

// ------------------------------------------------------------ timing

/// Append throughput and recovery time. Not a gate: prints the numbers
/// for the plan doc.
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
        let ops: Vec<Op> = (0..ops_n).map(|i| {
            let mut v: Vec<f32> = (0..dim).map(|_| gauss(&mut rng)).collect(); normalize(&mut v);
            Op::Insert { id: 10_000_000 + i as u64, ts: 1 + (i / 10) as u64, tags: 0, vec: v }
        }).collect();

        // Append only: 10 ops per tick, real clock, fsync every second.
        let mut w = durable::create(&dir, &base, 0, params(), Instant::now()).unwrap();
        let t = Instant::now();
        for (i, op) in ops.iter().enumerate() {
            durable::append(&mut w, (i / 10) as u64, op);
            if i % 10 == 9 { durable::end_tick(&mut w, Instant::now()).unwrap(); }
        }
        durable::sync(&mut w, Instant::now()).unwrap();
        let append_s = t.elapsed().as_secs_f64();
        let log_bytes = fs::metadata(&w.seg_path).unwrap().len();
        drop(w);

        let t = Instant::now();
        let (recs, _) = oplog::decode_all(&fs::read(dir.join(format!("log-{:020}.log", 0))).unwrap()[oplog::SEGMENT_HEADER_LEN..], 0);
        let decode_s = t.elapsed().as_secs_f64();
        assert_eq!(recs.len(), ops_n);
        drop(recs);

        let t = Instant::now();
        let rec = durable::recover(&dir, Some(&base)).unwrap();
        let recover_s = t.elapsed().as_secs_f64();
        assert_eq!(rec.seq, ops_n as u64);

        let t = Instant::now();
        let snap_path = dir.join("timing.snap");
        snapshot::write(&snap_path, &rec.idx, rec.seq, rec.tick).unwrap();
        let snap_w = t.elapsed().as_secs_f64();
        let snap_bytes = fs::metadata(&snap_path).unwrap().len();
        let t = Instant::now();
        let snap = snapshot::read(&snap_path).unwrap();
        let snap_r = t.elapsed().as_secs_f64();
        assert_eq!(snap.idx.fingerprint(), rec.idx.fingerprint());

        println!("dim={dim} nlist={nlist} m={m}, {n} built rows + {ops_n} logged inserts:");
        println!("  append: {:.0} ops/s ({:.2} s, log {:.1} MB)", ops_n as f64 / append_s, append_s, log_bytes as f64 / 1e6);
        println!("  decode log only: {:.3} s", decode_s);
        println!("  recover (full replay of {ops_n} ops): {:.2} s", recover_s);
        println!("  snapshot write {:.3} s, read+verify {:.3} s, {:.1} MB", snap_w, snap_r, snap_bytes as f64 / 1e6);
        let _ = fs::remove_dir_all(&dir);
    }
}
