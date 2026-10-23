//! Stage 5, index side: real memories and the pending buffer.
//!
//! The game server flags a memory as real (and optionally LEGENDARY) when
//! it inserts it. A real memory goes into the 1-hour short-term index like
//! any other row, and also into a small pending buffer that keeps its f32
//! vector in RAM, searchable exactly, until the 12-hour commit drains it
//! into an outbox of plain records for the SQL side. These tests pin:
//! - a real memory is still found after the 1-hour expiry, until commit;
//! - after commit it is gone from RAM and in the outbox exactly once;
//! - commit order and contents are deterministic;
//! - a commit re-run after a crash adds no duplicate ids;
//! - delete / upsert of a pending or handed (committed, unacked) id;
//! - commit and ack happen only when the server sends the op;
//! - LEGENDARY is its own field, independent of the 64 tag bits;
//! - merged search (index + pending) is deterministic and never repeats an id.

mod common;
use common::*;

use std::collections::BTreeSet;

use raggedy_anndy::live::{self, Op, WriteError};
use raggedy_anndy::real_memory::{self as real, Outbox, Pending, RealApplied, RealError, RealOp, RecordKind, LEGENDARY, REAL};
use raggedy_anndy::seed::SplitMix64;
use raggedy_anndy::{Hit, IvfPqIndex, Metric};

const DIM: usize = 32;
const NLIST: usize = 16;
const M: usize = 8;
/// One hour in the test's time unit (seconds).
const HOUR: u64 = 3_600;

/// Rows with ts = 0..n, as if one arrived per second.
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

fn ids(hits: &[Hit]) -> Vec<u64> { hits.iter().map(|h| h.id).collect() }

fn bits(hits: &[Hit]) -> Vec<(u64, u32)> { hits.iter().map(|h| (h.id, h.score.to_bits())).collect() }

fn search(idx: &IvfPqIndex, buf: &Pending, q: &[f32], k: usize) -> Vec<Hit> {
    real::search(idx, buf, q, k, None, 0, 0.0)
}

/// Expire everything older than `before_ts`, uncapped.
fn expire_all(idx: &mut IvfPqIndex, s: &mut live::Scratch, before_ts: u64) {
    live::apply(idx, s, &Op::Expire { before_ts, max_rows: usize::MAX }).unwrap();
}

fn assert_ok(idx: &IvfPqIndex, buf: &Pending) {
    if let Err(e) = live::check_invariants(idx) { panic!("index invariant broken: {e}"); }
    if let Err(e) = real::check_invariants(buf) { panic!("pending invariant broken: {e}"); }
}

/// A seeded mix of normal and real inserts after the base rows, one per
/// second, with every 7th real and every 3rd real legendary.
fn mixed_inserts(n: usize, seed: u64, first_id: u64, first_ts: u64) -> Vec<RealOp> {
    let mut rng = SplitMix64::new(seed);
    let mut ops = Vec::with_capacity(n);
    let mut real_count = 0u64;
    for i in 0..n as u64 {
        let id = first_id + i;
        let ts = first_ts + i;
        let tags = rng.next_u64() & 0xF;
        let vec = random_unit(&mut rng);
        if i % 7 == 3 {
            real_count += 1;
            let flags = if real_count % 3 == 0 { REAL | LEGENDARY } else { REAL };
            ops.push(RealOp::InsertReal { id, ts, tags, flags, vec });
        } else {
            ops.push(RealOp::Live(Op::Insert { id, ts, tags, vec }));
        }
    }
    ops
}

fn replay(idx: &mut IvfPqIndex, buf: &mut Pending, outbox: &mut Outbox, ops: &[RealOp]) {
    let mut s = live::scratch_for(idx);
    for op in ops { real::apply(idx, &mut s, buf, outbox, op).unwrap(); }
}

// ------------------------------------------------------------ the gap

#[test]
fn real_memory_survives_the_one_hour_expiry_until_commit() {
    let mut idx = base_index(2000, 1);
    let mut buf = real::pending_for(&idx);
    let mut outbox = Outbox::default();
    let mut s = live::scratch_for(&idx);
    let mut rng = SplitMix64::new(2);

    let v_real = random_unit(&mut rng);
    let v_normal = random_unit(&mut rng);
    real::insert_real(&mut idx, &mut s, &mut buf, 900_001, 2_000, 0, REAL, &v_real).unwrap();
    live::insert(&mut idx, &mut s, 900_002, 2_000, 0, &v_normal).unwrap();
    assert_eq!(search(&idx, &buf, &v_real, 1)[0].id, 900_001);
    assert_eq!(search(&idx, &buf, &v_normal, 1)[0].id, 900_002);

    // One hour later the short-term window drops both rows.
    expire_all(&mut idx, &mut s, 2_000 + HOUR);
    assert!(live::locate(&idx, 900_001).is_none(), "real row should leave the short-term index");
    assert!(live::locate(&idx, 900_002).is_none());
    assert_ok(&idx, &buf);

    // The real one is still found (from the pending buffer); the normal one is gone.
    let hits = search(&idx, &buf, &v_real, 5);
    assert_eq!(hits[0].id, 900_001, "real memory lost between expiry and commit: {:?}", ids(&hits));
    assert!(!search(&idx, &buf, &v_normal, 20).iter().any(|h| h.id == 900_002));

    // Commit: it leaves RAM and is in the outbox exactly once, with its data.
    let committed = real::commit(&mut buf, &mut outbox);
    assert_eq!(committed, 1);
    assert_eq!(real::pending_len(&buf), 0);
    assert!(!real::is_pending(&buf, 900_001));
    assert!(!search(&idx, &buf, &v_real, 20).iter().any(|h| h.id == 900_001), "committed memory still in RAM");
    assert_eq!(outbox.records.iter().filter(|r| r.id == 900_001).count(), 1);
    let rec = outbox.records.iter().find(|r| r.id == 900_001).unwrap();
    assert_eq!((rec.ts, rec.tags, rec.flags), (2_000, 0, REAL));
    assert_eq!(rec.vec, v_real, "outbox must carry the vector as inserted");
    assert_ok(&idx, &buf);
}

#[test]
fn commit_before_expiry_leaves_the_short_term_row_alone() {
    // Commit only drains the pending buffer; the 1-hour window still owns the row.
    let mut idx = base_index(1000, 3);
    let mut buf = real::pending_for(&idx);
    let mut outbox = Outbox::default();
    let mut s = live::scratch_for(&idx);
    let v = random_unit(&mut SplitMix64::new(4));
    real::insert_real(&mut idx, &mut s, &mut buf, 700_077, 5_000, 0, REAL, &v).unwrap();
    real::commit(&mut buf, &mut outbox);
    assert!(live::locate(&idx, 700_077).is_some());
    assert_eq!(search(&idx, &buf, &v, 1)[0].id, 700_077);
    assert_eq!(outbox.records.len(), 1);
}

// ------------------------------------------------------------ commit

#[test]
fn commit_order_and_contents_are_deterministic() {
    let base = base_index(2000, 5);
    let ops = mixed_inserts(700, 6, 1_000_000, 10_000);
    let run = || {
        let mut idx = base.clone();
        let mut buf = real::pending_for(&idx);
        let mut outbox = Outbox::default();
        replay(&mut idx, &mut buf, &mut outbox, &ops);
        let mut s = live::scratch_for(&idx);
        expire_all(&mut idx, &mut s, 10_000 + 350);
        real::commit(&mut buf, &mut outbox);
        (idx.fingerprint(), outbox.records)
    };
    let (fa, a) = run();
    let (fb, b) = run();
    assert_eq!(fa, fb);
    assert_eq!(a, b);
    assert_eq!(a.len(), 100, "every 7th of 700 is real");
    // Order: ts ascending, then id ascending.
    assert!(a.windows(2).all(|w| (w[0].ts, w[0].id) < (w[1].ts, w[1].id)), "outbox not in (ts, id) order");
    // Contents match what was inserted.
    for r in &a {
        let op = ops.iter().find(|op| matches!(op, RealOp::InsertReal { id, .. } if *id == r.id)).unwrap();
        match op {
            RealOp::InsertReal { ts, tags, flags, vec, .. } => {
                assert_eq!((r.ts, r.tags, r.flags), (*ts, *tags, *flags));
                assert_eq!(r.vec.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), vec.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn commit_order_is_ts_then_id_not_insert_order() {
    // Same ts, ids inserted out of order: the outbox sorts them.
    let mut idx = base_index(500, 7);
    let mut buf = real::pending_for(&idx);
    let mut outbox = Outbox::default();
    let mut s = live::scratch_for(&idx);
    let mut rng = SplitMix64::new(8);
    for &(id, ts) in &[(10_030u64, 1_000u64), (10_010, 1_000), (10_020, 1_000), (10_005, 1_001)] {
        real::insert_real(&mut idx, &mut s, &mut buf, id, ts, 0, REAL, &random_unit(&mut rng)).unwrap();
    }
    real::commit(&mut buf, &mut outbox);
    assert_eq!(outbox.records.iter().map(|r| r.id).collect::<Vec<_>>(), vec![10_010, 10_020, 10_030, 10_005]);
}

#[test]
fn commit_rerun_after_crash_adds_no_duplicate_ids() {
    let base = base_index(1500, 9);
    let ops = mixed_inserts(350, 10, 2_000_000, 5_000);
    let mut idx = base.clone();
    let mut buf = real::pending_for(&idx);
    let mut outbox = Outbox::default();
    replay(&mut idx, &mut buf, &mut outbox, &ops);

    // The commit runs and its outbox is written, then the process dies
    // before the commit is recorded. Recovery replays the log, which
    // rebuilds the pending buffer, and the commit runs again.
    let before_crash = buf.clone();
    let first = real::commit(&mut buf, &mut outbox);
    let after_first = outbox.clone();
    buf = before_crash;
    let second = real::commit(&mut buf, &mut outbox);

    assert_eq!(first, 50);
    assert_eq!(second, 0, "re-run must not add records already in the outbox");
    assert_eq!(outbox, after_first);
    assert_eq!(real::pending_len(&buf), 0, "re-run still drains the buffer");
    let unique: BTreeSet<u64> = outbox.records.iter().map(|r| r.id).collect();
    assert_eq!(unique.len(), outbox.records.len(), "duplicate id in outbox");

    // A full replay from scratch into a fresh outbox gives the same records,
    // so the SQL side can drop a resend by `seq`.
    let mut idx2 = base.clone();
    let mut buf2 = real::pending_for(&idx2);
    let mut outbox2 = Outbox::default();
    replay(&mut idx2, &mut buf2, &mut outbox2, &ops);
    real::commit(&mut buf2, &mut outbox2);
    assert_eq!(outbox2, after_first);
}

#[test]
fn commit_as_an_op_replays_identically() {
    let base = base_index(1000, 11);
    let mut ops = mixed_inserts(200, 12, 3_000_000, 4_000);
    ops.push(RealOp::Commit);
    ops.extend(mixed_inserts(100, 13, 4_000_000, 4_200));
    let run = || {
        let mut idx = base.clone();
        let mut buf = real::pending_for(&idx);
        let mut outbox = Outbox::default();
        replay(&mut idx, &mut buf, &mut outbox, &ops);
        (idx.fingerprint(), buf, outbox)
    };
    let (fa, ba, oa) = run();
    let (fb, bb, ob) = run();
    assert_eq!(fa, fb);
    assert_eq!(ba, bb);
    assert_eq!(oa, ob);
    assert_eq!(oa.records.len(), 29, "reals in the first 200 ops");
    assert_eq!(real::pending_len(&ba), 14, "reals in the last 100 ops");
}

// ------------------------------------------------------------ LEGENDARY

#[test]
fn legendary_is_independent_of_tag_bits() {
    let mut idx = base_index(800, 14);
    let mut buf = real::pending_for(&idx);
    let mut outbox = Outbox::default();
    let mut s = live::scratch_for(&idx);
    let mut rng = SplitMix64::new(15);
    // Tags equal to the flags' bit patterns, and all 64 tag bits set.
    let tag_like_flags = [LEGENDARY as u64, (REAL | LEGENDARY) as u64, u64::MAX];
    for (i, &tags) in tag_like_flags.iter().enumerate() {
        real::insert_real(&mut idx, &mut s, &mut buf, 10_100 + i as u64, 1_000, tags, REAL, &random_unit(&mut rng)).unwrap();
    }
    real::insert_real(&mut idx, &mut s, &mut buf, 10_200, 1_000, 0, REAL | LEGENDARY, &random_unit(&mut rng)).unwrap();
    for i in 0..3u64 { assert!(!real::is_legendary(&buf, 10_100 + i).unwrap(), "tags made id {} legendary", 10_100 + i); }
    assert!(real::is_legendary(&buf, 10_200).unwrap());

    real::commit(&mut buf, &mut outbox);
    for r in &outbox.records {
        let want = r.id == 10_200;
        assert_eq!(r.flags & LEGENDARY != 0, want, "id {} flags {:#b} tags {:#x}", r.id, r.flags, r.tags);
    }
    // Tags pass through untouched; flags never leak into them.
    assert_eq!(outbox.records.iter().find(|r| r.id == 10_200).unwrap().tags, 0);
    assert_eq!(outbox.records.iter().find(|r| r.id == 10_102).unwrap().tags, u64::MAX);
}

#[test]
fn bad_flags_are_rejected_without_changes() {
    let mut idx = base_index(500, 16);
    let mut buf = real::pending_for(&idx);
    let mut s = live::scratch_for(&idx);
    let v = random_unit(&mut SplitMix64::new(17));
    let fp = idx.fingerprint();
    for flags in [0u8, LEGENDARY, 0b100, REAL | 0b1000_0000] {
        assert_eq!(real::insert_real(&mut idx, &mut s, &mut buf, 9_999, 1_000, 0, flags, &v), Err(RealError::BadFlags { id: 9_999, flags }));
        assert_eq!(idx.fingerprint(), fp);
        assert_eq!(real::pending_len(&buf), 0);
    }
}

#[test]
fn index_errors_leave_the_buffer_unchanged() {
    let mut idx = base_index(500, 18);
    let mut buf = real::pending_for(&idx);
    let mut s = live::scratch_for(&idx);
    let v = random_unit(&mut SplitMix64::new(19));
    // Duplicate of a live index row.
    assert_eq!(real::insert_real(&mut idx, &mut s, &mut buf, 3, 1_000, 0, REAL, &v), Err(RealError::Write(WriteError::DuplicateId { id: 3 })));
    // Wrong dim.
    assert!(matches!(real::insert_real(&mut idx, &mut s, &mut buf, 9_000, 1_000, 0, REAL, &v[..DIM - 1]), Err(RealError::Write(WriteError::WrongDim { .. }))));
    assert_eq!(real::pending_len(&buf), 0);

    // An id still pending (expired from the index, not committed) cannot be reused.
    real::insert_real(&mut idx, &mut s, &mut buf, 9_001, 1_000, 0, REAL, &v).unwrap();
    expire_all(&mut idx, &mut s, 1_000 + HOUR);
    let fp = idx.fingerprint();
    assert_eq!(real::insert_real(&mut idx, &mut s, &mut buf, 9_001, 1_000 + HOUR, 0, REAL, &v), Err(RealError::Write(WriteError::DuplicateId { id: 9_001 })));
    assert_eq!(idx.fingerprint(), fp);
    assert_eq!(real::pending_len(&buf), 1);
    assert_ok(&idx, &buf);
}

// ------------------------------------------------------------ merged search

#[test]
fn merged_search_never_returns_an_id_twice() {
    let mut idx = base_index(2000, 20);
    idx.params.nprobe = NLIST;
    let mut buf = real::pending_for(&idx);
    let mut outbox = Outbox::default();
    let ops = mixed_inserts(700, 21, 5_000_000, 10_000);
    replay(&mut idx, &mut buf, &mut outbox, &ops);
    // Half the reals are still in the index too, half only in the buffer.
    let mut s = live::scratch_for(&idx);
    expire_all(&mut idx, &mut s, 10_000 + 350);
    let in_both = (0..700u64).filter(|i| {
        let id = 5_000_000 + i;
        real::is_pending(&buf, id) && live::locate(&idx, id).is_some()
    }).count();
    assert!(in_both > 10, "test needs ids present in both: {in_both}");

    let mut rng = SplitMix64::new(22);
    for _ in 0..60 {
        let q = random_unit(&mut rng);
        for k in [1usize, 5, 20, 200] {
            let hits = real::search(&idx, &buf, &q, k, None, 11_000, 1e-4);
            let unique: BTreeSet<u64> = hits.iter().map(|h| h.id).collect();
            assert_eq!(unique.len(), hits.len(), "duplicate id at k={k}: {:?}", ids(&hits));
            assert_eq!(hits.len(), k.min(live::live_rows(&idx) + real::pending_len(&buf) - in_both));
            assert!(hits.windows(2).all(|w| w[0].score > w[1].score || (w[0].score == w[1].score && w[0].id < w[1].id)), "not score desc, id asc");
        }
    }
    // Querying a pending vector that is also live in the index: one hit, rank 1.
    for op in &ops {
        if let RealOp::InsertReal { id, vec, .. } = op {
            if live::locate(&idx, *id).is_some() {
                let hits = search(&idx, &buf, vec, 10);
                assert_eq!(hits[0].id, *id);
                assert_eq!(hits.iter().filter(|h| h.id == *id).count(), 1);
            }
        }
    }
}

#[test]
fn merged_search_is_deterministic() {
    let base = base_index(2000, 23);
    let ops = mixed_inserts(500, 24, 6_000_000, 10_000);
    let build = || {
        let mut idx = base.clone();
        let mut buf = real::pending_for(&idx);
        let mut outbox = Outbox::default();
        replay(&mut idx, &mut buf, &mut outbox, &ops);
        let mut s = live::scratch_for(&idx);
        expire_all(&mut idx, &mut s, 10_000 + 250);
        (idx, buf)
    };
    let (ia, ba) = build();
    let (ib, bb) = build();
    let mut rng = SplitMix64::new(25);
    for _ in 0..50 {
        let q = random_unit(&mut rng);
        let a = real::search(&ia, &ba, &q, 10, Some(0x3), 10_600, 1e-4);
        assert_eq!(bits(&a), bits(&real::search(&ib, &bb, &q, 10, Some(0x3), 10_600, 1e-4)));
        assert_eq!(bits(&a), bits(&real::search(&ia, &ba, &q, 10, Some(0x3), 10_600, 1e-4)), "same inputs, different run");
        assert_eq!(bits(&search(&ia, &ba, &q, 10)), bits(&search(&ib, &bb, &q, 10)));
    }
}

#[test]
fn merged_search_honours_tag_filter_on_pending_rows() {
    let mut idx = base_index(1000, 26);
    let mut buf = real::pending_for(&idx);
    let mut s = live::scratch_for(&idx);
    let v = random_unit(&mut SplitMix64::new(27));
    real::insert_real(&mut idx, &mut s, &mut buf, 800_001, 2_000, 0b100, REAL, &v).unwrap();
    expire_all(&mut idx, &mut s, 2_000 + HOUR);
    assert_eq!(real::search(&idx, &buf, &v, 1, Some(0b100), 0, 0.0)[0].id, 800_001);
    assert!(real::search(&idx, &buf, &v, 5, Some(0b010), 0, 0.0).iter().all(|h| h.id != 800_001));
}

#[test]
fn pending_scores_match_the_index_exact_rerank() {
    // With raw vectors stored, the index re-ranks with the exact score, so a
    // fresh real memory scores the same bits whichever side returns it.
    let mut idx = base_index(1000, 28);
    idx.params.nprobe = NLIST;
    let mut buf = real::pending_for(&idx);
    let mut s = live::scratch_for(&idx);
    let mut rng = SplitMix64::new(29);
    let v = random_unit(&mut rng);
    real::insert_real(&mut idx, &mut s, &mut buf, 600_001, 2_000, 0, REAL, &v).unwrap();
    let q = random_unit(&mut rng);
    let from_index = idx.search_temporal(&q, 2_000, None, 2_500, 1e-4).into_iter().find(|h| h.id == 600_001).unwrap();
    let merged = real::search(&idx, &buf, &q, 2_000, None, 2_500, 1e-4).into_iter().find(|h| h.id == 600_001).unwrap();
    assert_eq!(from_index.score.to_bits(), merged.score.to_bits());
}

// ------------------------------------------------------------ delete / upsert of a real id

fn apply_ok(idx: &mut IvfPqIndex, s: &mut live::Scratch, buf: &mut Pending, outbox: &mut Outbox, op: RealOp) -> RealApplied {
    real::apply(idx, s, buf, outbox, &op).unwrap()
}

#[test]
fn delete_of_a_pending_id_drops_it_and_it_is_never_committed() {
    let mut idx = base_index(1000, 30);
    let mut buf = real::pending_for(&idx);
    let mut outbox = Outbox::default();
    let mut s = live::scratch_for(&idx);
    let mut rng = SplitMix64::new(31);
    let (va, vb, vc) = (random_unit(&mut rng), random_unit(&mut rng), random_unit(&mut rng));
    for (id, v) in [(50_001u64, &va), (50_002, &vb), (50_003, &vc)] {
        apply_ok(&mut idx, &mut s, &mut buf, &mut outbox, RealOp::InsertReal { id, ts: 2_000, tags: 0, flags: REAL, vec: v.clone() });
    }
    // Deleted while still live in the index: both sides drop it.
    let r = apply_ok(&mut idx, &mut s, &mut buf, &mut outbox, RealOp::Live(Op::Delete { id: 50_001 }));
    assert!(matches!(r, RealApplied::DeletedReal { loc: Some(_), was_pending: true, outboxed: false }), "{r:?}");
    assert!(!real::is_pending(&buf, 50_001) && live::locate(&idx, 50_001).is_none());
    assert_ok(&idx, &buf);

    // Deleted after the 1-hour expiry (pending only): still a valid delete.
    expire_all(&mut idx, &mut s, 2_000 + HOUR);
    let r = apply_ok(&mut idx, &mut s, &mut buf, &mut outbox, RealOp::Live(Op::Delete { id: 50_002 }));
    assert_eq!(r, RealApplied::DeletedReal { loc: None, was_pending: true, outboxed: false });
    assert!(!search(&idx, &buf, &vb, 20).iter().any(|h| h.id == 50_002), "deleted real memory still searchable");
    assert_eq!(real::search(&idx, &buf, &vc, 1, None, 0, 0.0)[0].id, 50_003, "remaining row repointed wrongly");
    assert_ok(&idx, &buf);

    // A second delete of the same id is NotFound and changes nothing.
    let before = buf.clone();
    assert_eq!(real::apply(&mut idx, &mut s, &mut buf, &mut outbox, &RealOp::Live(Op::Delete { id: 50_002 })), Err(RealError::Write(WriteError::NotFound { id: 50_002 })));
    assert_eq!(buf, before);

    real::commit(&mut buf, &mut outbox);
    assert_eq!(outbox.records.iter().map(|r| r.id).collect::<Vec<_>>(), vec![50_003], "a deleted pending id reached the outbox");
}

#[test]
fn upsert_of_a_pending_id_replaces_the_row_and_keeps_legendary() {
    let mut idx = base_index(1000, 32);
    let mut buf = real::pending_for(&idx);
    let mut outbox = Outbox::default();
    let mut s = live::scratch_for(&idx);
    let mut rng = SplitMix64::new(33);
    let (v1, v2, v3) = (random_unit(&mut rng), random_unit(&mut rng), random_unit(&mut rng));
    apply_ok(&mut idx, &mut s, &mut buf, &mut outbox, RealOp::InsertReal { id: 60_001, ts: 2_000, tags: 0b1, flags: REAL | LEGENDARY, vec: v1.clone() });
    apply_ok(&mut idx, &mut s, &mut buf, &mut outbox, RealOp::InsertReal { id: 60_002, ts: 2_001, tags: 0, flags: REAL, vec: v3.clone() });

    // A plain upsert carries no flags: the pending row keeps REAL | LEGENDARY.
    let r = apply_ok(&mut idx, &mut s, &mut buf, &mut outbox, RealOp::Live(Op::Upsert { id: 60_001, ts: 2_100, tags: 0b10, vec: v2.clone() }));
    assert!(matches!(r, RealApplied::UpsertedReal { replaced: Some(_), flags, .. } if flags == REAL | LEGENDARY), "{r:?}");
    assert_eq!(real::pending_len(&buf), 2);
    assert_eq!(real::is_legendary(&buf, 60_001), Some(true));
    assert_ok(&idx, &buf);

    // After expiry the buffer answers with the new version only.
    expire_all(&mut idx, &mut s, 2_100 + HOUR);
    assert_eq!(search(&idx, &buf, &v2, 1)[0].id, 60_001);
    assert!(real::search(&idx, &buf, &v1, 20, Some(0b1), 0, 0.0).iter().all(|h| h.id != 60_001), "old tags still matched");

    real::commit(&mut buf, &mut outbox);
    assert_eq!(outbox.records.len(), 2);
    let rec = outbox.records.iter().find(|r| r.id == 60_001).unwrap();
    assert_eq!((rec.kind, rec.ts, rec.tags, rec.flags), (RecordKind::Put, 2_100, 0b10, REAL | LEGENDARY));
    assert_eq!(rec.vec, v2, "committed row is not the latest version");
    // Order is by the new ts: 60_002 (2_001) before 60_001 (2_100).
    assert_eq!(outbox.records.iter().map(|r| r.id).collect::<Vec<_>>(), vec![60_002, 60_001]);
}

#[test]
fn upsert_real_sets_flags_explicitly() {
    let mut idx = base_index(1000, 34);
    let mut buf = real::pending_for(&idx);
    let mut outbox = Outbox::default();
    let mut s = live::scratch_for(&idx);
    let mut rng = SplitMix64::new(35);
    apply_ok(&mut idx, &mut s, &mut buf, &mut outbox, RealOp::InsertReal { id: 61_001, ts: 2_000, tags: 0, flags: REAL | LEGENDARY, vec: random_unit(&mut rng) });
    // Explicit flags replace the old ones: LEGENDARY can be dropped on purpose.
    apply_ok(&mut idx, &mut s, &mut buf, &mut outbox, RealOp::UpsertReal { id: 61_001, ts: 2_001, tags: 0, flags: REAL, vec: random_unit(&mut rng) });
    assert_eq!(real::is_legendary(&buf, 61_001), Some(false));
    // A normal index row is promoted to real.
    live::insert(&mut idx, &mut s, 61_002, 2_002, 0, &random_unit(&mut rng)).unwrap();
    apply_ok(&mut idx, &mut s, &mut buf, &mut outbox, RealOp::UpsertReal { id: 61_002, ts: 2_003, tags: 0, flags: REAL | LEGENDARY, vec: random_unit(&mut rng) });
    assert_eq!(real::is_legendary(&buf, 61_002), Some(true));
    // A plain upsert of a normal row stays normal.
    live::insert(&mut idx, &mut s, 61_003, 2_004, 0, &random_unit(&mut rng)).unwrap();
    let r = apply_ok(&mut idx, &mut s, &mut buf, &mut outbox, RealOp::Live(Op::Upsert { id: 61_003, ts: 2_005, tags: 0, vec: random_unit(&mut rng) }));
    assert!(matches!(r, RealApplied::Live(live::Applied::Upserted { .. })));
    assert!(!real::is_pending(&buf, 61_003));
    // Bad flags change nothing.
    let (fp, before) = (idx.fingerprint(), buf.clone());
    for flags in [0u8, LEGENDARY, 0b100] {
        let op = RealOp::UpsertReal { id: 61_001, ts: 2_006, tags: 0, flags, vec: random_unit(&mut rng) };
        assert_eq!(real::apply(&mut idx, &mut s, &mut buf, &mut outbox, &op), Err(RealError::BadFlags { id: 61_001, flags }));
    }
    assert_eq!((idx.fingerprint(), &buf), (fp, &before));
    assert_ok(&idx, &buf);
}

#[test]
fn delete_of_a_handed_id_appends_a_delete_record_after_its_put() {
    let mut idx = base_index(1000, 36);
    let mut buf = real::pending_for(&idx);
    let mut outbox = Outbox::default();
    let mut s = live::scratch_for(&idx);
    let mut rng = SplitMix64::new(37);
    apply_ok(&mut idx, &mut s, &mut buf, &mut outbox, RealOp::InsertReal { id: 70_001, ts: 2_000, tags: 0, flags: REAL | LEGENDARY, vec: random_unit(&mut rng) });
    apply_ok(&mut idx, &mut s, &mut buf, &mut outbox, RealOp::InsertReal { id: 70_002, ts: 2_001, tags: 0, flags: REAL, vec: random_unit(&mut rng) });
    apply_ok(&mut idx, &mut s, &mut buf, &mut outbox, RealOp::Commit);
    assert!(real::is_handed(&buf, 70_001) && !real::is_pending(&buf, 70_001));

    let r = apply_ok(&mut idx, &mut s, &mut buf, &mut outbox, RealOp::Live(Op::Delete { id: 70_001 }));
    assert!(matches!(r, RealApplied::DeletedReal { loc: Some(_), was_pending: false, outboxed: true }), "{r:?}");
    assert!(!real::is_handed(&buf, 70_001));
    let kinds: Vec<(u64, RecordKind, u64, u8)> = outbox.records.iter().map(|r| (r.seq, r.kind, r.id, r.flags)).collect();
    assert_eq!(kinds, vec![
        (0, RecordKind::Put, 70_001, REAL | LEGENDARY),
        (1, RecordKind::Put, 70_002, REAL),
        (2, RecordKind::Delete, 70_001, REAL | LEGENDARY),
    ]);
    assert!(outbox.records[2].vec.is_empty());

    // Once acked, an id is no longer tracked: a delete is a plain index delete.
    apply_ok(&mut idx, &mut s, &mut buf, &mut outbox, RealOp::Ack { through_seq: 2 });
    assert!(outbox.records.is_empty() && !real::is_handed(&buf, 70_002));
    let r = apply_ok(&mut idx, &mut s, &mut buf, &mut outbox, RealOp::Live(Op::Delete { id: 70_002 }));
    assert!(matches!(r, RealApplied::Live(live::Applied::Deleted { .. })));
    assert!(outbox.records.is_empty());
    assert_ok(&idx, &buf);
}

#[test]
fn upsert_of_a_handed_id_is_pending_again_and_commits_a_newer_put() {
    let mut idx = base_index(1000, 38);
    let mut buf = real::pending_for(&idx);
    let mut outbox = Outbox::default();
    let mut s = live::scratch_for(&idx);
    let mut rng = SplitMix64::new(39);
    let (v1, v2) = (random_unit(&mut rng), random_unit(&mut rng));
    apply_ok(&mut idx, &mut s, &mut buf, &mut outbox, RealOp::InsertReal { id: 71_001, ts: 2_000, tags: 0, flags: REAL | LEGENDARY, vec: v1 });
    apply_ok(&mut idx, &mut s, &mut buf, &mut outbox, RealOp::Commit);
    // Insert of a handed id is refused, like a pending one.
    let ins = RealOp::Live(Op::Insert { id: 71_001, ts: 2_100, tags: 0, vec: v2.clone() });
    expire_all(&mut idx, &mut s, 2_000 + HOUR);
    assert_eq!(real::apply(&mut idx, &mut s, &mut buf, &mut outbox, &ins), Err(RealError::Write(WriteError::DuplicateId { id: 71_001 })));

    let r = apply_ok(&mut idx, &mut s, &mut buf, &mut outbox, RealOp::Live(Op::Upsert { id: 71_001, ts: 2_000 + HOUR, tags: 0, vec: v2.clone() }));
    assert!(matches!(r, RealApplied::UpsertedReal { replaced: None, flags, .. } if flags == REAL | LEGENDARY), "{r:?}");
    assert!(real::is_pending(&buf, 71_001) && real::is_handed(&buf, 71_001));
    assert_ok(&idx, &buf);

    // Acking the first Put keeps nothing stale: the id is still pending.
    apply_ok(&mut idx, &mut s, &mut buf, &mut outbox, RealOp::Ack { through_seq: 0 });
    assert!(!real::is_handed(&buf, 71_001) && real::is_pending(&buf, 71_001));
    apply_ok(&mut idx, &mut s, &mut buf, &mut outbox, RealOp::Commit);
    assert_eq!(outbox.records.len(), 1);
    let rec = &outbox.records[0];
    assert_eq!((rec.seq, rec.kind, rec.id, rec.flags), (1, RecordKind::Put, 71_001, REAL | LEGENDARY));
    assert_eq!(rec.vec, v2);
}

#[test]
fn ack_ahead_of_emitted_records_is_refused() {
    let mut idx = base_index(500, 40);
    let mut buf = real::pending_for(&idx);
    let mut outbox = Outbox::default();
    let mut s = live::scratch_for(&idx);
    apply_ok(&mut idx, &mut s, &mut buf, &mut outbox, RealOp::InsertReal { id: 72_001, ts: 2_000, tags: 0, flags: REAL, vec: random_unit(&mut SplitMix64::new(41)) });
    assert_eq!(real::apply(&mut idx, &mut s, &mut buf, &mut outbox, &RealOp::Ack { through_seq: 0 }), Err(RealError::AckAhead { through_seq: 0, next_seq: 0 }));
    apply_ok(&mut idx, &mut s, &mut buf, &mut outbox, RealOp::Commit);
    let (b, o) = (buf.clone(), outbox.clone());
    assert_eq!(real::apply(&mut idx, &mut s, &mut buf, &mut outbox, &RealOp::Ack { through_seq: 1 }), Err(RealError::AckAhead { through_seq: 1, next_seq: 1 }));
    assert_eq!((buf, outbox), (b, o));
}

/// A seeded log mixing real and normal inserts, deletes and upserts of
/// real ids (pending, and handed after a commit), expiry, commits and
/// acks. Deletes and upserts only name ids this side still tracks: ids
/// inserted since the commit before last (acks trail one commit behind).
fn churn_log(seed: u64, first_id: u64, first_ts: u64) -> Vec<RealOp> {
    let mut rng = SplitMix64::new(seed);
    let mut ops = Vec::new();
    let mut handed: Vec<u64> = Vec::new();
    let mut pending: Vec<u64> = Vec::new();
    for i in 0..1_200u64 {
        let ts = first_ts + i * 60;
        let roll = rng.next_u64() % 100;
        let known = handed.len() + pending.len();
        if roll < 20 && known > 0 {
            let k = (rng.next_u64() % known as u64) as usize;
            let id = if k < handed.len() { handed.remove(k) } else { pending.remove(k - handed.len()) };
            if roll < 10 {
                ops.push(RealOp::Live(Op::Delete { id }));
            } else {
                ops.push(RealOp::Live(Op::Upsert { id, ts, tags: rng.next_u64() & 0xF, vec: random_unit(&mut rng) }));
                pending.push(id);
            }
        } else if roll < 40 {
            let flags = if roll % 3 == 0 { REAL | LEGENDARY } else { REAL };
            pending.push(first_id + i);
            ops.push(RealOp::InsertReal { id: first_id + i, ts, tags: rng.next_u64() & 0xF, flags, vec: random_unit(&mut rng) });
        } else {
            ops.push(RealOp::Live(Op::Insert { id: first_id + i, ts, tags: rng.next_u64() & 0xF, vec: random_unit(&mut rng) }));
        }
        if i % 60 == 59 { ops.push(RealOp::Live(Op::Expire { before_ts: ts.saturating_sub(HOUR), max_rows: usize::MAX })); }
        if i % 240 == 239 {
            // Ack trails one commit: it covers what the previous commit emitted.
            ops.push(RealOp::Ack { through_seq: u64::MAX });
            ops.push(RealOp::Commit);
            handed = std::mem::take(&mut pending);
        }
    }
    ops
}

/// Replay `ops`, turning each `Ack` into "ack everything before the last
/// commit's records" (skipped before the first commit). `last_commit` is
/// the `next_seq` seen at the last commit; it is part of the state a
/// snapshot must keep, like `buf`.
fn replay_churn(idx: &mut IvfPqIndex, buf: &mut Pending, outbox: &mut Outbox, last_commit: &mut Option<u64>, ops: &[RealOp]) {
    let mut s = live::scratch_for(idx);
    for op in ops {
        let op = match op {
            RealOp::Ack { .. } => match *last_commit {
                Some(seq) if seq < buf.next_seq => RealOp::Ack { through_seq: buf.next_seq - 1 },
                _ => continue,
            },
            RealOp::Commit => { *last_commit = Some(buf.next_seq); RealOp::Commit }
            other => other.clone(),
        };
        real::apply(idx, &mut s, buf, outbox, &op).unwrap();
    }
}

#[test]
fn churn_with_deletes_upserts_and_acks_replays_identically_and_reruns_append_nothing() {
    let base = base_index(1500, 42);
    let ops = churn_log(43, 8_000_000, 10_000);
    let run = || {
        let mut idx = base.clone();
        let mut buf = real::pending_for(&idx);
        let mut outbox = Outbox::default();
        replay_churn(&mut idx, &mut buf, &mut outbox, &mut None, &ops);
        assert_ok(&idx, &buf);
        (idx.fingerprint(), buf, outbox)
    };
    let (fa, ba, oa) = run();
    let (fb, bb, ob) = run();
    assert_eq!((fa, &ba, &oa), (fb, &bb, &ob));
    assert!(ba.next_seq > 100, "log should emit records: {}", ba.next_seq);
    assert!(oa.records.windows(2).all(|w| w[0].seq < w[1].seq), "outbox not in seq order");

    // Every Delete record follows a Put of the same id; no deleted pending
    // id was ever committed after its delete.
    let mut idx = base.clone();
    let mut buf = real::pending_for(&idx);
    let mut all = Outbox::default();
    let mut s = live::scratch_for(&idx);
    let mut deletes_of_handed = 0;
    for op in &ops {
        if matches!(op, RealOp::Ack { .. }) { continue; }
        if let RealApplied::DeletedReal { outboxed: true, .. } = real::apply(&mut idx, &mut s, &mut buf, &mut all, op).unwrap() { deletes_of_handed += 1; }
    }
    assert!(deletes_of_handed > 5, "test needs deletes of handed ids: {deletes_of_handed}");
    for (i, r) in all.records.iter().enumerate().filter(|(_, r)| r.kind == RecordKind::Delete) {
        assert!(all.records[..i].iter().any(|p| p.id == r.id && p.kind == RecordKind::Put), "Delete of {} with no Put before it", r.id);
        assert!(!all.records[i + 1..].iter().any(|p| p.id == r.id), "id {} committed after its delete", r.id);
    }

    // Crash: the outbox survives; index, buffer and ack bookkeeping come
    // back from a snapshot taken halfway, and the tail is replayed against
    // the surviving outbox. Nothing it already took is appended again.
    let cut = ops.len() / 2;
    let mut idx = base.clone();
    let mut buf = real::pending_for(&idx);
    let mut outbox = Outbox::default();
    let mut last = None;
    replay_churn(&mut idx, &mut buf, &mut outbox, &mut last, &ops[..cut]);
    let snap = (idx.clone(), buf.clone(), last);
    replay_churn(&mut idx, &mut buf, &mut outbox, &mut last, &ops[cut..]);
    assert_eq!((&buf, &outbox), (&ba, &oa), "split replay differs from one pass");
    let (mut idx, mut buf, mut last) = snap;
    replay_churn(&mut idx, &mut buf, &mut outbox, &mut last, &ops[cut..]);
    assert_eq!((&buf, &outbox), (&ba, &oa), "replayed tail appended records twice");
}

// ------------------------------------------------------------ the server owns the clock

#[test]
fn nothing_commits_without_the_commit_op() {
    // Two days of server time, expiry running, no Commit op: every real
    // memory stays pending and the outbox stays empty.
    let mut idx = base_index(1000, 44);
    let mut buf = real::pending_for(&idx);
    let mut outbox = Outbox::default();
    let ops = mixed_inserts(2_000, 45, 9_000_000, 0);
    let mut s = live::scratch_for(&idx);
    for (i, op) in ops.iter().enumerate() {
        // Stretch time: one insert every 87 s, so the log spans over 48 hours.
        let ts = 1_000 + i as u64 * 87;
        let op = match op {
            RealOp::InsertReal { id, tags, flags, vec, .. } => RealOp::InsertReal { id: *id, ts, tags: *tags, flags: *flags, vec: vec.clone() },
            RealOp::Live(Op::Insert { id, tags, vec, .. }) => RealOp::Live(Op::Insert { id: *id, ts, tags: *tags, vec: vec.clone() }),
            _ => unreachable!(),
        };
        real::apply(&mut idx, &mut s, &mut buf, &mut outbox, &op).unwrap();
        if i % 100 == 99 { expire_all(&mut idx, &mut s, ts.saturating_sub(HOUR)); }
    }
    const { assert!(1_999 * 87 > 48 * HOUR) };
    assert_eq!(real::pending_len(&buf), 286, "every 7th of 2000 is real");
    assert!(outbox.records.is_empty() && outbox.next_seq == 0 && buf.next_seq == 0, "something committed without the op");
    // The one Commit the server sends drains it all.
    real::apply(&mut idx, &mut s, &mut buf, &mut outbox, &RealOp::Commit).unwrap();
    assert_eq!(outbox.records.len(), 286);
}

#[test]
fn write_path_reads_no_clock() {
    // Commit timing is the server's: the write-path modules hold no timer
    // and no wall-clock read.
    for (name, src) in [("real_memory.rs", include_str!("../src/real_memory.rs")), ("live.rs", include_str!("../src/live.rs"))] {
        for banned in ["SystemTime", "Instant", "std::time", "thread::sleep", "spawn("] {
            assert!(!src.contains(banned), "{name} uses {banned}");
        }
    }
}
