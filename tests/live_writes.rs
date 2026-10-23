//! Live write path for IVF-PQ: insert, delete, upsert, expire, compact.
//!
//! Every write goes through `live::apply` with an `Op`, in log order, on
//! one thread. These tests pin the guarantees from the design doc:
//! - an inserted row is searchable at once;
//! - live inserts produce the same bytes as a batch build with the same
//!   model (so batch recall numbers hold for live data);
//! - the same op log always gives the same index and the same results;
//! - deleted and expired rows never come back, and k results still come
//!   back when k live rows match;
//! - bad input is rejected before anything changes;
//! - compaction never changes search results.

mod common;
use common::*;

use std::collections::BTreeMap;

use raggedy_anndy::live::{self, Applied, Op, WriteError};
use raggedy_anndy::seed::SplitMix64;
use raggedy_anndy::{FlatIndex, Hit, IvfPqIndex, Metric};

const DIM: usize = 32;
const NLIST: usize = 16;
const M: usize = 8;

fn base_index(n: usize, seed: u64) -> (IvfPqIndex, Dataset) {
    let ds = uniform_sphere(n, DIM, seed);
    let rows: Vec<(u64, Vec<f32>, u64)> = (0..n).map(|i| (ds.ids[i], row(&ds, i).to_vec(), i as u64)).collect();
    let idx = IvfPqIndex::build(Metric::Cosine, DIM, &rows, ivfpq_params(NLIST, 4, 64, M, 7));
    (idx, ds)
}

fn random_unit(rng: &mut SplitMix64) -> Vec<f32> {
    let mut v: Vec<f32> = (0..DIM).map(|_| gauss(rng)).collect();
    normalize(&mut v);
    v
}

fn ids(hits: &[Hit]) -> Vec<u64> { hits.iter().map(|h| h.id).collect() }

fn bits(hits: &[Hit]) -> Vec<(u64, u32)> { hits.iter().map(|h| (h.id, h.score.to_bits())).collect() }

fn insert(idx: &mut IvfPqIndex, s: &mut live::Scratch, id: u64, ts: u64, tags: u64, v: &[f32]) -> Result<Applied, WriteError> {
    live::apply(idx, s, &Op::Insert { id, ts, tags, vec: v.to_vec() })
}

fn assert_ok(idx: &IvfPqIndex) {
    if let Err(e) = live::check_invariants(idx) { panic!("invariant broken: {e}"); }
}

// ------------------------------------------------------------ visibility

#[test]
fn inserted_row_is_rank_1_immediately() {
    let (mut idx, _) = base_index(3000, 1);
    let mut s = live::scratch_for(&idx);
    let mut rng = SplitMix64::new(77);
    for t in 0..20u64 {
        let v = random_unit(&mut rng);
        let id = 1_000_000 + t;
        insert(&mut idx, &mut s, id, 10_000 + t, 0, &v).unwrap();
        let hits = idx.search(&v, 5);
        assert_eq!(hits[0].id, id, "insert {t}: got {:?}", ids(&hits));
    }
    assert_ok(&idx);
}

#[test]
fn inserted_row_is_in_pq_shortlist_without_raw_vectors() {
    let ds = uniform_sphere(3000, DIM, 2);
    let rows: Vec<(u64, Vec<f32>, u64)> = (0..3000).map(|i| (ds.ids[i], row(&ds, i).to_vec(), 0)).collect();
    let mut p = ivfpq_params(NLIST, 4, 64, M, 7);
    p.store_vecs = false;
    let mut idx = IvfPqIndex::build(Metric::Cosine, DIM, &rows, p);
    let mut s = live::scratch_for(&idx);
    let mut rng = SplitMix64::new(78);
    for t in 0..20u64 {
        let v = random_unit(&mut rng);
        insert(&mut idx, &mut s, 2_000_000 + t, 1, 0, &v).unwrap();
        let hits = idx.search(&v, idx.params.refine);
        assert!(hits.iter().any(|h| h.id == 2_000_000 + t), "insert {t} missing from PQ shortlist");
    }
}

// ------------------------------------------------------------ G4: live == build

#[test]
fn live_inserts_are_byte_identical_to_build() {
    // Model from build(A). Path Y: build(A), then insert B. Path X: same
    // model with no rows, insert A then B. Same op order, same bytes.
    let (built_a, ds_a) = base_index(3000, 3);
    let ds_b = uniform_sphere(1000, DIM, 4);

    let mut y = built_a.clone();
    let mut x = live::empty_like(&built_a);
    let mut s = live::scratch_for(&built_a);
    for i in 0..3000 {
        insert(&mut x, &mut s, ds_a.ids[i], i as u64, 0, row(&ds_a, i)).unwrap();
    }
    assert_eq!(x.fingerprint(), built_a.fingerprint(), "inserting A into an empty index differs from build(A)");

    for i in 0..1000 {
        let id = 10_000 + i as u64;
        let ts = 5_000 + i as u64;
        insert(&mut x, &mut s, id, ts, 0, row(&ds_b, i)).unwrap();
        insert(&mut y, &mut s, id, ts, 0, row(&ds_b, i)).unwrap();
    }
    assert_eq!(x.fingerprint(), y.fingerprint());
    assert_ok(&x);
    assert_ok(&y);
}

// ------------------------------------------------------------ G1/G2: replay

/// Seeded random op log over a mix of inserts, upserts, deletes, expiries
/// and compactions. Timestamps only move forward, as server ticks do.
fn random_log(n_ops: usize, seed: u64, first_id: u64) -> Vec<Op> {
    let mut rng = SplitMix64::new(seed);
    let mut live_ids: Vec<u64> = Vec::new();
    let mut next_id = first_id;
    let mut ts = 100_000u64;
    let mut ops = Vec::with_capacity(n_ops);
    for _ in 0..n_ops {
        ts += rng.gen_range(3) as u64;
        let roll = rng.gen_range(100);
        if roll < 55 || live_ids.is_empty() {
            let id = next_id; next_id += 1;
            live_ids.push(id);
            ops.push(Op::Insert { id, ts, tags: rng.next_u64() & 0xF, vec: random_unit(&mut rng) });
        } else if roll < 70 {
            let id = live_ids[rng.gen_range(live_ids.len())];
            ops.push(Op::Upsert { id, ts, tags: 0, vec: random_unit(&mut rng) });
        } else if roll < 90 {
            let id = live_ids.swap_remove(rng.gen_range(live_ids.len()));
            ops.push(Op::Delete { id });
        } else if roll < 97 {
            ops.push(Op::Expire { before_ts: ts.saturating_sub(400), max_rows: 64 });
        } else {
            ops.push(Op::Compact { list: rng.gen_range(NLIST) as u32 });
        }
    }
    ops
}

/// Applies `ops` in order. Expire may drop ids the generator still thinks
/// are live, so later Delete/Upsert of those ids may fail with NotFound;
/// that outcome is part of the deterministic result and is returned.
fn replay(idx: &mut IvfPqIndex, ops: &[Op]) -> Vec<Result<Applied, WriteError>> {
    let mut s = live::scratch_for(idx);
    ops.iter().map(|op| live::apply(idx, &mut s, op)).collect()
}

#[test]
fn same_op_log_gives_same_index_and_results() {
    let (base, _) = base_index(2000, 5);
    let ops = random_log(6000, 99, 1_000_000);
    let mut a = base.clone();
    let mut b = base.clone();
    let ra = replay(&mut a, &ops);
    let rb = replay(&mut b, &ops);
    assert_eq!(ra, rb);
    assert_eq!(a.fingerprint(), b.fingerprint());
    assert_ok(&a);

    let mut rng = SplitMix64::new(6);
    for _ in 0..50 {
        let q = random_unit(&mut rng);
        assert_eq!(bits(&a.search(&q, 10)), bits(&b.search(&q, 10)));
        assert_eq!(bits(&a.search_temporal(&q, 10, Some(0x3), 120_000, 1e-4)),
                   bits(&b.search_temporal(&q, 10, Some(0x3), 120_000, 1e-4)));
    }
}

// ------------------------------------------------------------ deletes

#[test]
fn deleted_ids_never_return_and_k_is_kept() {
    let (mut idx, ds) = base_index(3000, 7);
    let mut s = live::scratch_for(&idx);
    let q = row(&ds, 10).to_vec();
    let before = idx.search(&q, 10);
    for h in &before[..5] {
        assert!(matches!(live::apply(&mut idx, &mut s, &Op::Delete { id: h.id }), Ok(Applied::Deleted { .. })));
    }
    let after = idx.search(&q, 10);
    assert_eq!(after.len(), 10, "deletes must not shrink results below k");
    for h in &before[..5] { assert!(!after.iter().any(|a| a.id == h.id), "deleted id {} returned", h.id); }
    assert_eq!(live::live_rows(&idx), 2995);
    assert_ok(&idx);
}

#[test]
fn delete_then_reinsert_returns_only_the_new_vector() {
    let (mut idx, ds) = base_index(3000, 8);
    let mut s = live::scratch_for(&idx);
    let id = ds.ids[42];
    live::apply(&mut idx, &mut s, &Op::Delete { id }).unwrap();
    let mut rng = SplitMix64::new(9);
    let v = random_unit(&mut rng);
    insert(&mut idx, &mut s, id, 9_000, 0, &v).unwrap();

    let hits = idx.search(&v, 5);
    assert_eq!(hits[0].id, id);
    assert_eq!(hits.iter().filter(|h| h.id == id).count(), 1);
    let old = idx.search(row(&ds, 42), 50);
    assert!(old.iter().filter(|h| h.id == id).count() <= 1, "old and new copies both live");
    assert_ok(&idx);
}

#[test]
fn upsert_into_another_list_leaves_no_stale_copy() {
    let (mut idx, ds) = base_index(3000, 10);
    let mut s = live::scratch_for(&idx);
    let id = ds.ids[7];
    let old_loc = live::locate(&idx, id).unwrap();
    // Pick a target vector that lands in a different list.
    let mut rng = SplitMix64::new(11);
    let v = loop {
        let v = random_unit(&mut rng);
        if live::nearest_list(&idx, &v) != old_loc.list as usize { break v; }
    };
    let live_before = live::live_rows(&idx);
    live::apply(&mut idx, &mut s, &Op::Upsert { id, ts: 9_000, tags: 0, vec: v.clone() }).unwrap();
    let new_loc = live::locate(&idx, id).unwrap();
    assert_ne!(new_loc.list, old_loc.list);
    assert_eq!(idx.lists[old_loc.list as usize].tomb[old_loc.row as usize], 1, "old row still live");
    assert_eq!(live::live_rows(&idx), live_before);
    assert_eq!(idx.search(&v, 1)[0].id, id);
    assert_ok(&idx);
}

#[test]
fn upsert_of_missing_id_inserts() {
    let (mut idx, _) = base_index(1000, 12);
    let mut s = live::scratch_for(&idx);
    let mut rng = SplitMix64::new(13);
    let v = random_unit(&mut rng);
    live::apply(&mut idx, &mut s, &Op::Upsert { id: 555_555, ts: 2_000, tags: 0, vec: v.clone() }).unwrap();
    assert_eq!(idx.search(&v, 1)[0].id, 555_555);
}

// ------------------------------------------------------------ compaction

#[test]
fn compaction_does_not_change_results() {
    let (mut idx, _) = base_index(3000, 14);
    let ops = random_log(4000, 15, 1_000_000);
    let _ = replay(&mut idx, &ops);
    let mut rng = SplitMix64::new(16);
    let queries: Vec<Vec<f32>> = (0..40).map(|_| random_unit(&mut rng)).collect();
    let before: Vec<_> = queries.iter().map(|q| bits(&idx.search(q, 10))).collect();
    let tagged: Vec<_> = queries.iter().map(|q| bits(&idx.search_temporal(q, 10, Some(0x5), 110_000, 1e-4))).collect();

    let mut s = live::scratch_for(&idx);
    for l in 0..NLIST as u32 { live::apply(&mut idx, &mut s, &Op::Compact { list: l }).unwrap(); }
    assert!(idx.lists.iter().all(|l| l.tomb.iter().all(|&t| t == 0)), "compaction left dead rows");
    assert_ok(&idx);

    for (i, q) in queries.iter().enumerate() {
        assert_eq!(bits(&idx.search(q, 10)), before[i], "query {i}");
        assert_eq!(bits(&idx.search_temporal(q, 10, Some(0x5), 110_000, 1e-4)), tagged[i], "tagged query {i}");
    }
}

#[test]
fn compaction_policy_picks_lists_over_the_dead_ratio() {
    let (mut idx, _) = base_index(2000, 17);
    let mut s = live::scratch_for(&idx);
    assert!(live::lists_to_compact(&idx, 0.25).is_empty());
    // Kill 60% of list 0.
    let ids: Vec<u64> = idx.lists[0].ids.clone();
    for id in ids.iter().take(ids.len() * 6 / 10) { live::apply(&mut idx, &mut s, &Op::Delete { id: *id }).unwrap(); }
    assert_eq!(live::lists_to_compact(&idx, 0.25), vec![0]);
}

// ------------------------------------------------------------ oracle

/// Random op logs checked against a plain map of live rows and an exact
/// Flat search. With every list probed and refine >= live rows the index
/// must return exactly the Flat top-k.
#[test]
fn matches_exact_oracle_over_random_logs() {
    for seed in [21u64, 22, 23] {
        let (base, ds) = base_index(1500, seed);
        let mut idx = base.clone();
        idx.params.nprobe = NLIST;
        idx.params.refine = 100_000;
        let mut oracle: BTreeMap<u64, Vec<f32>> = (0..1500).map(|i| (ds.ids[i], row(&ds, i).to_vec())).collect();
        let ops = random_log(3000, seed * 31, 1_000_000);
        let mut s = live::scratch_for(&idx);
        for (n, op) in ops.iter().enumerate() {
            let r = live::apply(&mut idx, &mut s, op);
            match (op, &r) {
                (Op::Insert { id, vec, .. }, Ok(_)) | (Op::Upsert { id, vec, .. }, Ok(_)) => { oracle.insert(*id, vec.clone()); }
                (Op::Delete { id }, Ok(_)) => { oracle.remove(id); }
                (Op::Expire { .. }, Ok(Applied::Expired { ids, .. })) => { for id in ids { oracle.remove(id); } }
                _ => {}
            }
            if n % 750 == 749 {
                assert_ok(&idx);
                assert_eq!(live::live_rows(&idx), oracle.len());
                let mut flat = FlatIndex::new(DIM, Metric::Cosine);
                for (id, v) in &oracle { flat.add(*id, v); }
                let mut rng = SplitMix64::new(seed + n as u64);
                for _ in 0..10 {
                    let q = random_unit(&mut rng);
                    assert_eq!(ids(&idx.search(&q, 10)), ids(&flat.search(&q, 10)), "seed {seed} op {n}");
                }
            }
        }
    }
}

// ------------------------------------------------------------ G7: bad input

#[test]
fn bad_input_is_rejected_without_changes() {
    let (mut idx, ds) = base_index(1000, 30);
    let mut s = live::scratch_for(&idx);
    let fp = idx.fingerprint();
    let good = row(&ds, 0).to_vec();
    let mut nan = good.clone(); nan[3] = f32::NAN;
    let mut inf = good.clone(); inf[0] = f32::INFINITY;
    let max_ts = idx.max_ts;

    let cases: Vec<(Op, WriteError)> = vec![
        (Op::Insert { id: 9_999, ts: max_ts, tags: 0, vec: nan.clone() }, WriteError::NonFinite { id: 9_999 }),
        (Op::Insert { id: 9_999, ts: max_ts, tags: 0, vec: inf }, WriteError::NonFinite { id: 9_999 }),
        (Op::Insert { id: 9_999, ts: max_ts, tags: 0, vec: good[..DIM - 1].to_vec() }, WriteError::WrongDim { id: 9_999, got: DIM - 1, want: DIM }),
        (Op::Insert { id: ds.ids[5], ts: max_ts, tags: 0, vec: good.clone() }, WriteError::DuplicateId { id: ds.ids[5] }),
        (Op::Insert { id: 9_999, ts: max_ts - 1, tags: 0, vec: good.clone() }, WriteError::TsBackwards { id: 9_999, ts: max_ts - 1, last: max_ts }),
        (Op::Upsert { id: ds.ids[5], ts: max_ts, tags: 0, vec: nan }, WriteError::NonFinite { id: ds.ids[5] }),
        (Op::Delete { id: 123_456_789 }, WriteError::NotFound { id: 123_456_789 }),
        (Op::Compact { list: NLIST as u32 }, WriteError::NoSuchList { list: NLIST as u32 }),
    ];
    for (op, want) in cases {
        assert_eq!(live::apply(&mut idx, &mut s, &op), Err(want.clone()), "{op:?}");
        assert_eq!(idx.fingerprint(), fp, "{want:?} changed the index");
    }
    // A rejected upsert must not have deleted the old row.
    assert!(live::locate(&idx, ds.ids[5]).is_some());
}

// ------------------------------------------------------------ expiry

fn timed_index(n: usize, seed: u64) -> (IvfPqIndex, Dataset) {
    // Rows get ts = 0..n in input order, like events arriving tick by tick.
    base_index(n, seed)
}

#[test]
fn expired_rows_never_return() {
    let (mut idx, ds) = timed_index(3000, 40);
    let mut s = live::scratch_for(&idx);
    let r = live::apply(&mut idx, &mut s, &Op::Expire { before_ts: 1500, max_rows: usize::MAX }).unwrap();
    match r {
        Applied::Expired { ids, done } => { assert!(done); assert_eq!(ids.len(), 1500); }
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(live::live_rows(&idx), 1500);
    for i in (0..3000).step_by(37) {
        let hits = idx.search(row(&ds, i), 20);
        if i < 1500 { assert!(!hits.iter().any(|h| h.id == ds.ids[i]), "expired id {i} returned"); }
        else { assert_eq!(hits[0].id, ds.ids[i], "live id {i} lost"); }
    }
    assert_ok(&idx);
}

#[test]
fn capped_expiry_over_ticks_matches_one_uncapped_pass() {
    let (base, _) = timed_index(3000, 41);
    let mut a = base.clone();
    let mut b = base.clone();
    let mut s = live::scratch_for(&base);
    live::apply(&mut a, &mut s, &Op::Expire { before_ts: 2000, max_rows: usize::MAX }).unwrap();
    let mut ticks = 0;
    loop {
        ticks += 1;
        match live::apply(&mut b, &mut s, &Op::Expire { before_ts: 2000, max_rows: 150 }).unwrap() {
            Applied::Expired { ids, done } => { assert!(ids.len() <= 150); if done { break; } }
            other => panic!("unexpected {other:?}"),
        }
        assert!(ticks < 100, "capped expiry never finished");
    }
    assert!(ticks > 1);
    assert_eq!(a.fingerprint(), b.fingerprint());
}

#[test]
fn expiry_then_compaction_frees_rows() {
    let (mut idx, _) = timed_index(3000, 42);
    let mut s = live::scratch_for(&idx);
    live::apply(&mut idx, &mut s, &Op::Expire { before_ts: 2400, max_rows: usize::MAX }).unwrap();
    let mut lists = live::lists_to_compact(&idx, 0.25);
    assert!(!lists.is_empty());
    for l in lists.drain(..) { live::apply(&mut idx, &mut s, &Op::Compact { list: l }).unwrap(); }
    let stored: usize = idx.lists.iter().map(|l| l.ids.len()).sum();
    assert!(stored < 1000, "stored rows after compaction: {stored}");
    assert_eq!(live::live_rows(&idx), 600);
    assert_ok(&idx);
}

// ------------------------------------------------------------ tags

#[test]
fn tags_on_inserted_rows_filter_search() {
    let (mut idx, _) = base_index(2000, 50);
    let mut s = live::scratch_for(&idx);
    let mut rng = SplitMix64::new(51);
    let v = random_unit(&mut rng);
    insert(&mut idx, &mut s, 700_001, 5_000, 0b100, &v).unwrap();
    assert_eq!(idx.search_with_filter(&v, 1, Some(0b100))[0].id, 700_001);
    assert!(idx.search_with_filter(&v, 5, Some(0b010)).iter().all(|h| h.id != 700_001));
}

// ------------------------------------------------------------ timing

/// Insert cost on the writer thread. Not a gate: prints p50/p99 so the
/// numbers in the design doc can be replaced with measured ones.
/// cargo test --release --test live_writes insert_latency -- --ignored --nocapture
#[test]
#[ignore = "timing aid"]
fn insert_latency() {
    for &(n, dim, nlist, m) in &[(100_000usize, 128usize, 256usize, 32usize), (100_000, 256, 1024, 32)] {
        let ds = uniform_sphere(n, dim, 60);
        let rows: Vec<(u64, Vec<f32>, u64)> = (0..n).map(|i| (ds.ids[i], row(&ds, i).to_vec(), 0)).collect();
        let mut idx = IvfPqIndex::build(Metric::Cosine, dim, &rows, ivfpq_params(nlist, 8, 64, m, 7));
        live::reserve(&mut idx, 4_000);
        let mut s = live::scratch_for(&idx);
        let mut rng = SplitMix64::new(61);
        let vs: Vec<Vec<f32>> = (0..4_000).map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| gauss(&mut rng)).collect(); normalize(&mut v); v
        }).collect();
        let mut ns: Vec<u128> = Vec::with_capacity(vs.len());
        for (i, v) in vs.iter().enumerate() {
            let t = std::time::Instant::now();
            live::insert(&mut idx, &mut s, 10_000_000 + i as u64, 1, 0, v).unwrap();
            ns.push(t.elapsed().as_nanos());
        }
        ns.sort_unstable();
        let p = |q: f64| ns[((ns.len() as f64 - 1.0) * q) as usize] as f64 / 1000.0;
        println!("insert n={n} dim={dim} nlist={nlist} m={m}: p50 {:.1} us, p99 {:.1} us, max {:.1} us", p(0.5), p(0.99), p(1.0));
    }
}
