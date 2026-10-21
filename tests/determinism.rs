//! Determinism: builds must be bit-identical for a given (data, params,
//! seed) regardless of machine core count, and search must order results
//! by (score desc, id asc) with no panics on odd input.
//!
//! Tests marked `#[ignore = "KNOWN-FAIL ..."]` fail on master @ df23a59 and
//! document the target behaviour. Run them with:
//!     cargo test --release --test determinism -- --ignored
//! When a fix makes one pass, delete its `#[ignore]` line in the same PR.

mod common;
use common::*;

use raggedy_anndy::{FlatIndex, Hit, IvfIndex, IvfParams, IvfPqIndex, Metric};
use std::panic::{catch_unwind, AssertUnwindSafe};

const DIM: usize = 32;
const N: usize = 3_000;
const NLIST: usize = 32;
const M: usize = 8;

fn build_fixture() -> IvfPqIndex {
    let data = clustered(N, DIM, &cluster_centres(16, DIM, 1234), 0.2, 42);
    IvfPqIndex::build(Metric::Cosine, DIM, &ivfpq_rows(&data), ivfpq_params(NLIST, 8, 64, M, 7))
}

fn ids(h: &[Hit]) -> Vec<u64> { h.iter().map(|x| x.id).collect() }

// ------------------------------------------------------------ build identity

#[test]
fn ivfpq_two_builds_in_one_process_are_bit_identical() {
    let a = build_fixture();
    let b = build_fixture();
    assert_eq!(ivfpq_build_digest(&a), ivfpq_build_digest(&b));
}

/// Entry point for the cross-core-count test below. Does nothing unless
/// the parent sets RA_DIGEST_CHILD, so a plain `--ignored` run is a no-op.
#[test]
#[ignore = "helper: spawned by ivfpq_build_is_identical_on_1_core_and_all_cores"]
fn child_print_ivfpq_build_digest() {
    if std::env::var_os("RA_DIGEST_CHILD").is_none() { return; }
    let par = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
    println!("RA_PAR={par}");
    println!("RA_DIGEST={:016x}", ivfpq_build_digest(&build_fixture()));
}

/// IvfPqParams has no thread-count knob and build() sizes its k-means
/// fan-out from available_parallelism(), so the only way to vary it from a
/// test is to re-run this binary pinned to different CPU sets. A game
/// server and an offline builder with different core counts must produce
/// the same index from the same snapshot.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "KNOWN-FAIL: IVF-PQ k-means reduction order depends on core count (findings §3.3)"]
fn ivfpq_build_is_identical_on_1_core_and_all_cores() {
    use std::process::Command;
    let ncpu = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    assert!(ncpu >= 2, "needs >= 2 CPUs to compare against a 1-CPU build (have {ncpu})");
    let exe = std::env::current_exe().unwrap();
    let run = |cpus: &str| -> (String, String) {
        let out = Command::new("taskset")
            .args(["-c", cpus]).arg(&exe)
            .args(["child_print_ivfpq_build_digest", "--exact", "--ignored", "--nocapture", "--test-threads=1"])
            .env("RA_DIGEST_CHILD", "1")
            .output()
            .expect("taskset not found (util-linux)");
        let s = String::from_utf8_lossy(&out.stdout).into_owned();
        assert!(out.status.success(), "child failed: {s}\n{}", String::from_utf8_lossy(&out.stderr));
        // libtest may print "test name ... " on the same line, so match anywhere.
        let get = |key: &str| s.lines()
            .find_map(|l| l.find(key).map(|p| l[p + key.len()..].trim().to_string()))
            .unwrap_or_else(|| panic!("no {key} in {s}"));
        (get("RA_PAR="), get("RA_DIGEST="))
    };
    let (p1, d1) = run("0");
    let (pn, dn) = run(&format!("0-{}", ncpu - 1));
    assert_eq!(p1, "1", "pinning to one CPU did not reduce available_parallelism");
    assert_ne!(p1, pn);
    assert_eq!(d1, dn, "build digest differs: 1 cpu = {d1}, {pn} cpus = {dn}");
}

/// fingerprint() is how callers check that two indexes are the same. It
/// must cover everything search reads, not just list lengths.
#[test]
#[ignore = "KNOWN-FAIL: fingerprint() is XOR of list lengths only (findings §3.3)"]
fn ivfpq_fingerprint_detects_any_build_state_change() {
    let base = build_fixture();
    let fp = base.fingerprint();

    let mut c = base.clone();
    c.centroids[0][0] += 1e-3;
    assert_ne!(c.fingerprint(), fp, "centroid change not detected");

    let mut cb = base.clone();
    cb.pq.codebooks[0][0] += 1e-3;
    assert_ne!(cb.fingerprint(), fp, "codebook change not detected");

    let mut code = base.clone();
    let li = code.lists.iter().position(|l| !l.codes.is_empty()).unwrap();
    code.lists[li].codes[0] ^= 1;
    assert_ne!(code.fingerprint(), fp, "code byte change not detected");

    let mut id = base.clone();
    id.lists[li].ids[0] ^= 1 << 40;
    assert_ne!(id.fingerprint(), fp, "id change not detected");
}

// ------------------------------------------------------------ search identity

#[test]
fn ivfpq_repeated_search_is_identical() {
    let idx = build_fixture();
    let qs = clustered(50, DIM, &cluster_centres(16, DIM, 1234), 0.2, 999);
    for i in 0..qs.ids.len() {
        let a = idx.search(row(&qs, i), 10);
        let b = idx.search(row(&qs, i), 10);
        assert_eq!(a, b, "query {i}");
    }
}

/// Scores must be bit-identical too, not just ids: callers threshold and
/// merge on them. threads=1 currently re-normalises an already-normalised
/// query (search_parallel -> search), which shifts the last bits.
#[test]
#[ignore = "KNOWN-FAIL: IvfIndex::search_parallel(threads=1) normalises the query twice, scores differ in the last bits"]
fn ivf_flat_search_parallel_matches_serial_for_any_thread_count() {
    let data = uniform_sphere(2_000, DIM, 42);
    let rows: Vec<(u64, Vec<f32>)> = (0..data.ids.len()).map(|i| (data.ids[i], row(&data, i).to_vec())).collect();
    let ivf = IvfIndex::build(Metric::Cosine, DIM, &rows, IvfParams { nlist: 32, nprobe: 12, refine: 0, seed: 7 });
    let qs = uniform_sphere(30, DIM, 999);
    for i in 0..qs.ids.len() {
        let serial = ivf.search(row(&qs, i), 10);
        for t in [1usize, 2, 3, 4, 7, 16] {
            assert_eq!(ivf.search_parallel(row(&qs, i), 10, t), serial, "query {i}, threads {t}");
        }
    }
}

// ------------------------------------------------------------ tie-breaks

/// 40 exact duplicates inserted with *descending* ids, so any order that
/// just preserves insertion order fails. Expected: ascending id.
fn duplicate_rows(dim: usize) -> (Vec<f32>, Vec<(u64, Vec<f32>)>) {
    let filler = uniform_sphere(1_000, dim, 5);
    let mut dup = vec![0.0f32; dim];
    dup[0] = 1.0;
    let mut rows: Vec<(u64, Vec<f32>)> = (0..40u64).rev().map(|j| (10_000 + j, dup.clone())).collect();
    rows.extend((0..filler.ids.len()).map(|i| (filler.ids[i], row(&filler, i).to_vec())));
    (dup, rows)
}

#[test]
#[ignore = "KNOWN-FAIL: IVF-PQ search uses sort_unstable_by with no id tie-break (findings §3.3)"]
fn ivfpq_equal_scores_are_ordered_by_ascending_id() {
    let (q, rows) = duplicate_rows(DIM);
    let rows3: Vec<(u64, Vec<f32>, u64)> = rows.into_iter().map(|(id, v)| (id, v, 0)).collect();
    let idx = IvfPqIndex::build(Metric::Cosine, DIM, &rows3, ivfpq_params(16, 16, 200, M, 7));
    let got = ids(&idx.search(&q, 20));
    let want: Vec<u64> = (10_000..10_020).collect();
    assert_eq!(got, want);
}

#[test]
fn ivf_flat_equal_scores_are_ordered_by_ascending_id() {
    let (q, rows) = duplicate_rows(DIM);
    let ivf = IvfIndex::build(Metric::Cosine, DIM, &rows, IvfParams { nlist: 16, nprobe: 16, refine: 0, seed: 7 });
    let want: Vec<u64> = (10_000..10_020).collect();
    assert_eq!(ids(&ivf.search(&q, 20)), want);
    assert_eq!(ids(&ivf.search_parallel(&q, 20, 4)), want);
}

/// Was `stable_ties_and_zero_vectors_cosine` in src/eval.rs.
#[test]
fn flat_equal_scores_and_zero_query_are_ordered_by_ascending_id() {
    let dim = 8;
    let mut flat = FlatIndex::new(dim, Metric::Cosine);
    let v = vec![1.0f32; dim];
    let z = vec![0.0f32; dim];
    flat.add(2, &v);
    flat.add(1, &v);
    flat.add(3, &z);
    assert_eq!(ids(&flat.search(&v, 3)), vec![1, 2, 3]);
    // Zero query: cosine is defined as 0 for every row.
    assert_eq!(ids(&flat.search(&z, 3)), vec![1, 2, 3]);
}

// ------------------------------------------------------------ odd input

/// A NaN in a query (bad embedding, uninitialised buffer) must not take
/// down the game server. What it should *return* is a separate decision;
/// this only asserts no panic.
#[test]
#[ignore = "KNOWN-FAIL: partial_cmp(..).unwrap() panics on NaN (findings §3.3)"]
fn ivfpq_nan_query_does_not_panic() {
    let idx = build_fixture();
    let mut q = vec![0.1f32; DIM];
    q[3] = f32::NAN;
    let r = catch_unwind(AssertUnwindSafe(|| idx.search(&q, 10)));
    assert!(r.is_ok(), "IvfPqIndex::search panicked on a NaN query");
}

#[test]
#[ignore = "KNOWN-FAIL: stable_top_k uses partial_cmp(..).unwrap() (findings §3.3)"]
fn flat_nan_query_does_not_panic() {
    let data = uniform_sphere(100, DIM, 1);
    let flat = flat_from(&data, Metric::Cosine);
    let mut q = vec![0.1f32; DIM];
    q[0] = f32::NAN;
    let r = catch_unwind(AssertUnwindSafe(|| flat.search(&q, 10)));
    assert!(r.is_ok(), "FlatIndex::search panicked on a NaN query");
}
