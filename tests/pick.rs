//! Seeded "2 of top 5" pick (Brinn, 2026-10-21): one k=5 lookup, then a
//! uniform, seeded choice of 2 of those hits for URIEL to choose from.
//!
//! Contract under test:
//!   - same (tick, npc_id, query_id) -> same seed -> same picked ids, always;
//!   - returns exactly min(pick, hits.len()) hits, all distinct, all from
//!     the input, in the input's rank order (score desc, id asc);
//!   - every subset of size `pick` is equally likely (checked over fixed
//!     seeds, so the check itself is deterministic);
//!   - `IvfPqIndex::search_pick` == `pick_ranked(search_temporal(..))`.

mod common;
use common::*;

use raggedy_anndy::pick::{pick_ranked, pick_seed, PICK_DEFAULT, TOP_K_DEFAULT};
use raggedy_anndy::{Hit, IvfPqIndex, Metric};

fn hits(n: usize) -> Vec<Hit> {
    // Already in rank order: score desc, ids distinct.
    (0..n as u64).map(|i| Hit { id: 100 + i, score: 1.0 - i as f32 * 0.01 }).collect()
}

fn ids(h: &[Hit]) -> Vec<u64> { h.iter().map(|x| x.id).collect() }

/// Position in `src` of each picked hit.
fn positions(src: &[Hit], picked: &[Hit]) -> Vec<usize> {
    picked.iter().map(|p| src.iter().position(|s| s.id == p.id).expect("picked hit not in input")).collect()
}

#[test]
fn defaults_are_top_5_pick_2() {
    assert_eq!(TOP_K_DEFAULT, 5);
    assert_eq!(PICK_DEFAULT, 2);
}

// ------------------------------------------------------------------ seed

#[test]
fn seed_is_a_pure_function_of_its_inputs() {
    assert_eq!(pick_seed(10, 20, 30), pick_seed(10, 20, 30));
}

#[test]
fn seed_changes_with_every_input_and_with_argument_order() {
    let base = pick_seed(10, 20, 30);
    assert_ne!(base, pick_seed(11, 20, 30), "tick");
    assert_ne!(base, pick_seed(10, 21, 30), "npc_id");
    assert_ne!(base, pick_seed(10, 20, 31), "query_id");
    assert_ne!(pick_seed(1, 2, 3), pick_seed(3, 2, 1), "order");
    assert_ne!(pick_seed(1, 2, 3), pick_seed(2, 1, 3), "order");
    assert_ne!(pick_seed(0, 0, 0), 0, "all-zero inputs must still mix");
}

#[test]
fn seeds_do_not_collide_over_a_dense_grid() {
    // 64 ticks x 64 NPCs x 16 queries = 65536 adjacent inputs.
    let mut s: Vec<u64> = Vec::with_capacity(64 * 64 * 16);
    for t in 0..64 { for n in 0..64 { for q in 0..16 { s.push(pick_seed(t, n, q)); } } }
    s.sort_unstable();
    s.dedup();
    assert_eq!(s.len(), 64 * 64 * 16);
}

/// Pinned outputs. If these move, every replay recorded before the change
/// picks different memories: bump them only on purpose.
#[test]
fn seed_and_pick_are_pinned() {
    assert_eq!(pick_seed(0, 0, 0), PINNED_SEED_000);
    assert_eq!(pick_seed(1_000, 42, 7), PINNED_SEED_1000_42_7);
    let h = hits(5);
    assert_eq!(ids(&pick_ranked(&h, 2, pick_seed(0, 0, 0))), PINNED_PICK_000.to_vec());
    assert_eq!(ids(&pick_ranked(&h, 2, pick_seed(1_000, 42, 7))), PINNED_PICK_1000_42_7.to_vec());
}
const PINNED_SEED_000: u64 = 2558736989570252433;
const PINNED_SEED_1000_42_7: u64 = 16749230566864494860;
const PINNED_PICK_000: [u64; 2] = [100, 103];
const PINNED_PICK_1000_42_7: [u64; 2] = [101, 104];

// ------------------------------------------------------------------ shape

#[test]
fn pick_returns_distinct_input_hits_in_rank_order() {
    let h = hits(5);
    for t in 0..2_000u64 {
        let p = pick_ranked(&h, 2, pick_seed(t, 3, 9));
        assert_eq!(p.len(), 2);
        assert_eq!(p.capacity(), 2, "result must not carry spare capacity");
        let pos = positions(&h, &p);
        assert!(pos[0] < pos[1], "tick {t}: not in rank order: {pos:?}");
        for (x, &i) in p.iter().zip(&pos) {
            assert_eq!(x.score.to_bits(), h[i].score.to_bits(), "score must be passed through untouched");
        }
    }
}

#[test]
fn same_seed_same_pick() {
    let h = hits(5);
    for t in 0..500u64 {
        let s = pick_seed(t, 77, 1);
        assert_eq!(ids(&pick_ranked(&h, 2, s)), ids(&pick_ranked(&h, 2, s)));
    }
}

#[test]
fn short_inputs() {
    let s = pick_seed(5, 5, 5);
    assert!(pick_ranked(&hits(0), 2, s).is_empty(), "0 hits -> none");
    assert_eq!(ids(&pick_ranked(&hits(1), 2, s)), ids(&hits(1)), "1 hit -> it");
    assert_eq!(ids(&pick_ranked(&hits(2), 2, s)), ids(&hits(2)), "2 hits -> both, rank order");
    assert!(pick_ranked(&hits(5), 0, s).is_empty(), "pick 0 -> none");
    assert_eq!(ids(&pick_ranked(&hits(3), 9, s)), ids(&hits(3)), "pick > len -> all");
    // 3 or 4 hits: still 2, from what came back.
    for n in [3usize, 4] {
        let h = hits(n);
        for t in 0..200u64 {
            let p = pick_ranked(&h, 2, pick_seed(t, 0, 0));
            assert_eq!(p.len(), 2);
            let pos = positions(&h, &p);
            assert!(pos[0] < pos[1]);
        }
    }
}

/// Over 100k fixed seeds, each of the C(5,2) = 10 pairs should come up
/// 10% of the time. Expected count 10_000 each; sd ~95. A 6-sigma band
/// (+-570) cannot flake because the seeds are fixed, but it still catches
/// any real bias (an always-include-top-1 bug lands at 40%, not 10%).
#[test]
fn every_pair_is_equally_likely() {
    let h = hits(5);
    let trials = 100_000u64;
    let mut pair = [[0u32; 5]; 5];
    let mut slot = [0u32; 5];
    for t in 0..trials {
        let pos = positions(&h, &pick_ranked(&h, 2, pick_seed(t, t.wrapping_mul(31), 0)));
        pair[pos[0]][pos[1]] += 1;
        slot[pos[0]] += 1;
        slot[pos[1]] += 1;
    }
    let want = trials as f64 / 10.0;
    for a in 0..5 { for b in (a + 1)..5 {
        let got = pair[a][b] as f64;
        assert!((got - want).abs() < 570.0, "pair ({a},{b}) = {got}, want ~{want}");
    } }
    // Each position is in 2/5 of picks.
    for (i, &c) in slot.iter().enumerate() {
        let got = c as f64 / trials as f64;
        assert!((got - 0.4).abs() < 0.01, "position {i} picked {got:.4}, want 0.4");
    }
}

/// Different NPCs on the same tick asking the same thing get independent
/// picks: two NPCs should share the same pair ~10% of the time.
#[test]
fn npcs_on_one_tick_pick_independently() {
    let h = hits(5);
    let (npcs, tick, query) = (20_000u64, 9_999u64, 4u64);
    let mut same = 0u32;
    for n in 0..npcs {
        let a = ids(&pick_ranked(&h, 2, pick_seed(tick, 2 * n, query)));
        let b = ids(&pick_ranked(&h, 2, pick_seed(tick, 2 * n + 1, query)));
        if a == b { same += 1; }
    }
    let got = same as f64 / npcs as f64;
    assert!((got - 0.10).abs() < 0.015, "same-pair rate {got:.4}, want ~0.10");
}

// ------------------------------------------------------------------ index

fn small_index() -> (IvfPqIndex, Dataset) {
    let centres = cluster_centres(16, 32, 1234);
    let data = clustered(3_000, 32, &centres, 0.2, 42);
    let qs = clustered(64, 32, &centres, 0.2, 999);
    (IvfPqIndex::build(Metric::Cosine, 32, &ivfpq_rows(&data), ivfpq_params(32, 8, 64, 8, 7)), qs)
}

#[test]
fn search_pick_is_search_then_pick() {
    let (idx, qs) = small_index();
    for i in 0..qs.ids.len() {
        let q = row(&qs, i);
        let seed = pick_seed(i as u64, 11, 0);
        let top = idx.search_temporal(q, TOP_K_DEFAULT, None, 0, 0.0);
        assert_eq!(top.len(), TOP_K_DEFAULT);
        let want = pick_ranked(&top, PICK_DEFAULT, seed);
        let got = idx.search_pick(q, TOP_K_DEFAULT, PICK_DEFAULT, None, 0, 0.0, seed);
        assert_eq!(ids(&got), ids(&want), "query {i}");
        let gb: Vec<u32> = got.iter().map(|h| h.score.to_bits()).collect();
        let wb: Vec<u32> = want.iter().map(|h| h.score.to_bits()).collect();
        assert_eq!(gb, wb, "query {i}: score bits");
    }
}

#[test]
fn search_pick_on_bad_query_returns_nothing() {
    let (idx, _) = small_index();
    let q = vec![f32::NAN; 32];
    assert!(idx.search_pick(&q, 5, 2, None, 0, 0.0, pick_seed(1, 1, 1)).is_empty());
}
