//! Drift measurement for a live IVF-PQ index (stage 3).
//!
//! The model (centroids, OPQ, PQ codebooks) is frozen for an epoch. As
//! content moves away from what the model was trained on, rows land far
//! from their centroid, quantise badly, or pile into a few lists. This
//! module measures that with plain sums so a rebuild can be triggered:
//!
//! - `baseline`: per-list sums over the rows the model was trained on.
//! - `record`: the same per-list sums over every row inserted this epoch,
//!   plus delete counts, accumulated in op order as f64.
//! - `check`: compares the two against `Thresholds` and says what fired.
//!
//! Everything is a pure function of the index and the ops applied, in
//! order, so a signal fires at the same op on every run.

use crate::ivfpq::{rotate, IvfPqIndex};
use crate::live::{self, Applied, Scratch, WriteError};
use crate::pq::ProductQuantizer;

/// Per-list sums over the rows a model was trained on, measured right after
/// the build. Index = list.
#[derive(Clone, Debug, PartialEq)]
pub struct DriftBaseline {
    /// Σ ||r - decode(code)||² per list, r = OPQ(v - c_list).
    pub qerr: Vec<f64>,
    /// Σ ||r||² per list.
    pub rnorm: Vec<f64>,
    /// Rows per list.
    pub rows: Vec<u64>,
}

/// The same per-list sums over rows inserted this epoch, plus churn counts.
#[derive(Clone, Debug, PartialEq)]
pub struct DriftStats {
    pub qerr: Vec<f64>,
    pub rnorm: Vec<f64>,
    /// Inserted rows per list (upserts count as inserts).
    pub inserts: Vec<u64>,
    /// Rows removed this epoch: deletes, upsert replacements, expiries.
    pub deletes: u64,
    /// Live rows when the epoch started; the churn denominator.
    pub live_at_start: u64,
}

/// When a signal fires. Ratios are this epoch's mean per inserted row over
/// the baseline's mean per training row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Thresholds {
    pub qerr_ratio: f64,
    pub rnorm_ratio: f64,
    /// Largest live list over the mean live list.
    pub imbalance: f64,
    /// (inserts + deletes this epoch) / live rows at epoch start.
    pub churn: f64,
    /// The two ratios are not checked before this many inserts this epoch:
    /// one row's error says little about the distribution.
    pub min_inserts: u64,
    /// No new rebuild starts until this many logged writes have been applied
    /// since the last model decision (swap or reject; epoch 0 counts from
    /// the start). Signals are still measured; `rebuild::due` holds them.
    /// Counted in ops, not time, so the trigger lands at the same op on
    /// every run.
    pub min_ops_per_epoch: u64,
}

/// Starting thresholds from the design: 1.5, 1.5, 4.0, 0.5, 256 inserts.
/// `min_ops_per_epoch` 4096 = 16 x `min_inserts`: the next epoch's ratios
/// are then means over >= ~4k rows (a quarter of the 256-row noise), and a
/// shift that is still arriving gets 4k ops into the new model before it can
/// ask again (the test shift re-fired every ~320 ops without it). At the
/// 10 writes per tick ceiling that is >= 6.8 s at 60 Hz, about one 100k-row
/// rebuild (9 s on the dev container), so at most one rebuild runs at a time
/// without the worker falling behind.
pub fn default_thresholds() -> Thresholds {
    Thresholds { qerr_ratio: 1.5, rnorm_ratio: 1.5, imbalance: 4.0, churn: 0.5, min_inserts: 256, min_ops_per_epoch: 4096 }
}

/// Bits of `Signals::fired`.
pub const QERR: u8 = 1;
pub const RNORM: u8 = 2;
pub const IMBALANCE: u8 = 4;
pub const CHURN: u8 = 8;

/// Measured values and which ones crossed their threshold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Signals {
    pub qerr_ratio: f64,
    pub rnorm_ratio: f64,
    pub imbalance: f64,
    pub churn: f64,
    pub inserts: u64,
    /// OR of `QERR`, `RNORM`, `IMBALANCE`, `CHURN`; 0 = no drift.
    pub fired: u8,
}

/// (||r - decode(code)||², ||r||²) for one encoded row, in f64.
fn row_err(pq: &ProductQuantizer, r: &[f32], code: &[u8]) -> (f64, f64) {
    let dsub = pq.dsub;
    let (mut e, mut n) = (0.0f64, 0.0f64);
    for (j, &c) in code.iter().enumerate() {
        let cw = &pq.codebooks[j][c as usize * dsub..(c as usize + 1) * dsub];
        for (&x, &y) in r[j * dsub..(j + 1) * dsub].iter().zip(cw) {
            let (x, y) = (x as f64, y as f64);
            e += (x - y) * (x - y);
            n += x * x;
        }
    }
    (e, n)
}

/// Baseline over every live row of `idx`, lists in order, rows in order.
/// Call right after a build, when the live rows are the training rows.
/// Needs raw vectors (`store_vecs`).
pub fn baseline(idx: &IvfPqIndex) -> DriftBaseline {
    let (dim, m, nlist) = (idx.dim, idx.params.m, idx.lists.len());
    let mut b = DriftBaseline { qerr: vec![0.0; nlist], rnorm: vec![0.0; nlist], rows: vec![0; nlist] };
    let mut res = vec![0.0f32; dim];
    for (li, l) in idx.lists.iter().enumerate() {
        assert!(l.vecs.is_some(), "drift baseline needs store_vecs");
        for r in l.head..l.len() {
            if l.tomb[r] == 1 { continue; }
            for ((d, &x), &c) in res.iter_mut().zip(l.vec_row(dim, r)).zip(&idx.centroids[li]) { *d = x - c; }
            let (e, n) = row_err(&idx.pq, &rotate(&idx.opq, &res), &l.codes[r * m..(r + 1) * m]);
            b.qerr[li] += e;
            b.rnorm[li] += n;
            b.rows[li] += 1;
        }
    }
    b
}

/// Zeroed stats for a new epoch starting on `idx`.
pub fn new_stats(idx: &IvfPqIndex) -> DriftStats {
    let nlist = idx.lists.len();
    DriftStats { qerr: vec![0.0; nlist], rnorm: vec![0.0; nlist], inserts: vec![0; nlist], deletes: 0, live_at_start: live::live_rows(idx) as u64 }
}

/// Account for one applied op. `s` must be the scratch `live::apply` just
/// used on `idx`, and `result` what it returned: an insert's residual and codes are read from it,
/// so no row is encoded twice. Refused ops change nothing.
pub fn record(st: &mut DriftStats, idx: &IvfPqIndex, s: &Scratch, result: &Result<Applied, WriteError>) {
    let (loc, removed) = match result {
        Ok(Applied::Inserted { loc }) => (Some(*loc), 0),
        Ok(Applied::Upserted { loc, replaced }) => (Some(*loc), replaced.is_some() as u64),
        Ok(Applied::Deleted { .. }) => (None, 1),
        Ok(Applied::Expired { ids, .. }) => (None, ids.len() as u64),
        Ok(Applied::Compacted { .. }) | Err(_) => (None, 0),
    };
    st.deletes += removed;
    if let Some(loc) = loc {
        let (r, code) = live::last_encoded(idx, s);
        let (e, n) = row_err(&idx.pq, r, code);
        let li = loc.list as usize;
        st.qerr[li] += e;
        st.rnorm[li] += n;
        st.inserts[li] += 1;
    }
}

/// Current drift signals. O(nlist).
pub fn check(st: &DriftStats, b: &DriftBaseline, idx: &IvfPqIndex, th: &Thresholds) -> Signals {
    let sum = |v: &[f64]| v.iter().sum::<f64>();
    let inserts: u64 = st.inserts.iter().sum();
    let base_rows: u64 = b.rows.iter().sum();
    let ratio = |now: f64, base: f64| {
        if inserts == 0 || base_rows == 0 || base <= 0.0 { return 0.0; }
        (now / inserts as f64) / (base / base_rows as f64)
    };
    let qerr_ratio = ratio(sum(&st.qerr), sum(&b.qerr));
    let rnorm_ratio = ratio(sum(&st.rnorm), sum(&b.rnorm));

    let mut total = 0usize;
    let mut largest = 0usize;
    for l in &idx.lists {
        let live = l.len() - l.dead;
        total += live;
        largest = largest.max(live);
    }
    let imbalance = if total == 0 { 0.0 } else { largest as f64 * idx.lists.len() as f64 / total as f64 };
    let churn = (inserts + st.deletes) as f64 / st.live_at_start.max(1) as f64;

    let mut fired = 0u8;
    if inserts >= th.min_inserts && qerr_ratio > th.qerr_ratio { fired |= QERR; }
    if inserts >= th.min_inserts && rnorm_ratio > th.rnorm_ratio { fired |= RNORM; }
    if imbalance > th.imbalance { fired |= IMBALANCE; }
    if churn > th.churn { fired |= CHURN; }
    Signals { qerr_ratio, rnorm_ratio, imbalance, churn, inserts, fired }
}
