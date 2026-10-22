//! Live writes, stage 2: one writer thread, any number of query threads.
//!
//! The writer applies the ops of tick T while queries read the view
//! published at the end of tick T-1. Queries never wait on the writer's
//! work: fetching the latest view is one `Arc` clone under a mutex that
//! the writer holds only to swap in a pointer.
//!
//! Layout:
//! - Row data (ids, timestamps, tags, PQ codes, raw vectors) lives in
//!   fixed-capacity `arena::Block`s of `block_rows` rows. Row `r` of a list
//!   is in `blocks[r / block_rows]`, slot `r % block_rows`. Blocks are
//!   allocated whole and never move; growth adds a block. Written rows never
//!   change.
//! - The small mutable metadata (per-list block table, row count, head,
//!   dead count, tombstones) is `ListMeta`. The writer owns one set; each
//!   published `View` owns a copy. Rows past a view's `len` are invisible
//!   to it, so inserts, deletes and expiry in tick T are only seen by the
//!   view of tick T.
//! - The id directory is writer-only.
//!
//! Publish (`publish`): take a view no query holds (`Arc::get_mut`), copy
//! into it the lists changed since that view's own tick, give it the new
//! seq, and swap it into the shared slot. Only a view no one else holds is
//! ever written, so a held view never changes. No query holds it means no
//! other thread can reach it: the writer keeps a handle to every view it
//! has published, and the slot holds the newest one.
//!
//! Reclamation: compaction retires a list's old blocks. A retired block is
//! reset and reused only when its `Arc` is unique, i.e. no view (held by a
//! query or pooled by the writer) can still see it. The writer keeps one
//! idle view as a spare and drops the rest at publish, so an idle view does
//! not pin retired blocks for long.
//!
//! Results: search runs `ivfpq::search_rows`, the same scoring code as
//! `IvfPqIndex::search_temporal`, over the view's blocks. A query's result
//! is a function of (query, view seq): the same op log cut into the same
//! ticks gives the same view at every seq, and `to_index(view)` is
//! byte-identical (`fingerprint`) to `live::apply` of that log on an
//! `IvfPqIndex`.
//!
//! Design: "IVF-PQ live insert and delete design" (project doc), stage 2.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::arena::Block;
use crate::ivfpq::{search_rows, IvfPqIndex, ListRows, RowLoc, Run};
use crate::live::{self, Applied, Op, Scratch, WriteError};
use crate::types::Hit;

/// Per-list metadata. The writer owns one set; every view owns a copy.
#[derive(Clone)]
pub struct ListMeta {
    /// Row `r` lives in `blocks[r / block_rows]`, slot `r % block_rows`.
    /// May end with empty blocks (`reserve`).
    pub blocks: Vec<Arc<Block>>,
    pub len: usize,
    /// Rows before `head` were expired and are dead.
    pub head: usize,
    pub dead: usize,
    pub tomb: Vec<u8>,
}

/// Index contents at the end of tick `seq`. Never changes while shared.
pub struct View {
    pub seq: u64,
    pub max_ts: u64,
    pub block_rows: usize,
    /// Centroids, OPQ, codebooks and params; no rows (`live::empty_like`).
    pub model: Arc<IvfPqIndex>,
    pub lists: Vec<ListMeta>,
}

/// Where query threads get the newest view (`current`).
pub type Latest = Arc<Mutex<Arc<View>>>;

/// Block pool. Blocks in `ready` are empty and held by nothing else.
struct Pool {
    cap: usize,
    m: usize,
    vdim: usize,
    ready: Vec<Arc<Block>>,
    retired: Vec<Arc<Block>>,
    allocated: usize,
}

/// The single writer. Owned by the writer thread.
pub struct Writer {
    pub model: Arc<IvfPqIndex>,
    pub lists: Vec<ListMeta>,
    /// Seq of the tick that last changed each list.
    touched: Vec<u64>,
    /// id -> location of its live row. Writer-only, never iterated.
    pub dir: HashMap<u64, RowLoc>,
    pub max_ts: u64,
    /// Seq of the tick being applied; `publish` gives it to the view.
    pub seq: u64,
    pool: Pool,
    /// Every published view the writer may still recycle.
    views: Vec<Arc<View>>,
    latest: Latest,
    scratch: Scratch,
}

/// Block counts, for tests and monitoring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaStats { pub allocated: usize, pub ready: usize, pub retired: usize, pub views: usize }

fn empty_meta() -> ListMeta { ListMeta { blocks: Vec::new(), len: 0, head: 0, dead: 0, tomb: Vec::new() } }

fn take_block(p: &mut Pool) -> Arc<Block> {
    match p.ready.pop() {
        Some(b) => b,
        None => { p.allocated += 1; Arc::new(Block::new(p.cap, p.m, p.vdim)) }
    }
}

/// Append a live row to `l`; returns its row number.
fn push_row(p: &mut Pool, l: &mut ListMeta, id: u64, ts: u64, tags: u64, codes: &[u8], vec: &[f32]) -> usize {
    let r = l.len;
    if r == l.blocks.len() * p.cap { l.blocks.push(take_block(p)); }
    let slot = l.blocks[r / p.cap].push(id, ts, tags, codes, vec);
    debug_assert_eq!(slot, r % p.cap);
    l.tomb.push(0);
    l.len += 1;
    r
}

/// Move retired blocks no view can see any more to `ready`.
fn reclaim(p: &mut Pool) {
    let mut i = 0;
    while i < p.retired.len() {
        if Arc::get_mut(&mut p.retired[i]).map(|b| b.reset()).is_some() {
            let b = p.retired.swap_remove(i);
            p.ready.push(b);
        } else {
            i += 1;
        }
    }
}

/// Writer over a copy of `idx` (rows, tombstones, heads, directory), with
/// blocks of `block_rows` rows. Publishes `idx`'s state as view 0.
pub fn writer_from(idx: &IvfPqIndex, block_rows: usize) -> Writer {
    assert!(block_rows > 0, "block_rows must be > 0");
    let model = Arc::new(live::empty_like(idx));
    let (m, dim) = (idx.params.m, idx.dim);
    let vdim = if idx.params.store_vecs { dim } else { 0 };
    let mut pool = Pool { cap: block_rows, m, vdim, ready: Vec::new(), retired: Vec::new(), allocated: 0 };
    let mut lists = Vec::with_capacity(idx.lists.len());
    for l in &idx.lists {
        let mut meta = empty_meta();
        for r in 0..l.len() {
            let vec = if vdim > 0 { l.vec_row(dim, r) } else { &[] };
            push_row(&mut pool, &mut meta, l.ids[r], l.timestamps[r], l.tags[r], l.code_row(m, r), vec);
        }
        meta.tomb.copy_from_slice(&l.tomb);
        meta.head = l.head;
        meta.dead = l.dead;
        lists.push(meta);
    }
    let view = Arc::new(View { seq: 0, max_ts: idx.max_ts, block_rows, model: model.clone(), lists: lists.clone() });
    Writer {
        scratch: live::scratch_for(&model),
        touched: vec![0; lists.len()],
        model, lists, dir: idx.dir.clone(), max_ts: idx.max_ts, seq: 1, pool,
        views: vec![view.clone()],
        latest: Arc::new(Mutex::new(view)),
    }
}

/// Handle for query threads.
pub fn latest(w: &Writer) -> Latest { w.latest.clone() }

/// The newest published view. Holding it keeps its rows readable.
pub fn current(l: &Latest) -> Arc<View> { l.lock().unwrap_or_else(|e| e.into_inner()).clone() }

/// Apply one op to the writer's state; visible to queries after `publish`.
/// Same semantics and results as `live::apply`.
pub fn apply(w: &mut Writer, op: &Op) -> Result<Applied, WriteError> {
    match op {
        Op::Insert { id, ts, tags, vec } => insert(w, *id, *ts, *tags, vec).map(|loc| Applied::Inserted { loc }),
        Op::Upsert { id, ts, tags, vec } => upsert(w, *id, *ts, *tags, vec).map(|(loc, replaced)| Applied::Upserted { loc, replaced }),
        Op::Delete { id } => delete(w, *id).map(|loc| Applied::Deleted { loc }),
        Op::Expire { before_ts, max_rows } => {
            let (ids, done) = expire(w, *before_ts, *max_rows);
            Ok(Applied::Expired { ids, done })
        }
        Op::Compact { list } => compact_list(w, *list).map(|dropped| Applied::Compacted { list: *list, dropped }),
    }
}

/// Append a new row. No allocation once `reserve` has made room.
pub fn insert(w: &mut Writer, id: u64, ts: u64, tags: u64, v: &[f32]) -> Result<RowLoc, WriteError> {
    live::check_row(w.model.dim, w.max_ts, id, ts, v)?;
    if w.dir.contains_key(&id) { return Err(WriteError::DuplicateId { id }); }
    Ok(append(w, id, ts, tags, v))
}

/// Writes the row; all checks already passed.
fn append(w: &mut Writer, id: u64, ts: u64, tags: u64, v: &[f32]) -> RowLoc {
    let li = live::encode_row(&w.model, &mut w.scratch, v);
    let vec: &[f32] = if w.pool.vdim > 0 { &w.scratch.view } else { &[] };
    let row = push_row(&mut w.pool, &mut w.lists[li], id, ts, tags, &w.scratch.codes, vec);
    let loc = RowLoc { list: li as u32, row: row as u32 };
    w.dir.insert(id, loc);
    w.max_ts = ts;
    w.touched[li] = w.seq;
    loc
}

/// Mark `id` dead. O(1).
pub fn delete(w: &mut Writer, id: u64) -> Result<RowLoc, WriteError> {
    let loc = w.dir.remove(&id).ok_or(WriteError::NotFound { id })?;
    let l = &mut w.lists[loc.list as usize];
    l.tomb[loc.row as usize] = 1;
    l.dead += 1;
    w.touched[loc.list as usize] = w.seq;
    Ok(loc)
}

/// Delete `id` if live, then append the new row. As `live::upsert`.
pub fn upsert(w: &mut Writer, id: u64, ts: u64, tags: u64, v: &[f32]) -> Result<(RowLoc, Option<RowLoc>), WriteError> {
    live::check_row(w.model.dim, w.max_ts, id, ts, v)?;
    let replaced = delete(w, id).ok();
    Ok((append(w, id, ts, tags, v), replaced))
}

/// Expire live rows with `ts < before_ts`, at most `max_rows` walked. As `live::expire`.
pub fn expire(w: &mut Writer, before_ts: u64, max_rows: usize) -> (Vec<u64>, bool) {
    let cap = w.pool.cap;
    let mut ids = Vec::new();
    let mut budget = max_rows;
    for (li, l) in w.lists.iter_mut().enumerate() {
        while l.head < l.len && l.blocks[l.head / cap].ts(l.head % cap) < before_ts {
            if budget == 0 { return (ids, false); }
            budget -= 1;
            let r = l.head;
            if l.tomb[r] == 0 {
                l.tomb[r] = 1;
                l.dead += 1;
                let id = l.blocks[r / cap].id(r % cap);
                w.dir.remove(&id);
                ids.push(id);
            }
            l.head += 1;
            w.touched[li] = w.seq;
        }
    }
    (ids, true)
}

/// Rewrite `list` into fresh blocks keeping only live rows, and retire the
/// old blocks. As `live::compact_list`; views that hold the old blocks keep
/// reading them.
pub fn compact_list(w: &mut Writer, list: u32) -> Result<usize, WriteError> {
    let li = list as usize;
    if li >= w.lists.len() { return Err(WriteError::NoSuchList { list }); }
    let (cap, vdim) = (w.pool.cap, w.pool.vdim);
    let old = std::mem::replace(&mut w.lists[li], empty_meta());
    let new = &mut w.lists[li];
    new.tomb.reserve_exact(old.len - old.dead);
    new.blocks.reserve_exact((old.len - old.dead + cap - 1) / cap);
    for r in old.head..old.len {
        if old.tomb[r] == 1 { continue; }
        let (b, s) = (&old.blocks[r / cap], r % cap);
        let vec = if vdim > 0 { b.vec_row(s) } else { &[] };
        let row = push_row(&mut w.pool, new, b.id(s), b.ts(s), b.tags(s), b.code_row(s), vec);
        w.dir.insert(b.id(s), RowLoc { list, row: row as u32 });
    }
    w.pool.retired.extend(old.blocks);
    w.touched[li] = w.seq;
    Ok(old.dead)
}

/// Room for `rows_per_list` more rows in every list and in the directory,
/// so the next inserts do not allocate. Call between ticks.
pub fn reserve(w: &mut Writer, rows_per_list: usize) {
    let cap = w.pool.cap;
    for l in w.lists.iter_mut() {
        let need = (l.len + rows_per_list + cap - 1) / cap;
        l.blocks.reserve(need.saturating_sub(l.blocks.len()));
        while l.blocks.len() < need { l.blocks.push(take_block(&mut w.pool)); }
        l.tomb.reserve(rows_per_list);
    }
    w.dir.reserve(rows_per_list * w.lists.len());
}

/// End tick `w.seq`: publish its state as a view and return its seq.
pub fn publish(w: &mut Writer) -> u64 {
    let seq = w.seq;
    let view = match w.views.iter_mut().position(|v| Arc::get_mut(v).is_some()) {
        Some(i) => {
            let mut v = w.views.swap_remove(i);
            let dst = Arc::get_mut(&mut v).expect("unique");
            for ((d, s), &t) in dst.lists.iter_mut().zip(&w.lists).zip(&w.touched) {
                if t > dst.seq { d.clone_from(s); }
            }
            dst.seq = seq;
            dst.max_ts = w.max_ts;
            v
        }
        None => Arc::new(View { seq, max_ts: w.max_ts, block_rows: w.pool.cap, model: w.model.clone(), lists: w.lists.clone() }),
    };
    w.views.push(view.clone());
    let prev = std::mem::replace(&mut *w.latest.lock().unwrap_or_else(|e| e.into_inner()), view);
    drop(prev); // still in `views`: never the last handle, so no free under the lock
    w.seq += 1;
    // Keep one idle view as the next publish's buffer; the others would pin retired blocks.
    let mut spare = false;
    w.views.retain_mut(|v| Arc::get_mut(v).is_none() || !std::mem::replace(&mut spare, true));
    reclaim(&mut w.pool);
    seq
}

pub fn locate(w: &Writer, id: u64) -> Option<RowLoc> { w.dir.get(&id).copied() }

pub fn live_rows(w: &Writer) -> usize { w.dir.len() }

pub fn arena_stats(w: &Writer) -> ArenaStats {
    ArenaStats { allocated: w.pool.allocated, ready: w.pool.ready.len(), retired: w.pool.retired.len(), views: w.views.len() }
}

impl ListRows for View {
    #[inline] fn span(&self, li: usize) -> (usize, usize) { (self.lists[li].head, self.lists[li].len) }
    #[inline] fn tomb(&self, li: usize) -> &[u8] { &self.lists[li].tomb }
    #[inline] fn run(&self, li: usize, r: usize, end: usize) -> Run<'_> {
        let (b, cap) = (r / self.block_rows, self.block_rows);
        self.lists[li].blocks[b].rows(r - b * cap, (end - b * cap).min(cap))
    }
    #[inline] fn ts(&self, li: usize, r: usize) -> u64 { self.lists[li].blocks[r / self.block_rows].ts(r % self.block_rows) }
    #[inline] fn vec_row(&self, li: usize, r: usize) -> &[f32] { self.lists[li].blocks[r / self.block_rows].vec_row(r % self.block_rows) }
}

/// As `IvfPqIndex::search_temporal`, on this view's rows.
pub fn search_temporal(v: &View, q: &[f32], k: usize, required_tag: Option<u64>, current_time: u64, lambda: f32) -> Vec<Hit> {
    search_rows(&v.model, v, q, k, required_tag, current_time, lambda)
}

pub fn search(v: &View, q: &[f32], k: usize) -> Vec<Hit> { search_temporal(v, q, k, None, 0, 0.0) }

/// As `IvfPqIndex::search_pick`, on this view's rows.
pub fn search_pick(v: &View, q: &[f32], k: usize, pick: usize, required_tag: Option<u64>, current_time: u64, lambda: f32, seed: u64) -> Vec<Hit> {
    crate::pick::pick_ranked(&search_temporal(v, q, k, required_tag, current_time, lambda), pick, seed)
}

/// The view as a plain `IvfPqIndex` (rows copied into contiguous lists,
/// directory rebuilt). O(rows); for tests, fingerprints and snapshots.
pub fn to_index(v: &View) -> IvfPqIndex {
    let mut idx = (*v.model).clone();
    let cap = v.block_rows;
    for (li, l) in v.lists.iter().enumerate() {
        let out = &mut idx.lists[li];
        for r in 0..l.len {
            let (b, s) = (&l.blocks[r / cap], r % cap);
            let id = b.id(s);
            out.ids.push(id);
            out.timestamps.push(b.ts(s));
            out.tags.push(b.tags(s));
            out.codes.extend_from_slice(b.code_row(s));
            if let Some(vs) = &mut out.vecs { vs.extend_from_slice(b.vec_row(s)); }
            if l.tomb[r] == 0 { idx.dir.insert(id, RowLoc { list: li as u32, row: r as u32 }); }
        }
        out.tomb.clone_from(&l.tomb);
        out.head = l.head;
        out.dead = l.dead;
    }
    idx.max_ts = v.max_ts;
    idx
}

/// `IvfPqIndex::fingerprint` of the view's contents.
pub fn fingerprint(v: &View) -> u64 { to_index(v).fingerprint() }
