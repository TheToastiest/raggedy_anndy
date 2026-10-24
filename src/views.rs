//! Live writes, stage 2: one writer thread, any number of query threads.
//!
//! The writer applies the ops of tick T while queries read the view
//! published at the end of tick T-1. Queries never wait on the writer:
//! fetching the latest view is one lock-free `ArcSwap::load_full` (an
//! `Arc` clone), and the writer publishes with one `store`.
//!
//! Layout:
//! - Row data (ids, timestamps, tags, PQ codes, raw vectors) lives in
//!   fixed-capacity `arena::Block`s of `block_rows` rows (default
//!   `DEFAULT_BLOCK_ROWS`). Row `r` of a list is in `blocks[r / block_rows]`,
//!   slot `r % block_rows`. Blocks are allocated whole and never move;
//!   growth adds a block. Written rows never change.
//! - The small mutable metadata (per-list block table, row count, head,
//!   dead count, tombstones) is `ListMeta`. The writer owns one set; each
//!   published `View` owns a copy. Rows past a view's `len` are invisible
//!   to it, so inserts, deletes and expiry in tick T are only seen by the
//!   view of tick T.
//! - The id directory is writer-only.
//!
//! Publish (`publish`): the writer owns a fixed pool of views (`reserve`
//! sizes it). It takes the newest view no query holds (`Arc::get_mut`),
//! copies into it the lists changed since that view's own tick (into
//! buffers `reserve` made big enough, so no allocation), gives it the new
//! seq and stores it in the shared slot. Only a view no one else holds is
//! ever written, so a held view never changes. No query holds it means no
//! other thread can reach it: the writer keeps a handle to every view in
//! the pool, and the slot holds the newest one. If every view is held, the
//! publish is deferred: nothing is allocated or written, the previous view
//! stays current, and the next tick's publish shows both ticks.
//!
//! `ArcSwap` and `Arc::get_mut`: a `load` may hold the current view without
//! raising its count (a "debt"), but `store` pays every debt on the old
//! value before it returns, so a view that is not in the slot has an exact
//! count. The view in the slot is never unique (the pool holds it too).
//!
//! Reclamation: compaction retires a list's old blocks. A retired block is
//! reset and reused only when its `Arc` is unique, i.e. no view (held by a
//! query or pooled by the writer) can still see it. A compaction marks
//! every pooled view dirty; at the next publish where a dirty view is free
//! (and not the one being written), it drops its handles to the blocks of
//! every list changed since its seq, which it would copy again on reuse
//! anyway. So an idle view does not pin retired blocks past one publish.
//!
//! Compaction policy (`compactions_due`): once per tick the writer asks
//! which lists to compact and applies those `Op::Compact`s like any other
//! op, so they are in the log and replay. This is also what frees the
//! memory of expired rows.
//!
//! Pending real memories (stage 5): the writer keeps a copy of the
//! pending buffer's rows (`real_memory::Pending`: id, ts, tags, scored
//! vector) in blocks of its own (no PQ codes), with a tombstone per row
//! and the live pending ids sorted. `real_memory::apply_writer` keeps it
//! in step with the buffer (`pending_push`, `pending_kill`,
//! `pending_clear`); a view gets a copy of its metadata at `publish`, like
//! a list, so the view of tick T holds the buffer as it was at the end of
//! T. `search_temporal` on a view scans those rows exactly, as
//! `real_memory::search` does, and gives bit-identical results to it on
//! (`to_index(view)`, buffer at T). The commit retires the blocks, which
//! are reclaimed like compacted ones.
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
use std::sync::Arc;

use arc_swap::ArcSwap;

use crate::arena::Block;
use crate::ivfpq::{fingerprint_head, fnv, search_rows, IvfPqIndex, ListRows, RowLoc, Run};
use crate::ivfpq::{apply_decay, norm_inplace};
use crate::live::{self, Applied, Op, Scratch, WriteError};
use crate::metric::{self, Metric};
use crate::types::{stable_top_k, Hit};

/// Rows per block unless the caller picks another size.
pub const DEFAULT_BLOCK_ROWS: usize = 256;

/// Views in the pool after `writer_from`: the current one, one spare, and
/// room for this many minus two query threads holding a view at once.
pub const DEFAULT_VIEWS: usize = 8;

/// `dead_since` of a list with no dead rows.
pub const NEVER: u64 = u64::MAX;

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
    /// Pending real memories at the end of tick `seq`: rows in blocks with
    /// no codes and the scored vector (`real_memory::Pending::view`); `head` is 0.
    pub pending: ListMeta,
    /// Ids of the live rows of `pending`, ascending.
    pub pending_ids: Vec<u64>,
}

/// Where query threads get the newest view (`current`).
pub type Latest = Arc<ArcSwap<View>>;

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
    /// Per list, seq of the tick in which its oldest dead row died
    /// (`NEVER` if it has none). Read by `compactions_due`.
    pub dead_since: Vec<u64>,
    /// id -> location of its live row. Writer-only, never iterated.
    pub dir: HashMap<u64, RowLoc>,
    pub max_ts: u64,
    /// Seq of the tick being applied; `publish` gives it to the view.
    pub seq: u64,
    pool: Pool,
    /// The view pool. Every view the writer may recycle, including the current one.
    views: Vec<Arc<View>>,
    /// Per pooled view: a compaction ran since it was last written or
    /// swept, so it may still hold retired blocks.
    dirty: Vec<bool>,
    latest: Latest,
    /// Copy of the pending real-memory buffer (see the module doc).
    pend: ListMeta,
    /// Live pending ids ascending, and each one's row in `pend`.
    pend_ids: Vec<u64>,
    pend_rows: Vec<u32>,
    /// Seq of the tick that last changed `pend`.
    pend_touched: u64,
    /// Blocks of `pend`: no codes, a vector of `dim` floats.
    ppool: Pool,
    /// Seq of the view in `latest`.
    current: u64,
    /// Publishes deferred: in a row (since the last one that published), and in total.
    pending: u64,
    deferred: u64,
    /// Publishes that had to grow a view's buffers (allocated).
    grown: u64,
    scratch: Scratch,
}

/// What `publish` did at the end of a tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Published {
    /// The tick that ended.
    pub seq: u64,
    /// Seq of the view queries now get: `seq`, or an older one if deferred.
    pub current: u64,
    /// Ticks in a row not yet published, this one included; 0 if this
    /// publish went through. Always `seq - current`.
    pub deferred: u64,
}

/// Block and view counts, for tests and monitoring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaStats {
    pub allocated: usize,
    pub ready: usize,
    pub retired: usize,
    pub views: usize,
    /// Publishes deferred because every view was held.
    pub deferred: u64,
    /// Publishes that grew a view's buffers (the writer outgrew `reserve`).
    pub grown: u64,
}

/// When the writer compacts a list (`compactions_due`). Plain data.
///
/// A list with dead rows (deleted, or expired before `head`) is due when
/// any of these holds:
/// - its oldest dead row died `window_ticks` or more ticks ago (the rolling
///   window: no dead row is kept much longer than that);
/// - its dead rows are more than `max_dead_share` of its rows;
/// - it has `max_dead_rows` dead rows or more.
///
/// At most `max_per_tick` lists are compacted per tick, oldest dead row
/// first, then lower list index, so the writer's cost per tick is bounded
/// and no due list waits forever.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompactPolicy {
    pub window_ticks: u64,
    pub max_dead_share: f32,
    pub max_dead_rows: usize,
    pub max_per_tick: usize,
}

/// One minute at 60 Hz; a quarter dead; two default blocks; two per tick.
pub const COMPACT_DEFAULT: CompactPolicy = CompactPolicy {
    window_ticks: 3600,
    max_dead_share: 0.25,
    max_dead_rows: 2 * DEFAULT_BLOCK_ROWS,
    max_per_tick: 2,
};

fn empty_meta() -> ListMeta { ListMeta { blocks: Vec::new(), len: 0, head: 0, dead: 0, tomb: Vec::new() } }

/// `dst = src`, reusing `dst`'s buffers: no allocation when `fits`.
/// No block is freed either: every block is also held by the writer.
fn copy_meta(dst: &mut ListMeta, src: &ListMeta) {
    dst.blocks.clone_from(&src.blocks);
    dst.tomb.clone_from(&src.tomb);
    dst.len = src.len;
    dst.head = src.head;
    dst.dead = src.dead;
}

/// `copy_meta(dst, src)` would not allocate.
fn fits(dst: &ListMeta, src: &ListMeta) -> bool {
    dst.blocks.capacity() >= src.blocks.len() && dst.tomb.capacity() >= src.tomb.len()
}

/// `publish` into `v` would not allocate: every list changed since `v.seq` fits.
fn view_fits(v: &View, w: &Writer) -> bool {
    w.touched.iter().enumerate().all(|(li, &t)| t <= v.seq || fits(&v.lists[li], &w.lists[li]))
        && (w.pend_touched <= v.seq || (fits(&v.pending, &w.pend) && v.pending_ids.capacity() >= w.pend_ids.len()))
}

/// Grow `dst`'s buffers to at least `src`'s capacity.
fn fit(dst: &mut ListMeta, src: &ListMeta) {
    dst.blocks.reserve(src.blocks.capacity().saturating_sub(dst.blocks.len()));
    dst.tomb.reserve(src.tomb.capacity().saturating_sub(dst.tomb.len()));
}

/// A new pooled view of the writer's lists and pending rows, fitted to their capacity.
#[allow(clippy::too_many_arguments)]
fn new_view(w_lists: &[ListMeta], pend: &ListMeta, pend_ids: &[u64], seq: u64, max_ts: u64, block_rows: usize, model: &Arc<IvfPqIndex>) -> Arc<View> {
    let mut lists = w_lists.to_vec();
    for (d, s) in lists.iter_mut().zip(w_lists) { fit(d, s); }
    let mut pending = pend.clone();
    fit(&mut pending, pend);
    let mut ids = Vec::with_capacity(pend_ids.len().max(pend.tomb.capacity()));
    ids.extend_from_slice(pend_ids);
    Arc::new(View { seq, max_ts, block_rows, model: model.clone(), lists, pending, pending_ids: ids })
}

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

/// Move retired blocks no view can see any more to `ready`. No allocation:
/// `ready` has room for every retired block (`compact_list`).
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
/// blocks of `block_rows` rows (`DEFAULT_BLOCK_ROWS` unless there is a
/// reason) and `DEFAULT_VIEWS` pooled views. Publishes `idx`'s state as view 0.
pub fn writer_from(idx: &IvfPqIndex, block_rows: usize) -> Writer {
    writer_with_pending(idx, block_rows, &[], &[], &[], &[])
}

/// `writer_from`, plus the pending real-memory rows of a loaded buffer
/// (`real_memory::Pending`'s `ids`, `timestamps`, `tags` and scored
/// `view` vectors, row-major), so view 0 already searches them.
pub fn writer_with_pending(idx: &IvfPqIndex, block_rows: usize, ids: &[u64], ts: &[u64], tags: &[u64], vecs: &[f32]) -> Writer {
    assert!(block_rows > 0, "block_rows must be > 0");
    assert!(ts.len() == ids.len() && tags.len() == ids.len() && vecs.len() == ids.len() * idx.dim, "pending column lengths");
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
    let mut ppool = Pool { cap: block_rows, m: 0, vdim: dim, ready: Vec::new(), retired: Vec::new(), allocated: 0 };
    let mut pend = empty_meta();
    let mut pend_dir: Vec<(u64, u32)> = Vec::with_capacity(ids.len());
    for (i, &id) in ids.iter().enumerate() {
        let r = push_row(&mut ppool, &mut pend, id, ts[i], tags[i], &[], &vecs[i * dim..(i + 1) * dim]);
        pend_dir.push((id, r as u32));
    }
    pend_dir.sort_unstable();
    let pend_ids: Vec<u64> = pend_dir.iter().map(|p| p.0).collect();
    let pend_rows: Vec<u32> = pend_dir.iter().map(|p| p.1).collect();
    let views: Vec<Arc<View>> = (0..DEFAULT_VIEWS).map(|_| new_view(&lists, &pend, &pend_ids, 0, idx.max_ts, block_rows, &model)).collect();
    Writer {
        pend, pend_ids, pend_rows, pend_touched: 0, ppool,
        scratch: live::scratch_for(&model),
        touched: vec![0; lists.len()],
        dead_since: lists.iter().map(|l| if l.dead > 0 { 0 } else { NEVER }).collect(),
        model, lists, dir: idx.dir.clone(), max_ts: idx.max_ts, seq: 1, pool,
        latest: Arc::new(ArcSwap::new(views[0].clone())),
        dirty: vec![false; views.len()],
        views,
        current: 0, pending: 0, deferred: 0, grown: 0,
    }
}

/// Handle for query threads.
pub fn latest(w: &Writer) -> Latest { w.latest.clone() }

/// The newest published view. Lock-free. Holding it keeps its rows readable.
pub fn current(l: &Latest) -> Arc<View> { l.load_full() }

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

/// Count one more dead row in list `li`, noting when its first one died.
fn kill(l: &mut ListMeta, dead_since: &mut u64, seq: u64) {
    if l.dead == 0 { *dead_since = seq; }
    l.dead += 1;
}

/// Mark `id` dead. O(1).
pub fn delete(w: &mut Writer, id: u64) -> Result<RowLoc, WriteError> {
    let loc = w.dir.remove(&id).ok_or(WriteError::NotFound { id })?;
    let li = loc.list as usize;
    let l = &mut w.lists[li];
    l.tomb[loc.row as usize] = 1;
    kill(l, &mut w.dead_since[li], w.seq);
    w.touched[li] = w.seq;
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
                kill(l, &mut w.dead_since[li], w.seq);
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
/// reading them. Keeps the list's free room (`reserve`), so the inserts
/// that room was made for still do not allocate.
pub fn compact_list(w: &mut Writer, list: u32) -> Result<usize, WriteError> {
    let li = list as usize;
    if li >= w.lists.len() { return Err(WriteError::NoSuchList { list }); }
    let (cap, vdim) = (w.pool.cap, w.pool.vdim);
    let old = std::mem::replace(&mut w.lists[li], empty_meta());
    let live = old.len - old.dead;
    let free_rows = old.blocks.len() * cap - old.len;
    let need_blocks = (live + free_rows + cap - 1) / cap;
    let new = &mut w.lists[li];
    new.tomb.reserve_exact(live + (old.tomb.capacity() - old.len));
    new.blocks.reserve_exact(need_blocks + (old.blocks.capacity() - old.blocks.len()));
    for r in old.head..old.len {
        if old.tomb[r] == 1 { continue; }
        let (b, s) = (&old.blocks[r / cap], r % cap);
        let vec = if vdim > 0 { b.vec_row(s) } else { &[] };
        let row = push_row(&mut w.pool, new, b.id(s), b.ts(s), b.tags(s), b.code_row(s), vec);
        w.dir.insert(b.id(s), RowLoc { list, row: row as u32 });
    }
    while new.blocks.len() < need_blocks { new.blocks.push(take_block(&mut w.pool)); }
    w.pool.retired.extend(old.blocks);
    // `reclaim` moves retired blocks to `ready` without growing it.
    w.pool.ready.reserve(w.pool.retired.len());
    w.dirty.iter_mut().for_each(|d| *d = true);
    w.dead_since[li] = NEVER;
    w.touched[li] = w.seq;
    Ok(old.dead)
}

/// Room for `rows_per_list` more rows in every list and in the directory,
/// so the next inserts do not allocate, and a pool of at least `views`
/// views (max query threads holding a view at once + 2) whose buffers fit
/// the writer's, so `publish` does not allocate either. Call between ticks,
/// e.g. once per tick: a view that is held during one call is fitted by a
/// later one (cheap when nothing grew).
pub fn reserve(w: &mut Writer, rows_per_list: usize, views: usize) {
    let cap = w.pool.cap;
    for l in w.lists.iter_mut() {
        let need = (l.len + rows_per_list + cap - 1) / cap;
        l.blocks.reserve(need.saturating_sub(l.blocks.len()));
        while l.blocks.len() < need { l.blocks.push(take_block(&mut w.pool)); }
        l.tomb.reserve(rows_per_list);
    }
    w.dir.reserve(rows_per_list * w.lists.len());
    // A new view holds the writer's state now: the end of tick seq - 1, plus
    // any ops of tick seq already applied, which `publish` copies again anyway.
    w.views.reserve(views.saturating_sub(w.views.len()));
    w.dirty.reserve(views.saturating_sub(w.dirty.len()));
    while w.views.len() < views {
        w.views.push(new_view(&w.lists, &w.pend, &w.pend_ids, w.seq - 1, w.max_ts, cap, &w.model));
        w.dirty.push(false);
    }
    fit_views(w);
    w.pool.ready.reserve(w.pool.retired.len());
}

/// Grow every free pooled view's buffers to the writer's capacity.
fn fit_views(w: &mut Writer) {
    let ids_cap = w.pend_ids.capacity();
    for v in w.views.iter_mut() {
        if let Some(v) = Arc::get_mut(v) {
            for (d, s) in v.lists.iter_mut().zip(&w.lists) { fit(d, s); }
            fit(&mut v.pending, &w.pend);
            v.pending_ids.reserve(ids_cap.saturating_sub(v.pending_ids.len()));
        }
    }
}

/// Room for `rows` more pending real memories (`real_memory::apply_writer`
/// inserts and upserts of real ids), in the writer and in every free
/// pooled view, so they and `publish` do not allocate. Call between ticks,
/// with `reserve`; pair it with `real_memory::reserve` on the buffer.
pub fn reserve_pending(w: &mut Writer, rows: usize) {
    let cap = w.ppool.cap;
    let l = &mut w.pend;
    let need = (l.len + rows).div_ceil(cap);
    l.blocks.reserve(need.saturating_sub(l.blocks.len()));
    while l.blocks.len() < need { l.blocks.push(take_block(&mut w.ppool)); }
    l.tomb.reserve(rows);
    w.pend_ids.reserve(rows);
    w.pend_rows.reserve(rows);
    fit_views(w);
    w.ppool.ready.reserve(w.ppool.retired.len());
}

/// Add a pending real memory (`v` is its scored vector: unit length for
/// cosine). The id must not be pending already. Called by
/// `real_memory::apply_writer` right after the buffer's own push.
pub fn pending_push(w: &mut Writer, id: u64, ts: u64, tags: u64, v: &[f32]) {
    let at = w.pend_ids.binary_search(&id).expect_err("id already pending");
    let r = push_row(&mut w.ppool, &mut w.pend, id, ts, tags, &[], v);
    w.pend_ids.insert(at, id);
    w.pend_rows.insert(at, r as u32);
    w.pend_touched = w.seq;
}

/// Drop pending id `id` (deleted, or replaced by an upsert). Returns
/// whether it was pending. O(pending) for the sorted id list.
pub fn pending_kill(w: &mut Writer, id: u64) -> bool {
    let Ok(at) = w.pend_ids.binary_search(&id) else { return false };
    let r = w.pend_rows[at] as usize;
    w.pend_ids.remove(at);
    w.pend_rows.remove(at);
    w.pend.tomb[r] = 1;
    w.pend.dead += 1;
    w.pend_touched = w.seq;
    true
}

/// Empty the pending rows (the commit). Their blocks are retired and
/// reused once no view holds them. Keeps every buffer's capacity.
pub fn pending_clear(w: &mut Writer) {
    if w.pend.len == 0 { return; }
    w.ppool.retired.extend(w.pend.blocks.drain(..));
    w.ppool.ready.reserve(w.ppool.retired.len());
    w.pend.tomb.clear();
    w.pend.len = 0;
    w.pend.dead = 0;
    w.pend_ids.clear();
    w.pend_rows.clear();
    w.dirty.iter_mut().for_each(|d| *d = true);
    w.pend_touched = w.seq;
}

/// Live pending real memories in the writer's current state.
pub fn writer_pending_rows(w: &Writer) -> usize { w.pend_ids.len() }

/// Live pending real memories this view searches.
pub fn pending_rows(v: &View) -> usize { v.pending_ids.len() }

/// Let free dirty views other than `skip` drop their handles to blocks a
/// compaction may have retired, so an idle view does not pin them. Clears
/// the block table of every list changed since the view's seq: those lists
/// are copied again whenever the view is reused, and `clear` keeps the
/// buffer. No allocation, no block freed (the writer holds every block).
fn sweep(w: &mut Writer, skip: Option<usize>) {
    for (i, v) in w.views.iter_mut().enumerate() {
        if !w.dirty[i] || Some(i) == skip { continue; }
        let Some(v) = Arc::get_mut(v) else { continue };
        for (l, &t) in v.lists.iter_mut().zip(&w.touched) {
            if t > v.seq { l.blocks.clear(); }
        }
        if w.pend_touched > v.seq { v.pending.blocks.clear(); }
        w.dirty[i] = false;
    }
}

/// End tick `w.seq`: publish its state as a view, or defer if every pooled
/// view is held by a query. No allocation after `reserve`, deferred or not.
pub fn publish(w: &mut Writer) -> Published {
    let seq = w.seq;
    w.seq += 1;
    // Free view to reuse: the newest (its lists are the fewest to copy and
    // still in cache), lower index on ties. Only if it would have to grow
    // (rare: it was held during the `reserve` that grew the writer), the
    // newest one that fits instead, and if none fits, grow the newest.
    let mut pick: Option<(u64, usize)> = None;
    let mut fitting: Option<(u64, usize)> = None;
    let newer = |a: (u64, usize), b: Option<(u64, usize)>| b.map_or(true, |b| a.0 > b.0 || (a.0 == b.0 && a.1 < b.1));
    for (i, v) in w.views.iter_mut().enumerate() {
        let Some(v) = Arc::get_mut(v) else { continue };
        if newer((v.seq, i), pick) { pick = Some((v.seq, i)); }
    }
    let grow = match pick {
        Some((_, i)) if !view_fits(&w.views[i], w) => {
            for i in 0..w.views.len() {
                let v = &w.views[i];
                // Free: only the pool holds it (the slot never holds a free view).
                if Arc::strong_count(v) != 1 { continue; }
                if newer((v.seq, i), fitting) && view_fits(v, w) { fitting = Some((v.seq, i)); }
            }
            fitting.is_none()
        }
        _ => false,
    };
    sweep(w, fitting.or(pick).map(|p| p.1));
    let Some((_, i)) = fitting.or(pick) else {
        // Every view is held: keep the current one; the next publish covers this tick too.
        w.pending += 1;
        w.deferred += 1;
        reclaim(&mut w.pool);
        reclaim(&mut w.ppool);
        return Published { seq, current: w.current, deferred: w.pending };
    };
    if grow { w.grown += 1; }
    let dst = Arc::get_mut(&mut w.views[i]).expect("free view");
    for ((d, s), &t) in dst.lists.iter_mut().zip(&w.lists).zip(&w.touched) {
        if t > dst.seq { copy_meta(d, s); }
    }
    if w.pend_touched > dst.seq {
        copy_meta(&mut dst.pending, &w.pend);
        dst.pending_ids.clone_from(&w.pend_ids);
    }
    dst.seq = seq;
    dst.max_ts = w.max_ts;
    // The old view stays in `views`: never the last handle, so `store` frees nothing.
    w.latest.store(w.views[i].clone());
    w.dirty[i] = false;
    w.current = seq;
    w.pending = 0;
    reclaim(&mut w.pool);
    reclaim(&mut w.ppool);
    Published { seq, current: seq, deferred: 0 }
}

/// Lists to compact this tick under `p`, as `Op::Compact`s in `out`
/// (cleared first). The writer applies them with `apply` like any other op,
/// so they go into the log. Pure: depends only on the lists' row and dead
/// counts, `dead_since` and `seq`, all of which the op log determines.
/// No allocation if `out` has room for `p.max_per_tick` ops.
pub fn compactions_due(p: &CompactPolicy, w: &Writer, out: &mut Vec<Op>) {
    out.clear();
    let due = |li: usize| {
        let (l, since) = (&w.lists[li], w.dead_since[li]);
        l.dead > 0
            && (w.seq.saturating_sub(since) >= p.window_ticks
                || l.dead >= p.max_dead_rows
                || l.dead as f32 > p.max_dead_share * l.len as f32)
    };
    // Up to `max_per_tick` smallest (dead_since, list) keys among due
    // lists, by repeated scans: no sort buffer, and max_per_tick is small.
    let mut last: Option<(u64, usize)> = None;
    for _ in 0..p.max_per_tick {
        let mut best: Option<(u64, usize)> = None;
        for li in 0..w.lists.len() {
            let key = (w.dead_since[li], li);
            if last.map_or(false, |k| key <= k) || best.map_or(false, |b| key >= b) || !due(li) { continue; }
            best = Some(key);
        }
        let Some(b) = best else { break };
        out.push(Op::Compact { list: b.1 as u32 });
        last = best;
    }
}

pub fn locate(w: &Writer, id: u64) -> Option<RowLoc> { w.dir.get(&id).copied() }

pub fn live_rows(w: &Writer) -> usize { w.dir.len() }

pub fn arena_stats(w: &Writer) -> ArenaStats {
    ArenaStats {
        allocated: w.pool.allocated, ready: w.pool.ready.len(), retired: w.pool.retired.len(), views: w.views.len(),
        deferred: w.deferred, grown: w.grown,
    }
}

/// The view in the slot holds exactly the writer's state: the last
/// publish went through and nothing was applied since.
pub fn is_current(w: &Writer) -> bool {
    w.current + 1 == w.seq && w.touched.iter().all(|&t| t <= w.current) && w.pend_touched <= w.current
}

/// Rows per block.
pub fn block_rows(w: &Writer) -> usize { w.pool.cap }

/// The scratch the last write encoded with: `live::last_encoded(&w.model,
/// scratch(w))` is the last inserted row's encoded vector and codes, the
/// stats hook for `drift::record`.
pub fn scratch(w: &Writer) -> &Scratch { &w.scratch }

/// Swap in a rebuilt model (stage 3): replace the writer's model, rows and
/// directory with `idx`'s (the caught-up rebuild), keeping the tick (`seq`)
/// and the query handle. Queries keep the current view (old model and
/// blocks) until the next `publish`, which shows the new model. Lists that
/// arrive with dead rows count them as dead since this tick
/// (`dead_since = seq`), so the compaction policy is a function of the log.
/// O(rows) and allocates: every row is copied into new blocks once. The old
/// pool's views and blocks are freed when the last holder drops them (the
/// view in the slot at the next `publish`, on the writer thread).
pub fn rebase(w: &mut Writer, idx: &IvfPqIndex) {
    let mut n = writer_from(idx, w.pool.cap);
    n.seq = w.seq;
    n.dead_since = n.lists.iter().map(|l| if l.dead > 0 { w.seq } else { NEVER }).collect();
    n.latest = w.latest.clone();
    n.current = w.current;
    n.pending = w.pending;
    n.deferred = w.deferred;
    n.grown = w.grown;
    // The pending rows do not depend on the model: they move over as they are.
    std::mem::swap(&mut n.pend, &mut w.pend);
    std::mem::swap(&mut n.pend_ids, &mut w.pend_ids);
    std::mem::swap(&mut n.pend_rows, &mut w.pend_rows);
    std::mem::swap(&mut n.ppool, &mut w.ppool);
    n.pend_touched = n.seq;
    fit_views(&mut n);
    *w = n;
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

/// Top-k over this view's short-term rows and its pending real memories,
/// by (score desc, id asc), each id at most once: `real_memory::search`
/// on (`to_index(v)`, the buffer at `v.seq`), bit for bit. With no pending
/// rows, exactly `IvfPqIndex::search_temporal` on `to_index(v)`.
pub fn search_temporal(v: &View, q: &[f32], k: usize, required_tag: Option<u64>, current_time: u64, lambda: f32) -> Vec<Hit> {
    let p = v.pending_ids.len();
    if p == 0 { return search_rows(&v.model, v, q, k, required_tag, current_time, lambda); }
    if k == 0 || q.iter().any(|x| !x.is_finite()) { return Vec::new(); }
    // k + p from the index, so dropping the pending ids still leaves k.
    let mut hits = search_rows(&v.model, v, q, k + p, required_tag, current_time, lambda);
    hits.retain(|h| v.pending_ids.binary_search(&h.id).is_err());
    let mut q_eff = q.to_vec();
    if matches!(v.model.metric, Metric::Cosine) { norm_inplace(&mut q_eff); }
    pending_hits(v, &q_eff, required_tag, current_time, lambda, &mut hits);
    stable_top_k(hits, k)
}

/// Exact scores of the view's live pending rows that pass the tag filter,
/// appended to `out`. Same arithmetic as `real_memory::search`'s scan;
/// four rows per kernel call (bit-identical to one at a time).
fn pending_hits(v: &View, q_eff: &[f32], required_tag: Option<u64>, current_time: u64, lambda: f32, out: &mut Vec<Hit>) {
    let l = &v.pending;
    let cosine = matches!(v.model.metric, Metric::Cosine);
    let score4 = |rs: [&[f32]; 4]| if cosine { metric::dot_x4(q_eff, rs) } else {
        let d = metric::l2_distance_x4(q_eff, rs); [-d[0], -d[1], -d[2], -d[3]]
    };
    let score1 = |r: &[f32]| if cosine { metric::dot(q_eff, r) } else { -metric::l2_distance(q_eff, r) };
    // Up to 4 eligible rows (id, ts, vector) waiting for one kernel call.
    let mut batch: [(u64, u64, &[f32]); 4] = [(0, 0, &[]); 4];
    let mut nb = 0;
    let cap = v.block_rows;
    let mut r = 0;
    while r < l.len {
        let b = &l.blocks[r / cap];
        let run = b.rows(0, cap.min(l.len - r));
        for s in 0..run.ids.len() {
            if l.tomb[r + s] == 1 { continue; }
            if let Some(mask) = required_tag {
                if run.tags[s] & mask == 0 { continue; }
            }
            batch[nb] = (run.ids[s], run.ts[s], b.vec_row(s));
            nb += 1;
            if nb == 4 {
                let sc = score4([batch[0].2, batch[1].2, batch[2].2, batch[3].2]);
                for j in 0..4 { out.push(Hit { id: batch[j].0, score: apply_decay(sc[j], batch[j].1, current_time, lambda) }); }
                nb = 0;
            }
        }
        r += run.ids.len();
    }
    for &(id, ts, row) in &batch[..nb] {
        out.push(Hit { id, score: apply_decay(score1(row), ts, current_time, lambda) });
    }
}

pub fn search(v: &View, q: &[f32], k: usize) -> Vec<Hit> { search_temporal(v, q, k, None, 0, 0.0) }

/// As `IvfPqIndex::search_pick`, on this view's rows.
pub fn search_pick(v: &View, q: &[f32], k: usize, pick: usize, required_tag: Option<u64>, current_time: u64, lambda: f32, seed: u64) -> Vec<Hit> {
    crate::pick::pick_ranked(&search_temporal(v, q, k, required_tag, current_time, lambda), pick, seed)
}

/// Rows of `lists` (blocks of `cap` rows) as a plain `IvfPqIndex` with
/// `model`'s model, directory rebuilt. O(rows).
fn index_from(model: &IvfPqIndex, lists: &[ListMeta], cap: usize, max_ts: u64) -> IvfPqIndex {
    let mut idx = model.clone();
    for (li, l) in lists.iter().enumerate() {
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
    idx.max_ts = max_ts;
    idx
}

/// The view as a plain `IvfPqIndex` (rows copied into contiguous lists,
/// directory rebuilt). O(rows); for tests and fingerprints.
pub fn to_index(v: &View) -> IvfPqIndex { index_from(&v.model, &v.lists, v.block_rows, v.max_ts) }

/// The writer's current state (including ops of the tick not yet
/// published) as a plain `IvfPqIndex`. O(rows); for tests.
pub fn writer_index(w: &Writer) -> IvfPqIndex { index_from(&w.model, &w.lists, w.pool.cap, w.max_ts) }

/// `IvfPqIndex::fingerprint` of `lists`, folded straight from the blocks
/// (no copy). Same value as `index_from(..).fingerprint()`.
fn lists_fingerprint(model: &IvfPqIndex, lists: &[ListMeta], cap: usize, max_ts: u64) -> u64 {
    let mut h = fingerprint_head(model, max_ts);
    for l in lists {
        h = fnv(h, l.len as u64);
        h = fnv(h, l.head as u64);
        h = fnv(h, l.dead as u64);
        let runs = || (0..l.len.div_ceil(cap)).map(move |b| l.blocks[b].rows(0, cap.min(l.len - b * cap)));
        for r in runs() { for &id in r.ids { h = fnv(h, id); } }
        for r in runs() { for &ts in r.ts { h = fnv(h, ts); } }
        for &t in &l.tomb { h = fnv(h, t as u64); }
        for r in runs() { for &c in r.codes { h = fnv(h, c as u64); } }
        for r in runs() { for &t in r.tags { h = fnv(h, t); } }
        if model.params.store_vecs {
            for r in 0..l.len { for &x in l.blocks[r / cap].vec_row(r % cap) { h = fnv(h, x.to_bits() as u64); } }
        }
    }
    h
}

/// `IvfPqIndex::fingerprint` of the view's contents. O(rows), no copy;
/// safe on any thread while the view is held.
pub fn fingerprint(v: &View) -> u64 { lists_fingerprint(&v.model, &v.lists, v.block_rows, v.max_ts) }

/// `IvfPqIndex::fingerprint` of the writer's current state. O(rows), no copy.
pub fn writer_fingerprint(w: &Writer) -> u64 { lists_fingerprint(&w.model, &w.lists, w.pool.cap, w.max_ts) }
