//! Snapshot file: the whole `IvfPqIndex` (model, lists, max_ts) at one
//! log seq, written atomically and verified on load.
//!
//! This is the only code that knows the in-memory list layout on disk.
//! Stage 2 (segmented row arena) changes that layout: adapt `put_list` /
//! `get_list` and bump `SNAPSHOT_VERSION`; the header, the atomic write
//! and the checks stay as they are.
//!
//! File (little-endian, fixed width):
//!
//! | bytes | field |
//! | --- | --- |
//! | 8 | magic `RASNAP\0\0` |
//! | 4 | version (1) |
//! | 4 | reserved (0) |
//! | 8 | `seq`: last op applied |
//! | 8 | `tick` of that op |
//! | 8 | `IvfPqIndex::fingerprint` |
//! | 8 | payload length |
//! | 4 | CRC32 of the payload |
//! | 4 | CRC32 of the 52 bytes above |
//!
//! Payload, in order: metric u8 (0 L2, 1 Cosine), dim u64; params (nlist,
//! nprobe, refine, seed, m u64, nbits u8, iters u64, use_opq u8, opq_mode
//! u8, opq_sweeps u64, store_vecs u8); max_ts u64; centroids (count u64,
//! count x dim f32); centroids_u (flag u8, then as centroids); PQ (d, m,
//! dsub u64, nbits u8, k, iters, seed u64, codebook count u64, each: len
//! u64 + f32s); OPQ (flag u8, then dim, m u64, matrix flag u8 + dim x dim
//! f32, perm flag u8 + dim x u64); lists (count u64, each: n, head, dead
//! u64, ids n x u64, timestamps n x u64, tomb n x u8, tags n x u64, codes
//! n x m u8, vecs flag u8 + n x dim f32).
//!
//! The id directory is not stored: it is rebuilt from the lists (every
//! row with `tomb == 0`), which is exactly what the live directory holds.
//!
//! Load refuses the file unless header CRC, payload CRC, payload length,
//! `live::check_invariants` and the fingerprint all check out.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::ivfpq::{IvfPqIndex, IvfPqList, IvfPqParams, OpqMode, RowLoc};
use crate::metric::Metric;
use crate::oplog::crc32_update;
use crate::opq::Opq;
use crate::pq::ProductQuantizer;

pub const SNAPSHOT_MAGIC: [u8; 8] = *b"RASNAP\0\0";
pub const SNAPSHOT_VERSION: u32 = 1;
pub const SNAPSHOT_HEADER_LEN: usize = 56;

/// A loaded, verified snapshot.
pub struct Snapshot { pub idx: IvfPqIndex, pub seq: u64, pub tick: u64 }

fn bad(msg: &str) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, format!("snapshot: {msg}")) }

/// `path` with `.tmp` appended.
pub fn tmp_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".tmp");
    PathBuf::from(s)
}

/// fsync a directory so a create / rename / remove in it is durable.
pub fn sync_dir(dir: &Path) -> io::Result<()> { File::open(dir)?.sync_all() }

/// Directory holding `path` ("." for a bare file name).
pub fn parent_dir(path: &Path) -> &Path {
    match path.parent() { Some(p) if !p.as_os_str().is_empty() => p, _ => Path::new(".") }
}

// ------------------------------------------------------------ write

/// Byte sink that tracks length and CRC of everything written.
struct Sink { w: BufWriter<File>, crc: u32, len: u64 }

fn put(s: &mut Sink, b: &[u8]) -> io::Result<()> {
    s.crc = crc32_update(s.crc, b);
    s.len += b.len() as u64;
    s.w.write_all(b)
}
fn put_u8(s: &mut Sink, x: u8) -> io::Result<()> { put(s, &[x]) }
fn put_u64(s: &mut Sink, x: u64) -> io::Result<()> { put(s, &x.to_le_bytes()) }

/// Fixed-width little-endian columns, encoded through a stack buffer.
fn put_u64s(s: &mut Sink, xs: &[u64]) -> io::Result<()> {
    let mut buf = [0u8; 8 * 1024];
    for c in xs.chunks(1024) {
        for (o, x) in buf.chunks_exact_mut(8).zip(c) { o.copy_from_slice(&x.to_le_bytes()); }
        put(s, &buf[..c.len() * 8])?;
    }
    Ok(())
}
fn put_f32s(s: &mut Sink, xs: &[f32]) -> io::Result<()> {
    let mut buf = [0u8; 4 * 2048];
    for c in xs.chunks(2048) {
        for (o, x) in buf.chunks_exact_mut(4).zip(c) { o.copy_from_slice(&x.to_bits().to_le_bytes()); }
        put(s, &buf[..c.len() * 4])?;
    }
    Ok(())
}

fn put_rows(s: &mut Sink, rows: &[Vec<f32>]) -> io::Result<()> {
    put_u64(s, rows.len() as u64)?;
    for r in rows { put_f32s(s, r)?; }
    Ok(())
}

fn put_params(s: &mut Sink, p: &IvfPqParams) -> io::Result<()> {
    for x in [p.nlist, p.nprobe, p.refine] { put_u64(s, x as u64)?; }
    put_u64(s, p.seed)?;
    put_u64(s, p.m as u64)?;
    put_u8(s, p.nbits)?;
    put_u64(s, p.iters as u64)?;
    put_u8(s, p.use_opq as u8)?;
    put_u8(s, match p.opq_mode { OpqMode::Perm => 0, OpqMode::Pca => 1, OpqMode::PcaPerm => 2 })?;
    put_u64(s, p.opq_sweeps as u64)?;
    put_u8(s, p.store_vecs as u8)
}

fn put_pq(s: &mut Sink, pq: &ProductQuantizer) -> io::Result<()> {
    for x in [pq.d, pq.m, pq.dsub] { put_u64(s, x as u64)?; }
    put_u8(s, pq.nbits)?;
    for x in [pq.k, pq.iters] { put_u64(s, x as u64)?; }
    put_u64(s, pq.seed)?;
    put_u64(s, pq.codebooks.len() as u64)?;
    for cb in &pq.codebooks { put_u64(s, cb.len() as u64)?; put_f32s(s, cb)?; }
    Ok(())
}

fn put_opq(s: &mut Sink, o: &Option<Opq>) -> io::Result<()> {
    let Some(o) = o else { return put_u8(s, 0) };
    put_u8(s, 1)?;
    put_u64(s, o.dim as u64)?;
    put_u64(s, o.m as u64)?;
    match o.matrix() { Some(r) => { put_u8(s, 1)?; put_f32s(s, r)?; } None => put_u8(s, 0)? }
    match o.permutation() {
        Some(p) => { put_u8(s, 1)?; for &i in p { put_u64(s, i as u64)?; } }
        None => put_u8(s, 0)?,
    }
    Ok(())
}

/// Layout-specific: one inverted list. Stage 2 changes this.
fn put_list(s: &mut Sink, l: &IvfPqList) -> io::Result<()> {
    for x in [l.ids.len(), l.head, l.dead] { put_u64(s, x as u64)?; }
    put_u64s(s, &l.ids)?;
    put_u64s(s, &l.timestamps)?;
    put(s, &l.tomb)?;
    put_u64s(s, &l.tags)?;
    put(s, &l.codes)?;
    match &l.vecs { Some(v) => { put_u8(s, 1)?; put_f32s(s, v) } None => put_u8(s, 0) }
}

fn put_index(s: &mut Sink, idx: &IvfPqIndex) -> io::Result<()> {
    put_u8(s, match idx.metric { Metric::L2 => 0, Metric::Cosine => 1 })?;
    put_u64(s, idx.dim as u64)?;
    put_params(s, &idx.params)?;
    put_u64(s, idx.max_ts)?;
    put_rows(s, &idx.centroids)?;
    match &idx.centroids_u { Some(c) => { put_u8(s, 1)?; put_rows(s, c)?; } None => put_u8(s, 0)? }
    put_pq(s, &idx.pq)?;
    put_opq(s, &idx.opq)?;
    put_u64(s, idx.lists.len() as u64)?;
    for l in &idx.lists { put_list(s, l)?; }
    Ok(())
}

/// Write `idx` (the state after op `seq`, applied in `tick`) to `path`
/// atomically: stream to `path.tmp`, fsync, rename, fsync the directory.
/// A crash at any point leaves either the old file or the new one, never
/// a half-written `path`. Returns the fingerprint stored.
pub fn write(path: &Path, idx: &IvfPqIndex, seq: u64, tick: u64) -> io::Result<u64> {
    let fp = idx.fingerprint();
    let tmp = tmp_path(path);
    let f = File::create(&tmp)?;
    let mut s = Sink { w: BufWriter::with_capacity(1 << 20, f), crc: 0, len: 0 };
    s.w.write_all(&[0u8; SNAPSHOT_HEADER_LEN])?;
    put_index(&mut s, idx)?;
    let (crc, len) = (s.crc, s.len);
    let mut f = s.w.into_inner().map_err(|e| e.into_error())?;
    f.seek(SeekFrom::Start(0))?;
    f.write_all(&header_bytes(seq, tick, fp, len, crc))?;
    f.sync_all()?;
    drop(f);
    fs::rename(&tmp, path)?;
    sync_dir(parent_dir(path))?;
    Ok(fp)
}

fn header_bytes(seq: u64, tick: u64, fp: u64, len: u64, crc: u32) -> [u8; SNAPSHOT_HEADER_LEN] {
    let mut h = [0u8; SNAPSHOT_HEADER_LEN];
    h[0..8].copy_from_slice(&SNAPSHOT_MAGIC);
    h[8..12].copy_from_slice(&SNAPSHOT_VERSION.to_le_bytes());
    h[16..24].copy_from_slice(&seq.to_le_bytes());
    h[24..32].copy_from_slice(&tick.to_le_bytes());
    h[32..40].copy_from_slice(&fp.to_le_bytes());
    h[40..48].copy_from_slice(&len.to_le_bytes());
    h[48..52].copy_from_slice(&crc.to_le_bytes());
    let hc = crc32_update(0, &h[..52]);
    h[52..56].copy_from_slice(&hc.to_le_bytes());
    h
}

// ------------------------------------------------------------ read

/// Byte source over the payload: tracks CRC and refuses to read (or
/// allocate for) more than the payload length says is left.
struct Src { r: BufReader<File>, crc: u32, left: u64 }

fn get(s: &mut Src, out: &mut [u8]) -> io::Result<()> {
    if out.len() as u64 > s.left { return Err(bad("payload shorter than its contents")); }
    s.r.read_exact(out)?;
    s.left -= out.len() as u64;
    s.crc = crc32_update(s.crc, out);
    Ok(())
}
fn get_u8(s: &mut Src) -> io::Result<u8> { let mut b = [0u8; 1]; get(s, &mut b)?; Ok(b[0]) }
fn get_u64(s: &mut Src) -> io::Result<u64> { let mut b = [0u8; 8]; get(s, &mut b)?; Ok(u64::from_le_bytes(b)) }
fn get_usize(s: &mut Src) -> io::Result<usize> { usize::try_from(get_u64(s)?).map_err(|_| bad("length overflow")) }
fn get_bool(s: &mut Src) -> io::Result<bool> {
    match get_u8(s)? { 0 => Ok(false), 1 => Ok(true), _ => Err(bad("bad flag")) }
}

/// Bytes for `n` items of `width` bytes, if they can still be in the
/// payload. Checked before allocating, so a bad length cannot ask for
/// more memory than the file holds.
fn room(s: &Src, n: usize, width: usize) -> io::Result<usize> {
    let bytes = n.checked_mul(width).ok_or_else(|| bad("length overflow"))?;
    if bytes as u64 > s.left { return Err(bad("length past end of payload")); }
    Ok(bytes)
}

fn get_bytes(s: &mut Src, n: usize) -> io::Result<Vec<u8>> {
    let mut v = vec![0u8; room(s, n, 1)?];
    get(s, &mut v)?;
    Ok(v)
}
fn get_u64s(s: &mut Src, n: usize) -> io::Result<Vec<u64>> {
    let raw = get_bytes(s, room(s, n, 8)?)?;
    Ok(raw.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect())
}
fn get_f32s(s: &mut Src, n: usize) -> io::Result<Vec<f32>> {
    let raw = get_bytes(s, room(s, n, 4)?)?;
    Ok(raw.chunks_exact(4).map(|c| f32::from_bits(u32::from_le_bytes(c.try_into().unwrap()))).collect())
}

fn get_rows(s: &mut Src, dim: usize) -> io::Result<Vec<Vec<f32>>> {
    let n = get_usize(s)?;
    room(s, n, dim.max(1) * 4)?;
    (0..n).map(|_| get_f32s(s, dim)).collect()
}

fn get_params(s: &mut Src) -> io::Result<IvfPqParams> {
    let nlist = get_usize(s)?;
    let nprobe = get_usize(s)?;
    let refine = get_usize(s)?;
    let seed = get_u64(s)?;
    let m = get_usize(s)?;
    let nbits = get_u8(s)?;
    let iters = get_usize(s)?;
    let use_opq = get_bool(s)?;
    let opq_mode = match get_u8(s)? { 0 => OpqMode::Perm, 1 => OpqMode::Pca, 2 => OpqMode::PcaPerm, _ => return Err(bad("opq_mode")) };
    let opq_sweeps = get_usize(s)?;
    let store_vecs = get_bool(s)?;
    Ok(IvfPqParams { nlist, nprobe, refine, seed, m, nbits, iters, use_opq, opq_mode, opq_sweeps, store_vecs })
}

fn get_pq(s: &mut Src) -> io::Result<ProductQuantizer> {
    let d = get_usize(s)?;
    let m = get_usize(s)?;
    let dsub = get_usize(s)?;
    let nbits = get_u8(s)?;
    let k = get_usize(s)?;
    let iters = get_usize(s)?;
    let seed = get_u64(s)?;
    let ncb = get_usize(s)?;
    room(s, ncb, 8)?;
    let mut codebooks = Vec::with_capacity(ncb);
    for _ in 0..ncb { let n = get_usize(s)?; codebooks.push(get_f32s(s, n)?); }
    if m == 0 || d != m * dsub || !(1..=8).contains(&nbits) || k != 1usize << nbits || ncb != m
        || codebooks.iter().any(|cb| cb.len() != k * dsub) {
        return Err(bad("PQ shape"));
    }
    Ok(ProductQuantizer { d, m, dsub, nbits, k, iters, seed, codebooks })
}

fn get_opq(s: &mut Src) -> io::Result<Option<Opq>> {
    if !get_bool(s)? { return Ok(None); }
    let dim = get_usize(s)?;
    let m = get_usize(s)?;
    let mut o = Opq::identity(dim, m);
    if get_bool(s)? {
        let r = get_f32s(s, dim.checked_mul(dim).ok_or_else(|| bad("OPQ dim"))?)?;
        o = o.with_matrix(r);
    }
    if get_bool(s)? {
        let p = get_u64s(s, dim)?;
        let mut seen = vec![false; dim];
        let mut perm = Vec::with_capacity(dim);
        for x in p {
            let i = usize::try_from(x).map_err(|_| bad("OPQ perm"))?;
            if i >= dim || seen[i] { return Err(bad("OPQ perm")); }
            seen[i] = true;
            perm.push(i);
        }
        o = o.with_perm(perm);
    }
    Ok(Some(o))
}

/// Layout-specific: one inverted list. Stage 2 changes this.
fn get_list(s: &mut Src, m: usize, dim: usize) -> io::Result<IvfPqList> {
    let n = get_usize(s)?;
    let head = get_usize(s)?;
    let dead = get_usize(s)?;
    let ids = get_u64s(s, n)?;
    let timestamps = get_u64s(s, n)?;
    let tomb = get_bytes(s, n)?;
    let tags = get_u64s(s, n)?;
    let codes = get_bytes(s, n.checked_mul(m).ok_or_else(|| bad("codes length"))?)?;
    let vecs = if get_bool(s)? { Some(get_f32s(s, n.checked_mul(dim).ok_or_else(|| bad("vecs length"))?)?) } else { None };
    if head > n || dead > n || tomb.iter().any(|&t| t > 1) { return Err(bad("list shape")); }
    Ok(IvfPqList { ids, codes, tomb, timestamps, tags, vecs, head, dead })
}

fn get_index(s: &mut Src) -> io::Result<IvfPqIndex> {
    let metric = match get_u8(s)? { 0 => Metric::L2, 1 => Metric::Cosine, _ => return Err(bad("metric")) };
    let dim = get_usize(s)?;
    let params = get_params(s)?;
    let max_ts = get_u64(s)?;
    let centroids = get_rows(s, dim)?;
    let centroids_u = if get_bool(s)? { Some(get_rows(s, dim)?) } else { None };
    let pq = get_pq(s)?;
    let opq = get_opq(s)?;
    if pq.d != dim || pq.m != params.m || opq.as_ref().map_or(false, |o| o.dim != dim) {
        return Err(bad("model shape"));
    }
    let nl = get_usize(s)?;
    if nl != centroids.len() { return Err(bad("list count")); }
    room(s, nl, 25)?;
    let mut lists = Vec::with_capacity(nl);
    for _ in 0..nl { lists.push(get_list(s, params.m, dim)?); }
    if lists.iter().any(|l| l.vecs.is_some() != params.store_vecs) { return Err(bad("store_vecs")); }

    // The directory: every row not tombstoned, which is what live keeps.
    let live: usize = lists.iter().map(|l| l.ids.len() - l.dead).sum();
    let mut dir = HashMap::with_capacity(live);
    for (li, l) in lists.iter().enumerate() {
        for (r, (&id, &t)) in l.ids.iter().zip(&l.tomb).enumerate() {
            if t == 0 && dir.insert(id, RowLoc { list: li as u32, row: r as u32 }).is_some() {
                return Err(bad("duplicate live id"));
            }
        }
    }
    Ok(IvfPqIndex { metric, dim, params, centroids, centroids_u, pq, opq, lists, dir, max_ts })
}

/// Read and verify a snapshot. Any mismatch (header or payload CRC,
/// length, shape, `live::check_invariants`, fingerprint) is an
/// `InvalidData` error and nothing is returned.
pub fn read(path: &Path) -> io::Result<Snapshot> {
    let f = File::open(path)?;
    let file_len = f.metadata()?.len();
    let mut r = BufReader::with_capacity(1 << 20, f);
    let mut h = [0u8; SNAPSHOT_HEADER_LEN];
    r.read_exact(&mut h).map_err(|_| bad("short header"))?;
    let u32_at = |i: usize| u32::from_le_bytes(h[i..i + 4].try_into().unwrap());
    let u64_at = |i: usize| u64::from_le_bytes(h[i..i + 8].try_into().unwrap());
    if h[0..8] != SNAPSHOT_MAGIC || u32_at(8) != SNAPSHOT_VERSION { return Err(bad("magic or version")); }
    if u32_at(52) != crc32_update(0, &h[..52]) { return Err(bad("header CRC")); }
    let (seq, tick, fp, len, crc) = (u64_at(16), u64_at(24), u64_at(32), u64_at(40), u32_at(48));
    if file_len != SNAPSHOT_HEADER_LEN as u64 + len { return Err(bad("file length")); }
    let mut s = Src { r, crc: 0, left: len };
    let idx = get_index(&mut s)?;
    if s.left != 0 { return Err(bad("trailing payload")); }
    if s.crc != crc { return Err(bad("payload CRC")); }
    crate::live::check_invariants(&idx).map_err(|e| bad(&e))?;
    if idx.fingerprint() != fp { return Err(bad("fingerprint mismatch")); }
    Ok(Snapshot { idx, seq, tick })
}
