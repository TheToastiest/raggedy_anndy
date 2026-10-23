//! Fixed-capacity row blocks shared by the writer thread and the query
//! threads. All `unsafe` of the live-view path is in this file.
//!
//! A `Block` holds up to `cap` rows of one inverted list in SoA columns
//! (ids, timestamps, tags, PQ codes, raw vectors), allocated once at full
//! size, so rows never move. Rows are appended in slot order by `push`.
//! A written slot is never written again until `reset`, which takes
//! `&mut Block`: no other thread can be reading it then. `views` gets that
//! `&mut` with `Arc::get_mut`, i.e. only once no view holds the block.
//!
//! Slot protocol (why the shared reads and writes below are sound):
//! - `push` claims slot n with `claimed.fetch_add` (so exactly one caller
//!   writes slot n), writes it, waits for `filled == n` and stores
//!   `filled = n + 1` with Release.
//! - Readers load `filled` with Acquire and only form references to slots
//!   below it. Those slots were fully written before the Release store, and
//!   no `push` touches them again.
//! So no slot is written while a reference to it exists, and every read
//! sees the finished write.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::ivfpq::Run;

pub struct Block {
    cap: usize,
    m: usize,
    /// Raw-vector width; 0 when the index does not store raw vectors.
    vdim: usize,
    claimed: AtomicUsize,
    filled: AtomicUsize,
    ids: Box<[UnsafeCell<u64>]>,
    ts: Box<[UnsafeCell<u64>]>,
    tags: Box<[UnsafeCell<u64>]>,
    codes: Box<[UnsafeCell<u8>]>,
    vecs: Box<[UnsafeCell<f32>]>,
}

// SAFETY: every shared access goes through `push` (one writer per slot,
// before the slot is published) or the readers below (only slots under an
// Acquire load of `filled`). See the module doc.
unsafe impl Sync for Block {}

fn cells<T: Copy>(n: usize, zero: T) -> Box<[UnsafeCell<T>]> { (0..n).map(|_| UnsafeCell::new(zero)).collect() }

/// Shared slice of `cells[from..to]`.
///
/// # Safety
/// No element of `cells[from..to]` may be written while the slice lives.
#[inline]
unsafe fn view<T>(cells: &[UnsafeCell<T>], from: usize, to: usize) -> &[T] {
    let s = &cells[from..to]; // bounds check
    // UnsafeCell<T> is repr(transparent) over T.
    std::slice::from_raw_parts(s.as_ptr() as *const T, s.len())
}

/// Copy `src` into `cells[at..at + src.len()]`.
///
/// # Safety
/// No reference to those elements may exist, and no other thread may
/// access them, during the call.
#[inline]
unsafe fn write<T: Copy>(cells: &[UnsafeCell<T>], at: usize, src: &[T]) {
    let dst = &cells[at..at + src.len()]; // bounds check
    std::ptr::copy_nonoverlapping(src.as_ptr(), UnsafeCell::raw_get(dst.as_ptr()), src.len());
}

impl Block {
    /// Empty block for `cap` rows of `m` code bytes and `vdim` floats.
    pub fn new(cap: usize, m: usize, vdim: usize) -> Block {
        Block {
            cap, m, vdim,
            claimed: AtomicUsize::new(0), filled: AtomicUsize::new(0),
            ids: cells(cap, 0), ts: cells(cap, 0), tags: cells(cap, 0),
            codes: cells(cap * m, 0), vecs: cells(cap * vdim, 0.0),
        }
    }

    #[inline] pub fn cap(&self) -> usize { self.cap }

    /// Rows written so far.
    #[inline] pub fn filled(&self) -> usize { self.filled.load(Ordering::Acquire) }

    /// Append one row and return its slot. Panics if the block is full.
    /// `vec` must be `vdim` long (empty when raw vectors are not stored).
    pub fn push(&self, id: u64, ts: u64, tags: u64, codes: &[u8], vec: &[f32]) -> usize {
        assert_eq!(codes.len(), self.m, "code length");
        assert_eq!(vec.len(), self.vdim, "vector length");
        let n = self.claimed.fetch_add(1, Ordering::Relaxed);
        assert!(n < self.cap, "block full");
        // SAFETY: slot n is ours alone (fetch_add) and not yet below
        // `filled`, so no reader has a reference to it.
        unsafe {
            write(&self.ids, n, &[id]);
            write(&self.ts, n, &[ts]);
            write(&self.tags, n, &[tags]);
            write(&self.codes, n * self.m, codes);
            write(&self.vecs, n * self.vdim, vec);
        }
        while self.filled.load(Ordering::Acquire) != n { std::hint::spin_loop(); }
        self.filled.store(n + 1, Ordering::Release);
        n
    }

    #[inline]
    fn check(&self, to: usize) {
        let f = self.filled();
        assert!(to <= f, "read of slot {to} past filled {f}");
    }

    /// Slots `[from, to)` as one run.
    #[inline]
    pub fn rows(&self, from: usize, to: usize) -> Run<'_> {
        self.check(to);
        // SAFETY: to <= filled (Acquire): those slots are written and frozen.
        unsafe {
            Run {
                ids: view(&self.ids, from, to),
                ts: view(&self.ts, from, to),
                tags: view(&self.tags, from, to),
                codes: view(&self.codes, from * self.m, to * self.m),
            }
        }
    }

    #[inline]
    pub fn vec_row(&self, i: usize) -> &[f32] {
        assert!(self.vdim > 0, "raw vectors not stored");
        self.check(i + 1);
        // SAFETY: as in `rows`.
        unsafe { view(&self.vecs, i * self.vdim, (i + 1) * self.vdim) }
    }

    #[inline]
    pub fn code_row(&self, i: usize) -> &[u8] {
        self.check(i + 1);
        // SAFETY: as in `rows`.
        unsafe { view(&self.codes, i * self.m, (i + 1) * self.m) }
    }

    #[inline] pub fn id(&self, i: usize) -> u64 { self.rows(i, i + 1).ids[0] }
    #[inline] pub fn ts(&self, i: usize) -> u64 { self.rows(i, i + 1).ts[0] }
    #[inline] pub fn tags(&self, i: usize) -> u64 { self.rows(i, i + 1).tags[0] }

    /// Empty the block for reuse. `&mut` proves no other reference exists.
    pub fn reset(&mut self) {
        *self.claimed.get_mut() = 0;
        *self.filled.get_mut() = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn rows_read_back_and_reset_empties() {
        let mut b = Block::new(4, 2, 3);
        assert_eq!(b.push(7, 1, 0b10, &[1, 2], &[0.5, 1.5, 2.5]), 0);
        assert_eq!(b.push(8, 2, 0b01, &[3, 4], &[3.5, 4.5, 5.5]), 1);
        let r = b.rows(0, 2);
        assert_eq!((r.ids, r.ts, r.tags, r.codes), (&[7u64, 8][..], &[1u64, 2][..], &[2u64, 1][..], &[1u8, 2, 3, 4][..]));
        assert_eq!(b.vec_row(1), &[3.5, 4.5, 5.5]);
        assert_eq!(b.code_row(0), &[1, 2]);
        b.reset();
        assert_eq!(b.filled(), 0);
        assert_eq!(b.push(9, 3, 0, &[5, 6], &[0.0; 3]), 0);
        assert_eq!(b.id(0), 9);
    }

    #[test]
    #[should_panic(expected = "past filled")]
    fn reading_past_filled_panics() { Block::new(4, 1, 0).rows(0, 1); }

    #[test]
    #[should_panic(expected = "block full")]
    fn pushing_past_cap_panics() {
        let b = Block::new(1, 1, 0);
        b.push(1, 0, 0, &[0], &[]);
        b.push(2, 0, 0, &[0], &[]);
    }

    /// One thread appends while others read the published prefix. Small
    /// enough to run under Miri: `cargo +nightly miri test --lib arena`.
    #[test]
    fn concurrent_push_and_read() {
        let n = if cfg!(miri) { 16 } else { 4096 };
        let b = Arc::new(Block::new(n, 2, 1));
        let readers: Vec<_> = (0..2).map(|_| {
            let b = b.clone();
            std::thread::spawn(move || {
                let mut seen = 0;
                while seen < n {
                    let f = b.filled();
                    let r = b.rows(0, f);
                    for i in 0..f {
                        assert_eq!(r.ids[i], i as u64);
                        assert_eq!(r.codes[2 * i..2 * i + 2], [i as u8, !(i as u8)]);
                        assert_eq!(b.vec_row(i)[0], i as f32);
                    }
                    seen = f;
                    std::thread::yield_now();
                }
            })
        }).collect();
        for i in 0..n { b.push(i as u64, i as u64, 0, &[i as u8, !(i as u8)], &[i as f32]); }
        for r in readers { r.join().unwrap(); }
    }
}
