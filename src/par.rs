// src/par.rs
use std::sync::{Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

/// Parallel map over a slice, keeping output order deterministic.
/// Spawns `threads` workers; with `threads <= 1` it runs serially.
///
/// I: Sync so we can share &I across threads
/// T: Send because results cross thread boundaries into shared Vec
/// F: Sync so all workers can borrow the same callable
pub fn parallel_map_indexed<I, T, F>(items: &[I], threads: usize, f: F) -> Vec<T>
where
    I: Sync,
    T: Send,
    F: Fn(&I, usize) -> T + Sync,
{
    let n = items.len();
    if n == 0 || threads <= 1 {
        return (0..n).map(|i| f(&items[i], i)).collect();
    }

    // Pre-allocate result slots without requiring Option<T>: Clone
    let out = Mutex::new({
        let mut v: Vec<Option<T>> = Vec::with_capacity(n);
        v.resize_with(n, || None);
        v
    });

    // Lock-free work distribution
    let next = AtomicUsize::new(0);

    // Scoped threads let us borrow `items` and `f` (no 'static required)
    thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let f_ref = &f;
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n { break; }
                    let res = f_ref(&items[i], i);
                    let mut guard = out.lock().expect("par out lock");
                    guard[i] = Some(res);
                }
            });
        }
    });

    // Extract Vec<T> in order
    let vec_opt = out.into_inner().expect("par out poisoned");
    vec_opt.into_iter().map(|x| x.expect("missing result")).collect()
}
