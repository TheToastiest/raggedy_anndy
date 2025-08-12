//! raggedy_anndy — deterministic ANN + RAG building blocks.
//!
//! Modules:
//! - `metric`: distance/similarity helpers.
//! - `types`: Metric, Hit, stable_top_k.
//! - `seed`: SplitMix64 for deterministic RNG.
//! - `flat`: FlatIndex (exact baseline + refine).
//! - `kmeans`: seeded k-means++ trainer.
//! - `ivf`: IvfIndex (IVF-Flat with refine).
//! - `eval`: recall@k + determinism check.

pub mod metric;
pub mod types;
pub mod seed;
pub mod flat;
pub mod kmeans;
pub mod ivf;
pub mod persist;
pub mod header;
pub mod embed;
pub mod eval;

pub use metric::Metric;
pub use types::{Hit, stable_top_k};
pub use flat::FlatIndex;
pub use ivf::{IvfIndex, IvfParams};
pub use header::{IndexHeader, Seeds};