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
//! - `pick`: seeded pick of 2 of the top 5 for URIEL.
//! - `live`: deterministic insert / delete / upsert / expire / compact for IVF-PQ.
//! - `views`: one writer thread publishing per-tick views to query threads (`arena`: the row blocks).

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
pub mod pq;
pub mod opq;
pub mod ivfpq;
pub mod par;
pub mod context;
pub mod time_index;
pub mod recall;
pub mod curve;
pub mod pick;
pub mod live;
pub mod arena;
pub mod views;
pub use ivfpq::{IvfPqIndex, IvfPqParams, RowLoc};
// src/lib.rs
pub use context::{
    ContextCfg,
    ContextEncoder,
    ContextKey,
    QueryCtx,
    TimeKey,
    CTX_DIM,
    TIME_DIM,
};
pub use time_index::{AnnIndex, TimedIndex};


pub use metric::Metric;
pub use types::{Hit, stable_top_k};
pub use flat::FlatIndex;
pub use ivf::{IvfIndex, IvfParams};
pub use header::{IndexHeader, Seeds};