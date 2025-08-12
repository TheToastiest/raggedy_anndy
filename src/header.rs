use serde::{Serialize, Deserialize};
use crate::metric::Metric;
use crate::ivf::IvfParams;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Seeds { pub data: u64, pub queries: u64, pub kmeans: u64, pub hnsw: Option<u64> }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexHeader {
    pub version: u32,
    pub dim: usize,
    pub metric: Metric,
    pub params: IvfParams,
    pub seeds: Seeds,
    pub cpu_features: String,
    pub rustc: String,
    pub build_flags: String,
    pub has_tags: bool,
}

impl IndexHeader {
    pub fn new(dim: usize, metric: Metric, seeds: Seeds, params: IvfParams, has_tags: bool) -> Self {
        let cpu = format!("sse2:{} avx2:{} avx512f:{}",
                          cfg!(target_feature = "sse2"),
                          cfg!(target_feature = "avx2"),
                          cfg!(target_feature = "avx512f"));
        let rustc = env!("CARGO_PKG_VERSION").to_string();
        let build_flags = std::env::var("RUSTFLAGS").unwrap_or_default();
        Self { version: 2, dim, metric, params, seeds, cpu_features: cpu, rustc, build_flags, has_tags }
    }
}
