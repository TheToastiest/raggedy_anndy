// //! Minimal web UI to run `freeze` and `sweep` from a browser.
// //!
// //! Add these to Cargo.toml (top-level):
// //!
// //! [dependencies]
// //! axum = { version = "0.7", features = ["macros", "tracing", "multipart"] }
// //! tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
// //! tower-http = { version = "0.5", features = ["cors", "trace"] }
// //! serde = { version = "1", features = ["derive"] }
// //! serde_json = "1"
// //! thiserror = "1"
// //!
// //! # already in your crate
// //! # raggedy_anndy = { path = ".." }
// //!
// //! Run:
// //!   cargo run --release --bin rn_ui
// //! Then open http://127.0.0.1:8080/
//
// use std::{sync::Arc, time::Instant};
//
// use axum::{
//     extract::{DefaultBodyLimit, Multipart, State},
//     http::StatusCode,
//     response::{Html, IntoResponse},
//     routing::{get, post},
//     Json, Router,
// };
// use serde::{Deserialize, Serialize};
// use tokio::sync::RwLock;
// use tower_http::{cors::CorsLayer, trace::TraceLayer};
// use tokio::net::TcpListener;
// use raggedy_anndy as ann;
// use ann::{FlatIndex, Metric};
// use ann::eval::wilson_lower_bound;
// use ann::ivf::{IvfIndex, IvfParams};
// use ann::ivfpq::{IvfPqIndex, IvfPqParams, OpqMode};
// use ann::par::parallel_map_indexed;
//
// // ------------------------------
// // App state
// // ------------------------------
//
// #[derive(Clone, Default)]
// struct AppState {
//     // Currently loaded dataset (id, vec)
//     data: Arc<RwLock<Option<Vec<(u64, Vec<f32>)>>>>,
// }
//
// // ------------------------------
// // Request/Response types
// // ------------------------------
//
// #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
// #[serde(rename_all = "lowercase")]
// enum MetricArg { Cosine, L2 }
// impl From<MetricArg> for Metric {
//     fn from(m: MetricArg) -> Self { match m { MetricArg::Cosine => Metric::Cosine, MetricArg::L2 => Metric::L2 } }
// }
//
// #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
// #[serde(rename_all = "kebab-case")]
// enum Backend { IvfFlat, IvfPq }
//
// #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
// #[serde(rename_all = "kebab-case")]
// enum OpqModeArg { Perm, Pca, PcaPerm }
// impl From<OpqModeArg> for OpqMode {
//     fn from(m: OpqModeArg) -> Self { match m { OpqModeArg::Perm => OpqMode::Perm, OpqModeArg::Pca => OpqMode::Pca, OpqModeArg::PcaPerm => OpqMode::PcaPerm } }
// }
//
// #[derive(Debug, Clone, Serialize, Deserialize)]
// struct FreezeReq {
//     // dataset
//     n: usize,
//     dim: usize,
//     metric: MetricArg,
//     k: usize,
//     queries: usize,
//     warmup: usize,
//
//     // seeds
//     seed_data: u64,
//     seed_queries: u64,
//     seed_kmeans: u64,
//
//     // backend
//     backend: Backend,
//
//     // IVF
//     nlist: usize,
//     nprobe: usize,
//     refine: usize,
//
//     // PQ
//     m: Option<usize>,
//     nbits: Option<u8>,
//     iters: Option<usize>,
//     opq: Option<bool>,
//     opq_mode: Option<OpqModeArg>,
//     opq_sweeps: Option<usize>,
//     store_vecs: Option<bool>,
//
//     // gates (optional; UI can check client-side)
//     min_lb95: Option<f32>,
//     min_recall: Option<f32>,
//     max_p95_ms: Option<f64>,
//
//     // query-level threads
//     threads: usize,
// }
//
// #[derive(Debug, Clone, Serialize, Deserialize)]
// struct SweepReq {
//     // dataset
//     n: usize,
//     dim: usize,
//     metric: MetricArg,
//     k: usize,
//     queries: usize,
//     warmup: usize,
//
//     // seeds
//     seed_data: u64,
//     seed_queries: u64,
//     seed_kmeans: u64,
//
//     // backend
//     backend: Backend,
//
//     // IVF
//     nlist: usize,
//     nprobe: String,  // comma list
//     refine: String,  // comma list
//
//     // PQ
//     m: Option<usize>,
//     nbits: Option<u8>,
//     iters: Option<usize>,
//     opq: Option<bool>,
//     opq_mode: Option<OpqModeArg>,
//     opq_sweeps: Option<usize>,
//     store_vecs: Option<bool>,
//
//     // query-level threads
//     threads: usize,
// }
//
// #[derive(Debug, Clone, Serialize)]
// struct FreezeResp {
//     recall: f32,
//     lb95: f32,
//     p50_ms: f64,
//     p90_ms: f64,
//     p95_ms: f64,
//     p99_ms: f64,
//     qps: f64,
//     build_ms: u128,
//     eval_ms: u128,
//     build_det: bool,
//     search_det: bool,
// }
//
// #[derive(Debug, Clone, Serialize)]
// struct SweepPoint { nprobe: usize, refine: usize, recall: f32, lb95: f32, p50_ms: f64, p90_ms: f64, p95_ms: f64, p99_ms: f64, qps: f64, build_det: bool, build_ms: u128, eval_ms: u128 }
//
// #[derive(Debug, Clone, Serialize)]
// struct SweepResp { rows: Vec<SweepPoint> }
//
// #[derive(Debug, thiserror::Error)]
// enum UiErr {
//     #[error("no dataset loaded; upload CSV or request synthetic")] NoData,
//     #[error("bad CSV: {0}")] Csv(String),
//     #[error("invalid parameters: {0}")] Params(String),
// }
// impl IntoResponse for UiErr {
//     fn into_response(self) -> axum::response::Response {
//         let msg = self.to_string();
//         (StatusCode::BAD_REQUEST, msg).into_response()
//     }
// }
//
// // ------------------------------
// // Utilities shared with freeze/sweep
// // ------------------------------
//
// fn random_unit_vec(rng: &mut rand::rngs::StdRng, dim: usize) -> Vec<f32> {
//     use rand::{Rng};
//     let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
//     let mut n = 0.0f32; for x in &v { n += *x * *x; } let n = n.sqrt();
//     if n > 0.0 { for x in v.iter_mut() { *x /= n; } }
//     v
// }
//
// fn percentile_us(mut v: Vec<u128>, p: f64) -> u128 {
//     if v.is_empty() { return 0; }
//     v.sort_unstable();
//     let idx = ((p * (v.len() as f64 - 1.0)).round() as usize).min(v.len() - 1);
//     v[idx]
// }
//
// #[derive(Clone, Copy)]
// struct PerQ { hits: usize, poss: usize, dt_us: u128, det_ok: bool }
//
// // ------------------------------
// // Routes
// // ------------------------------
//
// #[tokio::main]
// async fn main() {
//     let state = AppState::default();
//
//     let app = Router::new()
//         .route("/", get(ui_html))
//         .route("/api/health", get(|| async { "ok" }))
//         .route("/api/upload", post(upload_csv))
//         .route("/api/synth", post(make_synth))
//         .route("/api/freeze", post(run_freeze))
//         .route("/api/sweep", post(run_sweep))
//         .with_state(state)
//         .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
//         .layer(CorsLayer::permissive())
//         .layer(TraceLayer::new_for_http());
//
//     let addr = std::net::SocketAddr::from(([127,0,0,1], 8080));
//     println!("UI on http://{addr}");
//     let addr = "127.0.0.1:8080";
//     let listener = TcpListener::bind(addr).await.unwrap();
//     println!("listening on http://{addr}");
//     axum::serve(listener, app).await.unwrap();
// }
//
// async fn ui_html() -> Html<&'static str> {
//     Html(INDEX_HTML)
// }
//
// /// Upload CSV: lines like `id,f1,f2,...`
// async fn upload_csv(State(state): State<AppState>, mut mp: Multipart) -> Result<Json<serde_json::Value>, UiErr> {
//     let mut loaded: Option<Vec<(u64, Vec<f32>)>> = None;
//
//     while let Some(field) = mp.next_field().await.map_err(|e| UiErr::Csv(e.to_string()))? {
//         let name = field.name().unwrap_or("").to_string();
//         if name == "file" {
//             let bytes = field.bytes().await.map_err(|e| UiErr::Csv(e.to_string()))?;
//             let s = String::from_utf8(bytes.to_vec()).map_err(|e| UiErr::Csv(e.to_string()))?;
//             let mut rows: Vec<(u64, Vec<f32>)> = Vec::new();
//             for (lineno, line) in s.lines().enumerate() {
//                 if line.trim().is_empty() { continue; }
//                 let mut it = line.split(',');
//                 let id_str = it.next().ok_or_else(|| UiErr::Csv(format!("line {lineno}: missing id")))?;
//                 let id: u64 = id_str.parse().map_err(|_| UiErr::Csv(format!("line {lineno}: bad id")))?;
//                 let mut vec: Vec<f32> = Vec::new();
//                 for (j, tok) in it.enumerate() {
//                     let x: f32 = tok.parse().map_err(|_| UiErr::Csv(format!("line {lineno} col {j}: bad float")))?;
//                     vec.push(x);
//                 }
//                 rows.push((id, vec));
//             }
//             loaded = Some(rows);
//         }
//     }
//
//     if let Some(rows) = loaded {
//         let mut guard = state.data.write().await;
//         *guard = Some(rows);
//         Ok(Json(serde_json::json!({"ok": true})))
//     } else {
//         Err(UiErr::Csv("multipart field `file` missing".into()))
//     }
// }
//
// /// Generate a synthetic dataset and store it as the active dataset.
// #[derive(Deserialize)]
// struct SynthReq { n: usize, dim: usize, metric: MetricArg, seed_data: u64 }
// async fn make_synth(State(state): State<AppState>, Json(req): Json<SynthReq>) -> Result<Json<serde_json::Value>, UiErr> {
//     use rand::SeedableRng;
//     let metric: Metric = req.metric.into();
//     let mut rng = rand::rngs::StdRng::seed_from_u64(req.seed_data);
//     let mut data: Vec<(u64, Vec<f32>)> = Vec::with_capacity(req.n);
//     for i in 0..req.n {
//         let v = match metric {
//             Metric::Cosine => random_unit_vec(&mut rng, req.dim),
//             Metric::L2 => {
//                 use rand::Rng; (0..req.dim).map(|_| rng.gen::<f32>() - 0.5).collect()
//             }
//         };
//         data.push((i as u64, v));
//     }
//     let mut guard = state.data.write().await;
//     *guard = Some(data);
//     Ok(Json(serde_json::json!({"ok": true})))
// }
//
// // ------------------------------
// // Freeze
// // ------------------------------
//
// async fn run_freeze(State(state): State<AppState>, Json(req): Json<FreezeReq>) -> Result<Json<FreezeResp>, UiErr> {
//     let data_opt = state.data.read().await.clone();
//     let data = if let Some(d) = data_opt { d } else { return Err(UiErr::NoData) };
//
//     let metric: Metric = req.metric.into();
//
//     // Build exact baseline
//     let t_build0 = Instant::now();
//     let mut flat = FlatIndex::new(req.dim, metric);
//     for (id, v) in &data { flat.add(*id, v); }
//     let dataset_ms = t_build0.elapsed().as_millis();
//
//     // Queries
//     use rand::{SeedableRng, Rng};
//     let mut q_rng = rand::rngs::StdRng::seed_from_u64(req.seed_queries);
//     let queries: Vec<Vec<f32>> = (0..(req.warmup + req.queries)).map(|_| match metric {
//         Metric::Cosine => random_unit_vec(&mut q_rng, req.dim),
//         Metric::L2 => (0..req.dim).map(|_| q_rng.gen::<f32>() - 0.5).collect(),
//     }).collect();
//
//     // Build index based on backend
//     let build_t0 = Instant::now();
//     let (build_det, search_fn): (bool, Box<dyn Fn(&[f32]) -> Vec<ann::Hit> + Send + Sync>) = match req.backend {
//         Backend::IvfFlat => {
//             let params = IvfParams { nlist: req.nlist, nprobe: req.nprobe, refine: req.refine, seed: req.seed_kmeans };
//             let idx = IvfIndex::build(metric, req.dim, &data, params);
//             let fp1 = idx.fingerprint();
//             let det = fp1 == IvfIndex::build(metric, req.dim, &data, params).fingerprint();
//             let f = move |q: &[f32]| idx.search(q, req.k);
//             (det, Box::new(f))
//         }
//         Backend::IvfPq => {
//             let params = IvfPqParams {
//                 nlist: req.nlist,
//                 nprobe: req.nprobe,
//                 refine: req.refine,
//                 seed: req.seed_kmeans,
//                 m: req.m.unwrap_or(64),
//                 nbits: req.nbits.unwrap_or(8),
//                 iters: req.iters.unwrap_or(60),
//                 use_opq: req.opq.unwrap_or(true),
//                 opq_mode: req.opq_mode.unwrap_or(OpqModeArg::PcaPerm).into(),
//                 opq_sweeps: req.opq_sweeps.unwrap_or(8),
//                 store_vecs: req.store_vecs.unwrap_or(true),
//             };
//             let idx = IvfPqIndex::build(metric, req.dim, &data, params);
//             let fp1 = idx.fingerprint();
//             let det = fp1 == IvfPqIndex::build(metric, req.dim, &data, params).fingerprint();
//             let f = move |q: &[f32]| idx.search(q, req.k);
//             (det, Box::new(f))
//         }
//     };
//     let build_ms = build_t0.elapsed().as_millis() + dataset_ms;
//
//     // Warmup
//     for qi in 0..req.warmup { let _ = (search_fn)(&queries[qi]); }
//
//     // Evaluate
//     let eval_t0 = Instant::now();
//     let eval_slice = &queries[req.warmup..];
//
//     let perq: Vec<PerQ> = if req.threads <= 1 {
//         let mut out = Vec::with_capacity(eval_slice.len());
//         for q in eval_slice {
//             let t = Instant::now();
//             let ann1 = (search_fn)(q);
//             let dt = t.elapsed().as_micros() as u128;
//             let ann2 = (search_fn)(q);
//             let det_ok = ann1 == ann2;
//             let truth = flat.search(q, req.k);
//             let truth_ids: Vec<u64> = truth.iter().map(|h| h.id).collect();
//             let ann_ids: Vec<u64> = ann1.iter().map(|h| h.id).collect();
//             let mut hits = 0usize; let mut poss = 0usize;
//             for id in truth_ids.iter().take(req.k) { if ann_ids.contains(id) { hits += 1; } poss += 1; }
//             out.push(PerQ { hits, poss, dt_us: dt, det_ok });
//         }
//         out
//     } else {
//         parallel_map_indexed(eval_slice, req.threads, |q, _i| {
//             let t = Instant::now();
//             let ann1 = (search_fn)(q);
//             let dt = t.elapsed().as_micros() as u128;
//             let ann2 = (search_fn)(q);
//             let det_ok = ann1 == ann2;
//             let truth = flat.search(q, req.k);
//             let truth_ids: Vec<u64> = truth.iter().map(|h| h.id).collect();
//             let ann_ids: Vec<u64> = ann1.iter().map(|h| h.id).collect();
//             let mut hits = 0usize; let mut poss = 0usize;
//             for id in truth_ids.iter().take(req.k) { if ann_ids.contains(id) { hits += 1; } poss += 1; }
//             PerQ { hits, poss, dt_us: dt, det_ok }
//         })
//     };
//
//     let eval_ms = eval_t0.elapsed().as_millis();
//
//     let hits: usize = perq.iter().map(|r| r.hits).sum();
//     let poss: usize = perq.iter().map(|r| r.poss).sum();
//     let det = perq.iter().all(|r| r.det_ok);
//
//     let recall = if poss > 0 { (hits as f32) / (poss as f32) } else { 0.0 };
//     let lb = if poss > 0 { wilson_lower_bound(hits, poss, 1.96) as f32 } else { 0.0 };
//
//     let lat_us: Vec<u128> = perq.iter().map(|r| r.dt_us).collect();
//     let p50 = percentile_us(lat_us.clone(), 0.50) as f64 / 1000.0;
//     let p90 = percentile_us(lat_us.clone(), 0.90) as f64 / 1000.0;
//     let p95 = percentile_us(lat_us.clone(), 0.95) as f64 / 1000.0;
//     let p99 = percentile_us(lat_us, 0.99) as f64 / 1000.0;
//
//     let sum_us: u128 = perq.iter().map(|r| r.dt_us).sum();
//     let qps = if sum_us > 0 { (perq.len() as f64) / (sum_us as f64 / 1_000_000.0) } else { 0.0 };
//
//     Ok(Json(FreezeResp {
//         recall, lb95: lb, p50_ms: p50, p90_ms: p90, p95_ms: p95, p99_ms: p99,
//         qps, build_ms, eval_ms, build_det: build_det, search_det: det,
//     }))
// }
//
// // ------------------------------
// // Sweep
// // ------------------------------
//
// async fn run_sweep(State(state): State<AppState>, Json(req): Json<SweepReq>) -> Result<Json<SweepResp>, UiErr> {
//     let data_opt = state.data.read().await.clone();
//     let data = if let Some(d) = data_opt { d } else { return Err(UiErr::NoData) };
//     let metric: Metric = req.metric.into();
//
//     // Build exact baseline once
//     let mut flat = FlatIndex::new(req.dim, metric);
//     for (id, v) in &data { flat.add(*id, v); }
//
//     // Queries
//     use rand::{SeedableRng, Rng};
//     let mut q_rng = rand::rngs::StdRng::seed_from_u64(req.seed_queries);
//     let queries: Vec<Vec<f32>> = (0..(req.warmup + req.queries)).map(|_| match metric {
//         Metric::Cosine => random_unit_vec(&mut q_rng, req.dim),
//         Metric::L2 => (0..req.dim).map(|_| q_rng.gen::<f32>() - 0.5).collect(),
//     }).collect();
//
//     let parse_list_usize = |s: &str| -> Result<Vec<usize>, UiErr> {
//         let mut out = Vec::new();
//         for tok in s.split(',') {
//             let t = tok.trim(); if t.is_empty() { continue; }
//             out.push(t.parse::<usize>().map_err(|_| UiErr::Params(format!("bad list: {s}")))?);
//         }
//         if out.is_empty() { return Err(UiErr::Params(format!("empty list: {s}"))); }
//         Ok(out)
//     };
//
//     let nprobe_list = parse_list_usize(&req.nprobe)?;
//     let refine_list = parse_list_usize(&req.refine)?;
//
//     let mut rows: Vec<SweepPoint> = Vec::new();
//
//     for &nprobe in &nprobe_list {
//         for &refine in &refine_list {
//             // Build index per grid point
//             let build_t0 = Instant::now();
//             let (build_det, search_fn): (bool, Box<dyn Fn(&[f32]) -> Vec<ann::Hit> + Send + Sync>) = match req.backend {
//                 Backend::IvfFlat => {
//                     let params = IvfParams { nlist: req.nlist, nprobe, refine, seed: req.seed_kmeans };
//                     let idx = IvfIndex::build(metric, req.dim, &data, params);
//                     let fp1 = idx.fingerprint();
//                     let det = fp1 == IvfIndex::build(metric, req.dim, &data, params).fingerprint();
//                     let f = move |q: &[f32]| idx.search(q, req.k);
//                     (det, Box::new(f))
//                 }
//                 Backend::IvfPq => {
//                     let params = IvfPqParams {
//                         nlist: req.nlist,
//                         nprobe,
//                         refine,
//                         seed: req.seed_kmeans,
//                         m: req.m.unwrap_or(64),
//                         nbits: req.nbits.unwrap_or(8),
//                         iters: req.iters.unwrap_or(60),
//                         use_opq: req.opq.unwrap_or(true),
//                         opq_mode: req.opq_mode.unwrap_or(OpqModeArg::PcaPerm).into(),
//                         opq_sweeps: req.opq_sweeps.unwrap_or(8),
//                         store_vecs: req.store_vecs.unwrap_or(true),
//                     };
//                     let idx = IvfPqIndex::build(metric, req.dim, &data, params);
//                     let fp1 = idx.fingerprint();
//                     let det = fp1 == IvfPqIndex::build(metric, req.dim, &data, params).fingerprint();
//                     let f = move |q: &[f32]| idx.search(q, req.k);
//                     (det, Box::new(f))
//                 }
//             };
//             let build_ms = build_t0.elapsed().as_millis();
//
//             // Warmup
//             for qi in 0..req.warmup { let _ = (search_fn)(&queries[qi]); }
//
//             // Evaluate
//             let eval_slice = &queries[req.warmup..];
//             let perq: Vec<PerQ> = if req.threads <= 1 {
//                 let mut out = Vec::with_capacity(eval_slice.len());
//                 for q in eval_slice {
//                     let t = Instant::now();
//                     let ann1 = (search_fn)(q);
//                     let dt = t.elapsed().as_micros() as u128;
//                     let ann2 = (search_fn)(q);
//                     let det_ok = ann1 == ann2;
//                     let truth = flat.search(q, req.k);
//                     let truth_ids: Vec<u64> = truth.iter().map(|h| h.id).collect();
//                     let ann_ids: Vec<u64> = ann1.iter().map(|h| h.id).collect();
//                     let mut hits = 0usize; let mut poss = 0usize;
//                     for id in truth_ids.iter().take(req.k) { if ann_ids.contains(id) { hits += 1; } poss += 1; }
//                     out.push(PerQ { hits, poss, dt_us: dt, det_ok });
//                 }
//                 out
//             } else {
//                 parallel_map_indexed(eval_slice, req.threads, |q, _i| {
//                     let t = Instant::now();
//                     let ann1 = (search_fn)(q);
//                     let dt = t.elapsed().as_micros() as u128;
//                     let ann2 = (search_fn)(q);
//                     let det_ok = ann1 == ann2;
//                     let truth = flat.search(q, req.k);
//                     let truth_ids: Vec<u64> = truth.iter().map(|h| h.id).collect();
//                     let ann_ids: Vec<u64> = ann1.iter().map(|h| h.id).collect();
//                     let mut hits = 0usize; let mut poss = 0usize;
//                     for id in truth_ids.iter().take(req.k) { if ann_ids.contains(id) { hits += 1; } poss += 1; }
//                     PerQ { hits, poss, dt_us: dt, det_ok }
//                 })
//             };
//
//             let hits: usize = perq.iter().map(|r| r.hits).sum();
//             let poss: usize = perq.iter().map(|r| r.poss).sum();
//             let recall = if poss > 0 { (hits as f32) / (poss as f32) } else { 0.0 };
//             let lb = if poss > 0 { wilson_lower_bound(hits, poss, 1.96) as f32 } else { 0.0 };
//
//             let lat_us: Vec<u128> = perq.iter().map(|r| r.dt_us).collect();
//             let p50 = percentile_us(lat_us.clone(), 0.50) as f64 / 1000.0;
//             let p90 = percentile_us(lat_us.clone(), 0.90) as f64 / 1000.0;
//             let p95 = percentile_us(lat_us.clone(), 0.95) as f64 / 1000.0;
//             let p99 = percentile_us(lat_us, 0.99) as f64 / 1000.0;
//
//             let sum_us: u128 = perq.iter().map(|r| r.dt_us).sum();
//             let qps = if sum_us > 0 { (perq.len() as f64) / (sum_us as f64 / 1_000_000.0) } else { 0.0 };
//
//             let eval_ms: u128 = 0; // elapsed is dominated by per-query sum already; keep simple.
//
//             rows.push(SweepPoint { nprobe, refine, recall, lb95: lb, p50_ms: p50, p90_ms: p90, p95_ms: p95, p99_ms: p99, qps, build_det, build_ms, eval_ms });
//         }
//     }
//
//     Ok(Json(SweepResp { rows }))
// }
//
// // ------------------------------
// // UI (very small HTML; no build step)
// // ------------------------------
//
// const INDEX_HTML: &str = r#"<!doctype html>
// <html lang=\"en\">
// <head>
// <meta charset=\"utf-8\" />
// <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
// <title>raggedy_anndy — UI</title>
// <style>
//   body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; }
//   h1 { margin-bottom: .25rem }
//   .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 1rem; margin: 1rem 0; }
//   .row { display: flex; gap: 1rem; flex-wrap: wrap; }
//   label { display: block; font-size: .85rem; color: #374151; }
//   input, select { padding: .4rem .5rem; border: 1px solid #d1d5db; border-radius: 8px; min-width: 6rem; }
//   button { padding: .5rem .8rem; border: 1px solid #0ea5e9; background: #0284c7; color: white; border-radius: 10px; cursor: pointer; }
//   pre { background: #0b1020; color: #cbe1ff; padding: 1rem; border-radius: 12px; max-height: 30rem; overflow: auto; }
//   small { color: #6b7280 }
// </style>
// </head>
// <body>
//   <h1>raggedy_anndy — test UI</h1>
//   <p><small>Upload a CSV (<code>id,f1,f2,...</code>) or generate synthetic data. Then run <b>Freeze</b> or <b>Sweep</b>.</small></p>
//
//   <div class=\"card\">
//     <h3>Dataset</h3>
//     <div class=\"row\">
//       <form id=\"uploadForm\">
//         <label>Upload CSV</label>
//         <input type=\"file\" name=\"file\" accept=\".csv,text/csv\" />
//         <button type=\"submit\">Upload</button>
//       </form>
//       <form id=\"synthForm\">
//         <label>Make synthetic</label>
//         <input name=\"n\" type=\"number\" value=\"50000\" />
//         <input name=\"dim\" type=\"number\" value=\"256\" />
//         <select name=\"metric\"><option>cosine</option><option>l2</option></select>
//         <input name=\"seed_data\" type=\"number\" value=\"42\" />
//         <button type=\"submit\">Generate</button>
//       </form>
//     </div>
//     <div id=\"dataStatus\"></div>
//   </div>
//
//   <div class=\"card\">
//     <h3>Freeze</h3>
//     <form id=\"freezeForm\" class=\"row\">
//       <div>
//         <label>Backend</label>
//         <select name=\"backend\"><option>ivf-pq</option><option>ivf-flat</option></select>
//       </div>
//       <div>
//         <label>nlist</label><input name=\"nlist\" type=\"number\" value=\"1536\" />
//       </div>
//       <div>
//         <label>nprobe</label><input name=\"nprobe\" type=\"number\" value=\"906\" />
//       </div>
//       <div>
//         <label>refine</label><input name=\"refine\" type=\"number\" value=\"2000\" />
//       </div>
//       <div>
//         <label>m</label><input name=\"m\" type=\"number\" value=\"128\" />
//       </div>
//       <div>
//         <label>iters</label><input name=\"iters\" type=\"number\" value=\"60\" />
//       </div>
//       <div>
//         <label>opq-mode</label>
//         <select name=\"opq_mode\"><option>pca-perm</option><option>perm</option><option>pca</option></select>
//       </div>
//       <div>
//         <label>opq-sweeps</label><input name=\"opq_sweeps\" type=\"number\" value=\"8\" />
//       </div>
//       <div>
//         <label>k</label><input name=\"k\" type=\"number\" value=\"5\" />
//       </div>
//       <div>
//         <label>queries</label><input name=\"queries\" type=\"number\" value=\"200\" />
//       </div>
//       <div>
//         <label>warmup</label><input name=\"warmup\" type=\"number\" value=\"5\" />
//       </div>
//       <div>
//         <label>threads</label><input name=\"threads\" type=\"number\" value=\"2\" />
//       </div>
//       <div>
//         <button type=\"submit\">Run Freeze</button>
//       </div>
//     </form>
//     <pre id=\"freezeOut\"></pre>
//   </div>
//
//   <div class=\"card\">
//     <h3>Sweep</h3>
//     <form id=\"sweepForm\" class=\"row\">
//       <div>
//         <label>Backend</label>
//         <select name=\"backend\"><option>ivf-pq</option><option>ivf-flat</option></select>
//       </div>
//       <div>
//         <label>nlist</label><input name=\"nlist\" type=\"number\" value=\"1536\" />
//       </div>
//       <div>
//         <label>nprobe (csv)</label><input name=\"nprobe\" type=\"text\" value=\"704,840,906\" />
//       </div>
//       <div>
//         <label>refine (csv)</label><input name=\"refine\" type=\"text\" value=\"1200,1600,2000\" />
//       </div>
//       <div>
//         <label>m</label><input name=\"m\" type=\"number\" value=\"128\" />
//       </div>
//       <div>
//         <label>iters</label><input name=\"iters\" type=\"number\" value=\"60\" />
//       </div>
//       <div>
//         <label>opq-mode</label>
//         <select name=\"opq_mode\"><option>pca-perm</option><option>perm</option><option>pca</option></select>
//       </div>
//       <div>
//         <label>opq-sweeps</label><input name=\"opq_sweeps\" type=\"number\" value=\"8\" />
//       </div>
//       <div>
//         <label>k</label><input name=\"k\" type=\"number\" value=\"10\" />
//       </div>
//       <div>
//         <label>queries</label><input name=\"queries\" type=\"number\" value=\"200\" />
//       </div>
//       <div>
//         <label>warmup</label><input name=\"warmup\" type=\"number\" value=\"25\" />
//       </div>
//       <div>
//         <label>threads</label><input name=\"threads\" type=\"number\" value=\"1\" />
//       </div>
//       <div>
//         <button type=\"submit\">Run Sweep</button>
//       </div>
//     </form>
//     <pre id=\"sweepOut\"></pre>
//   </div>
//
// <script>
// // helpers
// const asJSON = (form) => Object.fromEntries(new FormData(form).entries());
//
// // dataset upload
// const uploadForm = document.getElementById('uploadForm');
// uploadForm.addEventListener('submit', async (e) => {
//   e.preventDefault();
//   const fd = new FormData(uploadForm);
//   const res = await fetch('/api/upload', { method: 'POST', body: fd });
//   document.getElementById('dataStatus').textContent = res.ok ? 'Dataset: CSV loaded' : 'Upload failed';
// });
//
// // synthetic
// const synthForm = document.getElementById('synthForm');
// synthForm.addEventListener('submit', async (e) => {
//   e.preventDefault();
//   const body = asJSON(synthForm);
//   body.n = +body.n; body.dim = +body.dim; body.seed_data = +body.seed_data; body.metric = body.metric;
//   const res = await fetch('/api/synth', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
//   document.getElementById('dataStatus').textContent = res.ok ? 'Dataset: synthetic generated' : 'Synthetic failed';
// });
//
// // freeze
// const freezeForm = document.getElementById('freezeForm');
// freezeForm.addEventListener('submit', async (e) => {
//   e.preventDefault();
//   const B = asJSON(freezeForm);
//   const body = {
//     // dataset-ish defaults (match synth unless uploaded)
//     n: 50000, dim: 256, metric: 'cosine', k: +B.k, queries: +B.queries, warmup: +B.warmup,
//     seed_data: 42, seed_queries: 999, seed_kmeans: 7,
//     backend: B.backend === 'ivf-pq' ? 'ivf-pq' : 'ivf-flat',
//     nlist: +B.nlist, nprobe: +B.nprobe, refine: +B.refine,
//     m: +B.m, nbits: 8, iters: +B.iters,
//     opq: true, opq_mode: B.opq_mode, opq_sweeps: +B.opq_sweeps,
//     store_vecs: true, threads: +B.threads,
//   };
//   const res = await fetch('/api/freeze', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
//   const out = document.getElementById('freezeOut');
//   if (!res.ok) { out.textContent = 'Error: ' + await res.text(); return; }
//   const j = await res.json();
//   out.textContent = JSON.stringify(j, null, 2);
// });
//
// // sweep
// const sweepForm = document.getElementById('sweepForm');
// sweepForm.addEventListener('submit', async (e) => {
//   e.preventDefault();
//   const B = asJSON(sweepForm);
//   const body = {
//     n: 50000, dim: 256, metric: 'cosine', k: +B.k, queries: +B.queries, warmup: +B.warmup,
//     seed_data: 42, seed_queries: 999, seed_kmeans: 7,
//     backend: B.backend === 'ivf-pq' ? 'ivf-pq' : 'ivf-flat',
//     nlist: +B.nlist, nprobe: B.nprobe, refine: B.refine,
//     m: +B.m, nbits: 8, iters: +B.iters,
//     opq: true, opq_mode: B.opq_mode, opq_sweeps: +B.opq_sweeps, store_vecs: true,
//     threads: +B.threads,
//   };
//   const res = await fetch('/api/sweep', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
//   const out = document.getElementById('sweepOut');
//   if (!res.ok) { out.textContent = 'Error: ' + await res.text(); return; }
//   const j = await res.json();
//   const rows = j.rows || [];
//   // pretty table
//   const header = 'nprobe\trefine\trecall\tlb95\tp95(ms)\tQPS\n';
//   const lines = rows.map(r => `${r.nprobe}\t${r.refine}\t${r.recall.toFixed(3)}\t${r.lb95.toFixed(3)}\t${r.p95_ms.toFixed(3)}\t${r.qps.toFixed(1)}`).join('\n');
//   out.textContent = header + lines;
// });
// </script>
// </body>
// </html>"#;
