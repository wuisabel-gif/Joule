//! The HTTP proxy: an OpenAI-compatible front door that measures every request
//! and dispatches it through the configured provider/router plugins.

mod openai;

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::header::{ACCEPT, CONTENT_TYPE};
use axum::http::{HeaderMap, HeaderName, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router as AxumRouter};
use futures::StreamExt;
use serde_json::{json, Value};
use tracing::warn;

use crate::error::AppError;
use crate::estimator::{EnergyEstimate, Estimator};
use crate::metrics::Metrics;
use crate::optimizer::Optimizer;
use crate::provider::ProviderRegistry;
use crate::router::Router;
use crate::store::{RequestRecord, Store};
use crate::tokens::{
    approx_tokens, estimate_completion_tokens, estimate_prompt_tokens, TokenSource,
};

use openai::SseAccumulator;

/// Shared, cheaply-cloneable application state.
#[derive(Clone)]
pub struct AppState {
    pub estimator: Estimator,
    pub metrics: Arc<Metrics>,
    pub store: Arc<Store>,
    pub client: reqwest::Client,
    pub registry: Arc<ProviderRegistry>,
    pub router: Arc<dyn Router>,
    pub optimizer: Arc<Optimizer>,
}

/// Prompt-optimization outcome carried alongside a request.
#[derive(Clone, Default)]
struct Optimization {
    optimized: bool,
    tokens_saved: u64,
    energy_saved_j: f64,
    passes: String,
}

impl AppState {
    /// Estimate, export metrics, and persist a completed request. Returns the
    /// estimate so callers can also surface it in response headers.
    #[allow(clippy::too_many_arguments)]
    fn finalize(
        &self,
        model: &str,
        status: u16,
        input_tokens: u64,
        output_tokens: u64,
        latency: Duration,
        streamed: bool,
        source: TokenSource,
        opt: &Optimization,
    ) -> EnergyEstimate {
        let estimate = self.estimator.estimate(model, input_tokens, output_tokens);

        self.metrics.observe(
            model,
            status,
            input_tokens,
            output_tokens,
            estimate.energy_j,
            estimate.co2_g,
            estimate.cost_usd,
            latency.as_secs_f64(),
        );
        self.metrics
            .observe_savings(model, opt.tokens_saved, opt.energy_saved_j);

        let record = RequestRecord {
            ts: RequestRecord::now(),
            model: model.to_string(),
            input_tokens,
            output_tokens,
            latency_ms: latency.as_millis() as u64,
            energy_j: estimate.energy_j,
            electricity_wh: estimate.electricity_wh,
            co2_g: estimate.co2_g,
            cost_usd: estimate.cost_usd,
            status,
            streamed,
            token_source: source.as_str().to_string(),
            optimized: opt.optimized,
            tokens_saved: opt.tokens_saved,
            energy_saved_j: opt.energy_saved_j,
            optimizations: opt.passes.clone(),
        };
        if let Err(e) = self.store.record(&record) {
            warn!("failed to persist request record: {e}");
        }

        estimate
    }
}

/// Build the Axum router with all routes wired to shared state.
pub fn router(state: AppState) -> AxumRouter {
    AxumRouter::new()
        .route("/healthz", get(health))
        .route("/metrics", get(metrics_handler))
        .route("/stats", get(stats_handler))
        .route("/v1/chat/completions", post(chat_completions))
        .fallback(passthrough)
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}

async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    (
        [(CONTENT_TYPE, "text/plain; version=0.0.4")],
        state.metrics.render(),
    )
}

async fn stats_handler(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let totals = state.store.totals()?;
    let recent = state.store.recent(20)?;
    Ok(Json(json!({ "totals": totals, "recent": recent })))
}

/// Metered handler for `POST /v1/chat/completions`.
async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    let mut request: Value = serde_json::from_slice(&body)
        .map_err(|e| AppError::new(StatusCode::BAD_REQUEST, format!("invalid JSON body: {e}")))?;

    let requested_model = request
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let is_stream = request
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    // Optimize the prompt before anything else — the cheapest token is the one
    // we never send. Token estimates below are taken from the optimized request.
    let report = state.optimizer.optimize(&mut request);
    if report.changed() {
        tracing::info!(saved = report.tokens_saved(), passes = %report.pass_names(), "optimized prompt");
    }
    let prompt_tokens_est = estimate_prompt_tokens(&request);

    // Route, then capture everything we need as owned values so the borrow of
    // the registry ends before we move state into a streaming response.
    let decision = state
        .router
        .route(&state.registry, &requested_model)
        .map_err(|e| AppError::new(StatusCode::BAD_GATEWAY, format!("routing failed: {e}")))?;
    let provider_name = decision.provider.name().to_string();
    let model = decision.model.clone();
    let route_reason = decision.reason.clone();

    // Optimization savings are input-side: the prompt tokens we removed times
    // the chosen model's per-input-token energy.
    let optimization = Optimization {
        optimized: report.changed(),
        tokens_saved: report.tokens_saved(),
        energy_saved_j: state
            .estimator
            .estimate(&model, report.tokens_saved(), 0)
            .energy_j,
        passes: report.pass_names(),
    };

    let request_builder = decision
        .provider
        .build_chat_request(&state.client, &request, &model, &headers)
        .map_err(|e| AppError::new(StatusCode::BAD_REQUEST, e.to_string()))?;
    drop(decision);

    let started = Instant::now();
    let response = request_builder
        .send()
        .await
        .map_err(|e| AppError::upstream(format!("upstream request failed: {e}")))?;

    let status = response.status().as_u16();
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();

    let ctx = RequestCtx {
        provider_name,
        model,
        route_reason,
        prompt_tokens_est,
        started,
        status,
        content_type,
        optimization,
    };

    if is_stream {
        stream_response(state, ctx, response)
    } else {
        buffered_response(state, ctx, response).await
    }
}

/// Per-request context threaded into the two response paths.
struct RequestCtx {
    provider_name: String,
    model: String,
    route_reason: String,
    prompt_tokens_est: u64,
    started: Instant,
    status: u16,
    content_type: String,
    optimization: Optimization,
}

/// Handle a non-streaming upstream response: read it fully, translate it to
/// OpenAI shape, account for tokens, attach Joule headers, and return it.
async fn buffered_response(
    state: AppState,
    ctx: RequestCtx,
    response: reqwest::Response,
) -> Result<Response, AppError> {
    let bytes = response
        .bytes()
        .await
        .map_err(|e| AppError::upstream(format!("reading upstream body failed: {e}")))?;
    let latency = ctx.started.elapsed();

    let provider = state
        .registry
        .get(&ctx.provider_name)
        .ok_or_else(|| AppError::new(StatusCode::INTERNAL_SERVER_ERROR, "provider vanished"))?;

    let upstream_json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    let provider_usage = provider.usage_from_body(&upstream_json);
    let translated = provider.translate_response(upstream_json);

    let (input_tokens, output_tokens, source) = match provider_usage {
        Some((p, c)) => (p, c, TokenSource::Provider),
        None => (
            ctx.prompt_tokens_est,
            estimate_completion_tokens(&translated),
            TokenSource::Estimated,
        ),
    };

    let estimate = state.finalize(
        &ctx.model,
        ctx.status,
        input_tokens,
        output_tokens,
        latency,
        false,
        source,
        &ctx.optimization,
    );

    // Re-serialise the (possibly translated) body so clients see OpenAI shape.
    let out_bytes = serde_json::to_vec(&translated)
        .map_err(|e| AppError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let builder = Response::builder()
        .status(StatusCode::from_u16(ctx.status).unwrap_or(StatusCode::BAD_GATEWAY))
        .header(CONTENT_TYPE, &ctx.content_type);
    let builder = with_joule_headers(builder, &estimate, source, false, &ctx);

    builder
        .body(Body::from(out_bytes))
        .map_err(|e| AppError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

/// Handle a streaming upstream response: tee bytes to the client while
/// accumulating, then account for tokens once the stream ends.
fn stream_response(
    state: AppState,
    ctx: RequestCtx,
    response: reqwest::Response,
) -> Result<Response, AppError> {
    let mut upstream = Box::pin(response.bytes_stream());

    // Routing headers are known up front; the energy headers are omitted for
    // streams because accounting only completes once the stream ends.
    let builder = Response::builder()
        .status(StatusCode::from_u16(ctx.status).unwrap_or(StatusCode::BAD_GATEWAY))
        .header(CONTENT_TYPE, &ctx.content_type)
        .header("x-joule-provider", &ctx.provider_name)
        .header("x-joule-route", &ctx.route_reason)
        .header("x-joule-optimized", ctx.optimization.optimized.to_string())
        .header(
            "x-joule-prompt-saved-tokens",
            ctx.optimization.tokens_saved.to_string(),
        )
        .header("x-joule-optimizations", &ctx.optimization.passes)
        .header("x-joule-streamed", "true");

    let body = async_stream::stream! {
        let provider = match state.registry.get(&ctx.provider_name) {
            Some(p) => p,
            None => return,
        };
        let mut acc = SseAccumulator::default();

        while let Some(item) = upstream.next().await {
            match item {
                Ok(chunk) => {
                    acc.feed(&chunk, provider);
                    yield Ok::<Bytes, std::io::Error>(chunk);
                }
                Err(e) => {
                    yield Err(std::io::Error::other(e.to_string()));
                    break;
                }
            }
        }

        let latency = ctx.started.elapsed();
        let (input_tokens, output_tokens, source) = match acc.usage() {
            Some((p, c)) => (p, c, TokenSource::Provider),
            None => (
                ctx.prompt_tokens_est,
                approx_tokens(acc.content()),
                TokenSource::Estimated,
            ),
        };
        state.finalize(&ctx.model, ctx.status, input_tokens, output_tokens, latency, true, source, &ctx.optimization);
    };

    builder
        .body(Body::from_stream(body))
        .map_err(|e| AppError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

/// Transparent forwarder for any route Joule does not specifically meter
/// (embeddings, model listings, …), sent to the default provider.
async fn passthrough(
    State(state): State<AppState>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    let provider = state.registry.default();
    let path_and_query = uri
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or(uri.path());
    let url = format!("{}{}", provider.base_url(), path_and_query);

    let mut rb = state.client.request(method, &url);
    if let Some(ct) = headers.get(CONTENT_TYPE) {
        rb = rb.header(CONTENT_TYPE, ct.clone());
    }
    if let Some(ac) = headers.get(ACCEPT) {
        rb = rb.header(ACCEPT, ac.clone());
    }
    rb = provider.authorize(rb, &headers).body(body.to_vec());

    let response = rb
        .send()
        .await
        .map_err(|e| AppError::upstream(format!("upstream request failed: {e}")))?;

    let status = response.status().as_u16();
    let resp_headers = response.headers().clone();
    let bytes = response
        .bytes()
        .await
        .map_err(|e| AppError::upstream(format!("reading upstream body failed: {e}")))?;

    let mut builder =
        Response::builder().status(StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY));
    for (name, value) in resp_headers.iter() {
        if is_hop_by_hop(name) {
            continue;
        }
        builder = builder.header(name, value);
    }
    builder
        .body(Body::from(bytes))
        .map_err(|e| AppError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

/// Add the `x-joule-*` measurement headers to an outgoing response.
fn with_joule_headers(
    builder: axum::http::response::Builder,
    estimate: &EnergyEstimate,
    source: TokenSource,
    streamed: bool,
    ctx: &RequestCtx,
) -> axum::http::response::Builder {
    builder
        .header("x-joule-provider", &ctx.provider_name)
        .header("x-joule-route", &ctx.route_reason)
        .header("x-joule-model", &ctx.model)
        .header("x-joule-energy-j", format!("{:.4}", estimate.energy_j))
        .header(
            "x-joule-electricity-wh",
            format!("{:.6}", estimate.electricity_wh),
        )
        .header("x-joule-co2-g", format!("{:.6}", estimate.co2_g))
        .header("x-joule-cost-usd", format!("{:.6}", estimate.cost_usd))
        .header("x-joule-token-source", source.as_str())
        .header("x-joule-optimized", ctx.optimization.optimized.to_string())
        .header(
            "x-joule-prompt-saved-tokens",
            ctx.optimization.tokens_saved.to_string(),
        )
        .header(
            "x-joule-energy-saved-j",
            format!("{:.4}", ctx.optimization.energy_saved_j),
        )
        .header("x-joule-optimizations", &ctx.optimization.passes)
        .header("x-joule-streamed", streamed.to_string())
}

/// Hop-by-hop headers that must not be copied to the downstream response.
fn is_hop_by_hop(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailers"
            | "transfer-encoding"
            | "upgrade"
            | "content-length"
    )
}
