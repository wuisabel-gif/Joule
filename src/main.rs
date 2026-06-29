//! Joule — energy-aware optimization middleware for LLM inference.
//!
//! Phase 1: an OpenAI-compatible measuring proxy with token accounting, energy
//! estimation, Prometheus metrics, a SQLite request log, and a small CLI.
//! Requests are dispatched through pluggable provider and router components.

mod cache;
mod cli;
mod config;
mod error;
mod estimator;
mod metrics;
mod optimizer;
mod provider;
mod proxy;
mod router;
mod store;
mod tokens;

use std::io::Read;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use serde_json::json;
use tracing::info;
use tracing_subscriber::EnvFilter;

use cli::{Cli, Command, EstimateArgs, OptimizeArgs, ReportArgs, ServeArgs};
use config::Config;
use estimator::{models, Estimator};
use metrics::Metrics;
use optimizer::Optimizer;
use proxy::AppState;
use store::Store;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    match Cli::parse().command {
        Command::Serve(args) => serve(args).await,
        Command::Estimate(args) => {
            estimate(args);
            Ok(())
        }
        Command::Optimize(args) => optimize(args),
        Command::Report(args) => report(args),
        Command::Models => {
            list_models();
            Ok(())
        }
    }
}

async fn serve(args: ServeArgs) -> Result<()> {
    let config = match &args.config {
        Some(path) => Config::from_file(path)?,
        None => Config::single(
            args.upstream,
            args.api_key,
            args.provider_kind,
            args.router,
            args.optimize,
            !args.no_cache,
            args.cache_capacity,
            args.grid_intensity,
        ),
    };

    let estimator = config.estimator();
    let registry = config.build_registry().context("building providers")?;
    let router_plugin = config.build_router(estimator);
    let optimizer = config.optimizer();
    let cache = config.build_cache();

    let store = Store::open(&args.db).with_context(|| format!("opening database {}", args.db))?;

    let provider_names: Vec<&str> = registry.iter().map(|p| p.name()).collect();
    info!(
        listen = %args.listen,
        providers = ?provider_names,
        default = registry.default_name(),
        router = router_plugin.name(),
        optimize = optimizer.level().as_str(),
        cache = cache.enabled(),
        "joule proxy starting",
    );
    info!("metrics at /metrics, request log at /stats, health at /healthz");

    let state = AppState {
        estimator,
        metrics: Arc::new(Metrics::new()),
        store: Arc::new(store),
        client: reqwest::Client::new(),
        registry: Arc::new(registry),
        router: Arc::from(router_plugin),
        optimizer: Arc::new(optimizer),
        cache: Arc::new(cache),
    };

    let app = proxy::router(state);
    let listener = tokio::net::TcpListener::bind(&args.listen)
        .await
        .with_context(|| format!("binding {}", args.listen))?;

    axum::serve(listener, app).await.context("server error")?;
    Ok(())
}

fn estimate(args: EstimateArgs) {
    let estimator = Estimator::new(args.grid_intensity);
    let e = estimator.estimate(&args.model, args.input, args.output);
    let profile = models::profile_for(&args.model);

    println!(
        "Model:           {} (profile: {})",
        args.model, profile.family
    );
    println!("Input tokens:    {}", args.input);
    println!("Output tokens:   {}", args.output);
    println!("Energy:          {:.3} J", e.energy_j);
    println!("Electricity:     {:.6} Wh", e.electricity_wh);
    println!("CO2:             {:.6} g", e.co2_g);
    println!("Cost:            ${:.6}", e.cost_usd);
    println!("Grid intensity:  {:.0} g/kWh", args.grid_intensity);
}

fn optimize(args: OptimizeArgs) -> Result<()> {
    // Read the prompt from --text or stdin.
    let text = match args.text {
        Some(t) => t,
        None => {
            let mut buf = String::new();
            std::io::stdin()
                .read_to_string(&mut buf)
                .context("reading prompt from stdin")?;
            buf
        }
    };

    let mut request = json!({
        "model": args.model,
        "messages": [{ "role": "user", "content": text }],
    });

    let optimizer = Optimizer::new(args.level);
    let report = optimizer.optimize(&mut request);

    let estimator = Estimator::new(args.grid_intensity);
    let before = estimator.estimate(&args.model, report.tokens_before, 0);
    let after = estimator.estimate(&args.model, report.tokens_after, 0);
    let energy_saved = before.energy_j - after.energy_j;

    println!("{}", report.summary());
    println!();
    let verb = if energy_saved >= 0.0 {
        "saved"
    } else {
        "added"
    };
    println!(
        "Prompt energy (input side) for {}: {:.3} J \u{2192} {:.3} J ({} {:.3} J)",
        args.model,
        before.energy_j,
        after.energy_j,
        verb,
        energy_saved.abs()
    );
    if report.changed() {
        println!("\n--- optimized prompt ---");
        if let Some(content) = request
            .pointer("/messages/0/content")
            .and_then(|v| v.as_str())
        {
            println!("{content}");
        }
    }
    Ok(())
}

fn report(args: ReportArgs) -> Result<()> {
    let store = Store::open(&args.db).with_context(|| format!("opening database {}", args.db))?;
    let t = store.totals()?;
    let hits = store.cache_hits()?;
    let models = store.model_breakdown()?;

    // Savings are tracked in joules; derive Wh/kWh exactly and CO2 from the grid.
    let saved_wh = t.energy_saved_j / 3600.0;
    let saved_kwh = saved_wh / 1000.0;
    let saved_co2_g = saved_kwh * args.grid_intensity;
    let spent_wh = t.energy_j / 3600.0;

    println!("Joule report — {}", args.db);
    println!("{:-<52}", "");
    println!("Requests:        {}", t.requests);
    if t.requests > 0 {
        println!(
            "  cache hits:    {} ({:.0}%)",
            hits,
            hits as f64 / t.requests as f64 * 100.0
        );
    }
    println!(
        "Tokens:          {} in · {} out",
        t.input_tokens, t.output_tokens
    );
    println!();
    println!("Energy spent:    {:.1} J  ({:.4} Wh)", t.energy_j, spent_wh);
    println!("CO2 emitted:     {:.3} g", t.co2_g);
    println!("Cost:            ${:.4}", t.cost_usd);
    println!();
    println!("Tokens saved:    {}", t.tokens_saved);
    println!(
        "Energy saved:    {:.1} J  ({:.4} Wh / {:.6} kWh)",
        t.energy_saved_j, saved_wh, saved_kwh
    );
    println!(
        "CO2 avoided:     {:.3} g  (≈ at {:.0} g/kWh)",
        saved_co2_g, args.grid_intensity
    );

    if !models.is_empty() {
        println!();
        println!("Top models by energy:");
        println!(
            "  {:<24} {:>8} {:>12} {:>10} {:>10}",
            "MODEL", "REQS", "ENERGY(J)", "CO2(g)", "COST($)"
        );
        for m in models.iter().take(10) {
            println!(
                "  {:<24} {:>8} {:>12.1} {:>10.3} {:>10.4}",
                m.model, m.requests, m.energy_j, m.co2_g, m.cost_usd
            );
        }
    }
    Ok(())
}

fn list_models() {
    println!(
        "{:<22} {:>10} {:>10} {:>12} {:>12}",
        "MODEL", "J/in-tok", "J/out-tok", "$/M in", "$/M out"
    );
    for p in models::all() {
        println!(
            "{:<22} {:>10.3} {:>10.3} {:>12.3} {:>12.3}",
            p.family,
            p.j_per_input_token,
            p.j_per_output_token,
            p.usd_per_m_input,
            p.usd_per_m_output
        );
    }
}
