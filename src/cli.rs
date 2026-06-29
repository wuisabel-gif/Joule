//! Command-line interface.

use clap::{Args, Parser, Subcommand};

use crate::config::{ProviderKind, RouterKind};
use crate::estimator::DEFAULT_GRID_INTENSITY_G_PER_KWH;
use crate::optimizer::OptLevel;

/// Energy-aware optimization middleware for LLM inference.
#[derive(Debug, Parser)]
#[command(name = "joule", version, about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run the measuring proxy server.
    Serve(ServeArgs),
    /// Estimate the energy footprint of a hypothetical request.
    Estimate(EstimateArgs),
    /// Optimize a prompt and show the energy it would save (a prompt improver).
    Optimize(OptimizeArgs),
    /// List the known model energy/price profiles.
    Models,
}

#[derive(Debug, Args)]
pub struct ServeArgs {
    /// Address to bind the proxy to.
    #[arg(long, env = "JOULE_LISTEN", default_value = "127.0.0.1:8080")]
    pub listen: String,

    /// Path to a JSON config file describing providers and routing. When set,
    /// the single-provider flags below are ignored.
    #[arg(long, env = "JOULE_CONFIG")]
    pub config: Option<String>,

    /// Upstream provider base URL (single-provider quickstart).
    #[arg(long, env = "JOULE_UPSTREAM", default_value = "https://api.openai.com")]
    pub upstream: String,

    /// Wire protocol the upstream speaks (single-provider quickstart).
    #[arg(long, value_enum, default_value_t = ProviderKind::Openai)]
    pub provider_kind: ProviderKind,

    /// Routing policy.
    #[arg(long, value_enum, default_value_t = RouterKind::Static)]
    pub router: RouterKind,

    /// Prompt-optimization intensity.
    #[arg(long, value_enum, default_value_t = OptLevel::Lite)]
    pub optimize: OptLevel,

    /// Disable the exact-match response cache (enabled by default).
    #[arg(long)]
    pub no_cache: bool,

    /// Maximum entries in the response cache.
    #[arg(long, env = "JOULE_CACHE_CAPACITY", default_value_t = 1024)]
    pub cache_capacity: usize,

    /// API key injected when the client request omits credentials.
    #[arg(long, env = "JOULE_UPSTREAM_API_KEY")]
    pub api_key: Option<String>,

    /// Path to the SQLite request log.
    #[arg(long, env = "JOULE_DB", default_value = "joule.db")]
    pub db: String,

    /// Grid carbon intensity in grams CO2 per kWh.
    #[arg(long, env = "JOULE_GRID_INTENSITY", default_value_t = DEFAULT_GRID_INTENSITY_G_PER_KWH)]
    pub grid_intensity: f64,
}

#[derive(Debug, Args)]
pub struct OptimizeArgs {
    /// Optimization intensity.
    #[arg(long, value_enum, default_value_t = OptLevel::Full)]
    pub level: OptLevel,

    /// Model to estimate the energy saving against.
    #[arg(long, default_value = "gpt-4o")]
    pub model: String,

    /// Prompt text to optimize. If omitted, reads from stdin.
    #[arg(long)]
    pub text: Option<String>,

    /// Grid carbon intensity in grams CO2 per kWh.
    #[arg(long, default_value_t = DEFAULT_GRID_INTENSITY_G_PER_KWH)]
    pub grid_intensity: f64,
}

#[derive(Debug, Args)]
pub struct EstimateArgs {
    /// Model name to estimate for.
    #[arg(long, default_value = "gpt-4o")]
    pub model: String,

    /// Number of input (prompt) tokens.
    #[arg(long, default_value_t = 1000)]
    pub input: u64,

    /// Number of output (completion) tokens.
    #[arg(long, default_value_t = 300)]
    pub output: u64,

    /// Grid carbon intensity in grams CO2 per kWh.
    #[arg(long, default_value_t = DEFAULT_GRID_INTENSITY_G_PER_KWH)]
    pub grid_intensity: f64,
}
