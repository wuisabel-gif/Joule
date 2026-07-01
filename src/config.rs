//! Runtime configuration: providers, routing policy, and the estimator.
//!
//! Two ways to configure Joule:
//! - a JSON config file (`--config`) for multi-provider / routed setups, or
//! - the single-provider quickstart flags (`--upstream`, `--api-key`, …).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::ValueEnum;
use serde::Deserialize;

use crate::cache::Cache;
use crate::carbon::{CarbonFeed, CarbonMap, CarbonSourceKind};
use crate::estimator::{Estimator, DEFAULT_GRID_INTENSITY_G_PER_KWH};
use crate::optimizer::{OptLevel, Optimizer};
use crate::provider::{
    AnthropicProvider, GeminiProvider, OpenAiCompatibleProvider, Provider, ProviderRegistry,
};
use crate::resilience::Breakers;
use crate::router::{
    CarbonRouter, ComplexityRouter, GreenestRouter, ModelRouter, Router, StaticRouter,
};
use crate::semantic::SemanticCache;

/// Which wire protocol a provider speaks.
#[derive(Debug, Clone, Copy, Default, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    /// OpenAI-compatible (OpenAI, Ollama, vLLM, LM Studio, llama.cpp, …).
    #[default]
    Openai,
    /// Anthropic Messages API.
    Anthropic,
    /// Google Gemini generateContent API.
    Gemini,
}

/// Which routing policy to use.
#[derive(Debug, Clone, Copy, Default, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum RouterKind {
    /// Always the default provider.
    #[default]
    Static,
    /// First provider that supports the requested model.
    Model,
    /// Lowest-energy candidate model.
    Greenest,
    /// Provider whose region has the lowest grid carbon intensity.
    Carbon,
    /// Small model for simple tasks, capable model otherwise.
    Complexity,
}

/// One configured provider.
#[derive(Debug, Clone, Deserialize)]
pub struct ProviderConfig {
    pub name: String,
    #[serde(default)]
    pub kind: ProviderKind,
    pub base_url: String,
    #[serde(default)]
    pub api_key: Option<String>,
    /// Model-name prefixes this provider serves; empty = wildcard.
    #[serde(default)]
    pub models: Vec<String>,
    /// Anthropic API version (Anthropic providers only).
    #[serde(default)]
    pub anthropic_version: Option<String>,
    /// Region key for carbon-aware routing (e.g. "us-west", "norway").
    #[serde(default)]
    pub region: Option<String>,
}

fn default_grid() -> f64 {
    DEFAULT_GRID_INTENSITY_G_PER_KWH
}

fn default_cache() -> bool {
    true
}

fn default_cache_capacity() -> usize {
    1024
}

fn default_embed_model() -> String {
    "text-embedding-3-small".to_string()
}

fn default_semantic_threshold() -> f64 {
    0.92
}

fn default_semantic_capacity() -> usize {
    512
}

fn default_timeout_secs() -> u64 {
    60
}

fn default_connect_timeout_secs() -> u64 {
    10
}

fn default_max_retries() -> u32 {
    2
}

fn default_circuit_threshold() -> u32 {
    5
}

fn default_circuit_cooldown_secs() -> u64 {
    30
}

fn default_complexity_max_simple_tokens() -> usize {
    240
}

fn default_carbon_poll_secs() -> u64 {
    300
}

/// Environment variable holding the live carbon-feed auth token (kept out of
/// config files so it never lands in version control).
const CARBON_TOKEN_ENV: &str = "JOULE_CARBON_TOKEN";

/// A ready-to-run carbon feed: the source, the `(region, zone)` pairs to poll,
/// and the polling interval.
pub type CarbonFeedPlan = (CarbonFeed, Vec<(String, String)>, Duration);

/// Full runtime configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub providers: Vec<ProviderConfig>,
    #[serde(default)]
    pub default_provider: Option<String>,
    #[serde(default)]
    pub router: RouterKind,
    /// Candidate models for the `greenest` router.
    #[serde(default)]
    pub greenest_candidates: Vec<String>,
    /// Per-region carbon-intensity overrides (g CO₂/kWh) for the `carbon` router.
    #[serde(default)]
    pub carbon_overrides: HashMap<String, f64>,
    /// Optional live carbon feed. When unset, the `carbon` router uses the
    /// static table only. When set, a background poller refreshes intensities.
    #[serde(default)]
    pub carbon_source: Option<CarbonSourceKind>,
    /// Override the feed's API base URL (defaults per source).
    #[serde(default)]
    pub carbon_source_url: Option<String>,
    /// Feed auth token. Prefer the `JOULE_CARBON_TOKEN` env var — a token in a
    /// config file risks being committed. The env var wins if both are set.
    #[serde(default)]
    pub carbon_source_token: Option<String>,
    /// Region key → source zone code (e.g. "norway" → "NO"). The UK source
    /// ignores zones (national) and defaults to refreshing the "uk" region.
    #[serde(default)]
    pub carbon_zones: HashMap<String, String>,
    /// How often to refresh the live carbon feed, seconds.
    #[serde(default = "default_carbon_poll_secs")]
    pub carbon_poll_secs: u64,
    /// Model for simple tasks under the `complexity` router.
    #[serde(default)]
    pub complexity_simple: Option<String>,
    /// Model for complex tasks under the `complexity` router.
    #[serde(default)]
    pub complexity_complex: Option<String>,
    /// Max prompt tokens for a request to be considered "simple".
    #[serde(default = "default_complexity_max_simple_tokens")]
    pub complexity_max_simple_tokens: usize,
    /// Prompt-optimization intensity.
    #[serde(default)]
    pub optimize: OptLevel,
    /// Exact-match response cache (on by default).
    #[serde(default = "default_cache")]
    pub cache: bool,
    /// Maximum entries in the response cache.
    #[serde(default = "default_cache_capacity")]
    pub cache_capacity: usize,
    /// Semantic (embedding-similarity) cache (off by default; needs embeddings).
    #[serde(default)]
    pub semantic_cache: bool,
    #[serde(default = "default_embed_model")]
    pub embed_model: String,
    /// Embeddings endpoint base URL (defaults to the default provider's).
    #[serde(default)]
    pub embed_base_url: Option<String>,
    /// Embeddings API key (defaults to the default provider's).
    #[serde(default)]
    pub embed_api_key: Option<String>,
    #[serde(default = "default_semantic_threshold")]
    pub semantic_threshold: f64,
    #[serde(default = "default_semantic_capacity")]
    pub semantic_capacity: usize,
    /// Per-request upstream timeout (non-streaming), seconds.
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
    /// Connection-establishment timeout, seconds.
    #[serde(default = "default_connect_timeout_secs")]
    pub connect_timeout_secs: u64,
    /// Retries after a transient upstream failure (timeout / 5xx / 429).
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// Consecutive failures that trip a provider's circuit breaker.
    #[serde(default = "default_circuit_threshold")]
    pub circuit_threshold: u32,
    /// How long a tripped breaker stays open before a trial request, seconds.
    #[serde(default = "default_circuit_cooldown_secs")]
    pub circuit_cooldown_secs: u64,
    #[serde(default = "default_grid")]
    pub grid_intensity: f64,
}

impl Config {
    /// Load configuration from a JSON file.
    pub fn from_file(path: &str) -> Result<Self> {
        let raw =
            std::fs::read_to_string(path).with_context(|| format!("reading config file {path}"))?;
        let config: Config =
            serde_json::from_str(&raw).with_context(|| format!("parsing config file {path}"))?;
        if config.providers.is_empty() {
            anyhow::bail!("config must define at least one provider");
        }
        Ok(config)
    }

    /// Build a single-provider configuration from the quickstart flags.
    #[allow(clippy::too_many_arguments)]
    pub fn single(
        upstream: String,
        api_key: Option<String>,
        kind: ProviderKind,
        router: RouterKind,
        optimize: OptLevel,
        cache: bool,
        cache_capacity: usize,
        semantic_cache: bool,
        embed_model: String,
        timeout_secs: u64,
        max_retries: u32,
        grid_intensity: f64,
    ) -> Self {
        Config {
            providers: vec![ProviderConfig {
                name: "default".to_string(),
                kind,
                base_url: upstream,
                api_key,
                models: Vec::new(),
                anthropic_version: None,
                region: None,
            }],
            default_provider: Some("default".to_string()),
            router,
            greenest_candidates: Vec::new(),
            carbon_overrides: HashMap::new(),
            carbon_source: None,
            carbon_source_url: None,
            carbon_source_token: None,
            carbon_zones: HashMap::new(),
            carbon_poll_secs: default_carbon_poll_secs(),
            complexity_simple: None,
            complexity_complex: None,
            complexity_max_simple_tokens: default_complexity_max_simple_tokens(),
            optimize,
            cache,
            cache_capacity,
            semantic_cache,
            embed_model,
            embed_base_url: None,
            embed_api_key: None,
            semantic_threshold: default_semantic_threshold(),
            semantic_capacity: default_semantic_capacity(),
            timeout_secs,
            connect_timeout_secs: default_connect_timeout_secs(),
            max_retries,
            circuit_threshold: default_circuit_threshold(),
            circuit_cooldown_secs: default_circuit_cooldown_secs(),
            grid_intensity,
        }
    }

    /// Per-request upstream timeout.
    pub fn timeout(&self) -> Duration {
        Duration::from_secs(self.timeout_secs)
    }

    /// Connection-establishment timeout.
    pub fn connect_timeout(&self) -> Duration {
        Duration::from_secs(self.connect_timeout_secs)
    }

    /// Build a circuit breaker per configured provider.
    pub fn build_breakers(&self) -> Breakers {
        Breakers::new(
            self.providers.iter().map(|p| p.name.clone()),
            self.circuit_threshold,
            Duration::from_secs(self.circuit_cooldown_secs),
        )
    }

    /// Build the semantic cache, if enabled. Falls back to the default
    /// provider's endpoint and key for embeddings when not set explicitly.
    pub fn build_semantic(&self, client: reqwest::Client) -> Option<SemanticCache> {
        if !self.semantic_cache {
            return None;
        }
        let default = self.default_provider_name();
        let dp = self
            .providers
            .iter()
            .find(|p| p.name == default)
            .or_else(|| self.providers.first());
        let base = self
            .embed_base_url
            .clone()
            .or_else(|| dp.map(|p| p.base_url.clone()))
            .unwrap_or_default();
        let key = self
            .embed_api_key
            .clone()
            .or_else(|| dp.and_then(|p| p.api_key.clone()));
        Some(SemanticCache::new(
            client,
            base,
            self.embed_model.clone(),
            key,
            self.semantic_threshold as f32,
            self.semantic_capacity,
        ))
    }

    /// Build the configured optimizer.
    pub fn optimizer(&self) -> Optimizer {
        Optimizer::new(self.optimize)
    }

    /// Build the configured response cache.
    pub fn build_cache(&self) -> Cache {
        Cache::new(self.cache, self.cache_capacity)
    }

    /// The resolved default provider name (first provider if unset).
    pub fn default_provider_name(&self) -> String {
        self.default_provider
            .clone()
            .unwrap_or_else(|| self.providers[0].name.clone())
    }

    /// The estimator implied by this configuration.
    pub fn estimator(&self) -> Estimator {
        Estimator::new(self.grid_intensity)
    }

    /// Instantiate every provider into a registry.
    pub fn build_registry(&self) -> Result<ProviderRegistry> {
        let mut providers: Vec<Box<dyn Provider>> = Vec::new();
        for pc in &self.providers {
            let provider: Box<dyn Provider> = match pc.kind {
                ProviderKind::Openai => Box::new(OpenAiCompatibleProvider::new(
                    pc.name.clone(),
                    pc.base_url.clone(),
                    pc.api_key.clone(),
                    pc.models.clone(),
                )),
                ProviderKind::Anthropic => Box::new(AnthropicProvider::new(
                    pc.name.clone(),
                    pc.base_url.clone(),
                    pc.api_key.clone(),
                    pc.models.clone(),
                    pc.anthropic_version.clone(),
                )),
                ProviderKind::Gemini => Box::new(GeminiProvider::new(
                    pc.name.clone(),
                    pc.base_url.clone(),
                    pc.api_key.clone(),
                    pc.models.clone(),
                )),
            };
            providers.push(provider);
        }
        Ok(ProviderRegistry::new(
            providers,
            self.default_provider_name(),
        ))
    }

    /// The shared carbon-intensity map (static table + overrides). The `carbon`
    /// router reads it and the live feed poller writes it, so both must share
    /// the same instance.
    pub fn build_carbon_map(&self) -> Arc<CarbonMap> {
        Arc::new(CarbonMap::new(self.grid_intensity, &self.carbon_overrides))
    }

    /// Build the live carbon feed and its zone list, if one is configured.
    /// Returns the feed, the `(region, zone)` pairs to poll, and the interval.
    /// The token is read from `JOULE_CARBON_TOKEN` (preferred) or config;
    /// returns `None` (static table only) if a required token is missing or no
    /// zones can be determined.
    pub fn build_carbon_feed(&self, client: reqwest::Client) -> Option<CarbonFeedPlan> {
        let kind = self.carbon_source?;

        let token = std::env::var(CARBON_TOKEN_ENV)
            .ok()
            .filter(|s| !s.is_empty())
            .or_else(|| self.carbon_source_token.clone());
        if kind.needs_token() && token.is_none() {
            tracing::warn!(
                source = ?kind,
                "carbon_source set but no token ({CARBON_TOKEN_ENV} or carbon_source_token); \
                 using the static carbon table",
            );
            return None;
        }

        // Determine which (region, zone) pairs to poll.
        let mut zones: Vec<(String, String)> = self
            .carbon_zones
            .iter()
            .map(|(region, zone)| (region.clone(), zone.clone()))
            .collect();
        if zones.is_empty() {
            match kind {
                // National feed: refresh the "uk" region by default.
                CarbonSourceKind::Uk => zones.push(("uk".to_string(), "GB".to_string())),
                _ => {
                    tracing::warn!(
                        source = ?kind,
                        "carbon_source set but carbon_zones is empty; using the static table",
                    );
                    return None;
                }
            }
        }
        zones.sort();

        let feed = CarbonFeed::new(client, kind, self.carbon_source_url.clone(), token);
        Some((feed, zones, Duration::from_secs(self.carbon_poll_secs)))
    }

    /// Instantiate the configured routing policy, sharing `carbon` with the feed.
    pub fn build_router(&self, estimator: Estimator, carbon: Arc<CarbonMap>) -> Box<dyn Router> {
        let default = self.default_provider_name();
        match self.router {
            RouterKind::Static => Box::new(StaticRouter::new(default)),
            RouterKind::Model => Box::new(ModelRouter::new(default)),
            RouterKind::Greenest => Box::new(GreenestRouter::new(
                self.greenest_candidates.clone(),
                estimator,
                default,
            )),
            RouterKind::Carbon => {
                let regions = self
                    .providers
                    .iter()
                    .filter_map(|p| p.region.clone().map(|r| (p.name.clone(), r)))
                    .collect();
                Box::new(CarbonRouter::new(carbon, regions, default))
            }
            RouterKind::Complexity => Box::new(ComplexityRouter::new(
                self.complexity_simple.clone(),
                self.complexity_complex.clone(),
                self.complexity_max_simple_tokens,
                default,
            )),
        }
    }
}
