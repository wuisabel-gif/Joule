//! Runtime configuration: providers, routing policy, and the estimator.
//!
//! Two ways to configure Joule:
//! - a JSON config file (`--config`) for multi-provider / routed setups, or
//! - the single-provider quickstart flags (`--upstream`, `--api-key`, …).

use anyhow::{Context, Result};
use clap::ValueEnum;
use serde::Deserialize;

use crate::estimator::{Estimator, DEFAULT_GRID_INTENSITY_G_PER_KWH};
use crate::optimizer::{OptLevel, Optimizer};
use crate::provider::{
    AnthropicProvider, GeminiProvider, OpenAiCompatibleProvider, Provider, ProviderRegistry,
};
use crate::router::{GreenestRouter, ModelRouter, Router, StaticRouter};

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
}

fn default_grid() -> f64 {
    DEFAULT_GRID_INTENSITY_G_PER_KWH
}

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
    /// Prompt-optimization intensity.
    #[serde(default)]
    pub optimize: OptLevel,
    #[serde(default = "default_grid")]
    pub grid_intensity: f64,
}

impl Config {
    /// Load configuration from a JSON file.
    pub fn from_file(path: &str) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading config file {path}"))?;
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
            }],
            default_provider: Some("default".to_string()),
            router,
            greenest_candidates: Vec::new(),
            optimize,
            grid_intensity,
        }
    }

    /// Build the configured optimizer.
    pub fn optimizer(&self) -> Optimizer {
        Optimizer::new(self.optimize)
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
        Ok(ProviderRegistry::new(providers, self.default_provider_name()))
    }

    /// Instantiate the configured routing policy.
    pub fn build_router(&self, estimator: Estimator) -> Box<dyn Router> {
        let default = self.default_provider_name();
        match self.router {
            RouterKind::Static => Box::new(StaticRouter::new(default)),
            RouterKind::Model => Box::new(ModelRouter::new(default)),
            RouterKind::Greenest => Box::new(GreenestRouter::new(
                self.greenest_candidates.clone(),
                estimator,
                default,
            )),
        }
    }
}
