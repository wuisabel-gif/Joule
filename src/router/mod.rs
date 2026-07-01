//! Router plugins: how a request is mapped to a provider (and maybe a model).
//!
//! Routing is where energy-, cost-, and quality-aware policy lives. Each router
//! implements the same [`Router`] trait, so swapping policy is a config change.
//! Three are provided:
//!
//! - [`StaticRouter`]   — always the default provider (transparent proxy).
//! - [`ModelRouter`]     — the first provider that declares support for the model.
//! - [`GreenestRouter`]  — the candidate model with the lowest estimated energy.
//! - [`CarbonRouter`]    — the provider whose region has the cleanest grid.
//! - [`ComplexityRouter`] — a small model for simple tasks, a capable one otherwise.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;

use crate::carbon::CarbonMap;
use crate::estimator::Estimator;
use crate::provider::{Provider, ProviderRegistry};
use crate::semantic::prompt_text;

/// The outcome of routing: which provider to use, with what model, and why.
pub struct RouteDecision<'a> {
    pub provider: &'a dyn Provider,
    pub model: String,
    pub reason: String,
}

/// A pluggable routing policy.
pub trait Router: Send + Sync {
    /// Policy name, surfaced in the `x-joule-route` response header.
    fn name(&self) -> &str;

    /// Choose a provider and model. `model` is what the client requested;
    /// `request` is the full (optimized) chat request, for content-aware policies.
    fn route<'a>(
        &self,
        registry: &'a ProviderRegistry,
        model: &str,
        request: &Value,
    ) -> Result<RouteDecision<'a>, String>;
}

/// Always routes to the default provider, leaving the model untouched.
pub struct StaticRouter {
    default: String,
}

impl StaticRouter {
    pub fn new(default: String) -> Self {
        Self { default }
    }
}

impl Router for StaticRouter {
    fn name(&self) -> &str {
        "static"
    }

    fn route<'a>(
        &self,
        registry: &'a ProviderRegistry,
        model: &str,
        _request: &Value,
    ) -> Result<RouteDecision<'a>, String> {
        let provider = registry
            .get(&self.default)
            .ok_or_else(|| format!("unknown default provider '{}'", self.default))?;
        Ok(RouteDecision {
            provider,
            model: model.to_string(),
            reason: format!("static -> {}", provider.name()),
        })
    }
}

/// Routes to the first provider that declares support for the requested model,
/// falling back to the default provider.
pub struct ModelRouter {
    default: String,
}

impl ModelRouter {
    pub fn new(default: String) -> Self {
        Self { default }
    }
}

impl Router for ModelRouter {
    fn name(&self) -> &str {
        "model"
    }

    fn route<'a>(
        &self,
        registry: &'a ProviderRegistry,
        model: &str,
        _request: &Value,
    ) -> Result<RouteDecision<'a>, String> {
        if let Some(provider) = registry.supporting(model) {
            return Ok(RouteDecision {
                provider,
                model: model.to_string(),
                reason: format!("model '{model}' -> {}", provider.name()),
            });
        }
        let provider = registry
            .get(&self.default)
            .ok_or_else(|| format!("no provider supports '{model}' and no default"))?;
        Ok(RouteDecision {
            provider,
            model: model.to_string(),
            reason: format!("fallback -> {}", provider.name()),
        })
    }
}

/// Energy-aware router: among a configured set of candidate models, pick the
/// one with the lowest estimated energy for a representative workload, then
/// route to a provider that serves it. Overrides the requested model.
pub struct GreenestRouter {
    candidates: Vec<String>,
    estimator: Estimator,
    default: String,
}

impl GreenestRouter {
    /// Representative token shape used to compare candidate models.
    const SAMPLE_INPUT: u64 = 512;
    const SAMPLE_OUTPUT: u64 = 256;

    pub fn new(candidates: Vec<String>, estimator: Estimator, default: String) -> Self {
        Self {
            candidates,
            estimator,
            default,
        }
    }
}

impl Router for GreenestRouter {
    fn name(&self) -> &str {
        "greenest"
    }

    fn route<'a>(
        &self,
        registry: &'a ProviderRegistry,
        model: &str,
        _request: &Value,
    ) -> Result<RouteDecision<'a>, String> {
        // Rank candidates served by some provider by estimated energy.
        let mut best: Option<(&'a dyn Provider, String, f64)> = None;
        for candidate in &self.candidates {
            let Some(provider) = registry.supporting(candidate) else {
                continue;
            };
            let energy = self
                .estimator
                .estimate(candidate, Self::SAMPLE_INPUT, Self::SAMPLE_OUTPUT)
                .energy_j;
            let better = match &best {
                Some((_, _, b)) => energy < *b,
                None => true,
            };
            if better {
                best = Some((provider, candidate.clone(), energy));
            }
        }

        if let Some((provider, chosen, energy)) = best {
            return Ok(RouteDecision {
                provider,
                reason: format!("greenest: {chosen} (~{energy:.1} J / sample) instead of {model}"),
                model: chosen,
            });
        }

        // No candidate available: honour the requested model on a provider that
        // supports it, else the default.
        let provider = registry
            .supporting(model)
            .or_else(|| registry.get(&self.default))
            .ok_or_else(|| "no candidate, supporting, or default provider".to_string())?;
        Ok(RouteDecision {
            provider,
            model: model.to_string(),
            reason: format!("greenest: no candidate available -> {}", provider.name()),
        })
    }
}

/// Carbon-aware router: among providers that support the requested model, route
/// to the one whose region has the lowest current grid carbon intensity. The
/// intensity map can be refreshed at runtime from a live source.
pub struct CarbonRouter {
    carbon: Arc<CarbonMap>,
    /// Provider name → region key.
    regions: HashMap<String, String>,
    default: String,
}

impl CarbonRouter {
    pub fn new(carbon: Arc<CarbonMap>, regions: HashMap<String, String>, default: String) -> Self {
        Self {
            carbon,
            regions,
            default,
        }
    }

    fn region_of(&self, provider: &str) -> &str {
        self.regions.get(provider).map(String::as_str).unwrap_or("")
    }
}

impl Router for CarbonRouter {
    fn name(&self) -> &str {
        "carbon"
    }

    fn route<'a>(
        &self,
        registry: &'a ProviderRegistry,
        model: &str,
        _request: &Value,
    ) -> Result<RouteDecision<'a>, String> {
        // Pick the supporting provider whose region is cleanest right now.
        let mut best: Option<(&'a dyn Provider, f64)> = None;
        for provider in registry.iter() {
            if !provider.supports_model(model) {
                continue;
            }
            let gco2 = self.carbon.intensity(self.region_of(provider.name()));
            if best.is_none_or(|(_, b)| gco2 < b) {
                best = Some((provider, gco2));
            }
        }

        if let Some((provider, gco2)) = best {
            let region = self.region_of(provider.name());
            return Ok(RouteDecision {
                model: model.to_string(),
                reason: format!(
                    "carbon: {} @ ~{gco2:.0} gCO2/kWh{}",
                    provider.name(),
                    if region.is_empty() {
                        String::new()
                    } else {
                        format!(" ({region})")
                    }
                ),
                provider,
            });
        }

        let provider = registry
            .get(&self.default)
            .ok_or_else(|| format!("no provider supports '{model}' and no default"))?;
        Ok(RouteDecision {
            provider,
            model: model.to_string(),
            reason: format!("carbon: fallback -> {}", provider.name()),
        })
    }
}

/// Complexity-aware router: send clearly-simple requests (translate, summarize,
/// classify, format, short prompts) to a small model and everything else to a
/// capable one. Applied conservatively — it downgrades only when confident the
/// task is simple, defaulting to the capable model to protect answer quality.
pub struct ComplexityRouter {
    simple: Option<String>,
    complex: Option<String>,
    max_simple_tokens: usize,
    default: String,
}

impl ComplexityRouter {
    pub fn new(
        simple: Option<String>,
        complex: Option<String>,
        max_simple_tokens: usize,
        default: String,
    ) -> Self {
        Self {
            simple,
            complex,
            max_simple_tokens,
            default,
        }
    }
}

impl Router for ComplexityRouter {
    fn name(&self) -> &str {
        "complexity"
    }

    fn route<'a>(
        &self,
        registry: &'a ProviderRegistry,
        model: &str,
        request: &Value,
    ) -> Result<RouteDecision<'a>, String> {
        let text = prompt_text(request).unwrap_or_default();
        let simple = looks_simple(&text, self.max_simple_tokens);
        // Fall back to the requested model when a tier isn't configured, so an
        // unconfigured tier never silently downgrades.
        let (tier, chosen) = if simple {
            (
                "simple",
                self.simple.clone().unwrap_or_else(|| model.to_string()),
            )
        } else {
            (
                "complex",
                self.complex.clone().unwrap_or_else(|| model.to_string()),
            )
        };
        let provider = registry
            .supporting(&chosen)
            .or_else(|| registry.get(&self.default))
            .ok_or_else(|| format!("no provider supports '{chosen}' and no default"))?;
        Ok(RouteDecision {
            reason: format!("complexity: {tier} -> {chosen}"),
            model: chosen,
            provider,
        })
    }
}

/// Heuristic complexity check — true only when a request is *confidently* simple:
/// short, carrying a simple-task signal, and free of any complexity signal.
fn looks_simple(text: &str, max_simple_tokens: usize) -> bool {
    // ~4 chars/token is enough for a length gate.
    if text.chars().count() / 4 > max_simple_tokens {
        return false;
    }
    let t = text.to_ascii_lowercase();

    const COMPLEX: &[&str] = &[
        "```",
        "prove",
        "derive",
        "analyz",
        "debug",
        "refactor",
        "algorithm",
        "step by step",
        "explain why",
        "reason",
        "theorem",
        "architecture",
        "trade-off",
        "optimize",
        "implement",
    ];
    if COMPLEX.iter().any(|k| t.contains(k)) {
        return false;
    }

    const SIMPLE: &[&str] = &[
        "translate",
        "summar",
        "classif",
        "format",
        "extract",
        "spell",
        "grammar",
        "rewrite",
        "tl;dr",
        "label",
        "categor",
        "json",
    ];
    SIMPLE.iter().any(|k| t.contains(k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimator::DEFAULT_GRID_INTENSITY_G_PER_KWH;
    use crate::provider::{AnthropicProvider, OpenAiCompatibleProvider, ProviderRegistry};

    fn registry() -> ProviderRegistry {
        let openai = OpenAiCompatibleProvider::new(
            "openai".into(),
            "https://api.openai.com".into(),
            None,
            vec!["gpt-".into()],
        );
        let anthropic = AnthropicProvider::new(
            "anthropic".into(),
            "https://api.anthropic.com".into(),
            None,
            vec!["claude".into()],
            None,
        );
        ProviderRegistry::new(vec![Box::new(openai), Box::new(anthropic)], "openai".into())
    }

    #[test]
    fn model_router_picks_by_support() {
        let r = ModelRouter::new("openai".into());
        let reg = registry();
        assert_eq!(
            r.route(&reg, "claude-3-opus", &Value::Null)
                .unwrap()
                .provider
                .name(),
            "anthropic"
        );
        assert_eq!(
            r.route(&reg, "gpt-4o", &Value::Null)
                .unwrap()
                .provider
                .name(),
            "openai"
        );
    }

    #[test]
    fn carbon_router_prefers_cleaner_region() {
        // Two providers serving the same model, in different-carbon regions.
        let dirty = OpenAiCompatibleProvider::new(
            "dirty".into(),
            "http://a".into(),
            None,
            vec!["gpt-".into()],
        );
        let clean = OpenAiCompatibleProvider::new(
            "clean".into(),
            "http://b".into(),
            None,
            vec!["gpt-".into()],
        );
        let reg = ProviderRegistry::new(vec![Box::new(dirty), Box::new(clean)], "dirty".into());

        let carbon = Arc::new(CarbonMap::new(445.0, &HashMap::new()));
        let mut regions = HashMap::new();
        regions.insert("dirty".to_string(), "us-east".to_string()); // ~380
        regions.insert("clean".to_string(), "norway".to_string()); // ~30
        let r = CarbonRouter::new(carbon, regions, "dirty".into());

        assert_eq!(
            r.route(&reg, "gpt-4o", &Value::Null)
                .unwrap()
                .provider
                .name(),
            "clean"
        );
    }

    #[test]
    fn greenest_router_prefers_low_energy_model() {
        let est = Estimator::new(DEFAULT_GRID_INTENSITY_G_PER_KWH);
        // gpt-4o-mini is far lower energy than claude-3-opus.
        let r = GreenestRouter::new(
            vec!["claude-3-opus".into(), "gpt-4o-mini".into()],
            est,
            "openai".into(),
        );
        let reg = registry();
        let decision = r.route(&reg, "gpt-4", &Value::Null).unwrap();
        assert_eq!(decision.model, "gpt-4o-mini");
        assert_eq!(decision.provider.name(), "openai");
    }

    #[test]
    fn complexity_router_downgrades_only_simple_tasks() {
        use serde_json::json;
        let reg = registry(); // openai (gpt-), anthropic (claude)
        let r = ComplexityRouter::new(
            Some("gpt-4o-mini".into()),
            Some("gpt-4o".into()),
            240,
            "openai".into(),
        );

        let simple = json!({"messages":[{"role":"user","content":"Translate 'hello' to French."}]});
        let d = r.route(&reg, "gpt-4o", &simple).unwrap();
        assert_eq!(d.model, "gpt-4o-mini"); // downgraded

        let complex = json!({"messages":[{"role":"user","content":"Prove that sqrt(2) is irrational, step by step."}]});
        let d = r.route(&reg, "gpt-4o", &complex).unwrap();
        assert_eq!(d.model, "gpt-4o"); // kept capable

        // No simple signal → treated as complex (conservative).
        let plain = json!({"messages":[{"role":"user","content":"What time is it in Tokyo?"}]});
        assert_eq!(r.route(&reg, "gpt-4o", &plain).unwrap().model, "gpt-4o");
    }
}
