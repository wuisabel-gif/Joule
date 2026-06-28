//! Router plugins: how a request is mapped to a provider (and maybe a model).
//!
//! Routing is where energy-, cost-, and quality-aware policy lives. Each router
//! implements the same [`Router`] trait, so swapping policy is a config change.
//! Three are provided:
//!
//! - [`StaticRouter`]  — always the default provider (transparent proxy).
//! - [`ModelRouter`]   — the first provider that declares support for the model.
//! - [`GreenestRouter`] — among configured candidate models, pick the one with
//!   the lowest estimated energy and route to a provider that serves it.

use crate::estimator::Estimator;
use crate::provider::{Provider, ProviderRegistry};

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

    /// Choose a provider and model for a request that asked for `model`.
    fn route<'a>(
        &self,
        registry: &'a ProviderRegistry,
        model: &str,
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
                reason: format!(
                    "greenest: {chosen} (~{energy:.1} J / sample) instead of {model}"
                ),
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
        assert_eq!(r.route(&reg, "claude-3-opus").unwrap().provider.name(), "anthropic");
        assert_eq!(r.route(&reg, "gpt-4o").unwrap().provider.name(), "openai");
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
        let decision = r.route(&reg, "gpt-4").unwrap();
        assert_eq!(decision.model, "gpt-4o-mini");
        assert_eq!(decision.provider.name(), "openai");
    }
}
