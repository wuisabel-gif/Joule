//! Prompt optimization: the cheapest token is the one you never generate.
//!
//! This is the top layer of the energy stack — it reduces the work the model
//! does *before* inference starts. Optimization is a pipeline of composable
//! [`Pass`]es, each one small, explainable, and gated by an intensity
//! [`OptLevel`] (mirroring Ponytail's lite/full/ultra philosophy):
//!
//! - **lite**  — lossless formatting only (whitespace, exact-duplicate messages).
//! - **full**  — lossless content cleanup (collapse repeated lines, drop filler).
//! - **ultra** — behaviour-affecting levers (cap output length, ask for brevity).
//!
//! Nothing happens invisibly: every pass that fires returns a human-readable
//! note, and the [`OptimizationReport`] records exactly what changed and how
//! many prompt tokens it saved.

mod passes;

use serde_json::Value;

use crate::tokens::estimate_prompt_tokens;

pub use passes::default_passes;

/// Optimization intensity. Ordered: `Off < Lite < Full < Ultra`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, serde::Deserialize, clap::ValueEnum,
)]
#[serde(rename_all = "snake_case")]
pub enum OptLevel {
    /// No optimization; forward the prompt untouched.
    Off,
    /// Lossless formatting only. The safe default.
    #[default]
    Lite,
    /// Lossless content cleanup.
    Full,
    /// Aggressive, may change model behaviour (output length, brevity).
    Ultra,
}

impl OptLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            OptLevel::Off => "off",
            OptLevel::Lite => "lite",
            OptLevel::Full => "full",
            OptLevel::Ultra => "ultra",
        }
    }
}

/// A single optimization step. Object-safe so passes live behind `dyn`.
pub trait Pass: Send + Sync {
    /// Short identifier, e.g. `"collapse-whitespace"`.
    fn name(&self) -> &str;

    /// Lowest intensity at which this pass runs.
    fn min_level(&self) -> OptLevel;

    /// Apply the pass to the request in place. Returns a human-readable note
    /// describing what changed, or `None` if the pass was a no-op.
    fn apply(&self, request: &mut Value) -> Option<String>;
}

/// One pass that actually fired.
#[derive(Debug, Clone)]
pub struct AppliedPass {
    pub name: String,
    pub detail: String,
}

/// The outcome of running the optimizer over a request.
#[derive(Debug, Clone, Default)]
pub struct OptimizationReport {
    pub level: &'static str,
    pub applied: Vec<AppliedPass>,
    pub tokens_before: u64,
    pub tokens_after: u64,
}

impl OptimizationReport {
    /// Prompt tokens removed (never negative).
    pub fn tokens_saved(&self) -> u64 {
        self.tokens_before.saturating_sub(self.tokens_after)
    }

    /// Whether any pass changed the request.
    pub fn changed(&self) -> bool {
        !self.applied.is_empty()
    }

    /// Percentage of prompt tokens removed.
    pub fn percent_saved(&self) -> f64 {
        if self.tokens_before == 0 {
            0.0
        } else {
            self.tokens_saved() as f64 / self.tokens_before as f64 * 100.0
        }
    }

    /// Comma-separated names of the passes that fired.
    pub fn pass_names(&self) -> String {
        self.applied
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Whether any applied pass targets *output* tokens rather than the prompt.
    /// These trade a few input tokens for (typically larger) output savings the
    /// prompt-token count below cannot show.
    pub fn has_output_side_pass(&self) -> bool {
        self.applied
            .iter()
            .any(|p| matches!(p.name.as_str(), "output-limit" | "brevity-hint"))
    }

    /// A transparent, human-facing summary (used by the CLI and logs).
    pub fn summary(&self) -> String {
        if !self.changed() {
            return format!("Optimization Summary ({})\n  (no changes)", self.level);
        }
        let mut out = format!("Optimization Summary ({})\n", self.level);
        for p in &self.applied {
            out.push_str(&format!("  \u{2713} {}\n", p.detail));
        }

        // Show the real arithmetic, including increases, rather than clamping.
        let before = self.tokens_before as i64;
        let after = self.tokens_after as i64;
        let delta = before - after;
        let change = if delta > 0 {
            format!("\u{2212}{delta}, {:.0}% saved", self.percent_saved())
        } else if delta < 0 {
            format!("+{} added", -delta)
        } else {
            "no change".to_string()
        };
        out.push_str(&format!(
            "  Prompt tokens: {} \u{2192} {} ({change})",
            self.tokens_before, self.tokens_after,
        ));

        if self.has_output_side_pass() {
            out.push_str(
                "\n  Note: output-limit / brevity-hint target output tokens (the larger \
                 energy lever), which the prompt-token count above does not capture.",
            );
        }
        out
    }
}

/// A configured optimization pipeline.
pub struct Optimizer {
    level: OptLevel,
    passes: Vec<Box<dyn Pass>>,
}

impl Optimizer {
    /// Build an optimizer at `level` with the built-in pass set.
    pub fn new(level: OptLevel) -> Self {
        Self {
            level,
            passes: default_passes(),
        }
    }

    pub fn level(&self) -> OptLevel {
        self.level
    }

    /// Optimize a request in place, returning a report of what changed.
    pub fn optimize(&self, request: &mut Value) -> OptimizationReport {
        let tokens_before = estimate_prompt_tokens(request);
        let mut applied = Vec::new();

        if self.level != OptLevel::Off {
            for pass in &self.passes {
                if self.level < pass.min_level() {
                    continue;
                }
                if let Some(detail) = pass.apply(request) {
                    applied.push(AppliedPass {
                        name: pass.name().to_string(),
                        detail,
                    });
                }
            }
        }

        let tokens_after = estimate_prompt_tokens(request);
        OptimizationReport {
            level: self.level.as_str(),
            applied,
            tokens_before,
            tokens_after,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn off_level_is_a_noop() {
        let opt = Optimizer::new(OptLevel::Off);
        let mut req = json!({"messages":[{"role":"user","content":"hi   \n\n\n\nthere"}]});
        let before = req.clone();
        let report = opt.optimize(&mut req);
        assert!(!report.changed());
        assert_eq!(req, before);
    }

    #[test]
    fn lite_collapses_whitespace_and_dedup() {
        let opt = Optimizer::new(OptLevel::Lite);
        let mut req = json!({"messages":[
            {"role":"system","content":"You are helpful.\n\n\n\nBe nice."},
            {"role":"user","content":"Hello"},
            {"role":"user","content":"Hello"}
        ]});
        let report = opt.optimize(&mut req);
        assert!(report.changed());
        // Duplicate user message removed.
        assert_eq!(req["messages"].as_array().unwrap().len(), 2);
        assert!(report.tokens_saved() > 0);
    }

    #[test]
    fn ultra_caps_output_when_unset() {
        let opt = Optimizer::new(OptLevel::Ultra);
        let mut req = json!({"messages":[{"role":"user","content":"Write an essay."}]});
        let report = opt.optimize(&mut req);
        assert!(req.get("max_tokens").is_some());
        assert!(report.pass_names().contains("output-limit"));
    }
}
