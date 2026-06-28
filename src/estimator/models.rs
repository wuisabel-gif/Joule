//! Per-model energy and price profiles.
//!
//! These figures are deliberately rough first-order estimates derived from
//! published benchmarks and vendor pricing. Prefill (input) tokens are
//! processed in parallel and cost less energy per token than decode (output)
//! tokens, which are generated sequentially and are memory-bandwidth bound.
//!
//! Treat every number here as a starting point to be refined with real
//! hardware measurements (see the Estimator's longer-term roadmap).

/// Energy and pricing characteristics for one model.
#[derive(Debug, Clone, Copy)]
pub struct ModelProfile {
    /// Canonical model family this profile represents.
    pub family: &'static str,
    /// Estimated joules spent per input (prefill) token.
    pub j_per_input_token: f64,
    /// Estimated joules spent per output (decode) token.
    pub j_per_output_token: f64,
    /// USD per million input tokens.
    pub usd_per_m_input: f64,
    /// USD per million output tokens.
    pub usd_per_m_output: f64,
}

/// Profile used when a model name is not recognised. Intentionally on the
/// higher end so that unknown models are not silently treated as free.
pub const DEFAULT_PROFILE: ModelProfile = ModelProfile {
    family: "unknown",
    j_per_input_token: 0.6,
    j_per_output_token: 1.8,
    usd_per_m_input: 1.0,
    usd_per_m_output: 3.0,
};

/// Known model families, matched by case-insensitive prefix of the request's
/// `model` field. Order matters: longer / more specific prefixes first.
const PROFILES: &[(&str, ModelProfile)] = &[
    (
        "gpt-4o-mini",
        ModelProfile {
            family: "gpt-4o-mini",
            j_per_input_token: 0.10,
            j_per_output_token: 0.30,
            usd_per_m_input: 0.15,
            usd_per_m_output: 0.60,
        },
    ),
    (
        "gpt-4o",
        ModelProfile {
            family: "gpt-4o",
            j_per_input_token: 0.40,
            j_per_output_token: 1.20,
            usd_per_m_input: 2.50,
            usd_per_m_output: 10.0,
        },
    ),
    (
        "gpt-4",
        ModelProfile {
            family: "gpt-4",
            j_per_input_token: 0.80,
            j_per_output_token: 2.40,
            usd_per_m_input: 30.0,
            usd_per_m_output: 60.0,
        },
    ),
    (
        "gpt-3.5",
        ModelProfile {
            family: "gpt-3.5",
            j_per_input_token: 0.15,
            j_per_output_token: 0.45,
            usd_per_m_input: 0.50,
            usd_per_m_output: 1.50,
        },
    ),
    (
        "claude-3-5-haiku",
        ModelProfile {
            family: "claude-3-5-haiku",
            j_per_input_token: 0.12,
            j_per_output_token: 0.36,
            usd_per_m_input: 0.80,
            usd_per_m_output: 4.0,
        },
    ),
    (
        "claude-3-5-sonnet",
        ModelProfile {
            family: "claude-3-5-sonnet",
            j_per_input_token: 0.45,
            j_per_output_token: 1.35,
            usd_per_m_input: 3.0,
            usd_per_m_output: 15.0,
        },
    ),
    (
        "claude-3-opus",
        ModelProfile {
            family: "claude-3-opus",
            j_per_input_token: 0.90,
            j_per_output_token: 2.70,
            usd_per_m_input: 15.0,
            usd_per_m_output: 75.0,
        },
    ),
    (
        "gemini-1.5-flash",
        ModelProfile {
            family: "gemini-1.5-flash",
            j_per_input_token: 0.08,
            j_per_output_token: 0.24,
            usd_per_m_input: 0.075,
            usd_per_m_output: 0.30,
        },
    ),
    (
        "gemini-1.5-pro",
        ModelProfile {
            family: "gemini-1.5-pro",
            j_per_input_token: 0.50,
            j_per_output_token: 1.50,
            usd_per_m_input: 1.25,
            usd_per_m_output: 5.0,
        },
    ),
    (
        "llama-3",
        ModelProfile {
            family: "llama-3 (local)",
            j_per_input_token: 0.05,
            j_per_output_token: 0.20,
            usd_per_m_input: 0.0,
            usd_per_m_output: 0.0,
        },
    ),
];

/// Look up the profile for a model name, falling back to [`DEFAULT_PROFILE`].
pub fn profile_for(model: &str) -> ModelProfile {
    let needle = model.to_ascii_lowercase();
    for (prefix, profile) in PROFILES {
        if needle.starts_with(prefix) {
            return *profile;
        }
    }
    DEFAULT_PROFILE
}

/// All known profiles, for the `joule models` CLI command.
pub fn all() -> impl Iterator<Item = &'static ModelProfile> {
    PROFILES.iter().map(|(_, p)| p)
}
