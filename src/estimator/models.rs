//! Per-model energy and price profiles.
//!
//! These figures are first-order estimates, not measurements — but they are
//! calibrated to published benchmarks rather than guessed. Prefill (input)
//! tokens are processed in parallel and cost less energy per token than decode
//! (output) tokens, which are generated sequentially and are memory-bandwidth
//! bound, so output J/token ≈ 3× input across these profiles.
//!
//! Calibration anchor: `gpt-4o` is set so a ~500-output-token query lands near
//! **0.3 Wh** on optimized H100 serving, matching Epoch AI's estimate
//! (https://epoch.ai/gradient-updates/how-much-energy-does-chatgpt-use).
//! Other model families are scaled from that anchor by relative size/class.
//!
//! Supporting measurements these figures sit within:
//! - ~1.72 J/token on A100 (arXiv 2512.01644, systematic characterization)
//! - ~3–4 J/token on V100/A100 for LLaMA-65B ("From Words to Watts", Samsi 2023)
//! - ~0.39 J/token on H100 + FP8 + batch-128 (best-case, optimized serving)
//! - GPT-4o end-to-end 0.42–1.59 Wh by prompt length (arXiv 2505.09598,
//!   "How Hungry is AI?") — higher than the anchor because of less-optimized
//!   utilization/overhead assumptions; we anchor to the efficient H100 case.
//!
//! Real per-deployment numbers depend heavily on hardware, batching, and
//! quantization; treat these as a transparent default to be refined.

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
/// higher end (near a frontier model) so unknown models are not undercounted.
pub const DEFAULT_PROFILE: ModelProfile = ModelProfile {
    family: "unknown",
    j_per_input_token: 0.8,
    j_per_output_token: 2.5,
    usd_per_m_input: 1.0,
    usd_per_m_output: 3.0,
};

/// Known model families, matched by case-insensitive prefix of the request's
/// `model` field. Order matters: longer / more specific prefixes first.
///
/// Energy classes (output J/token), scaled from the `gpt-4o` anchor (2.0):
/// - small / "mini"   ~0.35–0.6   (Flash, mini, Haiku, 3.5)
/// - mid-large        ~2.0        (4o, Sonnet, Gemini Pro)
/// - frontier / large ~3.5        (GPT-4, Opus)
/// - local mid (A100) ~1.7        (measured ~1.72 J/token, arXiv 2512.01644)
const PROFILES: &[(&str, ModelProfile)] = &[
    (
        "gpt-4o-mini",
        ModelProfile {
            family: "gpt-4o-mini",
            // Small model; note real deployments sometimes run mini on A100s,
            // raising per-token energy despite the smaller architecture.
            j_per_input_token: 0.15,
            j_per_output_token: 0.50,
            usd_per_m_input: 0.15,
            usd_per_m_output: 0.60,
        },
    ),
    (
        "gpt-4o",
        ModelProfile {
            family: "gpt-4o",
            // Anchor: ~0.3 Wh for a 500-output-token query on H100 (Epoch AI).
            j_per_input_token: 0.60,
            j_per_output_token: 2.00,
            usd_per_m_input: 2.50,
            usd_per_m_output: 10.0,
        },
    ),
    (
        "gpt-4",
        ModelProfile {
            family: "gpt-4",
            // Older frontier model, less inference-optimized than 4o.
            j_per_input_token: 1.10,
            j_per_output_token: 3.50,
            usd_per_m_input: 30.0,
            usd_per_m_output: 60.0,
        },
    ),
    (
        "gpt-3.5",
        ModelProfile {
            family: "gpt-3.5",
            j_per_input_token: 0.20,
            j_per_output_token: 0.60,
            usd_per_m_input: 0.50,
            usd_per_m_output: 1.50,
        },
    ),
    (
        "claude-3-5-haiku",
        ModelProfile {
            family: "claude-3-5-haiku",
            j_per_input_token: 0.15,
            j_per_output_token: 0.50,
            usd_per_m_input: 0.80,
            usd_per_m_output: 4.0,
        },
    ),
    (
        "claude-3-5-sonnet",
        ModelProfile {
            family: "claude-3-5-sonnet",
            // Mid-large, ~4o class.
            j_per_input_token: 0.60,
            j_per_output_token: 2.00,
            usd_per_m_input: 3.0,
            usd_per_m_output: 15.0,
        },
    ),
    (
        "claude-3-opus",
        ModelProfile {
            family: "claude-3-opus",
            // Frontier large.
            j_per_input_token: 1.10,
            j_per_output_token: 3.50,
            usd_per_m_input: 15.0,
            usd_per_m_output: 75.0,
        },
    ),
    (
        "gemini-1.5-flash",
        ModelProfile {
            family: "gemini-1.5-flash",
            // Very efficient small model.
            j_per_input_token: 0.10,
            j_per_output_token: 0.35,
            usd_per_m_input: 0.075,
            usd_per_m_output: 0.30,
        },
    ),
    (
        "gemini-1.5-pro",
        ModelProfile {
            family: "gemini-1.5-pro",
            j_per_input_token: 0.60,
            j_per_output_token: 2.00,
            usd_per_m_input: 1.25,
            usd_per_m_output: 5.0,
        },
    ),
    (
        "llama-3",
        ModelProfile {
            family: "llama-3 (local)",
            // Self-hosted mid-size on a single A100 (~1.72 J/token measured).
            j_per_input_token: 0.50,
            j_per_output_token: 1.70,
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
