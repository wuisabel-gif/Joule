//! Provider plugins: the vendor wire-protocol abstraction.
//!
//! A [`Provider`] knows how to talk to one kind of backend (OpenAI-compatible,
//! Anthropic, …). It is intentionally *declarative*: it builds the upstream
//! request and parses tokens out of responses, but it does not execute HTTP or
//! own streaming/metrics — that stays in the proxy so every provider shares the
//! same measuring path.
//!
//! The canonical request/response format on the *client* side is always the
//! OpenAI chat-completions JSON shape; providers translate to and from it.

pub mod anthropic;
pub mod gemini;
pub mod openai;
mod registry;

use axum::http::HeaderMap;
use serde_json::Value;

pub use anthropic::AnthropicProvider;
pub use gemini::GeminiProvider;
pub use openai::OpenAiCompatibleProvider;
pub use registry::ProviderRegistry;

/// Error raised while preparing a provider request.
#[derive(Debug)]
pub struct ProviderError(pub String);

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ProviderError {}

/// A pluggable LLM backend. Object-safe so providers live behind `dyn`.
pub trait Provider: Send + Sync {
    /// Unique registry name, e.g. `"openai"`, `"anthropic"`, `"ollama"`.
    fn name(&self) -> &str;

    /// Upstream base URL, used for transparent passthrough of unmetered routes.
    fn base_url(&self) -> &str;

    /// Whether this provider can serve `model`. An empty model filter means the
    /// provider is a wildcard (serves anything routed to it).
    fn supports_model(&self, model: &str) -> bool;

    /// Apply provider-specific authentication to any request builder. Prefers
    /// credentials the client supplied, falling back to configured keys.
    fn authorize(
        &self,
        rb: reqwest::RequestBuilder,
        client_headers: &HeaderMap,
    ) -> reqwest::RequestBuilder;

    /// Build the upstream chat-completions request from canonical OpenAI JSON,
    /// using `model` (which the router may have overridden).
    fn build_chat_request(
        &self,
        client: &reqwest::Client,
        canonical: &Value,
        model: &str,
        client_headers: &HeaderMap,
    ) -> Result<reqwest::RequestBuilder, ProviderError>;

    /// Extract `(prompt_tokens, completion_tokens)` from a buffered upstream
    /// response body, if the provider reports usage.
    fn usage_from_body(&self, body: &Value) -> Option<(u64, u64)>;

    /// Translate a buffered upstream body into OpenAI chat-completions shape so
    /// clients always see a consistent format. Defaults to identity.
    fn translate_response(&self, body: Value) -> Value {
        body
    }

    /// Assistant text contained in one parsed streaming SSE event, if any.
    fn stream_content_delta(&self, event: &Value) -> Option<String>;

    /// Prompt-token count carried by one streaming SSE event, if any.
    fn stream_prompt_tokens(&self, event: &Value) -> Option<u64>;

    /// Completion-token count carried by one streaming SSE event, if any.
    fn stream_completion_tokens(&self, event: &Value) -> Option<u64>;
}
