//! OpenAI-compatible provider.
//!
//! Covers OpenAI itself and every server that speaks the same wire format:
//! Ollama, LM Studio, vLLM, llama.cpp, and friends. Differences between them
//! are entirely configuration (base URL, API key, served models).

use axum::http::header::AUTHORIZATION;
use axum::http::HeaderMap;
use serde_json::{json, Value};

use super::{Provider, ProviderError};

/// A provider speaking the OpenAI chat-completions protocol.
pub struct OpenAiCompatibleProvider {
    name: String,
    base_url: String,
    api_key: Option<String>,
    /// Model-name prefixes this provider serves; empty = wildcard.
    model_prefixes: Vec<String>,
}

impl OpenAiCompatibleProvider {
    pub fn new(
        name: String,
        base_url: String,
        api_key: Option<String>,
        model_prefixes: Vec<String>,
    ) -> Self {
        Self {
            name,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            model_prefixes,
        }
    }
}

impl Provider for OpenAiCompatibleProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn supports_model(&self, model: &str) -> bool {
        if self.model_prefixes.is_empty() {
            return true;
        }
        let model = model.to_ascii_lowercase();
        self.model_prefixes
            .iter()
            .any(|p| model.starts_with(&p.to_ascii_lowercase()))
    }

    fn authorize(
        &self,
        rb: reqwest::RequestBuilder,
        client_headers: &HeaderMap,
    ) -> reqwest::RequestBuilder {
        if let Some(auth) = client_headers.get(AUTHORIZATION) {
            rb.header(AUTHORIZATION, auth.clone())
        } else if let Some(key) = &self.api_key {
            rb.header(AUTHORIZATION, format!("Bearer {key}"))
        } else {
            rb
        }
    }

    fn build_chat_request(
        &self,
        client: &reqwest::Client,
        canonical: &Value,
        model: &str,
        client_headers: &HeaderMap,
    ) -> Result<reqwest::RequestBuilder, ProviderError> {
        let mut body = canonical.clone();
        body["model"] = json!(model);
        let url = format!("{}/v1/chat/completions", self.base_url);
        let rb = client.post(url).json(&body);
        Ok(self.authorize(rb, client_headers))
    }

    fn usage_from_body(&self, body: &Value) -> Option<(u64, u64)> {
        let usage = body.get("usage")?;
        Some((
            usage.get("prompt_tokens")?.as_u64()?,
            usage.get("completion_tokens")?.as_u64()?,
        ))
    }

    fn stream_content_delta(&self, event: &Value) -> Option<String> {
        event
            .pointer("/choices/0/delta/content")
            .and_then(Value::as_str)
            .map(str::to_string)
    }

    fn stream_prompt_tokens(&self, event: &Value) -> Option<u64> {
        event.pointer("/usage/prompt_tokens").and_then(Value::as_u64)
    }

    fn stream_completion_tokens(&self, event: &Value) -> Option<u64> {
        event
            .pointer("/usage/completion_tokens")
            .and_then(Value::as_u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wildcard_and_prefix_matching() {
        let wild =
            OpenAiCompatibleProvider::new("a".into(), "http://x".into(), None, vec![]);
        assert!(wild.supports_model("anything"));

        let scoped = OpenAiCompatibleProvider::new(
            "b".into(),
            "http://x".into(),
            None,
            vec!["gpt-".into()],
        );
        assert!(scoped.supports_model("gpt-4o"));
        assert!(!scoped.supports_model("claude-3-opus"));
    }
}
