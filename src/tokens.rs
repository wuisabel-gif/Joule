//! Token accounting.
//!
//! Joule prefers the provider's reported `usage` whenever it is present — that
//! is ground truth. When it isn't (some streaming responses, local models that
//! omit usage), Joule counts tokens with a real BPE tokenizer ([`tiktoken-rs`])
//! rather than a character heuristic.
//!
//! Encoding is chosen by model family: OpenAI's newer models use `o200k_base`,
//! `gpt-4`/`gpt-3.5` use `cl100k_base`. Non-OpenAI models (Claude, Gemini,
//! local) don't publish their tokenizers, so we fall back to `cl100k_base` as a
//! close approximation — the request record marks counts as `provider` vs
//! `estimate` so the two are never conflated.

use std::sync::OnceLock;

use serde_json::Value;
use tiktoken_rs::{cl100k_base, o200k_base, CoreBPE};

fn cl100k() -> &'static CoreBPE {
    static ENC: OnceLock<CoreBPE> = OnceLock::new();
    ENC.get_or_init(|| cl100k_base().expect("load cl100k_base tokenizer"))
}

fn o200k() -> &'static CoreBPE {
    static ENC: OnceLock<CoreBPE> = OnceLock::new();
    ENC.get_or_init(|| o200k_base().expect("load o200k_base tokenizer"))
}

/// Pick the tokenizer closest to `model`.
fn encoder_for(model: &str) -> &'static CoreBPE {
    let m = model.to_ascii_lowercase();
    let o200k_model = m.starts_with("gpt-4o")
        || m.starts_with("chatgpt-4o")
        || m.starts_with("gpt-4.1")
        || m.starts_with("gpt-5")
        || m.starts_with("o1")
        || m.starts_with("o3")
        || m.starts_with("o4");
    if o200k_model {
        o200k()
    } else {
        cl100k()
    }
}

/// Count tokens in `text` using the tokenizer closest to `model`.
pub fn count_text_tokens(model: &str, text: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }
    encoder_for(model).encode_ordinary(text).len() as u64
}

/// Count prompt tokens from an OpenAI-style `messages` array, using the
/// tokenizer for the request's model. Adds the small per-message framing
/// overhead chat templates insert (≈4/message + 3 priming), matching OpenAI's
/// own counting guidance.
pub fn estimate_prompt_tokens(request: &Value) -> u64 {
    let model = request.get("model").and_then(Value::as_str).unwrap_or("");
    let bpe = encoder_for(model);
    let count = |s: &str| bpe.encode_ordinary(s).len() as u64;

    let mut total = 0u64;
    if let Some(messages) = request.get("messages").and_then(Value::as_array) {
        for message in messages {
            total += 4; // role + delimiters
            match message.get("content") {
                Some(Value::String(s)) => total += count(s),
                // Multimodal content arrays: count any text parts.
                Some(Value::Array(parts)) => {
                    for part in parts {
                        if let Some(t) = part.get("text").and_then(Value::as_str) {
                            total += count(t);
                        }
                    }
                }
                _ => {}
            }
        }
        total += 3; // priming tokens for the assistant reply
    } else if let Some(prompt) = request.get("prompt").and_then(Value::as_str) {
        total += count(prompt);
    }
    total
}

/// Count completion tokens from a non-streaming chat response body.
pub fn estimate_completion_tokens(model: &str, response: &Value) -> u64 {
    let bpe = encoder_for(model);
    let mut total = 0u64;
    if let Some(choices) = response.get("choices").and_then(Value::as_array) {
        for choice in choices {
            if let Some(content) = choice
                .get("message")
                .and_then(|m| m.get("content"))
                .and_then(Value::as_str)
            {
                total += bpe.encode_ordinary(content).len() as u64;
            } else if let Some(text) = choice.get("text").and_then(Value::as_str) {
                total += bpe.encode_ordinary(text).len() as u64;
            }
        }
    }
    total
}

/// Where a token count came from. Recorded so estimates and ground truth are
/// never silently mixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenSource {
    /// Reported by the upstream provider's `usage` field.
    Provider,
    /// Counted by Joule's tokenizer.
    Estimated,
    /// Served from the response cache (no inference ran).
    Cache,
}

impl TokenSource {
    pub fn as_str(self) -> &'static str {
        match self {
            TokenSource::Provider => "provider",
            TokenSource::Estimated => "estimate",
            TokenSource::Cache => "cache",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn counts_real_tokens_not_chars() {
        // "hello world" is 2 tokens in cl100k/o200k, not 11/4≈3.
        assert_eq!(count_text_tokens("gpt-4o", "hello world"), 2);
        assert_eq!(count_text_tokens("gpt-4", "hello world"), 2);
    }

    #[test]
    fn prompt_tokens_include_message_overhead() {
        let req = json!({
            "model": "gpt-4o-mini",
            "messages": [{ "role": "user", "content": "hello world" }]
        });
        // 4 (overhead) + 2 (content) + 3 (priming) = 9
        assert_eq!(estimate_prompt_tokens(&req), 9);
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(count_text_tokens("gpt-4o", ""), 0);
        assert_eq!(count_text_tokens("claude-3-5-sonnet", ""), 0);
    }
}
