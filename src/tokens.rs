//! Lightweight token accounting.
//!
//! Phase 1 prefers the provider's reported `usage` whenever it is present in
//! the upstream response — that is ground truth. This module provides a cheap
//! heuristic used only when the provider does not report usage (e.g. some
//! streaming responses, or local models that omit it).
//!
//! The heuristic is the well-known "~4 characters per token" rule. It is an
//! approximation, not a tokenizer; the request record marks whether a count
//! came from the provider or from estimation so the two are never conflated.

use serde_json::Value;

/// Approximate the number of tokens in a string (~4 chars/token).
pub fn approx_tokens(text: &str) -> u64 {
    let chars = text.chars().count() as u64;
    // Round up, and never report 0 tokens for non-empty text.
    if chars == 0 {
        0
    } else {
        chars.div_ceil(4)
    }
}

/// Estimate prompt tokens from an OpenAI-style `messages` array.
///
/// Adds a small fixed per-message overhead to approximate the role / framing
/// tokens that chat templates insert.
pub fn estimate_prompt_tokens(request: &Value) -> u64 {
    let mut total = 0u64;
    if let Some(messages) = request.get("messages").and_then(Value::as_array) {
        for message in messages {
            // Per-message structural overhead (role + delimiters).
            total += 4;
            match message.get("content") {
                Some(Value::String(s)) => total += approx_tokens(s),
                // Multimodal content arrays: count any text parts.
                Some(Value::Array(parts)) => {
                    for part in parts {
                        if let Some(t) = part.get("text").and_then(Value::as_str) {
                            total += approx_tokens(t);
                        }
                    }
                }
                _ => {}
            }
        }
        total += 3; // priming tokens for the assistant reply.
    } else if let Some(prompt) = request.get("prompt").and_then(Value::as_str) {
        // Legacy completions endpoint.
        total += approx_tokens(prompt);
    }
    total
}

/// Estimate completion tokens from a non-streaming chat response body.
pub fn estimate_completion_tokens(response: &Value) -> u64 {
    let mut total = 0u64;
    if let Some(choices) = response.get("choices").and_then(Value::as_array) {
        for choice in choices {
            if let Some(content) = choice
                .get("message")
                .and_then(|m| m.get("content"))
                .and_then(Value::as_str)
            {
                total += approx_tokens(content);
            } else if let Some(text) = choice.get("text").and_then(Value::as_str) {
                total += approx_tokens(text);
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
    /// Computed by Joule's heuristic.
    Estimated,
}

impl TokenSource {
    pub fn as_str(self) -> &'static str {
        match self {
            TokenSource::Provider => "provider",
            TokenSource::Estimated => "estimate",
        }
    }
}
