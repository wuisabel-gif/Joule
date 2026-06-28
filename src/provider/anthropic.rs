//! Anthropic provider.
//!
//! Translates the canonical OpenAI chat-completions request into Anthropic's
//! `/v1/messages` format and maps the response back, so an OpenAI-speaking
//! client can transparently reach Claude models through Joule.
//!
//! Non-streaming requests are fully translated in both directions. Streaming
//! requests are forwarded in Anthropic's native SSE format (token accounting
//! still works via the stream hooks); re-framing the stream into OpenAI's
//! `chat.completion.chunk` events is a follow-up.

use axum::http::header::AUTHORIZATION;
use axum::http::HeaderMap;
use serde_json::{json, Map, Value};

use super::{Provider, ProviderError};

const DEFAULT_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u64 = 1024;

/// A provider speaking Anthropic's Messages API.
pub struct AnthropicProvider {
    name: String,
    base_url: String,
    api_key: Option<String>,
    version: String,
    model_prefixes: Vec<String>,
}

impl AnthropicProvider {
    pub fn new(
        name: String,
        base_url: String,
        api_key: Option<String>,
        model_prefixes: Vec<String>,
        version: Option<String>,
    ) -> Self {
        let prefixes = if model_prefixes.is_empty() {
            vec!["claude".to_string()]
        } else {
            model_prefixes
        };
        Self {
            name,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            version: version.unwrap_or_else(|| DEFAULT_VERSION.to_string()),
            model_prefixes: prefixes,
        }
    }

    /// Translate an OpenAI chat request into an Anthropic Messages request.
    fn to_anthropic(&self, canonical: &Value, model: &str) -> Value {
        let mut messages = Vec::new();
        let mut system_parts = Vec::new();

        if let Some(arr) = canonical.get("messages").and_then(Value::as_array) {
            for m in arr {
                let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
                let text = extract_text(m.get("content"));
                if role == "system" {
                    if !text.is_empty() {
                        system_parts.push(text);
                    }
                } else {
                    messages.push(json!({
                        "role": role,
                        "content": [{ "type": "text", "text": text }],
                    }));
                }
            }
        }

        let max_tokens = canonical
            .get("max_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_MAX_TOKENS);

        let mut body = Map::new();
        body.insert("model".into(), json!(model));
        body.insert("max_tokens".into(), json!(max_tokens));
        body.insert("messages".into(), json!(messages));
        if !system_parts.is_empty() {
            body.insert("system".into(), json!(system_parts.join("\n\n")));
        }
        for key in ["temperature", "top_p", "stream"] {
            if let Some(v) = canonical.get(key) {
                body.insert(key.into(), v.clone());
            }
        }
        Value::Object(body)
    }
}

impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn supports_model(&self, model: &str) -> bool {
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
        let rb = rb.header("anthropic-version", &self.version);
        if let Some(key) = client_headers.get("x-api-key") {
            rb.header("x-api-key", key.clone())
        } else if let Some(bearer) = client_headers
            .get(AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.strip_prefix("Bearer "))
        {
            rb.header("x-api-key", bearer.to_string())
        } else if let Some(key) = &self.api_key {
            rb.header("x-api-key", key.clone())
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
        let body = self.to_anthropic(canonical, model);
        let url = format!("{}/v1/messages", self.base_url);
        let rb = client.post(url).json(&body);
        Ok(self.authorize(rb, client_headers))
    }

    fn usage_from_body(&self, body: &Value) -> Option<(u64, u64)> {
        let usage = body.get("usage")?;
        Some((
            usage.get("input_tokens")?.as_u64()?,
            usage.get("output_tokens")?.as_u64()?,
        ))
    }

    fn translate_response(&self, body: Value) -> Value {
        // Map Anthropic message response -> OpenAI chat.completion.
        let text = body
            .get("content")
            .and_then(Value::as_array)
            .map(|parts| {
                parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(Value::as_str))
                    .collect::<String>()
            })
            .unwrap_or_default();

        let (input, output) = self.usage_from_body(&body).unwrap_or((0, 0));
        let finish = match body.get("stop_reason").and_then(Value::as_str) {
            Some("end_turn") | Some("stop_sequence") => "stop",
            Some("max_tokens") => "length",
            other => other.unwrap_or("stop"),
        };

        json!({
            "id": body.get("id").cloned().unwrap_or(json!("msg")),
            "object": "chat.completion",
            "model": body.get("model").cloned().unwrap_or(Value::Null),
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": text },
                "finish_reason": finish,
            }],
            "usage": {
                "prompt_tokens": input,
                "completion_tokens": output,
                "total_tokens": input + output,
            },
        })
    }

    fn stream_content_delta(&self, event: &Value) -> Option<String> {
        // content_block_delta events carry { delta: { type: "text_delta", text } }.
        event
            .pointer("/delta/text")
            .and_then(Value::as_str)
            .map(str::to_string)
    }

    fn stream_prompt_tokens(&self, event: &Value) -> Option<u64> {
        // message_start: { message: { usage: { input_tokens } } }.
        event
            .pointer("/message/usage/input_tokens")
            .and_then(Value::as_u64)
    }

    fn stream_completion_tokens(&self, event: &Value) -> Option<u64> {
        // message_delta: { usage: { output_tokens } }.
        event
            .pointer("/usage/output_tokens")
            .and_then(Value::as_u64)
    }
}

/// Extract plain text from an OpenAI `content` field (string or parts array).
fn extract_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_request_system_and_messages() {
        let p = AnthropicProvider::new(
            "anthropic".into(),
            "https://api.anthropic.com".into(),
            None,
            vec![],
            None,
        );
        let canonical = json!({
            "model": "ignored",
            "messages": [
                { "role": "system", "content": "Be terse." },
                { "role": "user", "content": "Hi" }
            ],
            "max_tokens": 64
        });
        let out = p.to_anthropic(&canonical, "claude-3-5-sonnet");
        assert_eq!(out["model"], "claude-3-5-sonnet");
        assert_eq!(out["max_tokens"], 64);
        assert_eq!(out["system"], "Be terse.");
        assert_eq!(out["messages"][0]["role"], "user");
        assert_eq!(out["messages"][0]["content"][0]["text"], "Hi");
    }

    #[test]
    fn translates_response_to_openai_shape() {
        let p = AnthropicProvider::new(
            "anthropic".into(),
            "https://api.anthropic.com".into(),
            None,
            vec![],
            None,
        );
        let resp = json!({
            "id": "msg_1",
            "model": "claude-3-5-sonnet",
            "content": [{ "type": "text", "text": "Hello" }],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 12, "output_tokens": 3 }
        });
        let out = p.translate_response(resp);
        assert_eq!(out["object"], "chat.completion");
        assert_eq!(out["choices"][0]["message"]["content"], "Hello");
        assert_eq!(out["choices"][0]["finish_reason"], "stop");
        assert_eq!(out["usage"]["prompt_tokens"], 12);
        assert_eq!(out["usage"]["total_tokens"], 15);
    }
}
