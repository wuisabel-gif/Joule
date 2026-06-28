//! Google Gemini provider.
//!
//! Translates the canonical OpenAI chat-completions request into Gemini's
//! `generateContent` format and maps the response back, so an OpenAI-speaking
//! client can transparently reach Gemini models through Joule.
//!
//! Notable shape differences handled here:
//! - the model and call type are encoded in the URL
//!   (`/v1beta/models/{model}:generateContent`), not the body;
//! - roles are `user` / `model` (OpenAI's `assistant` becomes `model`);
//! - the system prompt lives in a top-level `systemInstruction`;
//! - generation limits live under `generationConfig`;
//! - usage is reported as `usageMetadata.{promptTokenCount,candidatesTokenCount}`.
//!
//! Non-streaming requests are fully translated in both directions. Streaming
//! requests use `:streamGenerateContent?alt=sse` and are forwarded in Gemini's
//! native SSE format (token accounting still works via the stream hooks).

use axum::http::header::AUTHORIZATION;
use axum::http::HeaderMap;
use serde_json::{json, Map, Value};

use super::{Provider, ProviderError};

const API_KEY_HEADER: &str = "x-goog-api-key";
const DEFAULT_BASE: &str = "https://generativelanguage.googleapis.com";

/// A provider speaking Google's Gemini `generateContent` API.
pub struct GeminiProvider {
    name: String,
    base_url: String,
    api_key: Option<String>,
    model_prefixes: Vec<String>,
}

impl GeminiProvider {
    pub fn new(
        name: String,
        base_url: String,
        api_key: Option<String>,
        model_prefixes: Vec<String>,
    ) -> Self {
        let base = base_url.trim_end_matches('/');
        let base = if base.is_empty() { DEFAULT_BASE } else { base };
        let prefixes = if model_prefixes.is_empty() {
            vec!["gemini".to_string()]
        } else {
            model_prefixes
        };
        Self {
            name,
            base_url: base.to_string(),
            api_key,
            model_prefixes: prefixes,
        }
    }

    /// Translate an OpenAI chat request into a Gemini `generateContent` body.
    fn to_gemini(&self, canonical: &Value) -> Value {
        let mut contents = Vec::new();
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
                    let gemini_role = if role == "assistant" { "model" } else { "user" };
                    contents.push(json!({
                        "role": gemini_role,
                        "parts": [{ "text": text }],
                    }));
                }
            }
        }

        let mut generation = Map::new();
        if let Some(max) = canonical.get("max_tokens").and_then(Value::as_u64) {
            generation.insert("maxOutputTokens".into(), json!(max));
        }
        if let Some(t) = canonical.get("temperature") {
            generation.insert("temperature".into(), t.clone());
        }
        if let Some(p) = canonical.get("top_p") {
            generation.insert("topP".into(), p.clone());
        }

        let mut body = Map::new();
        body.insert("contents".into(), json!(contents));
        if !system_parts.is_empty() {
            body.insert(
                "systemInstruction".into(),
                json!({ "parts": [{ "text": system_parts.join("\n\n") }] }),
            );
        }
        if !generation.is_empty() {
            body.insert("generationConfig".into(), Value::Object(generation));
        }
        Value::Object(body)
    }
}

impl Provider for GeminiProvider {
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
        if let Some(key) = client_headers.get(API_KEY_HEADER) {
            rb.header(API_KEY_HEADER, key.clone())
        } else if let Some(bearer) = client_headers
            .get(AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.strip_prefix("Bearer "))
        {
            rb.header(API_KEY_HEADER, bearer.to_string())
        } else if let Some(key) = &self.api_key {
            rb.header(API_KEY_HEADER, key.clone())
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
        let is_stream = canonical
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let (method, suffix) = if is_stream {
            ("streamGenerateContent", "?alt=sse")
        } else {
            ("generateContent", "")
        };
        let url = format!(
            "{}/v1beta/models/{}:{}{}",
            self.base_url, model, method, suffix
        );
        let body = self.to_gemini(canonical);
        let rb = client.post(url).json(&body);
        Ok(self.authorize(rb, client_headers))
    }

    fn usage_from_body(&self, body: &Value) -> Option<(u64, u64)> {
        let usage = body.get("usageMetadata")?;
        Some((
            usage.get("promptTokenCount")?.as_u64()?,
            usage.get("candidatesTokenCount")?.as_u64()?,
        ))
    }

    fn translate_response(&self, body: Value) -> Value {
        let text = body
            .pointer("/candidates/0/content/parts")
            .and_then(Value::as_array)
            .map(|parts| {
                parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(Value::as_str))
                    .collect::<String>()
            })
            .unwrap_or_default();

        let (input, output) = self.usage_from_body(&body).unwrap_or((0, 0));
        let finish = match body
            .pointer("/candidates/0/finishReason")
            .and_then(Value::as_str)
        {
            Some("STOP") => "stop",
            Some("MAX_TOKENS") => "length",
            Some("SAFETY") | Some("RECITATION") => "content_filter",
            _ => "stop",
        };

        json!({
            "id": "gemini",
            "object": "chat.completion",
            "model": body.get("modelVersion").cloned().unwrap_or(Value::Null),
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
        event
            .pointer("/candidates/0/content/parts/0/text")
            .and_then(Value::as_str)
            .map(str::to_string)
    }

    fn stream_prompt_tokens(&self, event: &Value) -> Option<u64> {
        event
            .pointer("/usageMetadata/promptTokenCount")
            .and_then(Value::as_u64)
    }

    fn stream_completion_tokens(&self, event: &Value) -> Option<u64> {
        event
            .pointer("/usageMetadata/candidatesTokenCount")
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

    fn provider() -> GeminiProvider {
        GeminiProvider::new("gemini".into(), DEFAULT_BASE.into(), None, vec![])
    }

    #[test]
    fn translates_request_roles_and_system() {
        let p = provider();
        let canonical = json!({
            "model": "gemini-1.5-flash",
            "messages": [
                { "role": "system", "content": "Be brief." },
                { "role": "user", "content": "Hi" },
                { "role": "assistant", "content": "Hello" }
            ],
            "max_tokens": 32
        });
        let out = p.to_gemini(&canonical);
        assert_eq!(out["systemInstruction"]["parts"][0]["text"], "Be brief.");
        assert_eq!(out["contents"][0]["role"], "user");
        assert_eq!(out["contents"][1]["role"], "model");
        assert_eq!(out["contents"][1]["parts"][0]["text"], "Hello");
        assert_eq!(out["generationConfig"]["maxOutputTokens"], 32);
    }

    #[test]
    fn translates_response_to_openai_shape() {
        let p = provider();
        let resp = json!({
            "candidates": [{
                "content": { "role": "model", "parts": [{ "text": "Hello" }] },
                "finishReason": "STOP"
            }],
            "usageMetadata": { "promptTokenCount": 7, "candidatesTokenCount": 2 }
        });
        let out = p.translate_response(resp);
        assert_eq!(out["object"], "chat.completion");
        assert_eq!(out["choices"][0]["message"]["content"], "Hello");
        assert_eq!(out["choices"][0]["finish_reason"], "stop");
        assert_eq!(out["usage"]["prompt_tokens"], 7);
        assert_eq!(out["usage"]["total_tokens"], 9);
    }

    #[test]
    fn stream_url_uses_sse() {
        let p = provider();
        let canonical = json!({ "stream": true, "messages": [] });
        let rb = p
            .build_chat_request(
                &reqwest::Client::new(),
                &canonical,
                "gemini-1.5-pro",
                &HeaderMap::new(),
            )
            .unwrap();
        let req = rb.build().unwrap();
        assert!(req.url().as_str().contains(":streamGenerateContent"));
        assert_eq!(req.url().query(), Some("alt=sse"));
    }
}
