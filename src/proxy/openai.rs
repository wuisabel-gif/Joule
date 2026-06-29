//! Incremental parser for Server-Sent Events streams.
//!
//! As streamed chunks arrive we account for tokens and return the parsed
//! events, so the proxy can either forward them untouched (OpenAI-compatible)
//! or re-frame them into OpenAI chunks (Anthropic, Gemini). Event *semantics*
//! are delegated to the active [`Provider`] — they differ only in where the
//! content and usage fields live within each event.

use serde_json::Value;

use crate::provider::Provider;

/// Accumulates streamed SSE bytes and extracts token-accounting signals.
#[derive(Default)]
pub struct SseAccumulator {
    /// Bytes not yet terminated by a newline, carried across chunks.
    buffer: String,
    /// Concatenated assistant content.
    content: String,
    /// Prompt tokens, once a provider event reports them.
    prompt: Option<u64>,
    /// Completion tokens, once a provider event reports them.
    completion: Option<u64>,
}

impl SseAccumulator {
    /// Feed a raw chunk of stream bytes, using `provider` to interpret events.
    /// Returns the parsed `data:` JSON events in this chunk (for re-framing).
    pub fn feed(&mut self, chunk: &[u8], provider: &dyn Provider) -> Vec<Value> {
        self.buffer.push_str(&String::from_utf8_lossy(chunk));

        let mut events = Vec::new();
        while let Some(idx) = self.buffer.find('\n') {
            let line: String = self.buffer.drain(..=idx).collect();
            if let Some(event) = self.handle_line(line.trim_end(), provider) {
                events.push(event);
            }
        }
        events
    }

    fn handle_line(&mut self, line: &str, provider: &dyn Provider) -> Option<Value> {
        let payload = line.strip_prefix("data:")?.trim();
        if payload.is_empty() || payload == "[DONE]" {
            return None;
        }
        let event = serde_json::from_str::<Value>(payload).ok()?;

        if let Some(text) = provider.stream_content_delta(&event) {
            self.content.push_str(&text);
        }
        if let Some(p) = provider.stream_prompt_tokens(&event) {
            self.prompt = Some(p);
        }
        if let Some(c) = provider.stream_completion_tokens(&event) {
            self.completion = Some(c);
        }
        Some(event)
    }

    /// Provider-reported usage, available only once both counts were seen.
    pub fn usage(&self) -> Option<(u64, u64)> {
        match (self.prompt, self.completion) {
            (Some(p), Some(c)) => Some((p, c)),
            _ => None,
        }
    }

    /// The accumulated assistant text, for heuristic token estimation.
    pub fn content(&self) -> &str {
        &self.content
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::OpenAiCompatibleProvider;

    fn provider() -> OpenAiCompatibleProvider {
        OpenAiCompatibleProvider::new("test".into(), "http://x".into(), None, vec![])
    }

    #[test]
    fn accumulates_content_across_split_chunks() {
        let p = provider();
        let mut acc = SseAccumulator::default();
        acc.feed(b"data: {\"choices\":[{\"delta\":{\"content\":\"Hel", &p);
        acc.feed(b"lo\"}}]}\n\n", &p);
        acc.feed(
            b"data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
            &p,
        );
        acc.feed(b"data: [DONE]\n\n", &p);
        assert_eq!(acc.content(), "Hello world");
        assert_eq!(acc.usage(), None);
    }

    #[test]
    fn captures_usage_from_final_event() {
        let p = provider();
        let mut acc = SseAccumulator::default();
        acc.feed(
            b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n\n",
            &p,
        );
        assert_eq!(acc.usage(), Some((10, 5)));
    }
}
