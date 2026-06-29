//! Semantic response cache.
//!
//! Where the exact cache matches byte-identical requests, the semantic cache
//! matches requests that *mean the same thing*: it embeds the prompt and returns
//! a cached answer if cosine similarity to a past prompt clears a threshold, so
//! "What is Newton's 2nd law?" and "explain Newton's second law" collapse to one
//! inference.
//!
//! It needs an OpenAI-compatible embeddings endpoint, so it is opt-in
//! (`--semantic-cache`). Each non-cached request then pays one small embedding
//! call to enable (much larger) generation hits.

use std::sync::Mutex;

use reqwest::header::AUTHORIZATION;
use serde_json::json;

use crate::cache::CachedResponse;

/// Cosine similarity of two equal-length vectors, in [-1, 1] (0 on mismatch).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Opt-in semantic cache: an embedder plus a bounded vector store.
pub struct SemanticCache {
    client: reqwest::Client,
    embed_base: String,
    embed_model: String,
    embed_key: Option<String>,
    threshold: f32,
    capacity: usize,
    store: Mutex<Vec<(Vec<f32>, CachedResponse)>>,
}

impl SemanticCache {
    pub fn new(
        client: reqwest::Client,
        embed_base: String,
        embed_model: String,
        embed_key: Option<String>,
        threshold: f32,
        capacity: usize,
    ) -> Self {
        Self {
            client,
            embed_base: embed_base.trim_end_matches('/').to_string(),
            embed_model,
            embed_key,
            threshold,
            capacity: capacity.max(1),
            store: Mutex::new(Vec::new()),
        }
    }

    /// Embed `text` via the configured OpenAI-compatible embeddings endpoint.
    /// Returns None on any failure (the request then proceeds uncached).
    pub async fn embed(&self, text: &str) -> Option<Vec<f32>> {
        let url = format!("{}/v1/embeddings", self.embed_base);
        let mut rb = self
            .client
            .post(url)
            .json(&json!({ "model": self.embed_model, "input": text }));
        if let Some(key) = &self.embed_key {
            rb = rb.header(AUTHORIZATION, format!("Bearer {key}"));
        }
        let resp = rb.send().await.ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let body: serde_json::Value = resp.json().await.ok()?;
        let arr = body.pointer("/data/0/embedding")?.as_array()?;
        let vec: Vec<f32> = arr
            .iter()
            .filter_map(|x| x.as_f64().map(|f| f as f32))
            .collect();
        (!vec.is_empty()).then_some(vec)
    }

    /// Return the cached response most similar to `query` if it clears the
    /// threshold, with the similarity score.
    pub fn lookup(&self, query: &[f32]) -> Option<(f32, CachedResponse)> {
        let store = self.store.lock().expect("semantic store");
        let mut best: Option<(f32, &CachedResponse)> = None;
        for (emb, resp) in store.iter() {
            let sim = cosine(query, emb);
            if best.is_none_or(|(b, _)| sim > b) {
                best = Some((sim, resp));
            }
        }
        match best {
            Some((sim, resp)) if sim >= self.threshold => Some((sim, resp.clone())),
            _ => None,
        }
    }

    /// Remember an embedding and its response (FIFO eviction at capacity).
    pub fn put(&self, embedding: Vec<f32>, response: CachedResponse) {
        let mut store = self.store.lock().expect("semantic store");
        if store.len() >= self.capacity {
            store.remove(0);
        }
        store.push((embedding, response));
    }
}

/// Extract the prompt text to embed from an OpenAI chat request.
pub fn prompt_text(request: &serde_json::Value) -> Option<String> {
    let messages = request.get("messages")?.as_array()?;
    let mut parts = Vec::new();
    for m in messages {
        match m.get("content") {
            Some(serde_json::Value::String(s)) => parts.push(s.clone()),
            Some(serde_json::Value::Array(items)) => {
                for it in items {
                    if let Some(t) = it.get("text").and_then(|v| v.as_str()) {
                        parts.push(t.to_string());
                    }
                }
            }
            _ => {}
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    fn entry() -> CachedResponse {
        CachedResponse {
            status: 200,
            content_type: "application/json".into(),
            body: Bytes::from_static(b"cached"),
            model: "gpt-4o-mini".into(),
            input_tokens: 5,
            output_tokens: 5,
        }
    }

    #[test]
    fn cosine_basics() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert_eq!(cosine(&[1.0], &[1.0, 2.0]), 0.0); // length mismatch
    }

    #[test]
    fn lookup_respects_threshold() {
        let sc = SemanticCache::new(
            reqwest::Client::new(),
            "http://x".into(),
            "m".into(),
            None,
            0.95,
            16,
        );
        sc.put(vec![1.0, 0.0, 0.0], entry());
        // Near-identical query → hit.
        assert!(sc.lookup(&[0.99, 0.01, 0.0]).is_some());
        // Orthogonal query → miss.
        assert!(sc.lookup(&[0.0, 1.0, 0.0]).is_none());
    }

    #[test]
    fn evicts_at_capacity() {
        let sc = SemanticCache::new(
            reqwest::Client::new(),
            "http://x".into(),
            "m".into(),
            None,
            0.5,
            2,
        );
        sc.put(vec![1.0, 0.0], entry());
        sc.put(vec![0.0, 1.0], entry());
        sc.put(vec![1.0, 1.0], entry()); // evicts first
        assert!(sc.lookup(&[1.0, 0.0]).is_some()); // matches the [1,1] or [0,1]
        assert_eq!(sc.store.lock().unwrap().len(), 2);
    }
}
