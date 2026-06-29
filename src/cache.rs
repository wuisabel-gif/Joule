//! Exact-match response cache.
//!
//! The cheapest inference is the one that never runs. Keyed by the canonical
//! request (model + messages + sampling params), the cache returns a
//! previously-computed response without touching the provider. A hit costs
//! ~0 J, $0, and near-zero latency; Joule reports the energy it *avoided*.
//!
//! This is an exact match: byte-identical requests (after JSON key
//! normalisation) hit. It is in-memory and bounded (LRU) and does not persist
//! across restarts. Streaming requests are never cached.
//!
//! Note: with `temperature > 0` a cached response is one prior sample replayed
//! verbatim, not a fresh draw — the expected, intended behaviour of an
//! exact-match cache. Disable it per-deployment with `--no-cache`.

use std::num::NonZeroUsize;
use std::sync::Mutex;

use bytes::Bytes;
use lru::LruCache;
use serde_json::Value;

/// A cached upstream response, replayed verbatim to the client on a hit.
#[derive(Clone)]
pub struct CachedResponse {
    pub status: u16,
    pub content_type: String,
    pub body: Bytes,
    /// Model that produced the response (used to price the avoided energy).
    pub model: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// Bounded in-memory exact-match cache. When disabled, every operation is a
/// no-op so callers need no branching.
pub struct Cache {
    inner: Option<Mutex<LruCache<Vec<u8>, CachedResponse>>>,
}

impl Cache {
    /// Create a cache. When `enabled` is false the cache is inert.
    pub fn new(enabled: bool, capacity: usize) -> Self {
        let inner = enabled.then(|| {
            let cap = NonZeroUsize::new(capacity.max(1)).expect("capacity >= 1");
            Mutex::new(LruCache::new(cap))
        });
        Self { inner }
    }

    pub fn enabled(&self) -> bool {
        self.inner.is_some()
    }

    /// Canonical cache key for a request. `serde_json` serialises object keys in
    /// sorted order, so formatting / key-order differences map to the same key.
    pub fn key(request: &Value) -> Option<Vec<u8>> {
        serde_json::to_vec(request).ok()
    }

    /// Look up a response, promoting it to most-recently-used on a hit.
    pub fn get(&self, key: &[u8]) -> Option<CachedResponse> {
        let mutex = self.inner.as_ref()?;
        let mut guard = mutex.lock().expect("cache mutex");
        guard.get(key).cloned()
    }

    /// Insert a response (no-op when the cache is disabled).
    pub fn put(&self, key: Vec<u8>, response: CachedResponse) {
        if let Some(mutex) = &self.inner {
            mutex.lock().expect("cache mutex").put(key, response);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn entry(body: &str) -> CachedResponse {
        CachedResponse {
            status: 200,
            content_type: "application/json".into(),
            body: Bytes::from(body.to_string()),
            model: "gpt-4o-mini".into(),
            input_tokens: 10,
            output_tokens: 5,
        }
    }

    #[test]
    fn hit_and_miss() {
        let cache = Cache::new(true, 8);
        let k = Cache::key(&json!({"model":"m","messages":[]})).unwrap();
        assert!(cache.get(&k).is_none());
        cache.put(k.clone(), entry("hello"));
        assert_eq!(cache.get(&k).unwrap().body, Bytes::from("hello"));
    }

    #[test]
    fn key_is_order_independent() {
        let a = Cache::key(&json!({"model":"m","stream":false})).unwrap();
        let b = Cache::key(&json!({"stream":false,"model":"m"})).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn disabled_cache_is_inert() {
        let cache = Cache::new(false, 8);
        assert!(!cache.enabled());
        let k = Cache::key(&json!({"x":1})).unwrap();
        cache.put(k.clone(), entry("x"));
        assert!(cache.get(&k).is_none());
    }

    #[test]
    fn evicts_least_recently_used() {
        let cache = Cache::new(true, 2);
        let k1 = Cache::key(&json!({"i":1})).unwrap();
        let k2 = Cache::key(&json!({"i":2})).unwrap();
        let k3 = Cache::key(&json!({"i":3})).unwrap();
        cache.put(k1.clone(), entry("1"));
        cache.put(k2.clone(), entry("2"));
        cache.put(k3.clone(), entry("3")); // evicts k1
        assert!(cache.get(&k1).is_none());
        assert!(cache.get(&k2).is_some());
        assert!(cache.get(&k3).is_some());
    }
}
