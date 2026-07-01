//! Per-provider circuit breakers.
//!
//! When an upstream keeps failing, forwarding more requests just piles up
//! timeouts and drags Joule down with it. A circuit breaker trips after a run of
//! failures and then **fails fast** — returning immediately without calling the
//! upstream — giving the struggling provider room to recover. After a cooldown it
//! lets a trial request through; success closes it, failure re-opens it.
//!
//! This is a consecutive-failure breaker: `threshold` failures in a row open it
//! for `cooldown`; any success resets the count.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

struct State {
    consecutive_failures: u32,
    open_until: Option<Instant>,
}

/// A single provider's breaker.
pub struct CircuitBreaker {
    threshold: u32,
    cooldown: Duration,
    state: Mutex<State>,
}

impl CircuitBreaker {
    pub fn new(threshold: u32, cooldown: Duration) -> Self {
        Self {
            threshold: threshold.max(1),
            cooldown,
            state: Mutex::new(State {
                consecutive_failures: 0,
                open_until: None,
            }),
        }
    }

    /// May a request proceed now? False only while open and still cooling down;
    /// once the cooldown elapses a trial request is allowed through.
    pub fn allow(&self) -> bool {
        let state = self.state.lock().expect("breaker");
        match state.open_until {
            Some(until) => Instant::now() >= until,
            None => true,
        }
    }

    /// Is the breaker currently open (tripped and cooling down)?
    pub fn is_open(&self) -> bool {
        let state = self.state.lock().expect("breaker");
        matches!(state.open_until, Some(until) if Instant::now() < until)
    }

    /// Record a successful call — closes the breaker.
    pub fn record_success(&self) {
        let mut state = self.state.lock().expect("breaker");
        state.consecutive_failures = 0;
        state.open_until = None;
    }

    /// Record a failed call — opens the breaker once the threshold is reached.
    pub fn record_failure(&self) {
        let mut state = self.state.lock().expect("breaker");
        state.consecutive_failures += 1;
        if state.consecutive_failures >= self.threshold {
            state.open_until = Some(Instant::now() + self.cooldown);
        }
    }
}

/// One circuit breaker per provider.
pub struct Breakers {
    map: HashMap<String, CircuitBreaker>,
}

impl Breakers {
    pub fn new<I: IntoIterator<Item = String>>(
        providers: I,
        threshold: u32,
        cooldown: Duration,
    ) -> Self {
        let map = providers
            .into_iter()
            .map(|name| (name, CircuitBreaker::new(threshold, cooldown)))
            .collect();
        Self { map }
    }

    /// The breaker for `provider`, if one is configured.
    pub fn get(&self, provider: &str) -> Option<&CircuitBreaker> {
        self.map.get(provider)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opens_after_threshold_and_denies() {
        let b = CircuitBreaker::new(3, Duration::from_secs(30));
        assert!(b.allow());
        b.record_failure();
        b.record_failure();
        assert!(b.allow(), "still closed below threshold");
        b.record_failure(); // 3rd → opens
        assert!(b.is_open());
        assert!(!b.allow(), "open breaker denies");
    }

    #[test]
    fn success_resets_failure_count() {
        let b = CircuitBreaker::new(2, Duration::from_secs(30));
        b.record_failure();
        b.record_success();
        b.record_failure();
        assert!(
            b.allow(),
            "count reset by success, so one failure doesn't open"
        );
        assert!(!b.is_open());
    }

    #[test]
    fn reopens_and_recovers_after_cooldown() {
        let b = CircuitBreaker::new(1, Duration::from_millis(20));
        b.record_failure(); // opens immediately (threshold 1)
        assert!(!b.allow());
        std::thread::sleep(Duration::from_millis(30));
        assert!(b.allow(), "cooldown elapsed → trial allowed");
        b.record_success(); // trial succeeds → closed
        assert!(!b.is_open());
        assert!(b.allow());
    }
}
