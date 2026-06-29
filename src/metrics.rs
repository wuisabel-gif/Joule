//! Prometheus metrics registry.
//!
//! Exposes the Phase 1 signal set: requests, tokens, energy, carbon, cost, and
//! latency — all labelled by model so per-model efficiency is observable.

use prometheus::{
    CounterVec, Encoder, HistogramOpts, HistogramVec, IntCounterVec, Opts, Registry, TextEncoder,
};

/// All Joule metrics plus the registry that renders them.
pub struct Metrics {
    registry: Registry,
    requests_total: IntCounterVec,
    input_tokens_total: IntCounterVec,
    output_tokens_total: IntCounterVec,
    energy_joules_total: CounterVec,
    co2_grams_total: CounterVec,
    cost_usd_total: CounterVec,
    latency_seconds: HistogramVec,
    tokens_saved_total: IntCounterVec,
    energy_saved_joules_total: CounterVec,
    cache_hits_total: IntCounterVec,
}

impl Metrics {
    pub fn new() -> Self {
        let registry = Registry::new();

        let requests_total = IntCounterVec::new(
            Opts::new("joule_requests_total", "Total proxied inference requests."),
            &["model", "status"],
        )
        .expect("valid metric");

        let input_tokens_total = IntCounterVec::new(
            Opts::new("joule_input_tokens_total", "Total prompt tokens."),
            &["model"],
        )
        .expect("valid metric");

        let output_tokens_total = IntCounterVec::new(
            Opts::new("joule_output_tokens_total", "Total completion tokens."),
            &["model"],
        )
        .expect("valid metric");

        let energy_joules_total = CounterVec::new(
            Opts::new("joule_energy_joules_total", "Estimated energy in joules."),
            &["model"],
        )
        .expect("valid metric");

        let co2_grams_total = CounterVec::new(
            Opts::new("joule_co2_grams_total", "Estimated CO2 emissions in grams."),
            &["model"],
        )
        .expect("valid metric");

        let cost_usd_total = CounterVec::new(
            Opts::new("joule_cost_usd_total", "Estimated provider cost in USD."),
            &["model"],
        )
        .expect("valid metric");

        let latency_seconds = HistogramVec::new(
            HistogramOpts::new("joule_latency_seconds", "Upstream request latency.")
                .buckets(vec![0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]),
            &["model"],
        )
        .expect("valid metric");

        let tokens_saved_total = IntCounterVec::new(
            Opts::new(
                "joule_prompt_tokens_saved_total",
                "Prompt tokens removed by optimization.",
            ),
            &["model"],
        )
        .expect("valid metric");

        let energy_saved_joules_total = CounterVec::new(
            Opts::new(
                "joule_energy_saved_joules_total",
                "Estimated energy saved by optimization and cache hits, in joules.",
            ),
            &["model"],
        )
        .expect("valid metric");

        let cache_hits_total = IntCounterVec::new(
            Opts::new(
                "joule_cache_hits_total",
                "Requests served from the exact-match cache (no inference).",
            ),
            &["model"],
        )
        .expect("valid metric");

        registry
            .register(Box::new(requests_total.clone()))
            .expect("register");
        registry
            .register(Box::new(input_tokens_total.clone()))
            .expect("register");
        registry
            .register(Box::new(output_tokens_total.clone()))
            .expect("register");
        registry
            .register(Box::new(energy_joules_total.clone()))
            .expect("register");
        registry
            .register(Box::new(co2_grams_total.clone()))
            .expect("register");
        registry
            .register(Box::new(cost_usd_total.clone()))
            .expect("register");
        registry
            .register(Box::new(latency_seconds.clone()))
            .expect("register");
        registry
            .register(Box::new(tokens_saved_total.clone()))
            .expect("register");
        registry
            .register(Box::new(energy_saved_joules_total.clone()))
            .expect("register");
        registry
            .register(Box::new(cache_hits_total.clone()))
            .expect("register");

        Self {
            registry,
            requests_total,
            input_tokens_total,
            output_tokens_total,
            energy_joules_total,
            co2_grams_total,
            cost_usd_total,
            latency_seconds,
            tokens_saved_total,
            energy_saved_joules_total,
            cache_hits_total,
        }
    }

    /// Record one completed request.
    #[allow(clippy::too_many_arguments)]
    pub fn observe(
        &self,
        model: &str,
        status: u16,
        input_tokens: u64,
        output_tokens: u64,
        energy_j: f64,
        co2_g: f64,
        cost_usd: f64,
        latency_secs: f64,
    ) {
        let status = status.to_string();
        self.requests_total
            .with_label_values(&[model, &status])
            .inc();
        self.input_tokens_total
            .with_label_values(&[model])
            .inc_by(input_tokens);
        self.output_tokens_total
            .with_label_values(&[model])
            .inc_by(output_tokens);
        self.energy_joules_total
            .with_label_values(&[model])
            .inc_by(energy_j);
        self.co2_grams_total
            .with_label_values(&[model])
            .inc_by(co2_g);
        self.cost_usd_total
            .with_label_values(&[model])
            .inc_by(cost_usd);
        self.latency_seconds
            .with_label_values(&[model])
            .observe(latency_secs);
    }

    /// Record a cache hit and the energy it avoided.
    pub fn observe_cache_hit(&self, model: &str, energy_saved_j: f64) {
        self.cache_hits_total.with_label_values(&[model]).inc();
        if energy_saved_j > 0.0 {
            self.energy_saved_joules_total
                .with_label_values(&[model])
                .inc_by(energy_saved_j);
        }
    }

    /// Record prompt-optimization savings for a request.
    pub fn observe_savings(&self, model: &str, tokens_saved: u64, energy_saved_j: f64) {
        if tokens_saved > 0 {
            self.tokens_saved_total
                .with_label_values(&[model])
                .inc_by(tokens_saved);
        }
        if energy_saved_j > 0.0 {
            self.energy_saved_joules_total
                .with_label_values(&[model])
                .inc_by(energy_saved_j);
        }
    }

    /// Render the registry in the Prometheus text exposition format.
    pub fn render(&self) -> String {
        let mut buf = Vec::new();
        let families = self.registry.gather();
        TextEncoder::new()
            .encode(&families, &mut buf)
            .expect("encode metrics");
        String::from_utf8(buf).expect("utf8 metrics")
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}
