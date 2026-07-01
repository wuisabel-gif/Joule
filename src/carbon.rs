//! Grid carbon intensity by region, for carbon-aware routing.
//!
//! The same kWh is not equally clean everywhere — published averages span from
//! ~30 g CO₂/kWh on hydro/nuclear grids to >600 on coal. This module holds a
//! region → gCO₂/kWh map, seeded from rough published averages, overridable in
//! config, and updatable at runtime (`set`) by a **live feed**.
//!
//! The live feed is pluggable ([`CarbonFeed`]): the static table is always the
//! default (no token, always works), and an optional HTTP source can refresh it
//! in the background. Three sources are understood out of the box — the free
//! [UK Carbon Intensity API] (no token), [CO2 Signal], and [Electricity Maps] —
//! and they all reduce to "GET → a gCO₂/kWh number", so adding another is a
//! small match arm. This keeps carbon routing useful without committing to any
//! one paid/trial API: point it at the free UK feed to validate the pipeline,
//! or at Electricity Maps if you have a key, and it degrades to the static
//! table when no feed is configured.
//!
//! [UK Carbon Intensity API]: https://carbonintensity.org.uk
//! [CO2 Signal]: https://www.co2signal.com
//! [Electricity Maps]: https://www.electricitymaps.com

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use serde::Deserialize;
use serde_json::Value;
use tracing::{info, warn};

use crate::metrics::Metrics;

/// Built-in regional averages (g CO₂-equivalent per kWh). Rough, public-data
/// figures — override per-deployment for accuracy, or refresh from a live API.
const BUILTIN: &[(&str, f64)] = &[
    ("iceland", 28.0), // geothermal + hydro
    ("norway", 30.0),  // hydro
    ("quebec", 32.0),  // hydro
    ("sweden", 45.0),
    ("france", 60.0), // nuclear
    ("eu-north", 50.0),
    ("oregon", 120.0),
    ("uk", 200.0),
    ("us-west", 210.0), // CAISO-ish
    ("eu-west", 230.0),
    ("us-central", 430.0),
    ("us-east", 380.0), // PJM-ish
    ("germany", 380.0),
    ("australia", 530.0),
    ("india", 630.0),
    ("poland", 650.0), // coal
];

/// Thread-safe region → carbon-intensity map.
pub struct CarbonMap {
    intensity: RwLock<HashMap<String, f64>>,
    /// Used when a region is unknown.
    fallback: f64,
}

impl CarbonMap {
    /// Build from the built-in table plus `overrides`, with `fallback` for
    /// unknown regions.
    pub fn new(fallback: f64, overrides: &HashMap<String, f64>) -> Self {
        let mut map: HashMap<String, f64> =
            BUILTIN.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        for (k, v) in overrides {
            map.insert(k.to_ascii_lowercase(), *v);
        }
        Self {
            intensity: RwLock::new(map),
            fallback,
        }
    }

    /// Carbon intensity for `region` (g CO₂/kWh), or the fallback if unknown.
    pub fn intensity(&self, region: &str) -> f64 {
        self.intensity
            .read()
            .expect("carbon map")
            .get(&region.to_ascii_lowercase())
            .copied()
            .unwrap_or(self.fallback)
    }

    /// Insert or replace a region's intensity — the hook the live feed poller
    /// ([`spawn_poller`]) calls after each successful fetch.
    pub fn set(&self, region: &str, gco2_per_kwh: f64) {
        self.intensity
            .write()
            .expect("carbon map")
            .insert(region.to_ascii_lowercase(), gco2_per_kwh);
    }
}

/// A live carbon-intensity source. Each variant knows one public API's URL
/// shape, auth, and response field; all reduce to a gCO₂/kWh number per zone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CarbonSourceKind {
    /// UK Carbon Intensity API — free, no token, national grid only.
    Uk,
    /// CO2 Signal (Electricity Maps' community API) — free token, per country.
    Co2signal,
    /// Electricity Maps v3 — token (trial/paid), per zone.
    ElectricityMaps,
}

impl CarbonSourceKind {
    /// Does this source require an auth token?
    pub fn needs_token(self) -> bool {
        !matches!(self, CarbonSourceKind::Uk)
    }

    /// Default API base URL for the source.
    fn default_base(self) -> &'static str {
        match self {
            CarbonSourceKind::Uk => "https://api.carbonintensity.org.uk",
            CarbonSourceKind::Co2signal => "https://api.co2signal.com/v1",
            CarbonSourceKind::ElectricityMaps => "https://api.electricitymap.org/v3",
        }
    }
}

/// An HTTP carbon-intensity feed: one source, polled per zone.
pub struct CarbonFeed {
    client: reqwest::Client,
    kind: CarbonSourceKind,
    base_url: String,
    token: Option<String>,
}

impl CarbonFeed {
    pub fn new(
        client: reqwest::Client,
        kind: CarbonSourceKind,
        base_url: Option<String>,
        token: Option<String>,
    ) -> Self {
        Self {
            client,
            kind,
            base_url: base_url.unwrap_or_else(|| kind.default_base().to_string()),
            token,
        }
    }

    pub fn kind(&self) -> CarbonSourceKind {
        self.kind
    }

    /// Fetch the current intensity (gCO₂eq/kWh) for a source-specific `zone`
    /// code (e.g. "GB", "NO", "FR"). The UK source ignores the zone (national).
    pub async fn fetch(&self, zone: &str) -> Result<f64, String> {
        let base = self.base_url.trim_end_matches('/');
        let url = match self.kind {
            CarbonSourceKind::Uk => format!("{base}/intensity"),
            CarbonSourceKind::Co2signal => format!("{base}/latest?countryCode={zone}"),
            CarbonSourceKind::ElectricityMaps => {
                format!("{base}/carbon-intensity/latest?zone={zone}")
            }
        };

        let mut req = self.client.get(&url);
        if let Some(token) = &self.token {
            // Both Electricity Maps and CO2 Signal use the `auth-token` header.
            req = req.header("auth-token", token);
        }
        let resp = req.send().await.map_err(|e| format!("request: {e}"))?;
        let status = resp.status();
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("decoding {status}: {e}"))?;
        if !status.is_success() {
            return Err(format!("{status}: {body}"));
        }

        extract_intensity(self.kind, &body)
            .ok_or_else(|| format!("no intensity field in response: {body}"))
    }
}

/// Pull the gCO₂/kWh value out of a source's response body.
fn extract_intensity(kind: CarbonSourceKind, body: &Value) -> Option<f64> {
    match kind {
        // { "data": [ { "intensity": { "actual": 123, "forecast": 130 } } ] }
        CarbonSourceKind::Uk => body
            .pointer("/data/0/intensity/actual")
            .filter(|v| !v.is_null())
            .or_else(|| body.pointer("/data/0/intensity/forecast")),
        // { "data": { "carbonIntensity": 123 } }
        CarbonSourceKind::Co2signal => body.pointer("/data/carbonIntensity"),
        // { "carbonIntensity": 123 }
        CarbonSourceKind::ElectricityMaps => body.pointer("/carbonIntensity"),
    }
    .and_then(Value::as_f64)
}

/// Spawn a background task that refreshes `map` from `feed` every `interval`.
///
/// `zones` maps each of Joule's region keys to the source's zone code; after a
/// successful fetch the region's entry in the shared [`CarbonMap`] is updated so
/// the `carbon` router sees live numbers. A failed fetch is logged and the last
/// known value (live or static) is kept — the feed never takes routing down.
/// The first refresh runs immediately.
pub fn spawn_poller(
    map: Arc<CarbonMap>,
    metrics: Arc<Metrics>,
    feed: CarbonFeed,
    zones: Vec<(String, String)>,
    interval: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            ticker.tick().await;
            for (region, zone) in &zones {
                match feed.fetch(zone).await {
                    Ok(gco2) => {
                        map.set(region, gco2);
                        metrics.set_grid_intensity(region, gco2);
                        info!(
                            source = ?feed.kind(),
                            region = %region,
                            zone = %zone,
                            gco2_per_kwh = gco2,
                            "refreshed grid carbon intensity",
                        );
                    }
                    Err(e) => warn!(
                        source = ?feed.kind(),
                        region = %region,
                        zone = %zone,
                        error = %e,
                        "carbon feed fetch failed; keeping last known value",
                    ),
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_and_overrides_and_fallback() {
        let mut ov = HashMap::new();
        ov.insert("us-west".to_string(), 99.0);
        let m = CarbonMap::new(445.0, &ov);
        assert_eq!(m.intensity("norway"), 30.0); // builtin
        assert_eq!(m.intensity("us-west"), 99.0); // override wins
        assert_eq!(m.intensity("mars"), 445.0); // fallback
    }

    #[test]
    fn set_updates_intensity() {
        let m = CarbonMap::new(445.0, &HashMap::new());
        m.set("us-east", 120.0);
        assert_eq!(m.intensity("us-east"), 120.0);
    }

    #[test]
    fn extract_intensity_per_source() {
        use serde_json::json;

        let uk = json!({"data": [{"intensity": {"actual": 123, "forecast": 130}}]});
        assert_eq!(extract_intensity(CarbonSourceKind::Uk, &uk), Some(123.0));

        // UK sometimes reports null `actual` (current half-hour) — fall back.
        let uk_pending = json!({"data": [{"intensity": {"actual": null, "forecast": 130}}]});
        assert_eq!(
            extract_intensity(CarbonSourceKind::Uk, &uk_pending),
            Some(130.0)
        );

        let co2 = json!({"data": {"carbonIntensity": 246.0}, "countryCode": "GB"});
        assert_eq!(
            extract_intensity(CarbonSourceKind::Co2signal, &co2),
            Some(246.0)
        );

        let em = json!({"zone": "NO", "carbonIntensity": 31});
        assert_eq!(
            extract_intensity(CarbonSourceKind::ElectricityMaps, &em),
            Some(31.0)
        );

        let missing = json!({"error": "invalid zone"});
        assert_eq!(
            extract_intensity(CarbonSourceKind::ElectricityMaps, &missing),
            None
        );
    }

    #[test]
    fn uk_needs_no_token_others_do() {
        assert!(!CarbonSourceKind::Uk.needs_token());
        assert!(CarbonSourceKind::Co2signal.needs_token());
        assert!(CarbonSourceKind::ElectricityMaps.needs_token());
    }
}
