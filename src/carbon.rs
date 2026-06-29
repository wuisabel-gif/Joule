//! Grid carbon intensity by region, for carbon-aware routing.
//!
//! The same kWh is not equally clean everywhere — published averages span from
//! ~30 g CO₂/kWh on hydro/nuclear grids to >600 on coal. This module holds a
//! region → gCO₂/kWh map, seeded from rough published averages, overridable in
//! config, and updatable at runtime (`set`) by a live source such as
//! ElectricityMaps or WattTime.

use std::collections::HashMap;
use std::sync::RwLock;

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

    /// Insert or replace a region's intensity — the hook a live refresher
    /// (e.g. an ElectricityMaps poller) calls. Not yet wired to a live source.
    #[allow(dead_code)]
    pub fn set(&self, region: &str, gco2_per_kwh: f64) {
        self.intensity
            .write()
            .expect("carbon map")
            .insert(region.to_ascii_lowercase(), gco2_per_kwh);
    }
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
}
