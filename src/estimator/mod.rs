//! Energy, electricity, carbon, and cost estimation for an inference.
//!
//! The estimator is deliberately a pure, side-effect-free calculation: given a
//! model and a token count it returns an [`EnergyEstimate`]. Everything that is
//! uncertain (per-token joules, grid carbon intensity) is an explicit input so
//! the estimate can be reproduced and refined.

pub mod models;

pub use models::{profile_for, ModelProfile};

/// Default carbon intensity of grid electricity, grams CO2-equivalent per kWh.
///
/// Set to the IEA 2024 global-average generation intensity (~445 g/kWh). For
/// reference: US grid ~321, EU ~174, US data centers ~548 (they cluster in
/// carbon-intensive regions), Norway hydro <20, Poland coal >700. Override
/// per-deployment with `--grid-intensity` for accuracy.
pub const DEFAULT_GRID_INTENSITY_G_PER_KWH: f64 = 445.0;

/// The estimated footprint of a single inference.
#[derive(Debug, Clone, Copy)]
pub struct EnergyEstimate {
    pub energy_j: f64,
    pub electricity_wh: f64,
    pub co2_g: f64,
    pub cost_usd: f64,
}

/// Stateless estimator parameterised by the local grid's carbon intensity.
#[derive(Debug, Clone, Copy)]
pub struct Estimator {
    grid_intensity_g_per_kwh: f64,
}

impl Estimator {
    /// Create an estimator for a grid with the given carbon intensity.
    pub fn new(grid_intensity_g_per_kwh: f64) -> Self {
        Self {
            grid_intensity_g_per_kwh,
        }
    }

    /// Estimate the footprint of an inference on `model` that consumed
    /// `input_tokens` of prompt and produced `output_tokens` of completion.
    pub fn estimate(&self, model: &str, input_tokens: u64, output_tokens: u64) -> EnergyEstimate {
        let profile = profile_for(model);
        self.estimate_with(&profile, input_tokens, output_tokens)
    }

    /// Estimate using an explicit profile (used by the `estimate` CLI command).
    pub fn estimate_with(
        &self,
        profile: &ModelProfile,
        input_tokens: u64,
        output_tokens: u64,
    ) -> EnergyEstimate {
        let energy_j = input_tokens as f64 * profile.j_per_input_token
            + output_tokens as f64 * profile.j_per_output_token;

        // 1 Wh = 3600 J.
        let electricity_wh = energy_j / 3600.0;

        // g = kWh * (g / kWh); kWh = Wh / 1000.
        let co2_g = (electricity_wh / 1000.0) * self.grid_intensity_g_per_kwh;

        let cost_usd = input_tokens as f64 / 1_000_000.0 * profile.usd_per_m_input
            + output_tokens as f64 / 1_000_000.0 * profile.usd_per_m_output;

        EnergyEstimate {
            energy_j,
            electricity_wh,
            co2_g,
            cost_usd,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joules_convert_to_watt_hours() {
        let est = Estimator::new(400.0);
        // A profile of exactly 3600 J/output-token over 1 token = 1 Wh.
        let profile = ModelProfile {
            family: "test",
            j_per_input_token: 0.0,
            j_per_output_token: 3600.0,
            usd_per_m_input: 0.0,
            usd_per_m_output: 0.0,
        };
        let e = est.estimate_with(&profile, 0, 1);
        assert!((e.electricity_wh - 1.0).abs() < 1e-9);
        // 1 Wh at 400 g/kWh = 0.4 g CO2.
        assert!((e.co2_g - 0.4).abs() < 1e-9);
    }

    #[test]
    fn output_tokens_cost_more_than_input() {
        let est = Estimator::new(DEFAULT_GRID_INTENSITY_G_PER_KWH);
        let only_input = est.estimate("gpt-4o", 1000, 0);
        let only_output = est.estimate("gpt-4o", 0, 1000);
        assert!(only_output.energy_j > only_input.energy_j);
    }
}
