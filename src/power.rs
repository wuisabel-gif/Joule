//! Measured power for self-hosted inference (NVIDIA Jetson via `tegrastats`).
//!
//! Everything else in Joule is *estimated* from tokens. On hardware you control,
//! we can do better: sample the board's real power draw during a request and
//! integrate it into **measured** joules, reported next to the estimate. On a
//! Jetson the on-board INA3221 monitor is exposed through `tegrastats`, whose
//! output carries per-rail power like `VDD_GPU_SOC 3200mW/3210mW`.
//!
//! A background task runs `tegrastats --interval <ms>`, sums the configured
//! rails each sample, and accumulates energy into an atomic counter. A request
//! reads the counter before and after; the delta is its measured energy. This is
//! exact for serial local inference; under concurrent requests the whole-board
//! draw is attributed to each, so treat measured energy as per-*window*, not a
//! clean per-request split, when several run at once.
//!
//! Rails differ across Jetson models and JetPack versions, so the rail set is
//! configurable and the sampler logs the rails it actually sees on first sample
//! — if none of the configured rails match, it warns instead of silently
//! reporting zero. On Orin AGX, `VDD_GPU_SOC` + `VDD_CPU_CV` is compute power;
//! `VIN_SYS_5V0` is whole-board input.

use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tracing::{info, warn};

use crate::metrics::Metrics;

/// Default rails summed on a Jetson Orin AGX (compute power). Override in config
/// with whatever `tegrastats` prints on your board (see the startup log).
pub fn default_rails() -> Vec<String> {
    vec!["VDD_GPU_SOC".to_string(), "VDD_CPU_CV".to_string()]
}

/// Accumulated measured energy from the power sampler. Cheap to read from the
/// request path; written only by the background sampler task.
pub struct PowerMeter {
    energy_mj: AtomicU64,
    rails: Vec<String>,
}

impl PowerMeter {
    pub fn new(rails: Vec<String>) -> Self {
        Self {
            energy_mj: AtomicU64::new(0),
            rails,
        }
    }

    /// Total measured energy since start, in millijoules.
    pub fn energy_mj(&self) -> u64 {
        self.energy_mj.load(Ordering::Relaxed)
    }

    /// Rails this meter sums.
    pub fn rails(&self) -> &[String] {
        &self.rails
    }

    /// Add one sample: `power_mw` held over `interval_ms` → energy in mJ.
    fn add_sample(&self, power_mw: u32, interval_ms: u64) {
        let mj = (power_mw as u64).saturating_mul(interval_ms) / 1000;
        self.energy_mj.fetch_add(mj, Ordering::Relaxed);
    }
}

/// Parse a `tegrastats` line into `(rail, milliwatts)` pairs. A rail appears as
/// a label token followed by a `<cur>mW/<avg>mW` token; we take the current value.
pub fn parse_tegrastats(line: &str) -> Vec<(String, u32)> {
    let toks: Vec<&str> = line.split_whitespace().collect();
    let mut out = Vec::new();
    for i in 1..toks.len() {
        if let Some(mw) = parse_power_value(toks[i]) {
            out.push((toks[i - 1].to_string(), mw));
        }
    }
    out
}

/// `"3200mW/3210mW"` → `Some(3200)`; anything else → `None`.
fn parse_power_value(tok: &str) -> Option<u32> {
    let (cur, avg) = tok.split_once("mW/")?;
    if !avg.ends_with("mW") {
        return None;
    }
    cur.parse::<u32>().ok()
}

/// Sum the milliwatts of the named rails present in a parsed sample. Returns
/// `None` if none of them appear (so the caller can distinguish "0 W" from
/// "wrong rail names").
pub fn sum_rails(parsed: &[(String, u32)], rails: &[String]) -> Option<u32> {
    let mut total = 0u32;
    let mut found = false;
    for (name, mw) in parsed {
        if rails.iter().any(|r| r == name) {
            total = total.saturating_add(*mw);
            found = true;
        }
    }
    found.then_some(total)
}

/// Spawn the background `tegrastats` sampler. Feeds [`PowerMeter`] and exports
/// the live board power gauge. On spawn failure it logs and disables measured
/// power (the request path just never sees a delta) — it never takes serving down.
pub fn spawn_sampler(
    meter: Arc<PowerMeter>,
    metrics: Arc<Metrics>,
    path: String,
    interval_ms: u64,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut child = match Command::new(&path)
            .arg("--interval")
            .arg(interval_ms.to_string())
            .stdout(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                warn!(tool = %path, error = %e, "could not start tegrastats; measured power disabled");
                return;
            }
        };
        let Some(stdout) = child.stdout.take() else {
            warn!("tegrastats produced no stdout; measured power disabled");
            return;
        };

        let mut lines = BufReader::new(stdout).lines();
        let mut first = true;
        while let Ok(Some(line)) = lines.next_line().await {
            let parsed = parse_tegrastats(&line);
            if first {
                let seen: Vec<&String> = parsed.iter().map(|(n, _)| n).collect();
                info!(rails = ?seen, using = ?meter.rails(), "tegrastats power rails");
                if sum_rails(&parsed, meter.rails()).is_none() {
                    warn!(
                        configured = ?meter.rails(),
                        available = ?seen,
                        "none of the configured power rails are present — measured energy \
                         will stay 0; set `power_rails` to one of the available rails",
                    );
                }
                first = false;
            }
            if let Some(mw) = sum_rails(&parsed, meter.rails()) {
                meter.add_sample(mw, interval_ms);
                metrics.set_board_power(mw as f64 / 1000.0);
            }
        }
        warn!("tegrastats stream ended; measured power will no longer update");
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // A representative Jetson Orin AGX tegrastats line (JetPack 5.x).
    const LINE: &str = "RAM 4096/30536MB (lfb 200x4MB) SWAP 0/15268MB (cached 0MB) \
        CPU [10%@2201,5%@2201,off,off] GR3D_FREQ 0% cpu@50C soc0@49C \
        VDD_GPU_SOC 3200mW/3210mW VDD_CPU_CV 1600mW/1610mW VIN_SYS_5V0 5000mW/5010mW \
        VDDQ_VDD2_1V8AO 400mW/400mW";

    #[test]
    fn parses_rails_from_a_real_line() {
        let p = parse_tegrastats(LINE);
        let get = |name: &str| p.iter().find(|(n, _)| n == name).map(|(_, v)| *v);
        assert_eq!(get("VDD_GPU_SOC"), Some(3200));
        assert_eq!(get("VDD_CPU_CV"), Some(1600));
        assert_eq!(get("VIN_SYS_5V0"), Some(5000));
        assert_eq!(get("VDDQ_VDD2_1V8AO"), Some(400));
        // Non-power tokens are ignored.
        assert!(get("GR3D_FREQ").is_none());
        assert!(get("RAM").is_none());
    }

    #[test]
    fn sums_only_configured_rails() {
        let p = parse_tegrastats(LINE);
        assert_eq!(sum_rails(&p, &default_rails()), Some(4800)); // GPU_SOC + CPU_CV
        assert_eq!(sum_rails(&p, &["VIN_SYS_5V0".into()]), Some(5000)); // whole board
        assert_eq!(sum_rails(&p, &["NO_SUCH_RAIL".into()]), None); // wrong label → None
    }

    #[test]
    fn integrates_energy_over_samples() {
        let m = PowerMeter::new(default_rails());
        // 5000 mW for 200 ms = 1000 mJ = 1 J.
        m.add_sample(5000, 200);
        assert_eq!(m.energy_mj(), 1000);
        // + 10000 mW for 100 ms = 1000 mJ → 2 J total.
        m.add_sample(10000, 100);
        assert_eq!(m.energy_mj(), 2000);
    }

    #[test]
    fn ignores_malformed_power_tokens() {
        assert_eq!(parse_power_value("3200mW/3210mW"), Some(3200));
        assert_eq!(parse_power_value("3200mW"), None); // no avg half
        assert_eq!(parse_power_value("50C"), None);
        assert_eq!(parse_power_value("xmW/ymW"), None);
    }
}
