//! Measured power for self-hosted inference.
//!
//! Everything else in Joule is *estimated* from tokens. On hardware you control,
//! we can do better: sample the machine's real power draw during a request and
//! integrate it into **measured** joules, reported next to the estimate.
//!
//! The measurement is one pluggable piece — a [`PowerSource`] — behind a shared
//! [`PowerMeter`]. Sources differ only in the tool they run and how they parse
//! its output; the meter, the energy integration, and the proxy reporting are
//! identical for all of them:
//!
//! - [`PowerSource::Tegrastats`]  — NVIDIA Jetson on-board INA3221 monitor.
//! - [`PowerSource::Powermetrics`] — Apple Silicon (macOS; needs sudo).
//! - [`PowerSource::NvidiaSmi`]    — desktop/server NVIDIA GPUs.
//!
//! A background task runs the source's tool on an interval, sums the configured
//! power rails each sample, and accumulates energy into an atomic counter; a
//! request reads the counter before and after and takes the delta. This is exact
//! for serial local inference; under concurrent requests the whole-machine draw
//! is attributed to each, so measured energy is per-*window*, cleanest when
//! requests run one at a time. Rails/labels differ by machine, so they're
//! configurable and logged on startup — a mismatch warns instead of silently
//! reporting zero.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use clap::ValueEnum;
use serde::Deserialize;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tracing::{info, warn};

use crate::metrics::Metrics;

/// Where measured power comes from. Each variant knows one tool's command,
/// sampling flag, output shape, and sensible default rails.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum PowerSource {
    /// NVIDIA Jetson `tegrastats` (one line per sample, rails inline).
    #[default]
    Tegrastats,
    /// Apple Silicon `powermetrics` (multi-line block per sample; needs sudo).
    Powermetrics,
    /// NVIDIA `nvidia-smi` power.draw polling (one wattage per line).
    NvidiaSmi,
}

impl PowerSource {
    /// Rails summed when the user doesn't configure `power_rails`. These sets
    /// don't overlap, so summing them is total compute power (no double count).
    pub fn default_rails(self) -> Vec<String> {
        match self {
            // Orin AGX compute rails; use VIN_SYS_5V0 for whole-board input.
            PowerSource::Tegrastats => vec!["VDD_GPU_SOC".into(), "VDD_CPU_CV".into()],
            // powermetrics reports these as "CPU/GPU/ANE Power: N mW".
            PowerSource::Powermetrics => vec!["CPU".into(), "GPU".into(), "ANE".into()],
            // nvidia-smi reports a single wattage; rails don't apply.
            PowerSource::NvidiaSmi => Vec::new(),
        }
    }

    /// Command + args to launch the sampler for `interval_ms`. `path` overrides
    /// the tool's binary (the Jetson/NVIDIA tools) where applicable.
    fn command(self, path: Option<&str>, interval_ms: u64) -> (String, Vec<String>) {
        match self {
            PowerSource::Tegrastats => (
                path.unwrap_or("tegrastats").to_string(),
                vec!["--interval".into(), interval_ms.to_string()],
            ),
            // Run under `sudo -n` (non-interactive): powermetrics needs root, so
            // add a NOPASSWD sudoers line scoped to powermetrics. Fails fast (and
            // we fall back to estimates) if that isn't set up.
            PowerSource::Powermetrics => (
                "sudo".to_string(),
                vec![
                    "-n".into(),
                    path.unwrap_or("powermetrics").to_string(),
                    "--samplers".into(),
                    "cpu_power,gpu_power".into(),
                    "-i".into(),
                    interval_ms.to_string(),
                ],
            ),
            PowerSource::NvidiaSmi => (
                path.unwrap_or("nvidia-smi").to_string(),
                vec![
                    "--query-gpu=power.draw".into(),
                    "--format=csv,noheader,nounits".into(),
                    "-lms".into(),
                    interval_ms.to_string(),
                ],
            ),
        }
    }
}

/// Accumulated measured energy from the power sampler. Cheap to read from the
/// request path; written only by the background sampler task.
pub struct PowerMeter {
    energy_mj: AtomicU64,
    source: PowerSource,
    rails: Vec<String>,
}

impl PowerMeter {
    /// `rails` empty ⇒ the source's default rails.
    pub fn new(source: PowerSource, rails: Vec<String>) -> Self {
        let rails = if rails.is_empty() {
            source.default_rails()
        } else {
            rails
        };
        Self {
            energy_mj: AtomicU64::new(0),
            source,
            rails,
        }
    }

    /// Total measured energy since start, in millijoules.
    pub fn energy_mj(&self) -> u64 {
        self.energy_mj.load(Ordering::Relaxed)
    }

    pub fn source(&self) -> PowerSource {
        self.source
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
        if let Some(mw) = tegrastats_power(toks[i]) {
            out.push((toks[i - 1].to_string(), mw));
        }
    }
    out
}

/// `"3200mW/3210mW"` → `Some(3200)`; anything else → `None`.
fn tegrastats_power(tok: &str) -> Option<u32> {
    let (cur, avg) = tok.split_once("mW/")?;
    avg.ends_with("mW").then_some(())?;
    cur.parse::<u32>().ok()
}

/// Parse one `powermetrics` line like `"GPU Power: 1234 mW"` → `("GPU", 1234)`.
/// Ignores the `"Combined Power (…): N mW"` summary line (would double-count).
pub fn parse_powermetrics(line: &str) -> Option<(String, u32)> {
    let (name, rest) = line.trim().split_once(" Power:")?;
    let mw = rest.trim().strip_suffix("mW")?.trim().parse::<u32>().ok()?;
    Some((name.trim().to_string(), mw))
}

/// Parse one `nvidia-smi` power.draw line (`"74.50"`, watts) → milliwatts.
pub fn parse_nvidia_smi(line: &str) -> Option<u32> {
    let watts: f64 = line.trim().parse().ok()?;
    (watts >= 0.0).then_some((watts * 1000.0) as u32)
}

/// Sum the milliwatts of the named rails present in a parsed sample. Returns
/// `None` if none appear (so the caller can tell "0 W" from "wrong rail names").
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

/// Accumulates a `powermetrics` multi-line sample block into one summed value.
/// `feed` returns `Some(mw)` for a completed block (emitted when the *next*
/// block's `*** Sampled …` header arrives).
struct PowermetricsAccumulator {
    rails: Vec<String>,
    current: HashMap<String, u32>,
    last_rails: Vec<String>,
}

impl PowermetricsAccumulator {
    fn new(rails: Vec<String>) -> Self {
        Self {
            rails,
            current: HashMap::new(),
            last_rails: Vec::new(),
        }
    }

    fn feed(&mut self, line: &str) -> Option<u32> {
        let mut out = None;
        if line.starts_with("*** Sampled") && !self.current.is_empty() {
            self.last_rails = self.current.keys().cloned().collect();
            let parsed: Vec<(String, u32)> = self.current.drain().collect();
            out = sum_rails(&parsed, &self.rails);
        }
        if let Some((name, mw)) = parse_powermetrics(line) {
            self.current.insert(name, mw);
        }
        out
    }

    /// Rail names seen in the most recently flushed block (for startup logging).
    fn last_rails(&self) -> Vec<String> {
        self.last_rails.clone()
    }
}

/// Spawn the background sampler for `source`. Feeds [`PowerMeter`] and exports
/// the live board-power gauge. On spawn failure it logs and disables measured
/// power — it never takes serving down.
pub fn spawn_sampler(
    meter: Arc<PowerMeter>,
    metrics: Arc<Metrics>,
    path: Option<String>,
    interval_ms: u64,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let source = meter.source();
        let (program, args) = source.command(path.as_deref(), interval_ms);
        let mut child = match Command::new(&program)
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                let hint = if source == PowerSource::Powermetrics {
                    " (powermetrics needs a NOPASSWD sudoers entry)"
                } else {
                    ""
                };
                warn!(source = ?source, tool = %program, error = %e,
                    "could not start power sampler; measured power disabled{hint}");
                return;
            }
        };
        let Some(stdout) = child.stdout.take() else {
            warn!("power sampler produced no stdout; measured power disabled");
            return;
        };

        let mut lines = BufReader::new(stdout).lines();
        let record = |mw: u32| {
            meter.add_sample(mw, interval_ms);
            metrics.set_board_power(mw as f64 / 1000.0);
        };

        match source {
            PowerSource::Tegrastats => {
                let mut first = true;
                while let Ok(Some(line)) = lines.next_line().await {
                    let parsed = parse_tegrastats(&line);
                    if first {
                        log_rails(&meter, parsed.iter().map(|(n, _)| n.clone()).collect());
                        first = false;
                    }
                    if let Some(mw) = sum_rails(&parsed, meter.rails()) {
                        record(mw);
                    }
                }
            }
            PowerSource::Powermetrics => {
                // powermetrics emits a multi-line block per sample; the
                // accumulator flushes one summed sample when the next block starts.
                let mut acc = PowermetricsAccumulator::new(meter.rails().to_vec());
                let mut first = true;
                while let Ok(Some(line)) = lines.next_line().await {
                    if let Some(mw) = acc.feed(&line) {
                        if first {
                            log_rails(&meter, acc.last_rails());
                            first = false;
                        }
                        record(mw);
                    }
                }
            }
            PowerSource::NvidiaSmi => {
                let mut first = true;
                while let Ok(Some(line)) = lines.next_line().await {
                    if let Some(mw) = parse_nvidia_smi(&line) {
                        if first {
                            info!(source = ?source, "sampling GPU power.draw");
                            first = false;
                        }
                        record(mw);
                    }
                }
            }
        }
        warn!("power sampler stream ended; measured power will no longer update");
    })
}

/// Log the rails a source exposed on first sample, and warn if none of the
/// configured rails matched (measured energy would otherwise stay silently 0).
fn log_rails(meter: &PowerMeter, seen: Vec<String>) {
    info!(source = ?meter.source(), rails = ?seen, using = ?meter.rails(), "power rails");
    let present = seen.iter().any(|s| meter.rails().iter().any(|r| r == s));
    if !present {
        warn!(
            configured = ?meter.rails(), available = ?seen,
            "none of the configured power rails are present — measured energy will \
             stay 0; set `power_rails` to one of the available rails",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A representative Jetson Orin AGX tegrastats line (JetPack 5.x).
    const TEGRA: &str = "RAM 4096/30536MB (lfb 200x4MB) SWAP 0/15268MB (cached 0MB) \
        CPU [10%@2201,5%@2201,off,off] GR3D_FREQ 0% cpu@50C soc0@49C \
        VDD_GPU_SOC 3200mW/3210mW VDD_CPU_CV 1600mW/1610mW VIN_SYS_5V0 5000mW/5010mW \
        VDDQ_VDD2_1V8AO 400mW/400mW";

    #[test]
    fn tegrastats_parses_and_sums() {
        let p = parse_tegrastats(TEGRA);
        let get = |n: &str| p.iter().find(|(k, _)| k == n).map(|(_, v)| *v);
        assert_eq!(get("VDD_GPU_SOC"), Some(3200));
        assert_eq!(get("VIN_SYS_5V0"), Some(5000));
        assert!(get("GR3D_FREQ").is_none());
        assert_eq!(
            sum_rails(&p, &["VDD_GPU_SOC".into(), "VDD_CPU_CV".into()]),
            Some(4800)
        );
        assert_eq!(sum_rails(&p, &["NOPE".into()]), None);
    }

    #[test]
    fn powermetrics_parses_named_rails() {
        assert_eq!(
            parse_powermetrics("GPU Power: 1234 mW"),
            Some(("GPU".into(), 1234))
        );
        assert_eq!(
            parse_powermetrics("CPU Power: 5678 mW"),
            Some(("CPU".into(), 5678))
        );
        assert_eq!(
            parse_powermetrics("ANE Power: 0 mW"),
            Some(("ANE".into(), 0))
        );
        // The summary line must be ignored (else CPU+GPU+ANE double-counts).
        assert_eq!(
            parse_powermetrics("Combined Power (CPU + GPU + ANE): 6912 mW"),
            None
        );
        assert_eq!(parse_powermetrics("GPU active residency: 42%"), None);
    }

    #[test]
    fn nvidia_smi_watts_to_mw() {
        assert_eq!(parse_nvidia_smi("74.50"), Some(74500));
        assert_eq!(parse_nvidia_smi(" 0 "), Some(0));
        assert_eq!(parse_nvidia_smi("[N/A]"), None);
    }

    #[test]
    fn default_rails_per_source() {
        assert_eq!(PowerSource::Tegrastats.default_rails().len(), 2);
        assert_eq!(
            PowerSource::Powermetrics.default_rails(),
            vec!["CPU", "GPU", "ANE"]
        );
        assert!(PowerSource::NvidiaSmi.default_rails().is_empty());
        // Empty configured rails ⇒ source defaults.
        let m = PowerMeter::new(PowerSource::Powermetrics, vec![]);
        assert_eq!(m.rails(), ["CPU", "GPU", "ANE"]);
    }

    #[test]
    fn powermetrics_accumulator_flushes_per_block() {
        let mut acc = PowermetricsAccumulator::new(vec!["CPU".into(), "GPU".into(), "ANE".into()]);
        let block1 = [
            "*** Sampled system activity (t1) ***",
            "GPU Power: 1500 mW",
            "CPU Power: 3000 mW",
            "ANE Power: 0 mW",
            "Combined Power (CPU + GPU + ANE): 4500 mW",
        ];
        // First block accumulates but doesn't flush (nothing precedes it).
        for l in block1 {
            assert_eq!(acc.feed(l), None);
        }
        // The next block's header flushes block 1: 1500 + 3000 + 0 = 4500.
        assert_eq!(acc.feed("*** Sampled system activity (t2) ***"), Some(4500));
        assert_eq!(acc.last_rails().len(), 3);
    }

    #[test]
    fn integrates_energy_over_samples() {
        let m = PowerMeter::new(PowerSource::Tegrastats, vec!["VDD_GPU_SOC".into()]);
        m.add_sample(5000, 200); // 5 W × 0.2 s = 1 J
        assert_eq!(m.energy_mj(), 1000);
        m.add_sample(10000, 100); // + 10 W × 0.1 s = 1 J
        assert_eq!(m.energy_mj(), 2000);
    }
}
