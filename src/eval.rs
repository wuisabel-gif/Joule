//! `joule eval` — does routing/optimization keep answer quality?
//!
//! Joule's core promise is "less energy, same answer". This harness puts a
//! number behind it: run a set of prompts through a **baseline** endpoint and a
//! **treatment** endpoint (typically the same Joule instance with a policy on vs
//! off), compare the energy each reports (`x-joule-energy-j`), and — if a judge
//! model is configured — score whether the treatment's answer is as good as the
//! baseline's. Out comes an energy-saved figure next to a quality figure, so a
//! routing/optimizer change can be accepted on evidence rather than faith.

use std::time::Duration;

use anyhow::{Context, Result};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde_json::{json, Value};

use crate::cli::EvalArgs;

/// One prompt run through both endpoints (and maybe the judge).
struct Trial {
    energy_a: Option<f64>,
    energy_b: Option<f64>,
    score: Option<u32>,
    ok: bool,
}

/// Aggregate outcome across all trials.
struct Summary {
    trials: usize,
    ok: usize,
    energy_a: f64,
    energy_b: f64,
    saved_pct: Option<f64>,
    mean_score: Option<f64>,
    pass_rate: Option<f64>,
}

pub async fn eval(args: EvalArgs) -> Result<()> {
    let prompts = read_prompts(&args.prompts)?;
    if prompts.is_empty() {
        anyhow::bail!("no prompts found in {}", args.prompts);
    }
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(args.timeout))
        .build()
        .context("building HTTP client")?;

    println!("Evaluating {} prompt(s):", prompts.len());
    println!("  baseline:  {} ({})", args.baseline, args.baseline_model);
    println!("  treatment: {} ({})", args.treatment, args.treatment_model);
    if let Some(j) = &args.judge {
        println!("  judge:     {} ({})", j, args.judge_model);
    }
    println!();

    let mut trials = Vec::new();
    for prompt in &prompts {
        let a = chat(&client, &args, &args.baseline, &args.baseline_model, prompt).await;
        let b = chat(
            &client,
            &args,
            &args.treatment,
            &args.treatment_model,
            prompt,
        )
        .await;

        let (answer_a, energy_a) = a.unwrap_or((String::new(), None));
        let (answer_b, energy_b) = b.clone().unwrap_or((String::new(), None));
        let ok = b.is_some() && !answer_a.is_empty();

        let score = if let (Some(judge), true) = (&args.judge, ok) {
            judge_pair(&client, &args, judge, prompt, &answer_a, &answer_b).await
        } else {
            None
        };

        println!(
            "  {:<44} {:>10}  {:>10}  {}",
            snippet(prompt, 44),
            fmt_j(energy_a),
            fmt_j(energy_b),
            score.map(|s| format!("judge {s}/100")).unwrap_or_default(),
        );
        trials.push(Trial {
            energy_a,
            energy_b,
            score,
            ok,
        });
    }

    print_summary(&summarize(&trials, args.judge_threshold));
    Ok(())
}

/// Send one chat completion, returning `(answer_text, energy_j_from_header)`.
async fn chat(
    client: &reqwest::Client,
    args: &EvalArgs,
    base: &str,
    model: &str,
    prompt: &str,
) -> Option<(String, Option<f64>)> {
    let url = format!("{}/v1/chat/completions", base.trim_end_matches('/'));
    let mut rb = client
        .post(url)
        .header(CONTENT_TYPE, "application/json")
        .json(&json!({
            "model": model,
            "messages": [{ "role": "user", "content": prompt }],
            "stream": false,
        }));
    if let Some(k) = &args.api_key {
        rb = rb.header(AUTHORIZATION, format!("Bearer {k}"));
    }
    if let Some(k) = &args.proxy_key {
        rb = rb.header("x-joule-key", k);
    }

    let resp = rb.send().await.ok()?;
    let energy = resp
        .headers()
        .get("x-joule-energy-j")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<f64>().ok());
    if !resp.status().is_success() {
        return None;
    }
    let body: Value = resp.json().await.ok()?;
    let answer = body
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    Some((answer, energy))
}

/// Ask the judge whether answer B is at least as good as A. Returns 0–100.
async fn judge_pair(
    client: &reqwest::Client,
    args: &EvalArgs,
    judge: &str,
    prompt: &str,
    a: &str,
    b: &str,
) -> Option<u32> {
    let instruction = format!(
        "You are grading two answers to a task. Reply with ONLY a single integer \
         0-100: how good is ANSWER B compared with ANSWER A (the reference)? \
         100 = B is as good or better, 50 = noticeably worse, 0 = wrong or useless.\n\n\
         TASK:\n{prompt}\n\nANSWER A:\n{a}\n\nANSWER B:\n{b}"
    );
    let (answer, _) = chat(client, args, judge, &args.judge_model, &instruction).await?;
    parse_score(&answer)
}

/// First integer in the text, clamped to 0–100. Judges sometimes add prose.
fn parse_score(s: &str) -> Option<u32> {
    let mut digits = String::new();
    for c in s.chars() {
        if c.is_ascii_digit() {
            digits.push(c);
        } else if !digits.is_empty() {
            break;
        }
    }
    digits.parse::<u32>().ok().map(|v| v.min(100))
}

/// Aggregate energy and (if judged) quality across trials.
fn summarize(trials: &[Trial], threshold: u32) -> Summary {
    let ok = trials.iter().filter(|t| t.ok).count();

    // Energy is only comparable for trials where both sides reported it.
    let paired: Vec<(f64, f64)> = trials
        .iter()
        .filter_map(|t| Some((t.energy_a?, t.energy_b?)))
        .collect();
    let energy_a: f64 = paired.iter().map(|(a, _)| a).sum();
    let energy_b: f64 = paired.iter().map(|(_, b)| b).sum();
    let saved_pct = (energy_a > 0.0).then(|| (energy_a - energy_b) / energy_a * 100.0);

    let scores: Vec<u32> = trials.iter().filter_map(|t| t.score).collect();
    let mean_score = (!scores.is_empty())
        .then(|| scores.iter().map(|&s| s as f64).sum::<f64>() / scores.len() as f64);
    let pass_rate = (!scores.is_empty()).then(|| {
        scores.iter().filter(|&&s| s >= threshold).count() as f64 / scores.len() as f64 * 100.0
    });

    Summary {
        trials: trials.len(),
        ok,
        energy_a,
        energy_b,
        saved_pct,
        mean_score,
        pass_rate,
    }
}

fn print_summary(s: &Summary) {
    println!("\n{:-<72}", "");
    println!("Trials:          {} ({} completed)", s.trials, s.ok);
    println!(
        "Energy:          baseline {:.1} J → treatment {:.1} J",
        s.energy_a, s.energy_b
    );
    if let Some(pct) = s.saved_pct {
        let verb = if pct >= 0.0 { "saved" } else { "added" };
        println!("Energy {verb}:     {:.1}%", pct.abs());
    }
    match (s.mean_score, s.pass_rate) {
        (Some(mean), Some(pass)) => {
            println!("Quality (judge): mean {mean:.0}/100, {pass:.0}% at or above threshold");
            let verdict = if pass >= 90.0 && s.saved_pct.unwrap_or(0.0) > 0.0 {
                "✓ energy down, quality held"
            } else if pass < 90.0 {
                "⚠ quality dropped on some prompts — inspect the low scorers"
            } else {
                "no energy change"
            };
            println!("Verdict:         {verdict}");
        }
        _ => println!("Quality:         no judge configured (--judge to score answers)"),
    }
}

fn read_prompts(path: &str) -> Result<Vec<String>> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("reading prompts {path}"))?;
    Ok(raw
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect())
}

fn snippet(s: &str, max: usize) -> String {
    let one_line = s.replace('\n', " ");
    if one_line.chars().count() <= max {
        one_line
    } else {
        let cut: String = one_line.chars().take(max - 1).collect();
        format!("{cut}…")
    }
}

fn fmt_j(e: Option<f64>) -> String {
    e.map(|v| format!("{v:.2} J"))
        .unwrap_or_else(|| "n/a".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_score_extracts_first_integer() {
        assert_eq!(parse_score("85"), Some(85));
        assert_eq!(parse_score("Score: 90."), Some(90));
        assert_eq!(parse_score("100/100 great"), Some(100));
        assert_eq!(parse_score("250"), Some(100)); // clamped
        assert_eq!(parse_score("no number here"), None);
    }

    fn trial(a: f64, b: f64, score: Option<u32>) -> Trial {
        Trial {
            energy_a: Some(a),
            energy_b: Some(b),
            score,
            ok: true,
        }
    }

    #[test]
    fn summarize_computes_savings_and_pass_rate() {
        let trials = vec![
            trial(10.0, 6.0, Some(95)),
            trial(10.0, 4.0, Some(60)),
            trial(10.0, 5.0, Some(80)),
        ];
        let s = summarize(&trials, 70);
        assert_eq!(s.trials, 3);
        assert!((s.energy_a - 30.0).abs() < 1e-9);
        assert!((s.energy_b - 15.0).abs() < 1e-9);
        assert!((s.saved_pct.unwrap() - 50.0).abs() < 1e-9); // 15/30 saved
        assert!((s.mean_score.unwrap() - 78.333).abs() < 0.01);
        // 2 of 3 scores >= 70
        assert!((s.pass_rate.unwrap() - 66.666).abs() < 0.01);
    }

    #[test]
    fn summarize_handles_missing_energy_and_no_judge() {
        let trials = vec![
            Trial {
                energy_a: None,
                energy_b: Some(5.0),
                score: None,
                ok: false,
            },
            trial(8.0, 4.0, None),
        ];
        let s = summarize(&trials, 70);
        // Only the fully-paired trial contributes to energy.
        assert!((s.energy_a - 8.0).abs() < 1e-9);
        assert!(s.mean_score.is_none());
        assert!(s.pass_rate.is_none());
    }
}
