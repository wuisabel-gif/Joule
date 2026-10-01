//! `context-recall` on LongMemEval: how much history does it cut, and does the
//! answer survive? Offline and deterministic; no model is called.
//!
//!   cargo run --release --example longmemeval -- longmemeval_s.json bench/longmemeval/
//!
//! LongMemEval (Wu et al., ICLR 2025, MIT) gives each question a chat history of
//! about 50 dated sessions and labels the turns that hold the answer
//! (`has_answer`). Each question becomes one chat request: a system prompt,
//! every session's turns in date order, then the question as the last user
//! message. We run only the `context-recall` pass and measure:
//!
//!   * prompt tokens before and after (gpt-4o tokenizer, Joule's estimator);
//!   * evidence kept: every `has_answer` turn survives (all) or at least one
//!     does (any).
//!
//! Baseline: plain truncation to the same token count (drop the oldest
//! messages until the prompt is as short as context-recall's), so the
//! comparison is ranking against recency at equal cost.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use joule_proxy::optimizer::recall::ContextRecall;
use joule_proxy::optimizer::Pass;
use joule_proxy::tokens::estimate_prompt_tokens;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Deserialize)]
struct Turn {
    role: String,
    content: String,
    #[serde(default)]
    has_answer: bool,
}

#[derive(Deserialize)]
struct Item {
    question_id: String,
    question: String,
    haystack_dates: Vec<String>,
    haystack_sessions: Vec<Vec<Turn>>,
}

#[derive(Default, Serialize, Clone)]
struct Score {
    questions: usize,
    tokens_before: f64,
    tokens_after: f64,
    saved_pct: f64,
    evidence_all: f64,
    evidence_any: f64,
}

impl Score {
    fn add(&mut self, before: u64, after: u64, kept: usize, total: usize) {
        self.questions += 1;
        self.tokens_before += before as f64;
        self.tokens_after += after as f64;
        self.saved_pct += 100.0 * (1.0 - after as f64 / before as f64);
        self.evidence_all += (kept == total) as u8 as f64;
        self.evidence_any += (kept > 0) as u8 as f64;
    }
    fn mean(&self) -> Score {
        let n = self.questions.max(1) as f64;
        Score {
            questions: self.questions,
            tokens_before: self.tokens_before / n,
            tokens_after: self.tokens_after / n,
            saved_pct: self.saved_pct / n,
            evidence_all: self.evidence_all / n,
            evidence_any: self.evidence_any / n,
        }
    }
}

const MODEL: &str = "gpt-4o";
const KEEP_RECENT: usize = 6;
const BUDGETS: [usize; 3] = [4, 16, 64];

fn tokens(messages: &[Value]) -> u64 {
    estimate_prompt_tokens(&json!({ "model": MODEL, "messages": messages }))
}

/// Evidence turns among `messages`.
fn evidence(messages: &[Value]) -> usize {
    messages.iter().filter(|m| m["_evidence"] == true).count()
}

/// Drop the oldest non-system messages until the prompt fits `budget` tokens.
/// The last message (the question) always stays.
fn truncate(messages: &[Value], costs: &[u64], budget: u64) -> Vec<Value> {
    let mut total: u64 = costs.iter().sum::<u64>() + 3;
    let mut drop = vec![false; messages.len()];
    for i in 0..messages.len() - 1 {
        if total <= budget {
            break;
        }
        if messages[i]["role"] != "system" {
            drop[i] = true;
            total -= costs[i];
        }
    }
    messages
        .iter()
        .zip(drop)
        .filter(|(_, d)| !d)
        .map(|(m, _)| m.clone())
        .collect()
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let input = PathBuf::from(
        args.next()
            .expect("usage: longmemeval <longmemeval_s.json> [out_dir]"),
    );
    let out_dir = PathBuf::from(args.next().unwrap_or_else(|| "bench/longmemeval".into()));
    let items: Vec<Item> = serde_json::from_slice(&fs::read(&input)?)?;

    let mut scores: BTreeMap<String, Score> = BTreeMap::new();
    let mut skipped_no_evidence = 0;
    for item in items.iter().filter(|i| !i.question_id.ends_with("_abs")) {
        anyhow::ensure!(
            item.haystack_dates.len() == item.haystack_sessions.len(),
            "{}: dates and sessions differ in length",
            item.question_id
        );
        // Sessions in date order ("2023/05/20 (Sat) 02:21" sorts as text).
        let mut order: Vec<usize> = (0..item.haystack_sessions.len()).collect();
        order.sort_by(|&a, &b| item.haystack_dates[a].cmp(&item.haystack_dates[b]));
        let mut messages =
            vec![json!({"role": "system", "content": "You are a helpful assistant."})];
        for s in order {
            for t in &item.haystack_sessions[s] {
                messages
                    .push(json!({"role": t.role, "content": t.content, "_evidence": t.has_answer}));
            }
        }
        messages.push(json!({"role": "user", "content": item.question}));
        let total = evidence(&messages);
        if total == 0 {
            skipped_no_evidence += 1;
            continue;
        }
        let before = tokens(&messages);
        let costs: Vec<u64> = messages
            .iter()
            .map(|m| tokens(std::slice::from_ref(m)) - 3)
            .collect();

        for k in BUDGETS {
            let mut req = json!({ "model": MODEL, "messages": messages });
            ContextRecall {
                keep_recent: KEEP_RECENT,
                keep_relevant: k,
            }
            .apply(&mut req);
            let kept = req["messages"].as_array().unwrap().clone();
            let after = tokens(&kept);
            scores
                .entry(format!("context-recall, {k} exchanges"))
                .or_default()
                .add(before, after, evidence(&kept), total);

            let cut = truncate(&messages, &costs, after);
            scores
                .entry(format!("truncation at the same tokens, {k}"))
                .or_default()
                .add(before, tokens(&cut), evidence(&cut), total);
        }
    }

    let means: BTreeMap<String, Score> =
        scores.iter().map(|(k, v)| (k.clone(), v.mean())).collect();
    fs::create_dir_all(&out_dir)?;
    fs::write(
        out_dir.join("results.json"),
        serde_json::to_string_pretty(&json!({
            "dataset": input.file_name().unwrap().to_string_lossy(),
            "tokenizer_model": MODEL,
            "keep_recent": KEEP_RECENT,
            "skipped_no_evidence": skipped_no_evidence,
            "systems": means,
        }))? + "\n",
    )?;

    println!(
        "LongMemEval ({}), {} questions, {} without labeled evidence skipped",
        input.display(),
        means.values().next().map_or(0, |s| s.questions),
        skipped_no_evidence
    );
    println!(
        "{:<38} {:>9} {:>9} {:>7} {:>9} {:>9}",
        "system", "before", "after", "saved", "ev. all", "ev. any"
    );
    for (name, s) in &means {
        println!(
            "{:<38} {:>9.0} {:>9.0} {:>6.1}% {:>9.3} {:>9.3}",
            name, s.tokens_before, s.tokens_after, s.saved_pct, s.evidence_all, s.evidence_any
        );
    }
    Ok(())
}
