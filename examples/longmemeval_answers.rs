//! Does `context-recall` keep answers right, not just evidence? Sends each
//! trimmed LongMemEval request to a real model and grades the reply.
//!
//!   cargo run --release --example longmemeval_answers -- longmemeval_s.json \
//!     --model qwen2.5:0.5b --sample 100 --seed 7 --out bench/longmemeval/answers-qwen2.5-0.5b.json
//!
//! For a seeded random sample of non-abstention questions, builds the same
//! request as `examples/longmemeval.rs`, trims it two ways at equal cost
//! (`context-recall` keeping `k` exchanges, and plain truncation to the same
//! token count), and sends both to an OpenAI-compatible endpoint (default:
//! local Ollama) at temperature 0. Both get the same system prompt with the
//! question date. The untrimmed history (~104k tokens) is not sent: it does
//! not fit a small local model, and equal cost is the comparison that matters.
//!
//! Grading: normalized answer match (lowercase, drop punctuation and articles,
//! number words to digits; correct if the gold answer appears in the reply, or
//! if the gold answer is a number, optionally with a unit, and that number
//! does). `--judge` also asks the same model a LongMemEval-style yes/no
//! question.
//!
//! Self-check for the grader: `cargo test --example longmemeval_answers`.

mod lme_common;

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;
use joule_proxy::optimizer::recall::ContextRecall;
use joule_proxy::optimizer::Pass;
use lme_common::{build_messages, costs, evidence, tokens, truncate, Turn, KEEP_RECENT, MODEL};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Parser)]
struct Args {
    /// Path to longmemeval_s.json.
    dataset: PathBuf,
    /// OpenAI-compatible base URL (without /chat/completions).
    #[arg(
        long,
        env = "LME_BASE_URL",
        default_value = "http://127.0.0.1:11434/v1"
    )]
    base_url: String,
    #[arg(long, env = "LME_MODEL", default_value = "qwen2.5:0.5b")]
    model: String,
    /// Bearer token, if the endpoint needs one.
    #[arg(long, env = "LME_API_KEY", hide_env_values = true)]
    api_key: Option<String>,
    /// Questions to sample (0 = all).
    #[arg(long, env = "LME_SAMPLE", default_value_t = 50)]
    sample: usize,
    #[arg(long, env = "LME_SEED", default_value_t = 7)]
    seed: u64,
    /// Older exchanges context-recall keeps.
    #[arg(long, default_value_t = 16)]
    k: usize,
    #[arg(long, default_value_t = 96)]
    max_tokens: u32,
    /// Stop starting new questions after this many minutes (0 = no limit), so a
    /// time-limited run still ends with a report for what it finished.
    #[arg(long, default_value_t = 0)]
    max_minutes: u64,
    /// Also grade with a yes/no judge prompt on the same endpoint.
    #[arg(long)]
    judge: bool,
    #[arg(long, default_value = "bench/longmemeval/answers.json")]
    out: PathBuf,
}

#[derive(Deserialize)]
struct Item {
    question_id: String,
    question_type: String,
    question: String,
    question_date: String,
    /// A string, or occasionally a bare number.
    answer: Value,
    haystack_dates: Vec<String>,
    haystack_sessions: Vec<Vec<Turn>>,
}

const SYSTEMS: [&str; 2] = ["context-recall", "truncation"];

/// splitmix64: a seeded shuffle without a new dependency.
fn shuffle<T>(v: &mut [T], mut seed: u64) {
    for i in (1..v.len()).rev() {
        seed = seed.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        v.swap(i, (z % (i as u64 + 1)) as usize);
    }
}

fn normalize(s: &str) -> Vec<String> {
    const NUMBERS: [&str; 21] = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
    ];
    s.to_lowercase()
        .replace(',', "") // 1,200 -> 1200
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .filter(|w| !matches!(*w, "a" | "an" | "the"))
        .map(|w| match NUMBERS.iter().position(|n| *n == w) {
            Some(i) => i.to_string(),
            None => w.to_string(),
        })
        .collect()
}

/// Gold answer appears in the reply as a word sequence, or the gold answer is
/// a number (optionally with a unit, like "3" or "2 weeks") and that number
/// appears in the reply.
fn answer_match(gold: &str, reply: &str) -> bool {
    let (g, r) = (normalize(gold), normalize(reply));
    if g.is_empty() {
        return false;
    }
    if r.windows(g.len()).any(|w| w == g.as_slice()) {
        return true;
    }
    let numeric = g.len() <= 3 && g[0].chars().all(|c| c.is_ascii_digit());
    numeric && r.contains(&g[0])
}

/// Exact two-sided McNemar (binomial) p-value for `b` vs `c` discordant pairs.
fn mcnemar_p(b: usize, c: usize) -> f64 {
    let n = b + c;
    if n == 0 {
        return 1.0;
    }
    let (mut p, mut term) = (0.0, 0.5f64.powi(n as i32));
    for i in 0..=b.min(c) {
        p += term;
        term *= (n - i) as f64 / (i + 1) as f64;
    }
    (2.0 * p).min(1.0)
}

struct Client {
    http: reqwest::Client,
    url: String,
    model: String,
    key: Option<String>,
}

impl Client {
    async fn chat(&self, messages: &[Value], max_tokens: u32) -> anyhow::Result<(String, Value)> {
        let body = json!({
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        });
        let mut last = None;
        for attempt in 0..8u32 {
            let mut req = self.http.post(&self.url).json(&body);
            if let Some(k) = &self.key {
                req = req.bearer_auth(k);
            }
            let resp = req.send().await;
            // Rate limits and server errors are worth waiting out: honor
            // Retry-After when given, else back off 2, 4, 8 ... up to 60 s.
            if let Ok(r) = &resp {
                let status = r.status();
                if status.as_u16() == 429 || status.is_server_error() {
                    let wait = r
                        .headers()
                        .get(reqwest::header::RETRY_AFTER)
                        .and_then(|v| v.to_str().ok()?.parse::<u64>().ok())
                        .unwrap_or(2u64 << attempt.min(5))
                        .min(60);
                    eprintln!("  {status}, retrying in {wait}s");
                    tokio::time::sleep(std::time::Duration::from_secs(wait)).await;
                    last = Some(resp.and_then(|r| r.error_for_status()).unwrap_err());
                    continue;
                }
            }
            match resp.and_then(|r| r.error_for_status()) {
                Ok(r) => {
                    let v: Value = r.json().await?;
                    let text = v["choices"][0]["message"]["content"]
                        .as_str()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    return Ok((text, v["usage"]["prompt_tokens"].clone()));
                }
                Err(e) => last = Some(e),
            }
        }
        Err(last.unwrap().into())
    }

    async fn judge(&self, question: &str, gold: &str, reply: &str) -> anyhow::Result<bool> {
        // After LongMemEval's evaluate_qa.py grading prompt.
        let prompt = format!(
            "I will give you a question, a correct answer, and a response from a model. \
             Please answer yes if the response contains the correct answer. Otherwise, answer no. \
             If the response is equivalent to the correct answer or contains all the intermediate \
             steps to get the correct answer, you should also answer yes. If the response only \
             contains a subset of the information required by the answer, answer no.\n\n\
             Question: {question}\n\nCorrect Answer: {gold}\n\nModel Response: {reply}\n\n\
             Is the model response correct? Answer yes or no only."
        );
        let (text, _) = self
            .chat(&[json!({"role": "user", "content": prompt})], 8)
            .await?;
        Ok(text.to_lowercase().contains("yes"))
    }
}

#[derive(Default, Clone, Copy)]
struct Tally {
    n: usize,
    matched: [usize; 2],
    judged: [usize; 2],
}

impl Tally {
    fn json(&self, judge: bool) -> Value {
        let acc = |c: usize| c as f64 / self.n.max(1) as f64;
        let mut v = json!({
            "questions": self.n,
            "match_accuracy": { SYSTEMS[0]: acc(self.matched[0]), SYSTEMS[1]: acc(self.matched[1]) },
        });
        if judge {
            v["judge_accuracy"] =
                json!({ SYSTEMS[0]: acc(self.judged[0]), SYSTEMS[1]: acc(self.judged[1]) });
        }
        v
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let started = Instant::now();
    let items: Vec<Item> = serde_json::from_slice(&fs::read(&args.dataset)?)?;
    let mut pool: Vec<&Item> = items
        .iter()
        .filter(|i| !i.question_id.ends_with("_abs"))
        .collect();
    shuffle(&mut pool, args.seed);
    if args.sample > 0 {
        pool.truncate(args.sample);
    }

    let client = Client {
        http: reqwest::Client::builder()
            .timeout(Duration::from_secs(600))
            .build()?,
        url: format!("{}/chat/completions", args.base_url.trim_end_matches('/')),
        model: args.model.clone(),
        key: args.api_key.clone().filter(|k| !k.is_empty()),
    };

    let mut overall = Tally::default();
    let mut by_type: BTreeMap<String, Tally> = BTreeMap::new();
    let (mut match_disc, mut judge_disc) = ([0usize; 2], [0usize; 2]);
    let mut tokens_after = [0u64; 2];
    let mut records = Vec::new();
    for (n, item) in pool.iter().enumerate() {
        if args.max_minutes > 0 && started.elapsed().as_secs() >= args.max_minutes * 60 {
            eprintln!(
                "time budget of {} min reached after {} questions; stopping",
                args.max_minutes, n
            );
            break;
        }
        let gold = match &item.answer {
            Value::String(s) => s.clone(),
            v => v.to_string(),
        };
        let messages = build_messages(
            &item.question_id,
            &item.question,
            &item.haystack_dates,
            &item.haystack_sessions,
        )?;
        let total_evidence = evidence(&messages);
        let mut req = json!({ "model": MODEL, "messages": messages });
        ContextRecall {
            keep_recent: KEEP_RECENT,
            keep_relevant: args.k,
        }
        .apply(&mut req);
        let recalled = req["messages"].as_array().unwrap().clone();
        let cut = truncate(&messages, &costs(&messages), tokens(&recalled));

        let system = format!(
            "You are a helpful assistant. Answer the last question using the conversation \
             history above. Answer briefly. Current date: {}.",
            item.question_date
        );
        let mut record = json!({
            "question_id": item.question_id,
            "question_type": item.question_type,
            "question": item.question,
            "gold": gold,
        });
        let mut ok = [false; 2];
        let mut jok = [false; 2];
        for (s, trimmed) in [recalled, cut].iter().enumerate() {
            let est = tokens(trimmed);
            tokens_after[s] += est;
            let ev = evidence(trimmed);
            let mut send: Vec<Value> = trimmed
                .iter()
                .map(|m| json!({"role": m["role"], "content": m["content"]}))
                .collect();
            send[0]["content"] = json!(system);
            let (reply, prompt_tokens) = client.chat(&send, args.max_tokens).await?;
            ok[s] = answer_match(&gold, &reply);
            let mut r = json!({
                "reply": reply,
                "match": ok[s],
                "tokens_estimated": est,
                "prompt_tokens_reported": prompt_tokens,
                "evidence_all": ev == total_evidence,
                "evidence_any": ev > 0,
            });
            if args.judge {
                jok[s] = client.judge(&item.question, &gold, &reply).await?;
                r["judge"] = json!(jok[s]);
            }
            record[SYSTEMS[s]] = r;
        }
        for t in [
            &mut overall,
            by_type.entry(item.question_type.clone()).or_default(),
        ] {
            t.n += 1;
            for s in 0..2 {
                t.matched[s] += ok[s] as usize;
                t.judged[s] += jok[s] as usize;
            }
        }
        if ok[0] != ok[1] {
            match_disc[ok[1] as usize] += 1;
        }
        if jok[0] != jok[1] {
            judge_disc[jok[1] as usize] += 1;
        }
        eprintln!(
            "[{}/{}] {} recall={} truncation={} ({:.0}s)",
            n + 1,
            pool.len(),
            item.question_id,
            ok[0],
            ok[1],
            started.elapsed().as_secs_f64()
        );
        records.push(record);
        let state = (&overall, &by_type, match_disc, judge_disc, tokens_after);
        write_report(&args, &started, pool.len(), state, &records)?;
    }

    let state = (&overall, &by_type, match_disc, judge_disc, tokens_after);
    write_report(&args, &started, pool.len(), state, &records)?;

    println!(
        "{} of {} questions, model {}, k={}, {:.0}s",
        records.len(),
        pool.len(),
        args.model,
        args.k,
        started.elapsed().as_secs_f64()
    );
    println!(
        "{:<28} {:>4} {:>8} {:>8} {:>8} {:>8}",
        "question type", "n", "rec.mt", "trc.mt", "rec.jd", "trc.jd"
    );
    for (name, t) in by_type.iter().chain([(&"overall".to_string(), &overall)]) {
        let a = |c: usize| c as f64 / t.n.max(1) as f64;
        println!(
            "{:<28} {:>4} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            name,
            t.n,
            a(t.matched[0]),
            a(t.matched[1]),
            a(t.judged[0]),
            a(t.judged[1])
        );
    }
    println!(
        "match disagreements: recall-only {} truncation-only {} (exact McNemar p={:.4})",
        match_disc[0],
        match_disc[1],
        mcnemar_p(match_disc[0], match_disc[1])
    );
    Ok(())
}

type State<'a> = (
    &'a Tally,
    &'a BTreeMap<String, Tally>,
    [usize; 2],
    [usize; 2],
    [u64; 2],
);

/// Write the report for the questions answered so far. Called after every
/// question, so an interrupted run keeps its results.
fn write_report(
    args: &Args,
    started: &Instant,
    planned: usize,
    (overall, by_type, match_disc, judge_disc, tokens_after): State,
    records: &[Value],
) -> anyhow::Result<()> {
    let disc = |d: [usize; 2]| {
        json!({
            "context_recall_only_correct": d[0],
            "truncation_only_correct": d[1],
            "mcnemar_exact_p": mcnemar_p(d[0], d[1]),
        })
    };
    let n = records.len().max(1) as f64;
    let mut summary = json!({
        "overall": overall.json(args.judge),
        "by_question_type": by_type.iter().map(|(k, t)| (k.clone(), t.json(args.judge))).collect::<BTreeMap<_, _>>(),
        "mean_prompt_tokens_estimated": { SYSTEMS[0]: tokens_after[0] as f64 / n, SYSTEMS[1]: tokens_after[1] as f64 / n },
        "match_disagreements": disc(match_disc),
    });
    if args.judge {
        summary["judge_disagreements"] = disc(judge_disc);
    }
    let out = json!({
        "dataset": args.dataset.file_name().unwrap().to_string_lossy(),
        "model": args.model,
        "base_url": args.base_url,
        "sample": records.len(),
        "planned_sample": planned,
        "seed": args.seed,
        "k": args.k,
        "keep_recent": KEEP_RECENT,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "runtime_seconds": started.elapsed().as_secs(),
        "summary": summary,
        "questions": records,
    });
    if let Some(dir) = args.out.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(&args.out, serde_json::to_string_pretty(&out)? + "\n")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grader() {
        assert!(answer_match(
            "Business Administration",
            "You studied business administration."
        ));
        assert!(answer_match("3", "You went three times."));
        assert!(answer_match("$1,200", "It cost 1200 dollars."));
        assert!(answer_match("2 weeks", "About 2 weeks ago."));
        assert!(!answer_match("3", "You went 13 times."));
        assert!(!answer_match("the red one", "the blue one"));
        assert!(!answer_match("Sarah", "I don't know."));
        assert_eq!(mcnemar_p(0, 0), 1.0);
        assert!((mcnemar_p(0, 5) - 0.0625).abs() < 1e-12);
        let mut v: Vec<u32> = (0..10).collect();
        shuffle(&mut v, 1);
        let mut w = v.clone();
        w.sort();
        assert_eq!(w, (0..10).collect::<Vec<_>>());
    }
}
