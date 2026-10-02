//! Shared by the LongMemEval examples: turning one question into a chat
//! request, counting tokens, and the same-size truncation baseline.

use joule_proxy::tokens::estimate_prompt_tokens;
use serde::Deserialize;
use serde_json::{json, Value};

pub const MODEL: &str = "gpt-4o";
pub const KEEP_RECENT: usize = 6;

#[derive(Deserialize)]
pub struct Turn {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub has_answer: bool,
}

/// A system prompt, every session's turns in date order, then the question as
/// the last user message. Turns carry `_evidence` (LongMemEval's `has_answer`).
pub fn build_messages(
    id: &str,
    question: &str,
    dates: &[String],
    sessions: &[Vec<Turn>],
) -> anyhow::Result<Vec<Value>> {
    anyhow::ensure!(
        dates.len() == sessions.len(),
        "{id}: dates and sessions differ in length"
    );
    // Sessions in date order ("2023/05/20 (Sat) 02:21" sorts as text).
    let mut order: Vec<usize> = (0..sessions.len()).collect();
    order.sort_by(|&a, &b| dates[a].cmp(&dates[b]));
    let mut messages = vec![json!({"role": "system", "content": "You are a helpful assistant."})];
    for s in order {
        for t in &sessions[s] {
            messages.push(json!({"role": t.role, "content": t.content, "_evidence": t.has_answer}));
        }
    }
    messages.push(json!({"role": "user", "content": question}));
    Ok(messages)
}

pub fn tokens(messages: &[Value]) -> u64 {
    estimate_prompt_tokens(&json!({ "model": MODEL, "messages": messages }))
}

/// Per-message token cost, without the request's fixed overhead.
pub fn costs(messages: &[Value]) -> Vec<u64> {
    messages
        .iter()
        .map(|m| tokens(std::slice::from_ref(m)) - 3)
        .collect()
}

/// Evidence turns among `messages`.
pub fn evidence(messages: &[Value]) -> usize {
    messages.iter().filter(|m| m["_evidence"] == true).count()
}

/// Drop the oldest non-system messages until the prompt fits `budget` tokens.
/// The last message (the question) always stays.
pub fn truncate(messages: &[Value], costs: &[u64], budget: u64) -> Vec<Value> {
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
