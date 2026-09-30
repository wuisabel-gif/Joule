//! `context-recall`: keep only the older conversation that matters.
//!
//! Long chats resend every earlier turn on every request. This pass keeps the
//! system prompt and the most recent turns as they are, and from the older
//! turns keeps only the exchanges most relevant to the latest user message,
//! ranked by MemoryWhale's retrieval engine (`memorywhale-core`, the same
//! ranker measured on the LongMemEval benchmark). Everything else is dropped.
//!
//! An exchange is a user message plus the assistant replies that follow it, so
//! kept history still alternates. Tool calls and tool results are never
//! dropped, which keeps every call paired with its result.

use std::collections::HashSet;

use chrono::{DateTime, Duration, Utc};
use memorywhale_core::engine::{BuiltinEngine, MemoryEngine};
use memorywhale_core::{Memory, Query};
use serde_json::Value;

use super::{OptLevel, Pass};

pub struct ContextRecall {
    /// Trailing messages always kept verbatim.
    pub keep_recent: usize,
    /// Older exchanges kept, chosen by relevance.
    pub keep_relevant: usize,
}

fn role(m: &Value) -> &str {
    m.get("role").and_then(Value::as_str).unwrap_or("")
}

/// Plain chat text: a user or assistant message with string content and no
/// tool calls. Anything else is left alone.
fn plain(m: &Value) -> Option<&str> {
    if !matches!(role(m), "user" | "assistant") || m.get("tool_calls").is_some() {
        return None;
    }
    m.get("content").and_then(Value::as_str)
}

impl Pass for ContextRecall {
    fn name(&self) -> &str {
        "context-recall"
    }
    fn min_level(&self) -> OptLevel {
        // Lossy: the model no longer sees the dropped turns.
        OptLevel::Ultra
    }
    fn apply(&self, request: &mut Value) -> Option<String> {
        let messages = request.get("messages")?.as_array()?;
        let query = messages
            .iter()
            .rev()
            .find(|m| role(m) == "user")
            .and_then(plain)?
            .to_string();
        let mut split = messages.len().checked_sub(self.keep_recent)?;
        // Don't cut an exchange in half: a reply in the recent window keeps its question.
        while split > 0 && role(&messages[split]) == "assistant" {
            split -= 1;
        }

        // Older exchanges: a user message and the plain assistant replies after it.
        let mut exchanges: Vec<Vec<usize>> = Vec::new();
        let mut i = 0;
        while i < split {
            if role(&messages[i]) == "user" && plain(&messages[i]).is_some() {
                let mut ex = vec![i];
                while i + 1 < split
                    && role(&messages[i + 1]) == "assistant"
                    && plain(&messages[i + 1]).is_some()
                {
                    i += 1;
                    ex.push(i);
                }
                exchanges.push(ex);
            }
            i += 1;
        }
        if exchanges.len() <= self.keep_relevant {
            return None;
        }

        // Older exchanges get older timestamps, so recency only breaks ties.
        let start = DateTime::<Utc>::UNIX_EPOCH;
        let memories: Vec<Memory> = exchanges
            .iter()
            .enumerate()
            .map(|(n, ex)| {
                let at = start + Duration::seconds(n as i64);
                Memory {
                    id: n as i64,
                    text: ex
                        .iter()
                        .filter_map(|&k| plain(&messages[k]))
                        .collect::<Vec<_>>()
                        .join("\n"),
                    created_at: at,
                    last_used: at,
                    mentions: 0,
                    importance: 0.5,
                    tags: Vec::new(),
                    embedding: None,
                    agent: None,
                }
            })
            .collect();
        let now = start + Duration::seconds(exchanges.len() as i64);
        let kept: HashSet<i64> = BuiltinEngine::new(memories)
            .retrieve(&Query::new(&query, now), self.keep_relevant)
            .into_iter()
            .map(|s| s.memory.id)
            .collect();
        let drop: HashSet<usize> = exchanges
            .iter()
            .enumerate()
            .filter(|(n, _)| !kept.contains(&(*n as i64)))
            .flat_map(|(_, ex)| ex.iter().copied())
            .collect();

        let total = exchanges.len();
        let messages = request.get_mut("messages")?.as_array_mut()?;
        let mut index = 0;
        messages.retain(|_| {
            index += 1;
            !drop.contains(&(index - 1))
        });
        Some(format!(
            "kept the {} of {total} older exchanges most relevant to the latest question \
             (MemoryWhale retrieval), dropped {} message(s)",
            kept.len(),
            drop.len()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn pass() -> ContextRecall {
        ContextRecall {
            keep_recent: 2,
            keep_relevant: 1,
        }
    }

    #[test]
    fn keeps_the_relevant_exchange_and_recent_turns() {
        let mut req = json!({"messages": [
            {"role": "system", "content": "You are a build assistant."},
            {"role": "user", "content": "The postgres migration fails with a lock timeout."},
            {"role": "assistant", "content": "Raise lock_timeout and retry the migration."},
            {"role": "user", "content": "What is a good name for a cat?"},
            {"role": "assistant", "content": "Miso."},
            {"role": "user", "content": "Recommend a pasta recipe."},
            {"role": "assistant", "content": "Cacio e pepe."},
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "Anytime."},
            {"role": "user", "content": "The postgres migration hit the lock timeout again, what did we change?"}
        ]});
        let note = pass().apply(&mut req).expect("pass fires");
        let texts: Vec<&str> = req["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["content"].as_str().unwrap())
            .collect();
        assert_eq!(
            texts,
            [
                "You are a build assistant.",
                "The postgres migration fails with a lock timeout.",
                "Raise lock_timeout and retry the migration.",
                "thanks",
                "Anytime.",
                "The postgres migration hit the lock timeout again, what did we change?"
            ]
        );
        assert!(note.contains("kept the 1 of 3"), "{note}");
    }

    #[test]
    fn never_drops_tool_messages_and_skips_short_chats() {
        let mut req = json!({"messages": [
            {"role": "user", "content": "list files"},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "a.txt"},
            {"role": "user", "content": "hi"},
            {"role": "user", "content": "bye"}
        ]});
        let before = req.clone();
        assert!(
            pass().apply(&mut req).is_none(),
            "one older exchange is not over the budget"
        );
        assert_eq!(req, before);

        let mut req = json!({"messages": [
            {"role": "user", "content": "unrelated one"},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "a.txt"},
            {"role": "user", "content": "unrelated two"},
            {"role": "user", "content": "files in the folder"},
            {"role": "user", "content": "which files are in the folder?"}
        ]});
        pass().apply(&mut req).expect("pass fires");
        let roles: Vec<&str> = req["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(role)
            .collect();
        assert!(roles.contains(&"tool"));
        assert_eq!(
            roles.iter().filter(|r| **r == "assistant").count(),
            1,
            "tool call kept"
        );
    }
}
