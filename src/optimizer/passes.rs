//! The built-in optimization passes.
//!
//! Each pass is deliberately small and conservative — it should never change
//! the *meaning* of a prompt within its intensity level. Lossy or
//! behaviour-affecting passes live at `Ultra` and say so.

use std::collections::HashSet;

use serde_json::{json, Value};

use super::{OptLevel, Pass};

/// The default pipeline, in execution order.
pub fn default_passes() -> Vec<Box<dyn Pass>> {
    vec![
        Box::new(CollapseWhitespace),
        Box::new(DedupMessages),
        Box::new(CollapseRepeatedLines),
        Box::new(DedupSystemLines),
        Box::new(StripFiller),
        Box::new(StripReasoning),
        Box::new(EnforceOutputLimit { cap: 512 }),
        Box::new(BrevityHint),
    ]
}

/// Borrow each message's string `content`, replacing it via `f`. Returns the
/// number of messages whose content changed.
fn rewrite_messages(request: &mut Value, mut f: impl FnMut(&str) -> String) -> usize {
    let Some(messages) = request.get_mut("messages").and_then(Value::as_array_mut) else {
        return 0;
    };
    let mut changed = 0;
    for message in messages.iter_mut() {
        let Some(content) = message.get("content").and_then(Value::as_str) else {
            continue;
        };
        let next = f(content);
        if next != content {
            message["content"] = json!(next);
            changed += 1;
        }
    }
    changed
}

// ---------------------------------------------------------------------------
// Lite: lossless formatting.
// ---------------------------------------------------------------------------

/// Strip trailing whitespace, collapse runs of blank lines, and trim edges.
/// Leaves leading indentation intact so code blocks survive.
struct CollapseWhitespace;

impl Pass for CollapseWhitespace {
    fn name(&self) -> &str {
        "collapse-whitespace"
    }
    fn min_level(&self) -> OptLevel {
        OptLevel::Lite
    }
    fn apply(&self, request: &mut Value) -> Option<String> {
        let changed = rewrite_messages(request, normalize_whitespace);
        (changed > 0).then(|| format!("normalized whitespace in {changed} message(s)"))
    }
}

fn normalize_whitespace(s: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    let mut prev_blank = false;
    for line in s.lines() {
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            if !prev_blank {
                out.push("");
            }
            prev_blank = true;
        } else {
            out.push(trimmed);
            prev_blank = false;
        }
    }
    out.join("\n").trim_matches('\n').to_string()
}

/// Remove later messages that exactly duplicate an earlier one (same role and
/// content) — repeated context the caller pasted twice.
struct DedupMessages;

impl Pass for DedupMessages {
    fn name(&self) -> &str {
        "dedup-messages"
    }
    fn min_level(&self) -> OptLevel {
        OptLevel::Lite
    }
    fn apply(&self, request: &mut Value) -> Option<String> {
        let messages = request.get("messages").and_then(Value::as_array)?;
        let mut seen: HashSet<String> = HashSet::new();
        let mut kept: Vec<Value> = Vec::with_capacity(messages.len());
        let mut removed = 0;
        for m in messages {
            let role = m.get("role").and_then(Value::as_str).unwrap_or("");
            let content = m.get("content").and_then(Value::as_str);
            // Only dedup plain-text messages; leave structured content alone.
            if let Some(content) = content {
                let key = format!("{role}\u{0}{content}");
                if seen.insert(key) {
                    kept.push(m.clone());
                } else {
                    removed += 1;
                }
            } else {
                kept.push(m.clone());
            }
        }
        if removed > 0 {
            request["messages"] = json!(kept);
            Some(format!("removed {removed} duplicate message(s)"))
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Full: lossless content cleanup.
// ---------------------------------------------------------------------------

/// Collapse consecutive identical lines within a message to a single copy.
struct CollapseRepeatedLines;

impl Pass for CollapseRepeatedLines {
    fn name(&self) -> &str {
        "collapse-repeated-lines"
    }
    fn min_level(&self) -> OptLevel {
        OptLevel::Full
    }
    fn apply(&self, request: &mut Value) -> Option<String> {
        let changed = rewrite_messages(request, |s| {
            let mut out: Vec<&str> = Vec::new();
            for line in s.lines() {
                if out.last() != Some(&line) {
                    out.push(line);
                }
            }
            out.join("\n")
        });
        (changed > 0).then(|| format!("collapsed repeated lines in {changed} message(s)"))
    }
}

/// Within system messages, drop duplicate identical non-empty lines (keeping the
/// first) — boilerplate instructions repeated across a long system prompt. Only
/// system prompts are touched, so repeated data lines in user content are safe.
struct DedupSystemLines;

impl Pass for DedupSystemLines {
    fn name(&self) -> &str {
        "dedup-lines"
    }
    fn min_level(&self) -> OptLevel {
        OptLevel::Full
    }
    fn apply(&self, request: &mut Value) -> Option<String> {
        let messages = request.get_mut("messages").and_then(Value::as_array_mut)?;
        let mut changed = 0;
        for m in messages.iter_mut() {
            if m.get("role").and_then(Value::as_str) != Some("system") {
                continue;
            }
            let content = match m.get("content").and_then(Value::as_str) {
                Some(c) => c.to_string(),
                None => continue,
            };
            let mut seen: HashSet<&str> = HashSet::new();
            let mut out: Vec<&str> = Vec::new();
            for line in content.lines() {
                let key = line.trim();
                // Drop only exact repeats of a non-empty line; keep blanks.
                if !key.is_empty() && !seen.insert(key) {
                    continue;
                }
                out.push(line);
            }
            let next = out.join("\n");
            if next != content {
                m["content"] = json!(next);
                changed += 1;
            }
        }
        (changed > 0).then(|| format!("deduped repeated lines in {changed} system message(s)"))
    }
}

/// Remove common filler / politeness phrases that add tokens but not meaning.
struct StripFiller;

/// Conservative list of phrases safe to drop. Matched case-insensitively.
const FILLER: &[&str] = &[
    "could you please ",
    "can you please ",
    "i would like you to ",
    "i want you to ",
    "please kindly ",
    "as an ai language model, ",
    "as a large language model, ",
    "please ",
    "kindly ",
];

impl Pass for StripFiller {
    fn name(&self) -> &str {
        "strip-filler"
    }
    fn min_level(&self) -> OptLevel {
        OptLevel::Full
    }
    fn apply(&self, request: &mut Value) -> Option<String> {
        let changed = rewrite_messages(request, |s| {
            let mut result = s.to_string();
            for phrase in FILLER {
                result = remove_ci(&result, phrase);
            }
            result
        });
        (changed > 0).then(|| format!("removed filler phrases in {changed} message(s)"))
    }
}

/// Case-insensitively remove every occurrence of `needle` from `haystack`.
fn remove_ci(haystack: &str, needle: &str) -> String {
    let lower_hay = haystack.to_ascii_lowercase();
    let lower_needle = needle.to_ascii_lowercase();
    let mut out = String::with_capacity(haystack.len());
    let mut cursor = 0;
    while let Some(rel) = lower_hay[cursor..].find(&lower_needle) {
        let start = cursor + rel;
        out.push_str(&haystack[cursor..start]);
        cursor = start + needle.len();
    }
    out.push_str(&haystack[cursor..]);
    out
}

// ---------------------------------------------------------------------------
// Ultra: behaviour-affecting levers (the biggest energy savings).
// ---------------------------------------------------------------------------

/// Cap output length when the caller did not set one. Output tokens are the
/// most energy-expensive, so bounding them is the single largest lever — but it
/// changes behaviour, hence Ultra only.
struct EnforceOutputLimit {
    cap: u64,
}

impl Pass for EnforceOutputLimit {
    fn name(&self) -> &str {
        "output-limit"
    }
    fn min_level(&self) -> OptLevel {
        OptLevel::Ultra
    }
    fn apply(&self, request: &mut Value) -> Option<String> {
        if request.get("max_tokens").and_then(Value::as_u64).is_some() {
            return None;
        }
        request["max_tokens"] = json!(self.cap);
        Some(format!("capped output at {} tokens (was unset)", self.cap))
    }
}

/// Strip chain-of-thought triggers ("think step by step", "show your
/// reasoning", …). These instructions multiply output tokens; removing them is
/// a large output-side saving but can reduce accuracy on hard tasks, so it is
/// Ultra-only.
struct StripReasoning;

/// Reasoning triggers to remove, longest/most-specific first so a shorter
/// phrase doesn't leave a fragment of a longer one. Matched case-insensitively.
const REASONING: &[&str] = &[
    "let's think step by step",
    "let us think step by step",
    "walk me through your reasoning",
    "walk through your reasoning",
    "explain your reasoning",
    "explain your thinking",
    "show your reasoning",
    "show your thinking",
    "show your work",
    "reason step by step",
    "think step by step",
    "think step-by-step",
];

impl Pass for StripReasoning {
    fn name(&self) -> &str {
        "strip-reasoning"
    }
    fn min_level(&self) -> OptLevel {
        OptLevel::Ultra
    }
    fn apply(&self, request: &mut Value) -> Option<String> {
        let changed = rewrite_messages(request, |s| {
            let mut result = s.to_string();
            for phrase in REASONING {
                result = remove_ci(&result, phrase);
            }
            result
        });
        (changed > 0).then(|| format!("removed reasoning triggers in {changed} message(s)"))
    }
}

/// Ask the model to answer directly, reducing output tokens. Behaviour-affecting.
struct BrevityHint;

const BREVITY: &str =
    "Answer concisely and directly — no preamble, no restating the question, only what was asked.";

impl Pass for BrevityHint {
    fn name(&self) -> &str {
        "brevity-hint"
    }
    fn min_level(&self) -> OptLevel {
        OptLevel::Ultra
    }
    fn apply(&self, request: &mut Value) -> Option<String> {
        let messages = request.get_mut("messages").and_then(Value::as_array_mut)?;

        // Append to an existing leading system message if present...
        if let Some(first) = messages.first_mut() {
            if first.get("role").and_then(Value::as_str) == Some("system") {
                if let Some(content) = first.get("content").and_then(Value::as_str) {
                    if content.contains(BREVITY) {
                        return None;
                    }
                    first["content"] = json!(format!("{content} {BREVITY}"));
                    return Some("appended brevity hint to system prompt".to_string());
                }
            }
        }
        // ...otherwise insert a fresh system message at the front.
        messages.insert(0, json!({ "role": "system", "content": BREVITY }));
        Some("inserted brevity system prompt".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_whitespace_preserves_indentation() {
        let input = "def f():\n    return 1   \n\n\n\nx = f()";
        let out = normalize_whitespace(input);
        assert_eq!(out, "def f():\n    return 1\n\nx = f()");
    }

    #[test]
    fn remove_ci_is_case_insensitive() {
        assert_eq!(
            remove_ci("Please do X. please do Y.", "please "),
            "do X. do Y."
        );
    }

    #[test]
    fn collapse_repeated_lines_dedups_consecutive() {
        let mut req = json!({"messages":[{"role":"user","content":"a\na\nb\nb\nb\na"}]});
        let pass = CollapseRepeatedLines;
        assert!(pass.apply(&mut req).is_some());
        assert_eq!(req["messages"][0]["content"], "a\nb\na");
    }

    #[test]
    fn dedup_system_lines_drops_repeats_but_spares_user() {
        let mut req = json!({"messages":[
            {"role":"system","content":"Be helpful.\nBe helpful.\nUse metric units."},
            {"role":"user","content":"row\nrow"}
        ]});
        let pass = DedupSystemLines;
        assert!(pass.apply(&mut req).is_some());
        assert_eq!(
            req["messages"][0]["content"],
            "Be helpful.\nUse metric units."
        );
        // user content untouched (repeated data lines are safe).
        assert_eq!(req["messages"][1]["content"], "row\nrow");
    }

    #[test]
    fn strip_reasoning_removes_cot_triggers() {
        let mut req = json!({"messages":[
            {"role":"user","content":"Solve 12*13. Let's think step by step and show your work."}
        ]});
        let pass = StripReasoning;
        assert!(pass.apply(&mut req).is_some());
        let out = req["messages"][0]["content"]
            .as_str()
            .unwrap()
            .to_lowercase();
        assert!(!out.contains("step by step"));
        assert!(!out.contains("show your work"));
        assert!(out.contains("solve 12*13"));
    }
}
