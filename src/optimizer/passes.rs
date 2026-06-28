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
        Box::new(StripFiller),
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

/// Ask the model to be concise, reducing output tokens. Behaviour-affecting.
struct BrevityHint;

const BREVITY: &str = "Be concise.";

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
}
