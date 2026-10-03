# `context-recall` on LongMemEval

How much chat history does Joule's `context-recall` pass cut, and does the part
that answers the question survive? Measured on
[LongMemEval](https://github.com/xiaowu0162/LongMemEval) (Wu et al., ICLR 2025,
MIT license), a public long-term memory benchmark we did not write.

The evidence benchmark below calls no model: it measures what reaches the
model. [Answer quality](#answer-quality) sends the trimmed requests to a real
model and grades the replies.

## Protocol

- Dataset: `longmemeval_s`, Hugging Face revision
  `2ec2a557f339b6c0369619b1ed5793734cc87533`, SHA-256
  `08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894`.
- 470 questions; the 30 abstention questions (`*_abs`) are excluded.
- One chat request per question: a system prompt, every turn of its ~50 history
  sessions in date order, then the question as the last user message. About
  104,000 prompt tokens on average.
- Only the `context-recall` pass runs, keeping the last 6 messages and the `k`
  older exchanges ranked most relevant to the question, for `k` = 4, 16, 64.
- **Evidence kept**: LongMemEval labels the turns that hold the answer
  (`has_answer`). *all* = every labeled turn survives; *any* = at least one does.
- **Baseline**: plain truncation to the same token count. Drop the oldest
  messages until the prompt is as short as `context-recall`'s output. Same
  cost, recency instead of relevance.
- Tokens: Joule's estimator with the gpt-4o tokenizer.

```bash
curl -L -o longmemeval_s.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/2ec2a557f339b6c0369619b1ed5793734cc87533/longmemeval_s
cargo run --release --example longmemeval -- longmemeval_s.json bench/longmemeval/
```

Deterministic; full numbers in [`results.json`](results.json).

## Results

| Kept | Prompt tokens | Saved | Evidence kept (all) | Evidence kept (any) |
|---|---|---|---|---|
| Full history | 103,674 | 0% | 1.000 | 1.000 |
| `context-recall`, 64 exchanges | 31,303 | 69.8% | **0.913** | **0.981** |
| Truncation, same tokens | 31,088 | 70.0% | 0.217 | 0.566 |
| `context-recall`, 16 exchanges (default) | 8,956 | 91.4% | **0.806** | **0.947** |
| Truncation, same tokens | 8,735 | 91.6% | 0.049 | 0.168 |
| `context-recall`, 4 exchanges (0.7.0 default) | 3,249 | 96.9% | **0.585** | **0.821** |
| Truncation, same tokens | 3,013 | 97.1% | 0.013 | 0.038 |

**At the same cost, ranking keeps the answer far more often than truncation.**
Cutting 91% of the prompt, `context-recall` keeps all of the answer's evidence
for 81% of questions; truncation keeps it for 5%. LongMemEval places answer
sessions anywhere in the history, so dropping the oldest turns usually drops
the answer.

**4 exchanges was too aggressive, so the default is now 16** (0.7.1). Keeping
4 saves 97% but loses some evidence for 41% of these questions; 16 still saves
91% and keeps all evidence for 81%. LongMemEval histories (about 500 messages)
are far longer than a typical chat, where the pass cuts much less.

## What this does not measure

- Answer accuracy: evidence that reaches the model can still be misread, and a
  model can sometimes answer without it. See [Answer quality](#answer-quality).
- Energy directly: saved prompt tokens convert to estimated joules through
  Joule's estimator, the same as for every other pass.
- Short chats, where the pass often does nothing because there is little older
  history to drop.

## Answer quality

Does keeping the evidence also keep the answer right? `longmemeval_answers`
sends trimmed requests to a model and grades what comes back.

### Protocol

- A seeded random sample of the 470 non-abstention questions (default 100,
  seed 7).
- Each question's request is built as above, then trimmed two ways at equal
  cost: `context-recall` keeping `k` older exchanges (default 16) plus the last
  6 messages, and truncation to the same token count.
- Both versions get the same system prompt, which asks for a brief answer and
  gives LongMemEval's question date (temporal questions need it). Temperature
  0, at most 96 output tokens.
- The full untrimmed history (about 104,000 tokens) is not sent. It does not fit
  a small model's context window, and the question that matters is ranking vs
  recency at the same cost.

### Grading

- **Match** (always on): lowercase both texts, drop punctuation, commas and the
  articles a/an/the, and turn number words up to twenty into digits. A reply is
  correct if the gold answer appears in it as a word sequence, or if the gold
  answer is a number with at most a short unit ("3", "2 weeks", "$1,200") and
  that number appears in the reply. Long gold answers, such as the
  `single-session-preference` rubrics, almost never match word for word, so
  this grader undercounts them.
- **Judge** (`--judge`): the same model is asked a yes/no question adapted from
  LongMemEval's grading prompt. A small model is an unreliable judge; treat it
  as a second opinion, not ground truth.
- Reported: accuracy for both systems overall and per `question_type`, the
  questions where exactly one system is right, and an exact McNemar p-value on
  those disagreements. Per-question replies are in the output JSON.

### Running it

On GitHub Actions: Actions tab, **LongMemEval**, **Run workflow**. Inputs:
`model` (default `qwen2.5:0.5b`), `sample` (default 100, 0 for all), `k`
(default 16), `base_url` (empty runs Ollama on the runner) and `judge`. The job
downloads the pinned dataset, fails unless its SHA-256 matches, reruns the
evidence benchmark and fails unless it reproduces [`results.json`](results.json),
then runs the answer benchmark. The summary table goes to the job summary and
the JSON to the `longmemeval-results` artifact.

The job stops starting new questions after `max_minutes` (default 240, under
the job's 300-minute limit) and reports the questions it finished; the JSON is
also rewritten after every question. On a CPU runner, `qwen2.5:0.5b` takes
roughly one to two minutes per question (two trimmed requests of about 9,000
tokens each), so about 100 questions fit in the budget at best. The first run
(100 questions, no budget) hit the job limit before finishing.

Hosted runners are CPU only, so models larger than about 1B parameters are too
slow there. For those, set `base_url` to an OpenAI-compatible API and add its
key as the repository secret `LME_API_KEY` (Settings, Secrets and variables,
Actions). The maintainer adds that secret; the workflow never prints it.

Locally, with Ollama (start it with a context window big enough for the
trimmed prompts, or it silently cuts them):

```bash
OLLAMA_CONTEXT_LENGTH=32768 ollama serve &
ollama pull qwen2.5:0.5b
cargo run --release --example longmemeval_answers -- longmemeval_s.json \
  --model qwen2.5:0.5b --sample 100 --seed 7 \
  --out bench/longmemeval/answers-qwen2.5-0.5b.json
```

With a stronger hosted model:

```bash
LME_API_KEY=... cargo run --release --example longmemeval_answers -- longmemeval_s.json \
  --base-url https://api.example.com/v1 --model <model-name> --sample 100 --judge \
  --out bench/longmemeval/answers-<model-name>.json
```

### Results

**Results: pending first run.**

### Limits

- A 0.5B model is weak. Absolute accuracy will be low; the useful number is
  the difference between the two systems on the same questions. Rerun with a
  stronger model before drawing conclusions about absolute quality.
- 100 questions is a sample, not the full set. Per-type rows have only a
  handful of questions each and are noisy.
- The match grader is strict on wording and can miss correct paraphrases; the
  judge is lenient and can accept wrong answers. Neither is LongMemEval's
  official GPT-4o judge.
