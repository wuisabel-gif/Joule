# `context-recall` on LongMemEval

How much chat history does Joule's `context-recall` pass cut, and does the part
that answers the question survive? Measured on
[LongMemEval](https://github.com/xiaowu0162/LongMemEval) (Wu et al., ICLR 2025,
MIT license), a public long-term memory benchmark we did not write.

No model is called. This measures what reaches the model, not whether the model
then answers correctly.

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
| `context-recall`, 16 exchanges | 8,956 | 91.4% | **0.806** | **0.947** |
| Truncation, same tokens | 8,735 | 91.6% | 0.049 | 0.168 |
| `context-recall`, 4 exchanges (default) | 3,249 | 96.9% | **0.585** | **0.821** |
| Truncation, same tokens | 3,013 | 97.1% | 0.013 | 0.038 |

**At the same cost, ranking keeps the answer far more often than truncation.**
Cutting 91% of the prompt, `context-recall` keeps all of the answer's evidence
for 81% of questions; truncation keeps it for 5%. LongMemEval places answer
sessions anywhere in the history, so dropping the oldest turns usually drops
the answer.

**The default of 4 exchanges is aggressive.** It saves 97% but loses some
evidence for 41% of these questions. LongMemEval histories are far longer than
a typical chat (about 500 messages), so on everyday traffic 4 exchanges cuts
much less; but on long histories, a larger `k` is the safer trade.

## What this does not measure

- Answer accuracy: evidence that reaches the model can still be misread, and a
  model can sometimes answer without it. `joule eval` against a real model is
  the check for that.
- Energy directly: saved prompt tokens convert to estimated joules through
  Joule's estimator, the same as for every other pass.
- Short chats, where the pass often does nothing because there is little older
  history to drop.
