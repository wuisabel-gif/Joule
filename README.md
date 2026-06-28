# Joule

[![CI](https://github.com/wuisabel-gif/Joule/actions/workflows/ci.yml/badge.svg)](https://github.com/wuisabel-gif/Joule/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Energy-aware optimization middleware for LLM inference.

Joule sits between your application and an LLM provider, speaking the
OpenAI-compatible API, and answers one question for every request:

> How many joules did this response cost, and could it have been lower?

This repository implements **Phase 1** (a transparent measuring proxy) plus the
prompt-optimization and routing pieces of Phases 2–3. Semantic caching and
carbon-aware scheduling are still ahead. See [`ROADMAP.md`](ROADMAP.md) for the
full vision and phase-by-phase status.

## Why bother — the energy stack

AI energy isn't burned in one place. There are opportunities to reduce it at
**every layer of the stack**, from the user's prompt all the way down to the
power grid. Organizing them as layers reveals where the savings are — and where
research is still needed:

| Layer | Technique | Typical impact | Joule today |
|-------|-----------|----------------|-------------|
| User | Better prompts | Fewer tokens generated | ✅ optimizer passes |
| Application | Semantic caching | Avoid repeated inference | 🔜 Phase 2 |
| Agent | Better planning | Avoid unnecessary tool calls | — |
| Model | Smaller / specialized models | Large energy savings | ✅ `greenest` router |
| Inference | Quantization | Lower computation & memory | provider-side |
| Serving | Batching & scheduling | Higher GPU utilization | provider-side |
| Hardware | Efficient accelerators | Better performance per watt | provider-side |
| Data center | Cooling & power optimization | Lower facility overhead | — |
| Grid | Carbon-aware scheduling | Lower emissions | 🔜 Phase 4 |

The cheapest token is the one you never generate. A few of these levers in more
detail:

- **Prompt optimization** (layer 1) — remove redundant context, drop repeated
  instructions, specify output length. `Summarize this paper.` →
  `Summarize in ≤150 words; focus on methodology and conclusions.` Less work
  before the model even starts. *Joule does this today — see below.*
- **Smaller models** (layer 4) — spell-check, JSON formatting, classification,
  and translation rarely need a frontier model. Routing simple requests to
  lightweight models is one of the biggest single savings. *Joule's `greenest`
  router moves in this direction.*
- **Semantic caching** (layer 3) — if someone already asked "What is Newton's
  Second Law?", return the previous answer. No GPU inference, near-zero energy.
- **Better memory** (layer 4) — retrieve only the *relevant* context (400
  tokens) instead of the whole conversation (40,000). Less attention compute,
  lower energy.
- **Quantization, sparsity, better decoding** (layers 5–7) — FP8/INT4,
  Mixture-of-Experts, speculative decoding: same answer, less computation.
  Largely provider-side, but Joule can *measure* and *prefer* the efficient path.
- **Carbon-aware scheduling** (layer 9) — the same kWh is not equally clean
  everywhere. Defer or relocate non-urgent batch work to cleaner grids.

**Measurement underpins all of it.** Most developers know latency, cost, and
tokens; very few know joules, Wh, or CO₂. Without measurement, optimization is
guesswork — which is why Joule starts by making energy observable.

### Where Joule fits

The exciting opportunity isn't inventing another model — it's becoming the
**LLVM of energy-efficient AI**: a single layer the request passes through that
applies whichever optimizations are safe and explains what it did.

```
Prompt
  │
  ▼
Joule ── Measure ─ Optimize ─ Cache ─ Retrieve only needed memory
       ─ Select model ─ Route ─ Carbon-aware schedule ─ Estimate ─ Explain
  │
  ▼
LLM
```

Instead of asking only *"How many tokens?"*, Joule asks the broader question:
*"Was this computation necessary?"*

## Grounding in measured data

Joule's per-token figures are estimates, but they are **calibrated to published
measurements**, not guessed. The `gpt-4o` profile is anchored so a ~500-output-
token query lands near **0.3 Wh** on optimized H100 serving — Epoch AI's
estimate — and other models are scaled from there by size/class.

| Quantity | Published figure | Source |
|----------|------------------|--------|
| GPT-4o query (~500 out, H100) | **~0.3 Wh** (≈2.0 J/output token) | [Epoch AI](https://epoch.ai/gradient-updates/how-much-energy-does-chatgpt-use) |
| GPT-4o by prompt length | **0.42 → 1.59 Wh** (short → long) | [How Hungry is AI? (2505.09598)](https://arxiv.org/html/2505.09598v1) |
| Per output token, A100 | **~1.72 J** | [Systematic Characterization (2512.01644)](https://arxiv.org/pdf/2512.01644) |
| Per token, LLaMA-65B V100/A100 | **~3–4 J** | From Words to Watts (Samsi 2023) |
| Per token, H100 + FP8 + batch-128 | **~0.39 J** (best case) | [Muxup](https://muxup.com/2026q1/per-query-energy-consumption-of-llms) |
| Long input (100k tokens) | **~40 Wh** (quadratic attention) | [Epoch AI](https://epoch.ai/gradient-updates/how-much-energy-does-chatgpt-use) |
| Small vs large model | **3–11×** less energy (up to 90% via routing) | [Nature s41598](https://www.nature.com/articles/s41598-026-45023-0) |
| FP8 / INT8 quantization | INT8 **≥1.6×** more efficient than FP16 | [van Baalen 2023 (2303.17951)](https://arxiv.org/pdf/2303.17951) |
| Batching (32 → 256) | **~25%** less J/token; batch-1 is 50–100× worse | [HotCarbon 2025](https://hotcarbon.org/assets/2025/paper-11.pdf) |
| Grid carbon intensity (global avg) | **445 g/kWh** (IEA 2024) — the default | [Carbon Brief](https://www.carbonbrief.org/ai-five-charts-that-put-data-centre-energy-use-and-emissions-into-context/) |
| Grid spread | **<20** (Norway hydro) → **>700** (Poland coal) | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666389925002788) |

Numbers vary by hardware, batching, and quantization, so these are a
transparent default — not a claim of precision. See
[`src/estimator/models.rs`](src/estimator/models.rs) for per-model values and
their calibration comments, and [ROADMAP.md § References](ROADMAP.md#references)
for the full academic bibliography (Strubell et al. 2019; Patterson et al. 2021;
Luccioni et al. 2024; Samsi et al. 2023; and others).

## What Phase 1 does

- **OpenAI-compatible proxy** — point your client's base URL at Joule; it
  forwards `/v1/chat/completions` (streaming and non-streaming) and transparently
  passes through every other `/v1/*` route (embeddings, model lists, …).
- **Token accounting** — prefers the provider's reported `usage`; falls back to a
  ~4-chars/token heuristic and records which source was used, so estimates and
  ground truth are never conflated.
- **Energy estimation** — converts tokens into estimated joules, watt-hours,
  grams of CO₂, and USD using a per-model profile table and a configurable grid
  carbon intensity.
- **Prompt optimization** — composable, explainable passes that strip redundant
  tokens before inference (and report exactly what they saved). See below.
- **Metrics** — Prometheus exposition at `/metrics`, labelled by model.
- **Request log** — every request is persisted to SQLite and summarised at
  `/stats`.
- **CLI** — `serve`, `estimate`, `optimize`, and `models`.

Per-request results are also returned to the client as response headers:
`x-joule-energy-j`, `x-joule-electricity-wh`, `x-joule-co2-g`,
`x-joule-cost-usd`, `x-joule-token-source`.

> The per-token energy figures in `src/estimator/models.rs` are estimates, not
> measurements — but they are calibrated to published benchmarks (see
> [Grounding in measured data](#grounding-in-measured-data)). Making energy
> *observable* is the point; per-deployment precision comes from
> `--grid-intensity` and refining the profiles.

## Build

```sh
cargo build --release
# single portable binary at target/release/joule
```

Or with Docker:

```sh
docker build -t joule .
docker run -p 8080:8080 -e JOULE_UPSTREAM=https://api.openai.com joule
```

## Quickstart (no API key, local Ollama)

The fastest way to see Joule work end-to-end is against a local
[Ollama](https://ollama.com) server — no API key, no cloud, no cost:

```sh
ollama serve &                      # start Ollama
ollama pull llama3.2                 # any local model

# Point Joule at Ollama's OpenAI-compatible endpoint
cargo run -- serve --upstream http://localhost:11434 --listen 127.0.0.1:8080

# Send a request through Joule and watch the energy headers come back
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.2","messages":[{"role":"user","content":"Hello!"}]}' \
  -i | grep -i x-joule
```

You'll see `x-joule-energy-j`, `x-joule-co2-g`, and friends on the response, and
running totals at `http://127.0.0.1:8080/stats`.

## Run the proxy

```sh
export JOULE_UPSTREAM_API_KEY=sk-...        # optional; or let clients send their own
cargo run -- serve --upstream https://api.openai.com --listen 127.0.0.1:8080
```

Then send an ordinary OpenAI request through it:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Hello!"}]}' \
  -i | grep -i x-joule
```

Inspect the measurements:

```sh
curl -s http://127.0.0.1:8080/metrics   # Prometheus
curl -s http://127.0.0.1:8080/stats     # JSON: lifetime totals + recent requests
```

Works with any OpenAI-compatible upstream — OpenAI, local Ollama
(`--upstream http://localhost:11434`), LM Studio, vLLM, llama.cpp, etc.

## Prompt optimization (plugins)

The cheapest token is the one you never generate. Joule optimizes the prompt
*before* inference through a pipeline of composable, explainable passes
([`src/optimizer`](src/optimizer)), gated by an intensity level:

| Level | Passes | Lossless? |
|-------|--------|-----------|
| `off` | none | — |
| `lite` (default) | `collapse-whitespace`, `dedup-messages` | yes — formatting only |
| `full` | + `collapse-repeated-lines`, `strip-filler` | yes — content cleanup |
| `ultra` | + `output-limit`, `brevity-hint` | no — changes model behaviour |

`lite`/`full` only remove redundancy (whitespace, duplicate messages, repeated
lines, filler like "could you please"). `ultra` adds the biggest lever —
bounding/encouraging shorter **output** — which changes behaviour, so it is
opt-in and clearly reported.

Nothing happens invisibly: each request returns `x-joule-optimized`,
`x-joule-prompt-saved-tokens`, `x-joule-energy-saved-j`, and
`x-joule-optimizations` headers, and savings are aggregated in `/stats` and
`/metrics` (`joule_prompt_tokens_saved_total`, `joule_energy_saved_joules_total`).

Set the level on the proxy with `--optimize <level>` (or `"optimize"` in the
config file).

### As a standalone prompt improver

```sh
printf 'Could you please summarize this paper.\nCould you please summarize this paper.\n\n\nKindly focus on the methodology and conclusions.' \
  | joule optimize --level full --model gpt-4o
```

```
Optimization Summary (full)
  ✓ normalized whitespace in 1 message(s)
  ✓ collapsed repeated lines in 1 message(s)
  ✓ removed filler phrases in 1 message(s)
  Prompt tokens: 49 → 23 (−26, 53% saved)

Prompt energy (input side) for gpt-4o: 29.400 J → 13.800 J (saved 15.600 J)
```

## Providers & routing (plugins)

Joule dispatches each request through two pluggable layers:

- **Providers** ([`src/provider`](src/provider)) — the vendor wire protocol. A
  provider builds the upstream request and parses tokens out of the response;
  HTTP execution, streaming, and metrics stay in the proxy so every provider is
  measured identically. Three are built in:
  - `openai` — OpenAI and any OpenAI-compatible server (Ollama, vLLM, …).
  - `anthropic` — translates the OpenAI request to `/v1/messages` and the
    response back to OpenAI shape, so OpenAI-speaking clients reach Claude
    transparently.
  - `gemini` — translates to Google's `generateContent` API (URL-encoded model,
    `user`/`model` roles, `systemInstruction`, `usageMetadata`) and back.
- **Routers** ([`src/router`](src/router)) — the provider-selection policy:
  - `static` — always the default provider (a transparent proxy).
  - `model` — the first provider that declares support for the requested model.
  - `greenest` — among configured candidate models, pick the lowest estimated
    energy and route there (energy-aware routing).

The chosen provider, model, and routing reason come back as
`x-joule-provider`, `x-joule-model`, and `x-joule-route` headers.

### Multi-provider config

For more than one provider, pass a JSON config instead of the single-provider
flags:

```json
{
  "providers": [
    { "name": "openai",    "kind": "openai",    "base_url": "https://api.openai.com",    "models": ["gpt-"] },
    { "name": "anthropic", "kind": "anthropic", "base_url": "https://api.anthropic.com", "models": ["claude"], "api_key": "sk-ant-..." },
    { "name": "gemini",    "kind": "gemini",    "base_url": "https://generativelanguage.googleapis.com", "models": ["gemini"], "api_key": "..." },
    { "name": "local",     "kind": "openai",    "base_url": "http://localhost:11434",    "models": ["llama"] }
  ],
  "default_provider": "openai",
  "router": "model"
}
```

```sh
joule serve --config joule.json
```

For the `greenest` router, add a candidate list:

```json
{ "router": "greenest", "greenest_candidates": ["claude-3-5-haiku", "gpt-4o-mini", "gemini-1.5-flash"] }
```

> Streaming requests to the Anthropic and Gemini providers are forwarded in
> their native SSE format (token accounting still works); re-framing the stream
> into OpenAI `chat.completion.chunk` events is a follow-up. The
> OpenAI-compatible provider streams OpenAI events natively.

## CLI

```sh
joule estimate --model gpt-4o --input 1200 --output 400
joule optimize --level full --model gpt-4o --text "Could you please help me"
joule models
```

## Configuration

| Flag | Env | Default | Meaning |
|------|-----|---------|---------|
| `--listen` | `JOULE_LISTEN` | `127.0.0.1:8080` | bind address |
| `--config` | `JOULE_CONFIG` | — | JSON multi-provider config (overrides single-provider flags) |
| `--upstream` | `JOULE_UPSTREAM` | `https://api.openai.com` | provider base URL |
| `--provider-kind` | — | `openai` | `openai`, `anthropic`, or `gemini` |
| `--router` | — | `static` | `static`, `model`, or `greenest` |
| `--optimize` | — | `lite` | `off`, `lite`, `full`, or `ultra` |
| `--api-key` | `JOULE_UPSTREAM_API_KEY` | — | fallback credential |
| `--db` | `JOULE_DB` | `joule.db` | SQLite request log |
| `--grid-intensity` | `JOULE_GRID_INTENSITY` | `445` | g CO₂ / kWh (IEA 2024 global avg) |

## Test

```sh
cargo test
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
