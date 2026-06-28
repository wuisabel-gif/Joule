# Joule

Energy-aware optimization middleware for LLM inference.

Joule sits between your application and an LLM provider, speaking the
OpenAI-compatible API, and answers one question for every request:

> How many joules did this response cost, and could it have been lower?

This repository currently implements **Phase 1**: a transparent measuring proxy.
Optimization, caching, and routing (Phases 2–4) build on top of these
measurements. See [`AGENT.md`](AGENT.md) for the full vision and roadmap.

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
- **CLI** — `serve`, `estimate`, and `models`.

Per-request results are also returned to the client as response headers:
`x-joule-energy-j`, `x-joule-electricity-wh`, `x-joule-co2-g`,
`x-joule-cost-usd`, `x-joule-token-source`.

> The per-token energy figures in `src/estimator/models.rs` are first-order
> estimates, not measurements. They are meant to be refined with real hardware
> data — making energy *observable* is the point; precision is a later phase.

## Build

```sh
cargo build --release
# single portable binary at target/release/joule
```

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
echo "Could you please summarize this paper." | joule optimize --level full --model gpt-4o
```

```
Optimization Summary (full)
  ✓ normalized whitespace in 1 message(s)
  ✓ removed filler phrases in 1 message(s)
  Prompt tokens: 49 → 18 (−31, 63% saved)

Prompt energy (input side) for gpt-4o: 19.600 J → 7.200 J (saved 12.400 J)
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
| `--grid-intensity` | `JOULE_GRID_INTENSITY` | `400` | g CO₂ / kWh |

## Test

```sh
cargo test
```

## License

MIT
