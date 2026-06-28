# Joule

Energy-aware optimization middleware for LLM inference.

Joule sits between your application and an LLM provider, speaking the
OpenAI-compatible API, and answers one question for every request:

> How many joules did this response cost, and could it have been lower?

This repository currently implements **Phase 1**: a transparent measuring proxy.
Optimization, caching, and routing (Phases 2–4) build on top of these
measurements. See [`AGENT.md`](AGENT.md) for the full vision and roadmap.

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
joule models
```

## Configuration

| Flag | Env | Default | Meaning |
|------|-----|---------|---------|
| `--listen` | `JOULE_LISTEN` | `127.0.0.1:8080` | bind address |
| `--config` | `JOULE_CONFIG` | — | JSON multi-provider config (overrides single-provider flags) |
| `--upstream` | `JOULE_UPSTREAM` | `https://api.openai.com` | provider base URL |
| `--provider-kind` | — | `openai` | `openai` or `anthropic` |
| `--router` | — | `static` | `static`, `model`, or `greenest` |
| `--api-key` | `JOULE_UPSTREAM_API_KEY` | — | fallback credential |
| `--db` | `JOULE_DB` | `joule.db` | SQLite request log |
| `--grid-intensity` | `JOULE_GRID_INTENSITY` | `400` | g CO₂ / kWh |

## Test

```sh
cargo test
```

## License

MIT
