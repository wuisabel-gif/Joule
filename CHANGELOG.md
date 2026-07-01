# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Resilience** for upstream calls: connect + per-request **timeouts**
  (`--timeout`), **retries** with exponential backoff on transient failures
  (`--max-retries`; `joule_upstream_retries_total`), and a per-provider
  **circuit breaker** that fails fast with `503` / `x-joule-circuit: open` after
  repeated failures and probes recovery after a cooldown
  (`joule_circuit_open{provider}`).
- Optimizer `full` pass **`dedup-lines`** — drops duplicate identical lines in
  system prompts (repeated boilerplate instructions), leaving user content
  untouched.
- Optimizer `ultra` pass **`strip-reasoning`** — removes chain-of-thought
  triggers ("think step by step", "show your reasoning", …) that multiply output
  tokens.

### Changed
- `brevity-hint` now asks the model to answer directly with no preamble or
  restating the question (was "Be concise."), covering more output waste.

Note: automatic **stop sequences** are intentionally not added — they would
truncate answers whose format Joule can't know.

## [0.2.0] — 2026-06-29

### Added
- **Exact-match response cache** — identical requests skip the upstream call and
  return the stored response (~0 J, $0, near-zero latency). Reports avoided
  energy via `x-joule-cache`, `x-joule-energy-saved-j`, the `joule_cache_hits_total`
  metric, and `/stats`. On by default; `--no-cache` / `--cache-capacity` to
  control. In-memory, LRU-bounded; streaming requests are never cached.
- **Real tokenizer** — token counts now use BPE tokenization (tiktoken:
  `o200k_base` for newer OpenAI models, `cl100k_base` otherwise and as a close
  approximation for non-OpenAI models), replacing the ~4-chars/token heuristic.
- **Streaming re-framing** — Anthropic and Gemini SSE streams are translated
  into OpenAI `chat.completion.chunk` frames (terminated with `[DONE]`), so
  streaming clients get a consistent format from any provider.
- **`joule report`** — summarises the request log: totals, cache hits, top
  models by energy, and cumulative energy / CO₂ saved.
- **Carbon-aware router** (`--router carbon`) — routes to the provider whose
  `region` has the lowest grid carbon intensity, from a built-in regional table
  plus `carbon_overrides` (a live source is the next increment).
- **Semantic cache** (`--semantic-cache`) — embeds the prompt and reuses a past
  answer when cosine similarity clears a threshold, so differently-worded but
  equivalent prompts share one inference. Opt-in; uses an OpenAI-compatible
  embeddings endpoint (`--embed-model`, defaults to the upstream).

### Changed
- `token_source` gains a `cache` value for cache-served responses.

## [0.1.0] — 2026-06-27

First public release. Phase 1 (measure) plus the prompt-optimization and routing
pieces of Phases 2–3.

### Added
- **OpenAI-compatible measuring proxy** — meters `/v1/chat/completions`
  (streaming and buffered) and transparently passes through other `/v1/*`
  routes.
- **Token accounting** — prefers the provider's reported `usage`, falls back to
  a ~4-chars/token heuristic, and records which source was used.
- **Energy estimator** — converts tokens to joules, watt-hours, CO₂, and USD via
  per-model profiles and a configurable grid carbon intensity. Defaults
  calibrated to published benchmarks (Epoch AI, "How Hungry is AI?", and others).
- **Provider plugins** — OpenAI-compatible, Anthropic (`/v1/messages`), and
  Google Gemini (`generateContent`), each translating to/from the canonical
  OpenAI shape.
- **Router plugins** — `static`, `model` (route by supported model), and
  `greenest` (lowest estimated energy among candidate models).
- **Prompt optimizer** — composable, intensity-gated passes (`off`/`lite`/
  `full`/`ultra`) that strip redundant tokens and report exactly what they saved.
- **Observability** — Prometheus metrics at `/metrics`, a SQLite request log,
  and a `/stats` summary. Per-request `x-joule-*` response headers.
- **CLI** — `serve`, `estimate`, `optimize` (a standalone prompt improver), and
  `models`.
- Docker image, GitHub Actions CI (fmt + clippy + build + test), and a cited
  bibliography in `ROADMAP.md`.

### Known limitations
- Token counts use a heuristic, not a real tokenizer.
- Anthropic/Gemini streaming is forwarded in native SSE format (not re-framed to
  OpenAI chunks); accounting still works.
- Energy figures are calibrated estimates, not meter readings — they do not
  model batching, hardware generation, or data-center overhead.
- No semantic cache or carbon-aware scheduling yet (Phases 2 and 4).

[0.2.0]: https://github.com/wuisabel-gif/Joule/releases/tag/v0.2.0
[0.1.0]: https://github.com/wuisabel-gif/Joule/releases/tag/v0.1.0
