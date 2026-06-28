# Joule — Vision & Roadmap

Energy-aware optimization for LLM inference.

Joule is a systems project that makes energy a first-class metric in AI
applications. Instead of optimizing solely for latency, token count, or API
cost, Joule estimates the energy and carbon footprint of every inference and
automatically applies optimizations that reduce electricity consumption while
preserving answer quality.

The project is designed as middleware between applications and LLM providers.
Applications continue to use familiar APIs (OpenAI-compatible, Anthropic,
Gemini, local models), while Joule transparently measures, optimizes, caches,
and routes requests.

## Vision

Modern AI applications generate billions of inference requests every day.
Although each request may consume only a small amount of energy, the aggregate
demand translates into significant electricity usage, cooling requirements, and
carbon emissions across data centers.

Joule aims to answer a simple question:

> How many joules did this response cost, and could it have been lower?

The long-term goal is to make energy optimization as commonplace as latency
optimization.

## Design Principles

**Measure before optimizing.** Energy should be observable. Every inference
should expose latency, input/output tokens, estimated energy (J), electricity
(Wh), CO₂, and cost.

**Quality first.** Reducing energy should never significantly reduce answer
quality. Optimizations should be measurable and reversible.

**Model agnostic.** Joule should support any provider — OpenAI, Anthropic,
Google Gemini, Ollama, LM Studio, vLLM, llama.cpp. The optimization pipeline
should remain independent of the model vendor.

**Transparent.** Joule should explain every optimization. Nothing should happen
invisibly:

```
Optimization Summary

✓ Reduced prompt size by 18%
✓ Removed duplicated context
✓ Retrieved cached embedding
✓ Selected GPT-4o-mini
Estimated energy reduction: 41%
```

## Architecture

```
Application
      │
      ▼
+----------------+
|     Joule      |
+----------------+
| Proxy          |
| Optimizer      |
| Cache          |
| Router         |
| Estimator      |
| Metrics        |
+----------------+
      │
      ▼
LLM Provider
```

- **Proxy** — receives requests from applications (OpenAI-compatible first).
- **Optimizer** — applies optimization passes: remove duplicated context,
  compress prompts, shorten instructions, reduce verbosity, recommend output
  limits.
- **Router** — selects an appropriate model by quality, latency, cost, and
  estimated energy. Future versions add carbon-aware routing.
- **Cache** — avoids repeated inference (exact, semantic, and embedding caches).
- **Estimator** — estimates inference energy from model, hardware, token count,
  provider measurements, and published benchmark data.
- **Metrics** — exports metrics via OpenTelemetry and Prometheus
  (joules/request, joules/token, CO₂/request, cache hit rate, savings, latency).

## Non-goals

Joule is **not** a foundation model, an LLM framework, a prompt-engineering
assistant, a model-serving engine, or a benchmark suite. It is infrastructure
that improves the efficiency of existing AI systems.

## Technology

Preferred stack: Rust, Tokio, Axum, Serde, SQLite, OpenTelemetry, Prometheus.
The project compiles into a single portable binary.

## Code Style

Prefer small modules, explicit types, immutable data, composition over
inheritance, and descriptive error messages. Avoid hidden global state,
unnecessary macros, premature abstraction, and unsafe code unless clearly
justified.

## Roadmap

Status: ✅ done · 🟡 partial · ⬜ planned

### Phase 1 — Measure
- ✅ OpenAI-compatible proxy
- ✅ request metrics (Prometheus)
- ✅ token accounting (provider usage + heuristic fallback)
- ✅ energy estimation (calibrated per-model profiles)
- ✅ CLI (`serve`, `estimate`, `optimize`, `models`)

### Phase 2 — Optimize & cache
- ✅ prompt optimization passes (composable, intensity-gated, explainable)
- ✅ optimization reports (per-request headers + summaries)
- ⬜ semantic cache (exact / embedding / semantic)

### Phase 3 — Route
- ✅ intelligent model routing (`static`, `model`, `greenest`)
- 🟡 provider comparison (multi-provider registry: OpenAI, Anthropic, Gemini)
- ⬜ dashboard

### Phase 4 — Carbon-aware
- ⬜ carbon-aware scheduling
- ⬜ electricity grid integration
- ⬜ automatic optimization recommendations

## Philosophy

Modern software engineers routinely optimize execution time, memory
consumption, and network bandwidth. Joule argues that AI applications should
also optimize energy.

The cheapest inference is not necessarily the greenest, the fastest is not
necessarily the most efficient, and the largest model is not always required.

By making energy visible, measurable, and optimizable, Joule encourages a new
performance metric for AI systems: build AI that is not only intelligent, but
efficient.
