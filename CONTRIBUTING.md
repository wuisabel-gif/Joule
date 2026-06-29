# Contributing to Joule

Thanks for your interest in Joule — energy-aware optimization for LLM inference.
Contributions of all kinds are welcome: bug reports, docs, new providers,
routers, optimization passes, and energy-data improvements.

By participating you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Getting started

You need a recent stable Rust toolchain (1.96+).

```sh
git clone https://github.com/wuisabel-gif/Joule
cd Joule
cargo build
cargo test
```

Run it against a local model with no API key (see the README for the full
quickstart):

```sh
cargo run -- serve --upstream http://localhost:11434   # Ollama
```

## Before you open a PR

CI runs the same checks — please run them locally first:

```sh
cargo fmt --all            # format
cargo clippy --all-targets -- -D warnings   # lint (warnings are errors)
cargo test                 # unit tests
```

All three must pass. Add tests for new behavior; if a change is observable
through the proxy, verify it end-to-end (the existing features were each tested
against a small mock upstream — see the PR history for examples).

## How the code is organized

Joule is a single binary built from small, focused modules:

| Area | Where | What |
|------|-------|------|
| Proxy / handlers | `src/proxy/` | request flow, streaming, headers |
| Providers | `src/provider/` | vendor wire protocols (OpenAI, Anthropic, Gemini) |
| Routers | `src/router/` | provider/model selection policies |
| Optimizer | `src/optimizer/` | prompt-optimization passes |
| Cache | `src/cache.rs`, `src/semantic.rs` | exact + semantic response caches |
| Estimator | `src/estimator/` | tokens → joules / CO₂ / cost |
| Metrics / store | `src/metrics.rs`, `src/store.rs` | Prometheus + SQLite |

A few design rules to keep in mind:

- **Providers are declarative.** They build the upstream request and parse
  tokens out of responses; HTTP execution, streaming, and metrics stay in the
  proxy so every provider is measured identically.
- **Plugins implement a trait.** Adding a provider, router, or optimization pass
  means implementing `Provider`, `Router`, or `Pass` and registering it — no
  changes to the request flow.
- **Nothing happens invisibly.** If a change alters a request or response,
  surface it (a header, the optimization report, a metric).
- **Honesty over precision.** Energy figures are calibrated estimates, not meter
  readings — keep them traceable (see the README's "Grounding in measured data")
  and never report a saving you didn't make.

### Adding things

- **A provider** → implement `Provider` in `src/provider/`, add a `ProviderKind`
  variant in `src/config.rs`, and wire it in `build_registry`.
- **A router** → implement `Router` in `src/router/`, add a `RouterKind` variant,
  and wire it in `build_router`.
- **An optimization pass** → implement `Pass` in `src/optimizer/passes.rs` and
  add it to `default_passes()`; gate it behind the right `OptLevel`.
- **A model profile** → add an entry to `src/estimator/models.rs` with a source
  comment for the per-token figures.

## Commit and PR conventions

- Keep PRs focused; one feature or fix per PR where practical.
- Write clear commit messages: a short imperative summary line, then a body
  explaining the *why*.
- Update `CHANGELOG.md` (under "Unreleased" / the next version) and any relevant
  docs when behavior changes.

## License

By contributing, you agree that your contributions will be licensed under the
project's [Apache-2.0](LICENSE) license.
