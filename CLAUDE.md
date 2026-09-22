# CLAUDE.md

Context for Claude when working in this repository. Every statement below is
verifiable in the code; where this file or other docs disagree with the code, the code
wins.

## What this repo is

An HTTP vector-memory store proxy for AI agents, written in Rust (edition 2021) on
axum + tokio. It embeds natural-language text through pluggable providers (Ollama,
OpenAI-compatible, Anthropic/Voyage AI as `claude`), stores memories in an in-memory
store and optionally in a Qdrant vector database, and can track sessions in SQLite
with optional API-key auth. Single binary: `penr-oz-agent-memory` (`src/main.rs`).

## Where the detail lives (link, don't re-derive)

- [AGENTS.md](AGENTS.md) — contributor guide: full module map, coding conventions,
  API modification rules (reserved metadata keys, provider selection, auth, optional
  features), commit-message format, and test expectations. **Follow it for any code
  change.**
- [README.md](README.md) — setup (Docker Compose and bare `cargo run`), the full API
  endpoint table, and the configuration reference for providers, Qdrant, and sessions.
- [config.toml](config.toml) — commented reference config; `src/config.rs` documents
  the environment-variable overrides.

## Module map (one line each; full table in AGENTS.md)

| Path | Responsibility |
|------|---------------|
| `src/main.rs` | Startup, `Router` wiring, `AppState`; all routes registered here |
| `src/config.rs` | `Config` from `config.toml` + git-ignored `config.local.toml` merge + `QDRANT_*` / `DATABASE_URL` env overrides |
| `src/routes.rs` | All axum handlers, request/response types, session auth helper |
| `src/memory.rs` | In-memory `MemoryStore` (`/memory` endpoints) |
| `src/vector_store.rs` | `QdrantStore` (`/api/memory`, `/api/search`) |
| `src/session_store.rs` | SQLite `SessionStore`; runs `migrations/` at startup |
| `src/embedding/` | `EmbeddingProvider` trait + registry; `ollama`/`openai`/`claude` |
| `src/error.rs` | Typed error enums implementing `IntoResponse` |
| `examples/agent_client.rs` | Self-contained demo client (full memory lifecycle) |

## Commands

```sh
cargo check                       # compiles without external services
cargo test                        # wiremock + in-memory SQLite; no live services or keys
cargo run                         # needs config.toml (path override: CONFIG_PATH)
cargo run --example agent_client  # needs a running server; optional URL argument
docker compose up -d              # Qdrant only; add --profile ollama for local embeddings
```

CI (`.github/workflows/ci.yml`) runs: `cargo fmt --all -- --check`,
`cargo clippy --all-targets --all-features -- -D warnings`, build, test.

## Conventions

See AGENTS.md for the authoritative list. Highlights: typed errors from `src/error.rs`
returned directly from handlers (no `anyhow`, no `unwrap` in handler code), `tracing`
with structured fields, request/response types next to their handler in
`src/routes.rs`, new config keys mirrored in `config.toml` (commented out if
optional). Note `thiserror` is pinned at 1.x here.

## Gotchas

- AGENTS.md claims there is no separate lint step, but CI **does** enforce
  `cargo fmt --all -- --check` and
  `cargo clippy --all-targets --all-features -- -D warnings` — run both before
  pushing.
- Qdrant is optional: without a `[qdrant]` section, `/api/memory` and `/api/search`
  return 503 while everything else works. `QDRANT_URL` alone can enable Qdrant via
  env; `QDRANT_COLLECTION` / `QDRANT_API_KEY` only override an already-enabled config
  and are warned about (ignored) otherwise.
- Sessions are optional: enabled by a `[database]` section or the `DATABASE_URL` env
  var. Migrations run automatically at startup via `sqlx::migrate!("./migrations")`.
- `SESSION_API_KEY` (env, not config.toml): when set, all `/api/sessions` endpoints
  require an `X-Api-Key` header; when unset they are unauthenticated.
- `.env` is consumed by Docker Compose only. The server reads env vars from the
  shell — exporting them in `.env` does nothing for `cargo run`.
- `"text"` and `"session_id"` are reserved metadata keys in the Qdrant payload
  (checked in `src/vector_store.rs`); do not remove or rename those checks.
- Embedding handlers honour a per-request `?provider=<name>` override; never
  hard-code a provider name in a handler.
- An embedding provider that cannot be reached (`EmbeddingError::HttpError`,
  always raised by a `send()` call) surfaces as **502** with the provider URL and
  the underlying cause; `503` stays reserved for "not configured". A bare
  `cargo run` with no Ollama running therefore 502s on `/api/embed`, `POST
  /memory` and `GET /memory/search`.
- Qdrant `dimensions` must match the embedding model's output (`nomic-embed-text`
  768, `text-embedding-3-small` 1536, `voyage-3` 1024).
- There are two parallel memory APIs: `/memory*` (in-memory store) and `/api/memory`
  + `/api/search` (Qdrant). They are separate stores — data written to one is not
  visible in the other.
