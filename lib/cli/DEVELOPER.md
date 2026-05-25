# Developer Guide

This document describes the layout of the `lib/cli` directory and what each file does. For user-facing documentation, see [README.md](README.md).

---

## Directory Layout

```
lib/cli/
├── Makefile            — Build orchestration (build, install, clean, test)
├── README.md           — User-facing documentation
├── DEVELOPER.md        — This file
├── config.yaml         — Sample / template configuration (copied to ~/.guido/config/ on install)
├── go.mod              — Go module definition (module: guido/lib/cli)
├── go.sum              — Dependency checksums
│
├── src/                — All Go source code
│   ├── harness/        — Core abstraction layer (interfaces, types, config)
│   ├── backends/       — LLM provider implementations
│   ├── httpserver/     — HTTP server layer
│   ├── tools/          — llama-server lifecycle + built-in tool calling
│   ├── mcp/            — MCP client (connect to external MCP servers, expose their tools)
│   └── cmd/            — Binary entry points
│       ├── cli/        — guido CLI binary
│       └── harness/    — guido-harness HTTP-only server binary
│
├── exec/               — Runtime artifacts
│   ├── bin/            — Compiled binaries and embedded llama.cpp tools
│   │   ├── guido            — Main CLI binary (after make build)
│   │   ├── guido-harness    — HTTP-only server binary (after make build)
│   │   └── llama-cpp-tools/ — Compiled llama.cpp executables + Python wrappers
│   └── scripts/        — Build scripts
│       ├── build-llama.sh          — Compiles llama.cpp and copies tools to exec/bin/llama-cpp-tools/
│       └── create-py-wrappers.sh   — Generates shell wrappers for llama.cpp Python scripts
│
└── modules/            — Git submodules
    └── llama.cpp/      — llama.cpp source (github.com/ggml-org/llama.cpp)
```

---

## `src/harness/` — Core abstraction layer

The harness defines the interfaces and types that every other package builds on.

| File | Purpose |
|------|---------|
| `llm.go` | `LLMProvider` interface, `Harness` struct (provider registry + router), `SimpleRouter`, `StatusReporter` interface |
| `models.go` | All shared request/response types: `Config`, `BackendConfig`, `MCPServerConfig`, `ChatRequest`, `ChatResponse`, `ChatMessage`, `CompletionRequest`, `CompletionResponse`, `ModelInfo`, `ModelStatusInfo` |
| `content.go` | `MessageContent` — dual-mode type that serializes as a plain JSON string (text-only) or an OpenAI-style content-part array (multimodal). Also defines `ContentPart`, `ImageURL`, and helpers `Text()`, `Parts()`, `TextPart()`, `ImageURLPart()` |
| `config.go` | `LoadConfig()` — reads `config.yaml` via Viper, expands `${ENV_VAR}` references in string fields including MCP server `args` and `env` entries |
| `errors.go` | Shared error types |

### Key interfaces

```go
// Every backend implements this.
type LLMProvider interface {
    Complete(ctx, *CompletionRequest) (*CompletionResponse, error)
    StreamTokens(ctx, *CompletionRequest) (<-chan string, error)
    Chat(ctx, *ChatRequest) (*ChatResponse, error)
    StreamChat(ctx, *ChatRequest) (<-chan string, error)
    ListModels(ctx) ([]ModelInfo, error)
}

// Lazy backends additionally implement this so the HTTP layer can
// report load state without triggering a load.
type StatusReporter interface {
    ModelStatus() ModelStatusInfo
}
```

### Routing

`SimpleRouter.Route(model string)` resolves a model name to a provider:
1. Exact match on backend key name in config
2. Match against the `model:` field inside each backend config
3. Fall back to the configured default backend

---

## `src/backends/` — LLM provider implementations

| File | Provider | Notes |
|------|----------|-------|
| `llamacpp.go` | llama.cpp HTTP API | Talks to a running `llama-server` via REST. Used as the `inner` backend by `LazyLlamaCppBackend` |
| `lazy_llamacpp.go` | Lazy-loading llama.cpp | State machine wrapper: `unloaded → loading → ready → unloaded`. Starts `llama-server` on the first request; optionally unloads after `idle_timeout_seconds` of inactivity. Implements `StatusReporter` |
| `openai.go` | OpenAI API | Streaming and non-streaming chat + completions via `api.openai.com` |
| `anthropic.go` | Anthropic API | Translates OpenAI-style `image_url` content parts to Anthropic's `image/source` format |
| `huggingface.go` | HuggingFace Inference API | Cloud inference for HuggingFace-hosted models |
| `mock.go` | In-process mock | Returns canned responses; no network calls. Used in tests and for development without a running model |
| `sse.go` | SSE parsing helper | Shared server-sent events reader used by `llamacpp.go` and `openai.go` |

### Lazy backend state machine

```
unloaded ──► loading ──► ready ──► unloaded (idle timeout)
                    └──► errored ──► loading  (retry on next request)
```

`EnsureLoaded(ctx)` is called before every request. If two requests race during loading, the second waits on a shared channel rather than starting a second load.

---

## `src/httpserver/` — HTTP server

| File | Purpose |
|------|---------|
| `serve.go` | Registers all routes using `go-chi/chi`. Constructs the `Handler` and wires it to `harness.Harness` |
| `handler.go` | One method per endpoint: `HandleCompletions`, `HandleChatCompletions`, `HandleModels`, `HandleModelStatus`, `HandleHealth`. `HandleModelStatus` calls `StatusReporter.ModelStatus()` on lazy backends without triggering a load |

### Endpoints

| Method | Path | Handler |
|--------|------|---------|
| `POST` | `/v1/completions` | `HandleCompletions` |
| `POST` | `/v1/chat/completions` | `HandleChatCompletions` |
| `GET` | `/v1/models` | `HandleModels` |
| `GET` | `/v1/model/status` | `HandleModelStatus` |
| `GET` | `/health` | `HandleHealth` |

---

## `src/tools/` — llama-server lifecycle and built-in tool calling

| File | Purpose |
|------|---------|
| `manager.go` | `Manager` — locates `llama-server` in `exec/bin/llama-cpp-tools/`, starts it as a subprocess (`StartLlamaServer`), waits for it to become healthy, and stops it on `Close()`. Accepts optional `mmProjPath` for vision models |
| `toolcall.go` | `ExecuteTool(name, argsJSON)` — dispatches built-in tool calls from the model (`web_search`, `fetch_url`) to their implementations. MCP tool calls are routed through `mcp.Registry` before this is called |
| `web.go` | `WebTools()` returns the tool schemas for `web_search` and `fetch_url`. Implementations: DuckDuckGo search and plain-text URL fetching |

---

## `src/mcp/` — MCP client

Connects to external [Model Context Protocol](https://modelcontextprotocol.io) servers and exposes their tools to the LLM in the agentic loop. Uses JSON-RPC 2.0 over stdio (subprocess) transport. Zero external dependencies — stdlib only.

| File | Purpose |
|------|---------|
| `types.go` | JSON-RPC 2.0 wire types (`Request`, `Notification`, `Response`, `IncomingMessage`, `RPCError`) and MCP-specific payloads (`InitializeParams/Result`, `ToolsListResult`, `MCPTool`, `ToolCallParams`, `ToolCallResult`, `ToolContent`) |
| `transport.go` | `Transport` interface + `StdioTransport` — spawns a subprocess, pipes stdin/stdout, forwards stderr. 1 MB read buffer for large tool-list responses. Write-side mutex for thread safety |
| `client.go` | `Client` — single server connection. Background `readLoop` goroutine demultiplexes responses via `pending map[int64]chan Response`. `initialize()` performs the MCP handshake. `loadTools()` fetches `tools/list` and caches results as namespaced `harness.Tool` values |
| `registry.go` | `Registry` — multi-server manager. `NewRegistry` connects best-effort (failed servers are logged and skipped). `Tools()` returns the aggregated tool list. `ExecuteTool(ctx, name, argsJSON)` returns `(result, handled, err)` — `handled=false` signals the caller to fall through to built-in tools |

### Tool namespacing

MCP tools are named `mcp__<server-name>__<tool-name>` (matching Claude Code conventions). For example, a tool `read_file` from a server named `filesystem` appears as `mcp__filesystem__read_file`. The double-underscore delimiter is chosen because tool names commonly contain single underscores.

### Concurrency model

Each `Client` runs one background `readLoop` goroutine. Callers use `client.call()` which places a channel in the `pending` map, sends the request, then blocks on the channel. The read loop pops responses and routes them by ID. Multiple concurrent tool calls are supported — the pending map handles fan-out and fan-in.

### Transport interface

`Transport` is the seam for adding alternative transports:

```go
type Transport interface {
    Send(msg interface{}) error
    Recv() (json.RawMessage, error)
    Close() error
}
```

Phase 2 will add `SSETransport` (HTTP server-sent events for remote MCP servers) without changing `Client` or `Registry`.

---

## `src/cmd/cli/` — `guido` CLI binary

Single `main.go` that uses `cobra` to expose four subcommands:

| Command | Behavior |
|---------|----------|
| `complete <prompt>` | One-shot prompt → response. Starts backend eagerly, cleans up on exit |
| `chat` | Interactive multi-turn session. Maintains in-memory history. Backends started eagerly |
| `serve` | Persistent HTTP server. Backends use lazy loading (start on first request) |
| `models` | Lists all models from all configured backends |

**Key helpers in `main.go`:**

- `filterBackends(cfg, target, all)` — restricts `cfg.Backends` to a single backend before initialization. Called by `complete`, `chat`, and `serve` using the `-m` flag value
- `initializeBackends(h, cfg, tm, lazy)` — instantiates and registers all backends in the (filtered) config. `lazy=true` wraps embedded llamacpp backends in `LazyLlamaCppBackend`; `lazy=false` starts them eagerly
- `buildMessageContent(text, contexts, files, images)` — assembles a `MessageContent` from CLI flags (`--context`, `--file`, `--image`)
- `llamaServerStatus` / `waitForServer` — used in the eager (non-lazy) path to detect and reuse an already-running `llama-server`

**Agentic tool-calling loop** (inside `chat`):

The loop activates whenever there are active tools — either `--search` is set or MCP servers are configured. It runs non-streaming so it can inspect `tool_calls` in the response before printing, and loops until the model returns a final answer with no tool calls.

Tool dispatch order:
1. `mcp.Registry.ExecuteTool` — handles `mcp__<server>__<tool>` names
2. `tools.ExecuteTool` — handles built-in names (`web_search`, `fetch_url`)

If `Registry.ExecuteTool` returns `handled=false` the call falls through to step 2, so built-in tools always work regardless of MCP configuration.

---

## `src/cmd/harness/` — `guido-harness` binary

Minimal HTTP-only server entry point. No `cobra`/CLI flags beyond `-config`, `-llama-port`, `-llama-gpu-layers`. Always uses lazy loading for embedded backends. Intended to be embedded in a GUI application that manages the process lifetime directly.

---

## `exec/scripts/`

### `build-llama.sh`
Compiles the llama.cpp submodule and copies the resulting binaries to `exec/bin/llama-cpp-tools/`. Called by `make build-llama`. Detects OS/arch and sets appropriate CMake flags (Metal on macOS, CUDA on Linux with `nvidia-smi`).

### `create-py-wrappers.sh`
Generates thin shell wrappers in `exec/bin/llama-cpp-tools/` for the llama.cpp Python conversion scripts (`convert_hf_to_gguf.py`, etc.). The wrappers resolve the llama.cpp source directory at runtime relative to their own location (`../../../modules/llama.cpp`).

---

## `modules/llama.cpp/`

Git submodule pinned to `github.com/ggml-org/llama.cpp`. The Go code does not import anything from here directly — it's a C++ project compiled separately by `build-llama.sh`. Guido uses the resulting `llama-server` executable to serve local GGUF models.

---

## Build system

```
make build          # build-llama + build-go
make build-llama    # cmake build of llama.cpp → exec/bin/llama-cpp-tools/
make build-go       # go build → exec/bin/guido + exec/bin/guido-harness
make install        # build + copy binary to ~/bin/guido + config to ~/.guido/config/
make clean          # remove exec/bin/ and modules/llama.cpp/build/
make test           # go test ./src/...
make dev-build      # go build only (fast, skips llama.cpp)
```

The `go.mod` module root is `lib/cli`, so all import paths are prefixed `guido/lib/cli/src/`:

```go
import (
    "guido/lib/cli/src/harness"
    "guido/lib/cli/src/backends"
    "guido/lib/cli/src/httpserver"
    "guido/lib/cli/src/mcp"
    "guido/lib/cli/src/tools"
)
```

---

## Adding a new backend

1. Create `src/backends/myprovider.go` with a struct implementing `harness.LLMProvider`
2. Add a case to `initializeBackends()` in `src/cmd/cli/main.go` (and mirror it in `src/cmd/harness/main.go`)
3. Add a config entry under `backends:` in `config.yaml`
4. (Optional) Add fields to `harness.BackendConfig` in `src/harness/models.go` if you need new config keys

## Adding a new HTTP endpoint

1. Add a handler method to the `Handler` struct in `src/httpserver/handler.go`
2. Register the route in `src/httpserver/serve.go`
3. If the endpoint needs to read model state without triggering a load, use `harness.StatusReporter` (see `HandleModelStatus` for the pattern)

## Adding a new MCP transport

1. Implement the `mcp.Transport` interface in a new file `src/mcp/transport_<name>.go`
2. Add a constructor (`NewSSETransport`, etc.) that returns a `*YourTransport`
3. In `mcp.Registry.NewRegistry`, select the transport based on whether `MCPServerConfig.URL` or `MCPServerConfig.Command` is set
4. No changes needed to `Client` or `Registry` — they operate on the `Transport` interface
