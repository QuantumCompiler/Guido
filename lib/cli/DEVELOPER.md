# Developer Guide

This document describes the layout of the `lib/cli` directory and what each file does. For user-facing documentation, see [README.md](README.md).

---

## Directory Layout

```
lib/cli/
├── Makefile            — Build orchestration (build, install, clean, test, build-embedded, …)
├── README.md           — User-facing documentation
├── DEVELOPER.md        — This file
├── .gitignore          — Excludes generated binaries and embed staging area
├── config.yaml         — Sample / template configuration (copied to ~/.guido/config/ on install)
├── go.mod              — Go module definition (module: guido/lib/cli)
├── go.sum              — Dependency checksums
│
├── src/                — All Go source code
│   ├── harness/        — Core abstraction layer (interfaces, types, config)
│   ├── backends/       — LLM provider implementations
│   ├── httpserver/     — HTTP server layer
│   ├── tools/          — guido-server lifecycle, built-in tool calling, embedded extraction
│   ├── embeddedtools/  — //go:embed staging package (data/ populated by make stage-embed; gitignored)
│   ├── mcp/            — MCP client (stdio, HTTP+SSE, Streamable HTTP transports)
│   └── cmd/            — Binary entry point
│       └── cli/        — guido binary (all subcommands: complete, chat, serve, harness, models)
│
├── exec/               — Runtime artifacts
│   ├── bin/            — Compiled binaries and llama.cpp tools
│   │   ├── guido            — Single self-contained binary with tools embedded (after make build)
│   │   └── guido-cpp-tools/ — Compiled llama.cpp executables (source for embedding; used directly in dev)
│   └── scripts/        — Build scripts
│       ├── build-llama.sh          — Compiles llama.cpp and copies tools to exec/bin/guido-cpp-tools/
│       └── create-py-wrappers.sh   — Generates shell wrappers for llama.cpp Python scripts
│
├── mcp/                — Built-in test MCP server
│   └── basic_mcp.py   — Test server (get_time, calculate, read_file, echo)
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
| `models.go` | All shared request/response types: `Config`, `BackendConfig`, `MCPServerConfig`, `ChatRequest`, `ChatResponse`, `ChatMessage`, `CompletionRequest`, `CompletionResponse`, `ModelInfo`, `ModelStatusInfo`, `Tool`, `ToolCall`, `ToolFunction`, `ToolCallFunction` |
| `content.go` | `MessageContent` — dual-mode type that serializes as a plain JSON string (text-only) or an OpenAI-style content-part array (multimodal). Also defines `ContentPart`, `ImageURL`, and helpers `Text()`, `Parts()`, `TextPart()`, `ImageURLPart()` |
| `config.go` | `LoadConfig()` — reads `config.yaml` via Viper, expands `${ENV_VAR}` references in string fields including MCP server `args` and `env` entries |
| `errors.go` | Sentinel error values: `ErrNoAvailableBackend`, `ErrModelNotFound`, `ErrInvalidConfig`, `ErrProviderNotRegistered` |
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

// Backends that embed tool calls inside the text stream (llamacpp system-prompt
// injection) implement this. The CLI agentic loop uses it to decide whether to
// take the streaming lookahead path for the final answer turn.
type InTextToolCaller interface {
    UsesInTextToolCalls() bool
}
```

`Harness.UsesInTextToolCalls(model string) bool` routes to the provider and returns false if it doesn't implement `InTextToolCaller`.

### Routing

`SimpleRouter.Route(model string)` resolves a model name to a provider:
1. Exact match on backend key name in config
2. Match against the `model:` field inside each backend config
3. Fall back to the configured default backend

---

## `src/backends/` — LLM provider implementations

| File | Provider | Notes |
|------|----------|-------|
| `llamacpp.go` | llama.cpp HTTP API | Talks to a running `guido-server` via REST. Used as the `inner` backend by `LazyLlamaCppBackend`. Tool calling uses **system-prompt injection** (see below) |
| `lazy_llamacpp.go` | Lazy-loading llama.cpp | State machine wrapper: `unloaded → loading → ready → unloaded`. Starts `guido-server` on the first request; optionally unloads after `idle_timeout_seconds` of inactivity. Implements `StatusReporter` |
| `openai.go` | OpenAI API | Full tool calling support via the `tools` / `tool_choice` API fields. Also used as the inner backend for Ollama |
| `ollama.go` | Ollama | Delegates all LLM calls to an inner `OpenAIBackend` pointed at Ollama's OpenAI-compatible endpoint. On first use, registers a GGUF file with Ollama via `ollama create` if `model_path` is set. Uses `sync.Once` so registration happens at most once per process |
| `anthropic.go` | Anthropic API | Translates OpenAI-style `image_url` content parts to Anthropic's `image/source` format |
| `huggingface.go` | HuggingFace Inference API | Cloud inference for HuggingFace-hosted models |
| `mock.go` | In-process mock | Returns canned responses; no network calls. Used in tests and for development without a running model |
| `sse.go` | SSE parsing helper | Shared server-sent events reader (`sseChunk` type) used by `llamacpp.go` and `openai.go` |

### Lazy backend state machine

```
unloaded ──► loading ──► ready ──► unloaded (idle timeout)
                    └──► errored ──► loading  (retry on next request)
```

`EnsureLoaded(ctx)` is called before every request. If two requests race during loading, the second waits on a shared channel rather than starting a second load.

### Tool calling in `llamacpp.go`

llama-server has a known bug in some versions where it returns HTTP 500 when a request includes a `tools` array. `LlamaCppBackend.Chat()` works around this transparently via **system-prompt injection**:

1. When `req.Tools` is non-empty, `Chat()` delegates to `chatWithToolPrompt()` instead of using the API `tools` field.
2. `toolSystemPrompt()` builds a system message listing every tool with its description and parameters schema, instructing the model to emit:
   ```
   TOOL_CALL: {"name": "tool_name", "arguments": {"key": "value"}}
   ```
3. `rewriteMessagesForTools()` converts the conversation history for a tool-unaware endpoint:
   - `role: "tool"` → `role: "user"` with `"Tool result for <name>: <content>"`
   - `role: "assistant"` with `tool_calls` → plain text with `TOOL_CALL:` lines
4. `parseToolCalls()` scans the model's text response with a regex and returns `[]harness.ToolCall`.
5. If tool calls are found, `FinishReason` is set to `"tool_calls"` so the agentic loop in `cmd/cli/main.go` continues.

This approach is backend-transparent — `LazyLlamaCppBackend` delegates to `LlamaCppBackend.Chat()` so it picks up the fix automatically.

**Streaming counterpart:** `streamChatWithToolPrompt()` applies the same message rewriting but calls guido-server with `"stream": true` and returns a `<-chan string` of raw tokens. This enables the CLI to stream the final answer to the terminal token-by-token. `StreamChat()` routes to it automatically when `req.Tools` is non-empty.

**Exported parser:** `ParseToolCalls(text string) []harness.ToolCall` wraps the internal `parseToolCalls` regex, allowing the CLI streaming loop to detect tool calls after buffering the full token stream without importing internal symbols.

---

## `src/httpserver/` — HTTP server

| File | Purpose |
|------|---------|
| `serve.go` | Registers all routes using `go-chi/chi`. Constructs the `Handler` (with optional `*ToolConfig`) and wires it to `harness.Harness` |
| `handler.go` | One method per endpoint. When `ToolConfig` is set, `HandleChat` runs the agentic tool loop internally before responding |

### `ToolConfig`

```go
type ToolConfig struct {
    Tools    []harness.Tool
    ExecTool func(ctx context.Context, name, argsJSON string) (string, error)
}
```

Passed to `NewHandler` and `Serve`. When non-nil, `HandleChat` runs the full tool loop and returns only the final text response — clients never see `finish_reason: "tool_calls"`. For streaming requests, the final answer is wrapped in a single SSE frame by `writeSSEText`.

### `Serve` signature

```go
func Serve(ctx context.Context, cfg *harness.Config, h *harness.Harness, tc *ToolConfig, onShutdown func()) error
```

`tc` is nil in `guido harness` mode (all backends, no tool injection). It is populated by `serveCmd` in `guido serve` when tool flags are active.

### Endpoints

| Method | Path | Handler |
|--------|------|---------|
| `POST` | `/v1/completions` | `HandleCompletion` |
| `POST` | `/v1/chat/completions` | `HandleChat` |
| `GET` | `/v1/models` | `HandleListModels` |
| `GET` | `/v1/model/status` | `HandleModelStatus` |
| `GET` | `/health` | `HandleHealth` |

`HandleModelStatus` calls `StatusReporter.ModelStatus()` on lazy backends without triggering a load — safe to poll from a GUI.

---

## `src/tools/` — guido-server lifecycle, built-in tool calling, embedded extraction

| File | Purpose |
|------|---------|
| `manager.go` | `Manager` — locates `guido-server`, starts it as a subprocess (`StartLlamaServer`), waits for it to become healthy, stops it on `Close()`. Has an optional `libDir` field: when set (embedded build), `DYLD_LIBRARY_PATH` / `LD_LIBRARY_PATH` is injected into the subprocess environment so the extracted dylibs are found at runtime |
| `extract.go` | `ExtractEmbedded(fs embed.FS, targetDir string)` — extracts the embedded tools from `embeddedtools.ToolsFS` to `~/.guido/tools/`. Version-stamped: if `targetDir/.version` already matches the build stamp, extraction is skipped. Returns `nil, nil` in stub (non-embedded) builds so the caller can fall through to the filesystem fallback |
| `toolcall.go` | `ExecuteTool(name, argsJSON)` — dispatches built-in tool calls from the model (`web_search`, `fetch_url`) to their implementations. MCP tool calls are routed through `mcp.Registry` before this is called |
| `web.go` | `WebTools()` returns the tool schemas for `web_search` and `fetch_url`. Implementations: DuckDuckGo search and plain-text URL fetching |

### Tools resolution order

The guido binary resolves the tools directory at startup:

1. `$GUIDO_TOOLS_DIR` — explicit override, useful for development or CI
2. `exec/bin/guido-cpp-tools` relative to CWD — the standard dev layout when running from the project root
3. `guido-cpp-tools/` adjacent to the binary — works for symlinked installs
4. Embedded extraction — `tools.ExtractEmbedded(embeddedtools.ToolsFS, "~/.guido/tools/")` — active only in binaries built with `-tags embed_tools`; no-op in regular builds

### Embedded dylibs and rpath

On macOS, the `guido-server` binary is dynamically linked against `@rpath/libllama.0.dylib` and several other llama.cpp / ggml shared libraries. The rpath baked at compile time points at the original build directory. When tools are extracted to `~/.guido/tools/`, those dylibs live in `~/.guido/tools/lib/`. `Manager` handles this by setting `DYLD_LIBRARY_PATH` (macOS) or `LD_LIBRARY_PATH` (Linux) to `libDir` when spawning guido-server — no binary patching required.

---

## `src/embeddedtools/` — embed staging package

| File | Purpose |
|------|---------|
| `fs_embedded.go` | Build tag `embed_tools` — declares `//go:embed all:data` and exports `var ToolsFS embed.FS`. Active only in `make build-embedded` builds. The `all:` prefix is required to include the hidden `.version` stamp file |
| `fs_stub.go` | Build tag `!embed_tools` (all regular builds) — exports a zero-value `embed.FS`. `tools.ExtractEmbedded` detects this and returns `nil, nil`, letting the CLI fall through to filesystem lookup |

The `data/` directory inside this package is populated by `make stage-embed` and is listed in `.gitignore`. It is never committed.

---

## `src/mcp/` — MCP client

Connects to external [Model Context Protocol](https://modelcontextprotocol.io) servers and exposes their tools to the LLM in the agentic loop. Zero external dependencies — stdlib only.

| File | Purpose |
|------|---------|
| `types.go` | JSON-RPC 2.0 wire types (`Request`, `Notification`, `Response`, `IncomingMessage`, `RPCError`) and MCP-specific payloads (`InitializeParams/Result`, `ToolsListResult`, `MCPTool`, `ToolCallParams`, `ToolCallResult`, `ToolContent`) |
| `transport.go` | `Transport` interface + `StdioTransport` — spawns a subprocess, pipes stdin/stdout, forwards stderr. 1 MB read buffer for large tool-list responses. Write-side mutex for thread safety |
| `transport_sse.go` | `SSETransport` — MCP spec 2024-11-05. A persistent `GET /sse` stream receives JSON-RPC responses; individual POST requests send messages. Session POST URL discovered from the first `endpoint` SSE event. Exports `ErrMethodNotAllowed` sentinel when the server returns HTTP 405 |
| `transport_streamable.go` | `StreamableTransport` — MCP spec 2025-03-26. Each `Send()` fires a POST to the MCP endpoint; the response (JSON or SSE body) is read in a background goroutine and forwarded to `recvCh`. Captures `Mcp-Session-Id` for session affinity. No persistent connection required |
| `client.go` | `Client` — single server connection. Background `readLoop` goroutine demultiplexes responses via `pending map[int64]chan Response`. `initialize()` performs the MCP handshake. `loadTools()` fetches `tools/list` and caches results as namespaced `harness.Tool` values |
| `registry.go` | `Registry` — multi-server manager. `NewRegistry` selects the transport automatically, connects best-effort, skips failures. `Tools()` returns the aggregated tool list. `ExecuteTool` returns `(result, handled, err)` — `handled=false` lets the caller fall through to built-in tools |

### Transport selection

`MCPServerConfig` in `harness/models.go` exposes both transport options:

```yaml
# stdio (local subprocess)
- name: local
  command: python3
  args: ["/path/to/server.py"]

# Remote — HTTP+SSE or Streamable HTTP, auto-detected
- name: remote
  url: "https://mcp.example.com"
  headers:
    Authorization: "Bearer ${TOKEN}"
```

`NewRegistry` selects the transport:
- `srv.Command` set → `StdioTransport`
- `srv.URL` set → tries `SSETransport`; if server returns HTTP 405, falls back to `StreamableTransport`
- Both set → `url` takes precedence

The 405 fallback is implemented via the `ErrMethodNotAllowed` sentinel exported from `transport_sse.go`:

```go
transport, err = NewSSETransport(ctx, srv.URL, srv.Headers)
if errors.Is(err, ErrMethodNotAllowed) {
    transport, err = NewStreamableTransport(ctx, srv.URL, srv.Headers)
}
```

### HTTP+SSE transport protocol (2024-11-05)

```
Client                              Server
  │                                   │
  ├─── GET /sse ──────────────────────►│  persistent SSE stream
  │◄── event: endpoint ───────────────┤  data: /messages?sessionId=abc
  │                                   │
  ├─── POST /messages?sessionId=abc ──►│  each JSON-RPC request
  │◄── event: message ────────────────┤  data: {"jsonrpc":"2.0","id":1,"result":{...}}
```

`SSETransport.Close()` cancels the GET context, which unblocks `readSSE` via Go's `net/http` context propagation.

### Streamable HTTP transport protocol (2025-03-26)

```
Client                          Server
  │                               │
  ├─── POST <url> ────────────────►│  JSON-RPC request body
  │◄── application/json ──────────┤  single response (or 202 for notifications)
  │       — or —                  │
  │◄── text/event-stream ─────────┤  streaming response (tool results, long outputs)
```

Each POST is independent. `Send()` fires the request and spawns a goroutine to read the response; `Recv()` blocks on the shared channel. The `readLoop` in `Client` is unaware of which transport is underneath.

### Tool namespacing

MCP tools are named `mcp__<server-name>__<tool-name>` (matching Claude Code conventions). For example, a tool `read_file` from a server named `filesystem` appears as `mcp__filesystem__read_file`. The double-underscore delimiter is chosen because tool names commonly contain single underscores.

### Concurrency model

Each `Client` runs one background `readLoop` goroutine. Callers use `client.call()` which places a channel in the `pending` map, sends the request, then blocks on the channel. The read loop pops responses and routes them by ID. Multiple concurrent tool calls are supported.

### Transport interface

`Transport` is the seam between `Client` (protocol) and connection details (subprocess vs. HTTP):

```go
type Transport interface {
    Send(msg interface{}) error   // thread-safe
    Recv() (json.RawMessage, error)
    Close() error
}
```

`Client` and `Registry` are transport-agnostic — adding a new transport (e.g. WebSocket) requires only implementing this interface.

---

## `src/cmd/cli/` — `guido` binary

Single `main.go` that uses `cobra` to expose five subcommands. All tool-related logic is centralized in package-level helpers so the three commands that support tools (`complete`, `chat`, `serve`) share identical behavior.

### Subcommands

| Command | Behavior |
|---------|----------|
| `complete <prompt>` | One-shot prompt → response. Runs the agentic loop when tools are active, streams directly when they are not |
| `chat` | Interactive multi-turn session. Agentic loop when tools active, streaming otherwise |
| `serve` | Persistent HTTP server with lazy-loading backends. Passes tool config to the HTTP handler |
| `harness` | Bare HTTP server — all backends, lazy loading, no tool injection. Replacement for the former `guido-harness` binary. Accepts `--llama-port` and `--llama-gpu-layers` to override defaults for embedded backends |
| `models` | Lists all models from all configured backends |

### Tool mode flags

All three active commands accept the same three mutually exclusive flags (enforced by `cobra.MarkFlagsMutuallyExclusive`):

| Flag | `useWeb` | `useMCP` |
|------|----------|----------|
| *(none)* | true | true |
| `--search` | true | false |
| `--mcp` | false | true |
| `--native` | false | false |

### Key helpers in `main.go`

| Helper | Purpose |
|--------|---------|
| `resolveToolMode()` | Reads `flagSearch`, `flagMCP`, `flagNative` and returns `(useWeb, useMCP bool)` |
| `setupTools(ctx, cfg, useWeb, useMCP)` | Connects to MCP servers (if `useMCP` and config has entries), calls `mcp.NewRegistry`, appends web tools and MCP tools to `activeTools`. Returns `([]harness.Tool, *mcp.Registry)` |
| `dispatchTool(ctx, tc, mcpReg)` | Executes one `harness.ToolCall` — tries `mcpReg.ExecuteTool` first, falls through to `tools.ExecuteTool` for built-ins |
| `runAgenticLoop(ctx, h, history, model, temp, maxTokens, tools, mcpReg, printProgress)` | Drives the model→tool→model cycle. Selects streaming or non-streaming inner loop based on `h.UsesInTextToolCalls(model)`. Appends each turn to `*history` in place. Returns the final assistant text |
| `streamingAgenticTurn(ctx, h, req, printProgress)` | Streaming turn for llamacpp backends. Opens `h.StreamChat`, buffers up to 10 non-whitespace chars to distinguish `TOOL_CALL:` responses (silent) from content responses (flushed + streamed live). Returns `(text, toolCalls, err)` |
| `dispatchToolCalls(ctx, calls, history, mcpReg, printProgress)` | Executes every `harness.ToolCall` in a slice, prints progress, and appends tool-result messages to `*history` |
| `filterBackends(cfg, target, all)` | Restricts `cfg.Backends` to a single backend before initialization |
| `initializeBackends(h, cfg, tm, lazy)` | Instantiates and registers all backends in the (filtered) config. `lazy=true` wraps embedded llamacpp backends in `LazyLlamaCppBackend` |
| `buildMessageContent(text, contexts, files, images)` | Assembles a `MessageContent` from CLI flags (`--context`, `--file`, `--image`) |

### Agentic tool-calling loop

`runAgenticLoop` is called by both `complete` and `chat` whenever `len(activeTools) > 0`. It is also mirrored inside `httpserver.Handler.runToolLoop` for the `serve` case.

**Streaming path (llamacpp):** When `h.UsesInTextToolCalls(model)` returns true (i.e. the backend satisfies `harness.InTextToolCaller`), each turn uses `streamingAgenticTurn`:

```
model turn (StreamChat)
  ├─ first ≥10 non-whitespace chars start with "TOOL_CALL:" ?
  │     YES → buffer silently, parse tool calls, loop
  │     NO  → flush buffer, stream remaining tokens live, done
  └─ short response (< lookahead) → treated as content
```

The 10-character lookahead is enough to distinguish `TOOL_CALL:` (10 chars) from any content response without noticeable latency.

**Non-streaming path (OpenAI, Anthropic):** `h.Chat()` is called; if `resp.Message.ToolCalls` is non-empty the loop continues, otherwise the final text is printed and returned.

**Tool dispatch order (both paths):**
1. `mcp.Registry.ExecuteTool` — handles `mcp__<server>__<tool>` names
2. `tools.ExecuteTool` — handles built-in names (`web_search`, `fetch_url`)

If `Registry.ExecuteTool` returns `handled=false`, the call falls through to step 2, so built-in tools always work regardless of MCP configuration.

---


## `exec/scripts/`

### `build-llama.sh`
Compiles the llama.cpp submodule and copies the resulting binaries to `exec/bin/guido-cpp-tools/`. Called by `make build-llama`. Detects OS/arch and sets appropriate CMake flags (Metal on macOS, CUDA on Linux with `nvidia-smi`).

### `create-py-wrappers.sh`
Generates thin shell wrappers in `exec/bin/guido-cpp-tools/` for the llama.cpp Python conversion scripts (`convert_hf_to_gguf.py`, etc.). The wrappers resolve the llama.cpp source directory at runtime relative to their own location (`../../../modules/llama.cpp`).

---

## `modules/llama.cpp/`

Git submodule pinned to `github.com/ggml-org/llama.cpp`. The Go code does not import anything from here directly — it's a C++ project compiled separately by `build-llama.sh`. Guido uses the resulting `guido-server` executable to serve local GGUF models.

---

## Build system

```
make build           # full build: build-llama → stage-embed → build-embedded (self-contained binary)
make build-llama     # cmake build of llama.cpp → exec/bin/guido-cpp-tools/
make stage-embed     # copy executables + dylibs into src/embeddedtools/data/, write .version stamp
make build-embedded  # stage-embed + go build -tags embed_tools → exec/bin/guido (~34 MB, self-contained)
make build-go        # go build (no embed tag) — fast dev build, finds tools on filesystem
make dev-build       # alias for build-go
make install         # build + copy config to ~/.guido/config/ + /usr/local/bin symlinks
make symlinks        # (re)create /usr/local/bin symlinks only
make clean           # remove exec/bin/ and modules/llama.cpp/build/
make test            # go test ./src/...
```

**When to use each:**
- `make build` — distribution builds; produces a binary you can copy anywhere
- `make build-go` / `make dev-build` — everyday development; fast, no re-embedding
- `make stage-embed` alone — after `make build-llama`, if you want to inspect what gets embedded before building

The `src/embeddedtools/data/` directory is listed in `.gitignore` and should never be committed.

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
2. Add a case to `initializeBackends()` in `src/cmd/cli/main.go`
3. Add a config entry under `backends:` in `config.yaml`
4. (Optional) Add fields to `harness.BackendConfig` in `src/harness/models.go` if you need new config keys

## Adding a new HTTP endpoint

1. Add a handler method to the `Handler` struct in `src/httpserver/handler.go`
2. Register the route in `src/httpserver/serve.go`
3. If the endpoint needs to read model state without triggering a load, use `harness.StatusReporter` (see `HandleModelStatus` for the pattern)

## Adding a new MCP transport

1. Implement the `mcp.Transport` interface in a new file `src/mcp/transport_<name>.go`
2. Add a constructor that returns `(*YourTransport, error)`
3. If the new transport is a fallback for an existing one (like Streamable HTTP is for SSE), export a sentinel error from the existing transport and check `errors.Is` in `registry.go`
4. Otherwise, add a field to `MCPServerConfig` in `harness/models.go` and dispatch in `NewRegistry` based on that field
5. No changes needed to `Client` — it operates on the `Transport` interface and is transport-agnostic

## Adding a new built-in tool

1. Add the tool schema to `WebTools()` (or a new `MyTools()` function) in `src/tools/web.go`
2. Add a dispatch case in `ExecuteTool()` in `src/tools/toolcall.go`
3. Append the new tool list in `setupTools()` in `src/cmd/cli/main.go`
