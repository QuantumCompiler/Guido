# Guido — LLM Harness

A unified Go-based harness for local and cloud LLM models. Run a single model from the command line, start a persistent OpenAI-compatible HTTP server, or use it as a library — all from one binary with embedded llama.cpp tooling.

---

## Quick Start

### Build & Install

```bash
cd lib/cli
make build    # compile llama.cpp + build guido binary
make install  # install to ~/bin/guido + config to ~/.guido/config/
```

`make install` places the binary at `~/bin/guido` and writes a starter config to `~/.guido/config/config.yaml` (skipped if the file already exists). It also creates `/usr/local/bin` symlinks for `guido`, `guido-harness`, and every tool in `exec/bin/llama-cpp-tools/`.

### Configure

Edit `~/.guido/config/config.yaml`:

```yaml
server:
  port: 8080

models:
  default: my-model   # which backend to use when -m is not specified

backends:
  my-model:
    type: llamacpp
    url: "embedded"   # auto-starts embedded llama-server on first request
    port: 8002
    model: "gemma4"
    model_path: "${HOME}/.cache/huggingface/hub/.../model.gguf"
    idle_timeout_seconds: 300   # unload from VRAM after 5 min idle (0 = never)

  mock:
    model: "test-model"   # no model file needed, useful for testing
```

### Use It

```bash
guido complete "Explain Go interfaces in one paragraph"
guido chat
guido serve          # OpenAI-compatible HTTP server on port 8080
```

---

## Commands

### `complete` — one-shot prompt

```bash
guido complete "<prompt>" [flags]
```

Sends a single prompt and prints the response, then exits. Any embedded llama-server started for this invocation is stopped on exit. When tools are active (the default), `complete` runs the full agentic loop and prints the final answer.

```bash
# Use the default model
guido complete "What is a transformer?"

# Use a specific backend
guido complete "Solve 3x + 7 = 22 step by step" -m my-reasoning-model

# Attach files and images
guido complete "Summarize this document" --file report.pdf
guido complete "What's in this image?" --image screenshot.png
guido complete "Explain this code" --file main.go --context "Focus on error handling"

# Tool mode control
guido complete "What time is it?"            # all tools (default)
guido complete "What time is it?" --search   # web search only
guido complete "What time is it?" --mcp      # MCP tools only
guido complete "What time is it?" --native   # no tools, plain model response
```

**Flags**

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--model` | `-m` | config default | Backend to initialize and query |
| `--temperature` | `-t` | `0.7` | Sampling temperature |
| `--max-tokens` | `-n` | `-1` (unlimited) | Max tokens to generate |
| `--context` | | | Raw string injected before the prompt |
| `--file` | | | File to attach (text → inline, image → base64) |
| `--image` | | | Image file to attach (base64-encoded) |
| `--all-backends` | | `false` | Initialize every configured backend instead of just the target |
| `--search` | | | Web search tools only (disables MCP) |
| `--mcp` | | | MCP tools only (disables web search) |
| `--native` | | | No tools — native model capabilities only |

`--search`, `--mcp`, and `--native` are mutually exclusive. Omitting all three enables all available tools.

---

### `chat` — interactive session

```bash
guido chat [flags]
```

Starts a multi-turn conversation in your terminal. Full message history is maintained in memory and re-sent each request (llama-server's prompt cache speeds up repeated prefixes). Type `exit` or press Ctrl+C to quit.

When tools are active the response loop is non-streaming — the model can call tools before producing a final answer. Without tools, responses stream token-by-token.

```bash
# All available tools (MCP from config + web search)
guido chat

# Web search only
guido chat --search

# MCP tools only
guido chat --mcp

# No tools — plain conversation
guido chat --native

# Specific model with a persona
guido chat -m my-model --system "You are a concise technical assistant."

# Attach context to the first message only
guido chat --image diagram.png --file architecture.md
```

**Flags** — same as `complete`, plus:

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--system` | `-s` | | System prompt injected as the first message |
| `--search` | | | Web search tools only |
| `--mcp` | | | MCP tools only |
| `--native` | | | No tools |

---

### `serve` — OpenAI-compatible HTTP server

```bash
guido serve [flags]
```

Starts a persistent HTTP server. Embedded llama-server processes use **lazy loading** — they start on the first request and optionally unload after the configured idle timeout. The server itself starts instantly with no VRAM usage.

When tool flags are active, the server runs the **agentic loop internally**: it calls tools on the model's behalf and returns only the final text response to clients. Clients do not need to implement tool calling themselves.

```bash
# All tools (default) — server handles MCP + web search internally
guido serve

# Serve a specific backend
guido serve -m my-model

# Web search only
guido serve --search

# No tool injection
guido serve --native

# Serve all configured backends (multi-model server)
guido serve --all-backends
```

**Flags**

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--model` | `-m` | config default | Backend to serve |
| `--all-backends` | | `false` | Serve every configured backend |
| `--search` | | | Web search tools only |
| `--mcp` | | | MCP tools only |
| `--native` | | | No tool injection |

---

### `models` — list available models

```bash
guido models
```

Queries all configured backends and prints their available models.

---

## Tool Modes

All three commands (`complete`, `chat`, `serve`) share the same tool flag semantics:

| Flag | Web Search | MCP Tools |
|------|-----------|-----------|
| *(none)* | ✓ | ✓ (if configured) |
| `--search` | ✓ | ✗ |
| `--mcp` | ✗ | ✓ (if configured) |
| `--native` | ✗ | ✗ |

Only one flag may be set at a time — Guido will error if two are combined.

### How tool calling works with local models

llama-server has a known serialization bug with its native tool-call API. Guido works around this transparently using **system-prompt injection**: tool definitions are described in a system message and the model is instructed to emit `TOOL_CALL: {"name": "...", "arguments": {...}}` lines. Guido parses these, dispatches the calls, and feeds results back — the end result is identical to native tool calling from the user's perspective.

---

## HTTP API

When running `guido serve`, the server exposes an OpenAI-compatible API on the configured port (default `8080`).

### Endpoints

#### `POST /v1/completions`
```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "model": "my-model", "max_tokens": 512, "stream": false}'
```

#### `POST /v1/chat/completions`
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "What is Go?"}],
    "stream": true
  }'
```

When `guido serve` is started with tool flags, this endpoint runs the full agentic loop and returns the final answer. The `tool_calls` turns are hidden from the client.

Multimodal messages use the OpenAI content-part format:
```json
{
  "model": "my-vision-model",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/..."}},
      {"type": "text", "text": "What is in this image?"}
    ]
  }]
}
```

#### `GET /v1/models`
```bash
curl http://localhost:8080/v1/models
```

#### `GET /v1/model/status`
Returns the load state of lazy backends (useful for a GUI loading indicator):
```bash
# All backends
curl http://localhost:8080/v1/model/status

# Specific backend
curl "http://localhost:8080/v1/model/status?backend=my-model"
```

Response:
```json
{
  "backends": {
    "my-model": {"model": "gemma4", "status": "ready", "idle_seconds": 42}
  }
}
```

States: `unloaded` → `loading` → `ready` → `unloaded` (after idle timeout)

#### `GET /health`
```bash
curl http://localhost:8080/health
```

---

## Configuration Reference

```yaml
server:
  port: 8080

models:
  default: my-model   # backend key used when no -m flag is given

backends:

  # ── Local model (embedded llama-server) ─────────────────────────────────────
  my-model:
    type: llamacpp
    url: "embedded"                           # guido manages the llama-server process
    port: 8002                                # port for this model's llama-server
    model: "gemma4"                           # model name reported to clients
    model_path: "${HOME}/.../model.gguf"
    mmproj_path: "${HOME}/.../mmproj.gguf"    # optional — required for vision models
    idle_timeout_seconds: 300                 # 0 = stay loaded until server stops
    gpu_layers: 99                            # layers to offload to GPU

  # ── External llama-server (you manage it) ───────────────────────────────────
  external:
    type: llamacpp
    url: "http://192.168.1.50:8000"
    model: "my-remote-model"

  # ── Ollama (local daemon — run `ollama serve` first) ────────────────────────
  # If model_path is set, Guido registers the GGUF with Ollama on first use.
  ollama:
    type: ollama
    model: "llama3.2"                         # any model pulled with `ollama pull`
    url: "http://localhost:11434"             # optional, this is the default
    model_path: "${HOME}/.../model.gguf"      # optional — auto-registers via `ollama create`

  # ── OpenAI ──────────────────────────────────────────────────────────────────
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4o"

  # ── Anthropic ───────────────────────────────────────────────────────────────
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    model: "claude-3-5-sonnet-20241022"

  # ── OpenAI-compatible endpoint (Azure, custom proxy, etc.) ──────────────────
  openai-azure:
    type: openai
    api_key: "${AZURE_OPENAI_KEY}"
    url: "https://my-resource.openai.azure.com/..."
    model: "gpt-4"

  # ── Mock (testing, no model file needed) ────────────────────────────────────
  mock:
    model: "test-model"

# ── MCP servers (optional) ──────────────────────────────────────────────────────
# Connect to Model Context Protocol servers. Their tools are available
# automatically when --mcp or no tool flag is used.
# Tools appear as mcp__<name>__<tool>.
mcp_servers:
  - name: devtools
    enabled: true
    command: python3
    args: ["/path/to/test-mcp-server.py"]

  - name: filesystem
    enabled: true
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "${HOME}/Documents"]

  - name: git
    enabled: true
    command: uvx
    args: ["mcp-server-git", "--repository", "."]

  - name: postgres
    enabled: false                            # set to true to activate
    command: npx
    args: ["-y", "@modelcontextprotocol/server-postgres"]
    env: ["DATABASE_URL=${DATABASE_URL}"]
```

Environment variables in `model_path`, `mmproj_path`, `api_key`, and MCP `args`/`env` entries are expanded at startup.

---

## Lazy Loading & Idle Timeout

Embedded llama-server backends use lazy loading in `serve` mode:

- **Server starts instantly** — no VRAM used at startup
- **Model loads on the first request** — clients see a ~6s delay while it warms up
- **`/v1/model/status`** lets your UI show a loading indicator instead of timing out
- **`idle_timeout_seconds`** — after this many seconds with no requests, the model unloads from VRAM automatically; the next request reloads it

For `complete` and `chat` commands, loading is eager (loads immediately, cleans up on exit). This is intentional — those commands are short-lived and you want the response right away.

---

## MCP Tools

Guido can connect to [Model Context Protocol](https://modelcontextprotocol.io) servers and expose their tools to the model. Any server runnable via `npx`, `uvx`, Python, or a direct executable is supported (stdio transport).

### Setup

Add an `mcp_servers` section to `~/.guido/config/config.yaml`:

```yaml
mcp_servers:
  - name: devtools
    enabled: true
    command: python3
    args: ["/path/to/test-mcp-server.py"]
```

MCP tools are active by default — no extra flag required. Use `--native` to disable them, or `--mcp` to enable only MCP and disable web search.

### Test server

Guido ships with a ready-to-use test MCP server at `lib/cli/test-mcp-server.py`:

| Tool | What it does |
|------|-------------|
| `get_time` | Returns the current UTC date and time (best forcing function — model cannot fake it) |
| `calculate` | Evaluates a math expression using Python's `math` module |
| `read_file` | Reads a file or lists a directory (capped at 4 KB) |
| `echo` | Returns its input unchanged — useful for debugging the tool dispatch loop |

Enable it in config:
```yaml
mcp_servers:
  - name: devtools
    enabled: true
    command: python3
    args: ["/path/to/lib/cli/test-mcp-server.py"]
```

Then test with:
```bash
guido chat
You: What time is it right now?
[tool] mcp__devtools__get_time {}
Guido: The current UTC time is 23:06:16 on May 25, 2026.
```

### Tool naming

Tools appear as `mcp__<server-name>__<tool-name>`:

| MCP server | Tool name | As seen by the model |
|---|---|---|
| `devtools` | `get_time` | `mcp__devtools__get_time` |
| `filesystem` | `read_file` | `mcp__filesystem__read_file` |
| `git` | `git_log` | `mcp__git__git_log` |

### Failed connections

Servers that fail to connect (missing command, wrong args, etc.) are logged and skipped. Guido starts normally with whichever servers did connect.

---

## Multimodal / Vision

Vision models require a multimodal projector (mmproj) file alongside the main model:

```yaml
backends:
  my-vision-model:
    type: llamacpp
    url: "embedded"
    port: 8002
    model: "gemma4"
    model_path: "${HOME}/.../model-q4km.gguf"
    mmproj_path: "${HOME}/.../model-mmproj.gguf"
```

Then from the CLI:
```bash
guido complete "Describe what you see" --image photo.jpg
guido chat --image diagram.png
```

Or via the API using OpenAI-compatible `image_url` content parts (data URIs supported).

---

## Architecture

```
lib/cli/
├── Makefile
├── README.md
├── DEVELOPER.md           # Package-by-package developer reference
├── config.yaml            # Sample configuration (copied to ~/.guido/config/ on install)
├── test-mcp-server.py     # Built-in MCP test server (get_time, calculate, read_file, echo)
├── go.mod / go.sum
│
├── src/                   # All Go source code
│   ├── harness/           # Core interfaces, types, and config
│   ├── backends/          # LLM provider implementations (llamacpp, openai, anthropic, ollama, …)
│   ├── httpserver/        # HTTP route registration and handlers
│   ├── tools/             # llama-server lifecycle and built-in tool calling
│   ├── mcp/               # MCP client (connects to external MCP servers)
│   └── cmd/
│       ├── cli/main.go    # guido CLI (complete, chat, serve, models)
│       └── harness/main.go # guido-harness (HTTP-only server)
│
├── exec/                  # Runtime artifacts
│   ├── bin/               # Compiled binaries and embedded llama.cpp tools
│   │   ├── guido          # Main binary (after make build)
│   │   ├── guido-harness  # HTTP-only server binary
│   │   └── llama-cpp-tools/
│   └── scripts/           # Build scripts
│       ├── build-llama.sh
│       └── create-py-wrappers.sh
│
└── modules/               # Git submodules
    └── llama.cpp/         # llama.cpp source (compiled separately)
```

For a detailed description of every file, see [DEVELOPER.md](DEVELOPER.md).

---

## Using as a Library

```go
import (
    "guido/lib/cli/src/backends"
    "guido/lib/cli/src/harness"
)

cfg, _ := harness.LoadConfig("~/.guido/config/config.yaml")
h := harness.NewHarness(cfg)

provider := backends.NewOpenAIBackend(os.Getenv("OPENAI_API_KEY"), "gpt-4o", "")
h.RegisterProvider("openai", provider)
h.SetRouter(harness.NewSimpleRouter(cfg, map[string]harness.LLMProvider{"openai": provider}))

ch, _ := h.StreamChat(ctx, &harness.ChatRequest{
    Messages: []harness.ChatMessage{
        {Role: "user", Content: harness.Text("Hello!")},
    },
    Model: "openai",
})
for token := range ch {
    fmt.Print(token)
}
```

---

## Building a New Backend

Implement the `LLMProvider` interface:

```go
type LLMProvider interface {
    Complete(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error)
    StreamTokens(ctx context.Context, req *CompletionRequest) (<-chan string, error)
    Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error)
    StreamChat(ctx context.Context, req *ChatRequest) (<-chan string, error)
    ListModels(ctx context.Context) ([]ModelInfo, error)
}
```

Register it in `initializeBackends` in `cmd/cli/main.go` (and mirror in `cmd/harness/main.go`), and add a config entry under `backends:`.

---

## Troubleshooting

**Model doesn't load / no output**
- Check `guido serve` logs — llama-server stderr is forwarded
- For vision models, confirm `mmproj_path` points to a valid mmproj file
- Run `curl http://localhost:<port>/health` to check the embedded server directly

**Tool calls failing with HTTP 500**
- This is a known llama-server version bug with its native tool API
- Guido works around it automatically via system-prompt injection — no action needed

**Port conflict**
```
a llama-server is already running on port 8002 but serves a different model
```
Kill the old process and retry:
```bash
pkill -f 'llama-server.*8002'
```

**Config not found**
```bash
guido --config /path/to/config.yaml complete "hello"
```
Default config path: `~/.guido/config/config.yaml`

**Build errors**
```bash
cd lib/cli
go mod tidy
make build
```
