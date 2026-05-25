# Guido — LLM Harness

A unified Go-based harness for local and cloud LLM models. Run a single model from the command line, start a persistent OpenAI-compatible HTTP server, or use it as a library — all from one binary with embedded llama.cpp tooling.

---

## Quick Start

### Build & Install

```bash
cd lib/cli
make build    # compile llama.cpp + build guido binary
make install  # install to ~/bin/guido + copy config to ~/.guido/config/
```

`make install` places the binary at `~/bin/guido` and writes a starter config to `~/.guido/config/config.yaml` (skipped if the file already exists).

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

Sends a single prompt and prints the response, then exits. Any embedded llama-server started for this invocation is stopped on exit.

```bash
# Use the default model
guido complete "What is a transformer?"

# Use a specific backend
guido complete "Solve 3x + 7 = 22 step by step" -m my-reasoning-model

# Attach files and images
guido complete "Summarize this document" --file report.pdf
guido complete "What's in this image?" --image screenshot.png
guido complete "Explain this code" --file main.go --context "Focus on error handling"
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

---

### `chat` — interactive session

```bash
guido chat [flags]
```

Starts a multi-turn conversation in your terminal. Full message history is maintained in memory and re-sent each request (llama-server's prompt cache speeds up repeated prefixes). Type `exit` or press Ctrl+C to quit.

```bash
# Default model, no system prompt
guido chat

# Specific model with a persona
guido chat -m my-model --system "You are a concise technical assistant."

# Attach context to the first message only
guido chat --image diagram.png --file architecture.md

# Enable web search tools
guido chat --search
```

**Flags** — same as `complete`, plus:

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--system` | `-s` | | System prompt |
| `--search` | | `false` | Give the model `web_search` and `fetch_url` tools |

MCP tools from servers listed under `mcp_servers:` in your config are always available in `chat` — no extra flag required. See [MCP Tools](#mcp-tools) below.

---

### `serve` — OpenAI-compatible HTTP server

```bash
guido serve [flags]
```

Starts a persistent HTTP server. Embedded llama-server processes use **lazy loading** — they start on the first request and optionally unload after the configured idle timeout. The server itself starts instantly with no VRAM usage.

```bash
# Serve the default model
guido serve

# Serve a specific backend
guido serve -m my-model

# Serve all configured backends (multi-model server)
guido serve --all-backends
```

**Flags**

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--model` | `-m` | config default | Backend to serve |
| `--all-backends` | | `false` | Serve every configured backend |

---

### `models` — list available models

```bash
guido models
```

Queries all configured backends and prints their available models.

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
Returns the load state of lazy backends (useful for showing a loading indicator in a GUI):
```bash
# All backends
curl http://localhost:8080/v1/model/status

# Specific backend
curl "http://localhost:8080/v1/model/status?backend=my-model"
```

Response:
```json
{
  "my-model": {"model": "gemma4", "status": "ready", "idle_seconds": 42}
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

  # ── Local model (embedded llama-server) ─────────────────────────────────
  my-model:
    type: llamacpp
    url: "embedded"                    # guido manages the llama-server process
    port: 8002                         # port for this model's llama-server
    model: "gemma4"                    # model name reported to clients
    model_path: "${HOME}/.../model.gguf"
    mmproj_path: "${HOME}/.../mmproj.gguf"  # optional — required for vision models
    idle_timeout_seconds: 300          # 0 = stay loaded until server stops
    gpu_layers: 99                     # layers to offload to GPU

  # ── External llama-server (you manage it) ────────────────────────────────
  external:
    type: llamacpp
    url: "http://192.168.1.50:8000"
    model: "my-remote-model"

  # ── OpenAI ───────────────────────────────────────────────────────────────
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4o"

  # ── Anthropic ────────────────────────────────────────────────────────────
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    model: "claude-3-5-sonnet-20241022"

  # ── Mock (testing, no model file needed) ────────────────────────────────
  mock:
    model: "test-model"

# ── MCP servers (optional) ──────────────────────────────────────────────────
# Connect to Model Context Protocol servers and make their tools available
# in the chat agentic loop. Tools appear as mcp__<name>__<tool>.
mcp_servers:
  - name: filesystem
    enabled: true
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "${HOME}/Documents"]

  - name: git
    enabled: true
    command: uvx
    args: ["mcp-server-git", "--repository", "."]

  - name: postgres
    enabled: false                # set to true to activate
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
    mmproj_path: "${HOME}/.../model-mmproj.gguf"   # required for image input
```

Then from the CLI:
```bash
guido complete "Describe what you see" --image photo.jpg
guido chat --image diagram.png
```

Or via the API using OpenAI-compatible `image_url` content parts (data URIs supported).

---

## MCP Tools

Guido can connect to [Model Context Protocol](https://modelcontextprotocol.io) servers and expose their tools to the model in the `chat` agentic loop. Any server runnable via `npx`, `uvx`, or a direct executable is supported (stdio transport).

### Setup

Add an `mcp_servers` section to `~/.guido/config/config.yaml`:

```yaml
mcp_servers:
  - name: filesystem
    enabled: true
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "${HOME}/Documents"]
```

Then run `guido chat` — no extra flag needed. Configured servers connect at startup and their tools are automatically included in every request.

### Tool naming

Tools appear as `mcp__<server-name>__<tool-name>`. For example:

| MCP server | Tool name | As seen by the model |
|---|---|---|
| `filesystem` | `read_file` | `mcp__filesystem__read_file` |
| `git` | `git_log` | `mcp__git__git_log` |
| `postgres` | `query` | `mcp__postgres__query` |

### Built-in tools alongside MCP

`--search` and MCP tools are independent and compose freely:

```bash
guido chat --search   # web_search + fetch_url + all MCP tools
guido chat            # MCP tools only (if mcp_servers configured)
```

### Failed connections

Servers that fail to connect (missing `npx`, wrong args, etc.) are logged and skipped. Guido starts normally with whichever servers did connect.

---

## Architecture

```
lib/cli/
├── Makefile
├── README.md
├── DEVELOPER.md           # Package-by-package developer reference
├── config.yaml            # Sample configuration (copied to ~/.guido/config/ on install)
├── go.mod / go.sum
│
├── src/                   # All Go source code
│   ├── harness/           # Core interfaces, types, and config
│   ├── backends/          # LLM provider implementations (llamacpp, openai, anthropic, …)
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

provider := backends.NewOpenAIBackend(os.Getenv("OPENAI_API_KEY"), "gpt-4o")
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

Register it in `initializeBackends` in `cmd/cli/main.go` and add a config entry.

---

## Troubleshooting

**Model doesn't load / no output**
- Check `guido serve` logs — llama-server stderr is forwarded
- For vision models, confirm `mmproj_path` points to a valid mmproj file
- Run `curl http://localhost:<port>/health` to check the embedded server directly

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
