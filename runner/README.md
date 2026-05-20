# runner

A minimal, self-contained local LLM runner written in Go (~600 lines, zero external dependencies).

```
client (curl / Open WebUI / ollama CLI)
        │  Ollama-compatible REST API
        ▼
     runner
        │  manages model files
        │  starts llama-server subprocess per model
        ▼
  llama-server  (from llama.cpp — the actual inference engine)
```

---

## Prerequisites

1. **Go 1.22+** — <https://go.dev/dl/>
2. **llama-server** binary from llama.cpp:
   ```bash
   # macOS (Metal GPU)
   brew install llama.cpp
   ```
3. A **GGUF model file**, e.g. from Hugging Face:
   ```bash
   huggingface-cli download \
     bartowski/Mistral-7B-Instruct-v0.3-GGUF \
     Mistral-7B-Instruct-v0.3-Q4_K_M.gguf \
     --local-dir ~/.runner/models/
   ```

---

## Quick Start

```bash
# Build
make build

# Run (default port 11434, models in ~/.runner/models/)
./bin/guido-runner

# Or with custom paths
./bin/guido-runner \
  --port 11434 \
  --models /path/to/your/models \
  --llama-server /usr/local/bin/llama-server
```

---

## API

runner exposes an Ollama-compatible REST API, so any existing Ollama client works out of the box.

### List models
```bash
curl http://localhost:11434/api/tags
```

### Generate (streaming)
```bash
curl http://localhost:11434/api/generate \
  -d '{"model":"Mistral-7B-Instruct-v0.3:Q4_K_M","prompt":"Why is the sky blue?"}'
```

### Generate (non-streaming)
```bash
curl http://localhost:11434/api/generate \
  -d '{"model":"Mistral-7B-Instruct-v0.3:Q4_K_M","prompt":"Why is the sky blue?","stream":false}'
```

### Chat (multi-turn)
```bash
curl http://localhost:11434/api/chat \
  -d '{
    "model": "Mistral-7B-Instruct-v0.3:Q4_K_M",
    "messages": [
      {"role":"user","content":"Hello! Who are you?"}
    ]
  }'
```

### With inference options
```bash
curl http://localhost:11434/api/generate \
  -d '{
    "model": "Mistral-7B-Instruct-v0.3:Q4_K_M",
    "prompt": "Write a haiku about Go.",
    "stream": false,
    "options": {
      "temperature": 0.7,
      "top_p": 0.95,
      "num_predict": 100
    }
  }'
```

### Use with the official Ollama CLI
```bash
OLLAMA_HOST=http://localhost:11434 ollama run Mistral-7B-Instruct-v0.3:Q4_K_M
```

---

## Project Structure

```
runner/
├── main.go                     Entry point, CLI flags
├── go.mod
├── Makefile
├── bin/                        Compiled binaries (gitignored)
├── internal/
│   ├── api/
│   │   ├── server.go           HTTP server, middleware, routing
│   │   ├── handlers.go         All endpoint handlers
│   │   └── types.go            Request/response types
│   ├── llm/
│   │   └── runner.go           llama-server subprocess management
│   └── registry/
│       └── registry.go         Model discovery & metadata
└── README.md
```

---

## How It Works

### 1. Model Discovery
On startup and on each `/api/tags` request, runner scans the models directory for `.gguf` files. It extracts a friendly name from the filename (e.g. `mistral-7b.Q4_K_M.gguf` → `mistral-7b:Q4_K_M`) and computes a partial SHA256 digest for identity.

### 2. Model Loading
When a request arrives for a model:
- Already loaded and subprocess alive → reuse it (no reload cost)
- Different model requested → kill the old subprocess, start a new one
- Waits up to 45s for `llama-server /health` to return 200

### 3. Inference
- `/api/generate` → proxied to llama-server `POST /completion`
- `/api/chat` → proxied to llama-server `POST /v1/chat/completions`
- llama-server returns SSE; runner re-emits as NDJSON to the client

### 4. GPU Support
`--n-gpu-layers 99` is passed to llama-server automatically. llama.cpp handles Metal (macOS), CUDA, ROCm, and Vulkan with no extra configuration.

---

## What's Missing

| Feature | runner | Ollama |
|---|---|---|
| Ollama-compatible REST API | ✅ | ✅ |
| Streaming NDJSON | ✅ | ✅ |
| GPU offloading | ✅ | ✅ |
| Open WebUI compatible | ✅ | ✅ |
| Model pulling | ❌ drop .gguf manually | ✅ |
| Multiple concurrent models | ❌ | ✅ |
| Modelfile system | ❌ | ✅ |
| Embedded llama.cpp (no subprocess) | ❌ | ✅ |
| Multi-modal (vision) | ❌ | ✅ |
| Model keep-alive timeout | ❌ | ✅ |