# Guido - LLM Model Harness

A unified Go-based abstraction layer for interacting with both local and cloud LLM models. Guido provides a consistent interface for seamlessly switching between different model providers.

## Features

- **Multi-backend support**: Local models (via llama.cpp), OpenAI, and Anthropic APIs
- **Dual mode operation**: HTTP server mode or command-line interface
- **Streaming support**: Stream tokens in real-time from any provider
- **Configuration management**: YAML-based configuration with environment variable overrides
- **Provider abstraction**: Clean, reusable interface for building tools on top

## Architecture

```
lib/cli/
├── harness/              # Core abstraction layer
│   ├── llm.go           # LLMProvider interface & routing logic
│   ├── config.go        # Configuration loading
│   ├── models.go        # Type definitions
│   └── errors.go        # Error types
│
├── backends/            # Provider implementations
│   ├── llamacpp.go      # llama.cpp HTTP adapter
│   ├── openai.go        # OpenAI API adapter
│   └── anthropic.go     # Anthropic API adapter
│
├── cmd/
│   ├── server/          # HTTP server entry point
│   │   ├── main.go
│   │   └── handler.go   # HTTP request handlers
│   │
│   └── cli/             # CLI entry point
│       └── main.go
│
├── bin/                 # Compiled binaries
│   ├── guido-server
│   └── guido-cli
│
├── config.yaml          # Sample configuration
├── go.mod               # Go module definition
└── go.sum               # Dependency checksums
```

## Installation

### Build from source

```bash
cd lib/cli
go build -o bin/guido-server ./cmd/server
go build -o bin/guido-cli ./cmd/cli
```

## Configuration

Create a `config.yaml` file (or use the template):

```yaml
server:
  port: 8080
  mode: http

models:
  default: openai

backends:
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4"

  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    model: "claude-3-sonnet-20240229"

  llamacpp:
    url: "http://localhost:8000"
    model: "llama-2"
```

### Environment Variables

- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `LLAMACPP_URL` - URL to llama.cpp HTTP server

## Usage

### Server Mode

Start the HTTP server:

```bash
cd lib/cli
./bin/guido-server -config config.yaml
```

The server exposes these endpoints:

#### POST /v1/completions
Submit a completion request:

```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is Go?",
    "model": "openai",
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": false
  }'
```

#### GET /v1/models
List available models:

```bash
curl http://localhost:8080/v1/models
```

#### GET /health
Health check:

```bash
curl http://localhost:8080/health
```

### CLI Mode

Get a single completion:

```bash
cd lib/cli
./bin/guido-cli complete "What is Go?" -m openai -n 256 -t 0.7
```

Available options:
- `-m, --model` - Model to use (default from config)
- `-t, --temperature` - Sampling temperature (default: 0.7)
- `-n, --max-tokens` - Maximum tokens to generate (default: 256)
- `-c, --config` - Path to config file (default: config.yaml)

List available models:

```bash
./bin/guido-cli models
```

Interactive chat (placeholder):

```bash
./bin/guido-cli chat
```

## Setting up Local Models with llama.cpp

1. Install llama.cpp:
```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
make
```

2. Start the HTTP server:
```bash
./llama-server -m model.gguf -c 2048
```

3. Update your config to use llamacpp backend:
```yaml
backends:
  llamacpp:
    url: "http://localhost:8000"
    model: "llama-2"
```

## Using as a Library

Import the harness in your own Go project:

```go
package main

import (
	"context"
	"log"

	"guido/lib/cli/backends"
	"guido/lib/cli/harness"
)

func main() {
	// Load config
	cfg, err := harness.LoadConfig("config.yaml")
	if err != nil {
		log.Fatal(err)
	}

	// Create harness
	h := harness.NewHarness(cfg)

	// Register providers
	openaiProvider := backends.NewOpenAIBackend("your-api-key", "gpt-4")
	h.RegisterProvider("openai", openaiProvider)

	// Set up router
	providers := map[string]harness.LLMProvider{
		"openai": openaiProvider,
	}
	router := harness.NewSimpleRouter(cfg, providers)
	h.SetRouter(router)

	// Use the harness
	req := &harness.CompletionRequest{
		Prompt:      "What is Go?",
		Model:       "openai",
		MaxTokens:   256,
		Temperature: 0.7,
	}

	ctx := context.Background()
	resp, err := h.Complete(ctx, req)
	if err != nil {
		log.Fatal(err)
	}

	log.Println(resp.Text)
}
```

## Extending with New Backends

To add support for a new provider, implement the `LLMProvider` interface:

```go
type MyBackend struct {
	// Your backend state
}

func (mb *MyBackend) Complete(ctx context.Context, req *harness.CompletionRequest) (*harness.CompletionResponse, error) {
	// Implement completion logic
}

func (mb *MyBackend) StreamTokens(ctx context.Context, req *harness.CompletionRequest) (<-chan string, error) {
	// Implement streaming logic
}

func (mb *MyBackend) ListModels(ctx context.Context) ([]harness.ModelInfo, error) {
	// Implement model listing
}
```

Then register it:

```go
h.RegisterProvider("mybackend", myBackendInstance)
```

## API Reference

### LLMProvider Interface

```go
type LLMProvider interface {
    Complete(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error)
    StreamTokens(ctx context.Context, req *CompletionRequest) (<-chan string, error)
    ListModels(ctx context.Context) ([]ModelInfo, error)
}
```

### CompletionRequest

```go
type CompletionRequest struct {
    Prompt      string  // The prompt text
    Temperature float32 // Sampling temperature (0-2)
    MaxTokens   int     // Maximum tokens to generate
    StreamMode  bool    // Whether to stream tokens
    Model       string  // Which model to use
}
```

### CompletionResponse

```go
type CompletionResponse struct {
    Text         string // Generated text
    FinishReason string // "stop", "length", or "error"
    TokensUsed   int    // Number of tokens used
    Model        string // Model that generated response
}
```

## Testing

### Manual Testing

Start the server:
```bash
cd lib/cli
./bin/guido-server -config config.yaml
```

In another terminal:
```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "model": "openai"}'
```

### Unit Tests

```bash
cd lib/cli
go test ./...
```

## Troubleshooting

### No backends configured
Ensure at least one backend is configured in `config.yaml` with the required API keys set via environment variables.

### Connection refused (llama.cpp)
Verify llama.cpp server is running on the configured URL:
```bash
curl http://localhost:8000/health
```

### API errors
Check that your API keys are correctly set in environment variables:
```bash
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

### Build errors after moving files
Make sure all import paths have been updated to use `guido/lib/cli/` instead of `guido/lib/`:
- `guido/lib/cli/harness`
- `guido/lib/cli/backends`

Run `go mod tidy` to clean up dependencies:
```bash
cd lib/cli
go mod tidy
```

## Future Enhancements

- [ ] Batch request support
- [ ] Token counting utilities
- [ ] Cost tracking
- [ ] Request caching
- [ ] Rate limiting
- [ ] Structured output support (JSON schema)
- [ ] Tool/function calling interface
- [ ] Vision/multimodal support
- [ ] Interactive chat mode with history
- [ ] Provider-specific optimization

## License

See LICENSE file in project root

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Support

For issues, questions, or suggestions, please open an issue on GitHub.
