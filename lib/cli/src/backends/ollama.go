package backends

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"guido/lib/cli/src/harness"
)

const defaultOllamaBase = "http://localhost:11434"

// OllamaBackend implements harness.LLMProvider for a locally-running Ollama
// instance. It delegates all LLM calls to an OpenAIBackend pointed at Ollama's
// OpenAI-compatible endpoint, and uses Ollama's native /api/tags for model listing.
//
// If modelPath is set, the GGUF file is registered with Ollama on the first
// request using `ollama create <model> -f <Modelfile>`. Subsequent requests
// skip registration because the model is already in Ollama's local store.
type OllamaBackend struct {
	inner      *OpenAIBackend
	baseURL    string
	modelName  string
	modelPath  string // optional: path to GGUF file to register
	ensureOnce sync.Once
	ensureErr  error
}

// NewOllamaBackend creates an Ollama backend.
//
//   - model     — Ollama model name (e.g. "llama3.2", "gemma4-q4km"). Used as
//     the registered name when modelPath is provided.
//   - baseURL   — Ollama server URL; defaults to http://localhost:11434.
//   - modelPath — Optional path to a local GGUF file. When non-empty, Guido
//     registers the file with Ollama on first use via `ollama create`.
func NewOllamaBackend(model, baseURL, modelPath string) *OllamaBackend {
	if baseURL == "" {
		baseURL = defaultOllamaBase
	}
	return &OllamaBackend{
		inner:     NewOpenAIBackend("ollama", model, baseURL),
		baseURL:   strings.TrimRight(baseURL, "/"),
		modelName: model,
		modelPath: modelPath,
	}
}

// ── GGUF registration ─────────────────────────────────────────────────────────

// ensureModel registers the GGUF with Ollama if model_path is set and the
// model isn't already present. Safe to call concurrently — runs at most once.
func (ob *OllamaBackend) ensureModel(ctx context.Context) error {
	ob.ensureOnce.Do(func() {
		if ob.modelPath == "" {
			return
		}
		if ob.isRegistered() {
			log.Printf("[ollama] model %q already registered — skipping import", ob.modelName)
			return
		}
		log.Printf("[ollama] registering %q from %s ...", ob.modelName, ob.modelPath)
		ob.ensureErr = ob.registerGGUF(ctx)
		if ob.ensureErr == nil {
			log.Printf("[ollama] model %q ready", ob.modelName)
		}
	})
	return ob.ensureErr
}

// isRegistered returns true when modelName (or modelName:latest) already
// appears in Ollama's local model list.
func (ob *OllamaBackend) isRegistered() bool {
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(ob.baseURL + "/api/tags")
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	var payload struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return false
	}

	want := ob.modelName
	wantLatest := ob.modelName + ":latest"
	for _, m := range payload.Models {
		if m.Name == want || m.Name == wantLatest {
			return true
		}
	}
	return false
}

// registerGGUF writes a temporary Modelfile and runs `ollama create` to import
// the GGUF into Ollama's local store under ob.modelName.
func (ob *OllamaBackend) registerGGUF(ctx context.Context) error {
	// Locate the ollama binary.
	ollamaPath, err := exec.LookPath("ollama")
	if err != nil {
		return fmt.Errorf("ollama binary not found in PATH: %w", err)
	}

	// Write a temporary Modelfile.
	mf, err := os.CreateTemp("", "guido-modelfile-*.txt")
	if err != nil {
		return fmt.Errorf("create temp Modelfile: %w", err)
	}
	defer os.Remove(mf.Name())

	if _, err := fmt.Fprintf(mf, "FROM %s\n", ob.modelPath); err != nil {
		mf.Close()
		return fmt.Errorf("write Modelfile: %w", err)
	}
	mf.Close()

	// Run `ollama create <name> -f <Modelfile>`.
	cmd := exec.CommandContext(ctx, ollamaPath, "create", ob.modelName, "-f", mf.Name())
	cmd.Stdout = os.Stderr // progress lines go to stderr so they don't pollute stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("ollama create %q: %w", ob.modelName, err)
	}
	return nil
}

// ── LLMProvider — delegate everything to the inner OpenAI-compat backend ──────

func (ob *OllamaBackend) Complete(ctx context.Context, req *harness.CompletionRequest) (*harness.CompletionResponse, error) {
	if err := ob.ensureModel(ctx); err != nil {
		return nil, err
	}
	return ob.inner.Complete(ctx, req)
}

func (ob *OllamaBackend) StreamTokens(ctx context.Context, req *harness.CompletionRequest) (<-chan string, error) {
	if err := ob.ensureModel(ctx); err != nil {
		return nil, err
	}
	return ob.inner.StreamTokens(ctx, req)
}

func (ob *OllamaBackend) Chat(ctx context.Context, req *harness.ChatRequest) (*harness.ChatResponse, error) {
	if err := ob.ensureModel(ctx); err != nil {
		return nil, err
	}
	return ob.inner.Chat(ctx, req)
}

func (ob *OllamaBackend) StreamChat(ctx context.Context, req *harness.ChatRequest) (<-chan string, error) {
	if err := ob.ensureModel(ctx); err != nil {
		return nil, err
	}
	return ob.inner.StreamChat(ctx, req)
}

// ListModels queries Ollama's /api/tags endpoint for the full list of locally
// pulled models, giving richer results than the OpenAI /v1/models equivalent.
func (ob *OllamaBackend) ListModels(_ context.Context) ([]harness.ModelInfo, error) {
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(ob.baseURL + "/api/tags")
	if err != nil {
		return nil, fmt.Errorf("ollama /api/tags: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama /api/tags returned status %d", resp.StatusCode)
	}

	var payload struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("ollama /api/tags decode: %w", err)
	}

	out := make([]harness.ModelInfo, len(payload.Models))
	for i, m := range payload.Models {
		out[i] = harness.ModelInfo{
			ID:       m.Name,
			Name:     m.Name,
			Provider: "ollama",
			Type:     "chat",
		}
	}
	return out, nil
}
