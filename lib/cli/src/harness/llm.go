package harness

import (
	"context"
)

// StatusReporter is an optional interface that LLMProvider implementations
// can satisfy to expose their loading/idle state to the HTTP layer.
type StatusReporter interface {
	ModelStatus() ModelStatusInfo
}

// InTextToolCaller is an optional interface implemented by backends that embed
// tool calls inside the text stream (e.g. llamacpp system-prompt injection).
// When a backend satisfies this interface the CLI's agentic loop can use a
// streaming response with a short lookahead buffer to detect tool calls,
// rather than waiting for the full non-streaming response.
type InTextToolCaller interface {
	UsesInTextToolCalls() bool
}

// LLMProvider is the interface that all model backends must implement
type LLMProvider interface {
	// Complete sends a single-turn completion request and returns a response
	Complete(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error)

	// StreamTokens returns a channel that streams response tokens one by one
	StreamTokens(ctx context.Context, req *CompletionRequest) (<-chan string, error)

	// Chat sends a multi-turn chat request and returns the assistant's response
	Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error)

	// StreamChat sends a multi-turn chat request and streams the response tokens
	StreamChat(ctx context.Context, req *ChatRequest) (<-chan string, error)

	// ListModels returns a list of available models for this provider
	ListModels(ctx context.Context) ([]ModelInfo, error)
}

// Harness is the main entry point for the LLM abstraction layer
type Harness struct {
	config    *Config
	providers map[string]LLMProvider // Map of backend name -> provider instance
	router    ModelRouter
}

// ModelRouter is responsible for routing requests to the appropriate backend
type ModelRouter interface {
	// Route determines which provider should handle a model
	Route(modelName string) (LLMProvider, error)
}

// SimpleRouter implements ModelRouter with basic model->backend mapping
type SimpleRouter struct {
	backends   map[string]LLMProvider
	config     *Config
	modelIndex map[string]LLMProvider // model name / backend key → provider
}

// NewHarness creates a new harness instance
func NewHarness(cfg *Config) *Harness {
	return &Harness{
		config:    cfg,
		providers: make(map[string]LLMProvider),
	}
}

// RegisterProvider registers a backend provider with a given name
func (h *Harness) RegisterProvider(name string, provider LLMProvider) {
	h.providers[name] = provider
}

// SetRouter sets the model router
func (h *Harness) SetRouter(router ModelRouter) {
	h.router = router
}

// Complete sends a completion request to the appropriate backend
func (h *Harness) Complete(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
	provider, err := h.router.Route(req.Model)
	if err != nil {
		return nil, err
	}

	return provider.Complete(ctx, req)
}

// StreamTokens returns a channel of tokens from the appropriate backend
func (h *Harness) StreamTokens(ctx context.Context, req *CompletionRequest) (<-chan string, error) {
	provider, err := h.router.Route(req.Model)
	if err != nil {
		return nil, err
	}

	return provider.StreamTokens(ctx, req)
}

// Chat sends a multi-turn chat request to the appropriate backend
func (h *Harness) Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	provider, err := h.router.Route(req.Model)
	if err != nil {
		return nil, err
	}

	return provider.Chat(ctx, req)
}

// StreamChat streams a multi-turn chat response from the appropriate backend
func (h *Harness) StreamChat(ctx context.Context, req *ChatRequest) (<-chan string, error) {
	provider, err := h.router.Route(req.Model)
	if err != nil {
		return nil, err
	}

	return provider.StreamChat(ctx, req)
}

// UsesInTextToolCalls returns true when the backend for model embeds tool calls
// in the text response (llamacpp system-prompt injection). The CLI uses this to
// decide whether to stream the agentic loop with lookahead detection.
func (h *Harness) UsesInTextToolCalls(model string) bool {
	provider, err := h.router.Route(model)
	if err != nil {
		return false
	}
	itc, ok := provider.(InTextToolCaller)
	return ok && itc.UsesInTextToolCalls()
}

// ModelStatus returns the status of a named backend if it implements StatusReporter.
// The second return value is false when the backend doesn't exist or doesn't report status.
func (h *Harness) ModelStatus(backendName string) (ModelStatusInfo, bool) {
	p, ok := h.providers[backendName]
	if !ok {
		return ModelStatusInfo{}, false
	}
	sr, ok := p.(StatusReporter)
	if !ok {
		return ModelStatusInfo{}, false
	}
	return sr.ModelStatus(), true
}

// AllModelStatuses returns status for every backend that implements StatusReporter.
func (h *Harness) AllModelStatuses() map[string]ModelStatusInfo {
	result := make(map[string]ModelStatusInfo)
	for name, p := range h.providers {
		if sr, ok := p.(StatusReporter); ok {
			result[name] = sr.ModelStatus()
		}
	}
	return result
}

// ListAllModels returns all available models from all providers
func (h *Harness) ListAllModels(ctx context.Context) ([]ModelInfo, error) {
	var allModels []ModelInfo

	for _, provider := range h.providers {
		models, err := provider.ListModels(ctx)
		if err != nil {
			// Log error but continue with other providers
			continue
		}
		allModels = append(allModels, models...)
	}

	return allModels, nil
}

// Route implements ModelRouter by mapping model names to backends.
// It matches by backend key name first, then by the model name configured
// in the backend (e.g. "gemma4:31b" → the "gemma4" provider).
func (sr *SimpleRouter) Route(modelName string) (LLMProvider, error) {
	if modelName == "" {
		modelName = sr.config.Models.Default
	}

	// Direct lookup in index (backend key OR configured model name)
	if provider, ok := sr.modelIndex[modelName]; ok {
		return provider, nil
	}

	// Fall back to the default provider
	if defaultProvider, ok := sr.modelIndex[sr.config.Models.Default]; ok {
		return defaultProvider, nil
	}

	// Last resort: any registered provider
	for _, provider := range sr.backends {
		return provider, nil
	}

	return nil, ErrNoAvailableBackend
}

// NewSimpleRouter creates a new SimpleRouter and pre-builds the model index
// so that both backend key names and configured model names resolve correctly.
func NewSimpleRouter(cfg *Config, backends map[string]LLMProvider) *SimpleRouter {
	sr := &SimpleRouter{
		backends:   backends,
		config:     cfg,
		modelIndex: make(map[string]LLMProvider),
	}

	for backendName, bcfg := range cfg.Backends {
		provider, ok := backends[backendName]
		if !ok {
			continue
		}
		// Register by backend key (e.g. "gemma4")
		sr.modelIndex[backendName] = provider
		// Register by the model name from config (e.g. "gemma4:31b")
		if bcfg.Model != "" {
			sr.modelIndex[bcfg.Model] = provider
		}
	}

	return sr
}
