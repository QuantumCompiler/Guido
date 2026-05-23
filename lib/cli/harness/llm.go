package harness

import (
	"context"
)

// LLMProvider is the interface that all model backends must implement
type LLMProvider interface {
	// Complete sends a completion request and returns a response
	Complete(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error)

	// StreamTokens returns a channel that streams response tokens one by one
	StreamTokens(ctx context.Context, req *CompletionRequest) (<-chan string, error)

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
	backends map[string]LLMProvider
	config   *Config
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

// Route implements ModelRouter by mapping model names to backends
func (sr *SimpleRouter) Route(modelName string) (LLMProvider, error) {
	// If the model name doesn't exist, use the default
	if modelName == "" {
		modelName = sr.config.Models.Default
	}

	// Look for a backend that claims to handle this model
	for backendName, provider := range sr.backends {
		if backendName == modelName || sr.canHandle(backendName, modelName) {
			return provider, nil
		}
	}

	// Default to the first available backend
	for _, provider := range sr.backends {
		return provider, nil
	}

	return nil, ErrNoAvailableBackend
}

// canHandle checks if a backend can handle a specific model
func (sr *SimpleRouter) canHandle(backendName, modelName string) bool {
	// For now, we assume backend names map to model names
	// This can be extended to support model groups or aliases
	return false
}

// NewSimpleRouter creates a new SimpleRouter
func NewSimpleRouter(cfg *Config, backends map[string]LLMProvider) *SimpleRouter {
	return &SimpleRouter{
		backends: backends,
		config:   cfg,
	}
}
