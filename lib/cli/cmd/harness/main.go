package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"

	"guido/lib/cli/backends"
	"guido/lib/cli/harness"
)

func main() {
	configPath := flag.String("config", "config.yaml", "Path to config file")
	flag.Parse()

	// Load configuration
	cfg, err := harness.LoadConfig(*configPath)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Initialize harness
	h := harness.NewHarness(cfg)

	// Register backends based on configuration
	providers := make(map[string]harness.LLMProvider)

	if openaiCfg, ok := cfg.Backends["openai"]; ok && openaiCfg.APIKey != "" {
		model := openaiCfg.Model
		if model == "" {
			model = "gpt-4"
		}
		providers["openai"] = backends.NewOpenAIBackend(openaiCfg.APIKey, model)
		h.RegisterProvider("openai", providers["openai"])
	}

	if anthropicCfg, ok := cfg.Backends["anthropic"]; ok && anthropicCfg.APIKey != "" {
		model := anthropicCfg.Model
		if model == "" {
			model = "claude-3-sonnet"
		}
		providers["anthropic"] = backends.NewAnthropicBackend(anthropicCfg.APIKey, model)
		h.RegisterProvider("anthropic", providers["anthropic"])
	}

	if llamacppCfg, ok := cfg.Backends["llamacpp"]; ok && llamacppCfg.URL != "" {
		model := llamacppCfg.Model
		if model == "" {
			model = "llama"
		}
		providers["llamacpp"] = backends.NewLlamaCppBackend(llamacppCfg.URL, model)
		h.RegisterProvider("llamacpp", providers["llamacpp"])
	}

	if _, ok := cfg.Backends["mock"]; ok {
		providers["mock"] = backends.NewMockBackend("test-model")
		h.RegisterProvider("mock", providers["mock"])
	}

	if hfCfg, ok := cfg.Backends["huggingface"]; ok && hfCfg.Model != "" {
		cacheDir := hfCfg.Extra["cache_dir"].(string)
		if cacheDir == "" {
			cacheDir = hfCfg.Extra["cache_dir"].(string)
		}
		providers["huggingface"] = backends.NewHuggingFaceBackend(hfCfg.Model, cacheDir)
		h.RegisterProvider("huggingface", providers["huggingface"])
	}

	if len(providers) == 0 {
		log.Fatal("No backends configured. Please set up at least one backend in config.")
	}

	// Set up the model router
	router := harness.NewSimpleRouter(cfg, providers)
	h.SetRouter(router)

	// Create HTTP router
	r := chi.NewRouter()

	// Add middleware
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)

	// Register handlers
	handler := NewHandler(h)
	r.Post("/v1/completions", handler.HandleCompletion)
	r.Get("/v1/models", handler.HandleListModels)
	r.Get("/health", handler.HandleHealth)

	// Start server
	addr := fmt.Sprintf(":%d", cfg.Server.Port)
	server := &http.Server{
		Addr:    addr,
		Handler: r,
	}

	// Graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		sig := <-sigChan
		log.Printf("Received signal: %v", sig)
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		server.Shutdown(ctx)
	}()

	log.Printf("Starting server on %s", addr)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Server error: %v", err)
	}
}
