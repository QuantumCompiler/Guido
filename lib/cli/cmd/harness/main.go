package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"

	"guido/lib/cli/backends"
	"guido/lib/cli/harness"
	"guido/lib/cli/tools"
)

func main() {
	configPath := flag.String("config", "config.yaml", "Path to config file")
	llamaPort := flag.Int("llama-port", 8000, "Port for llama-server")
	llamaGPU := flag.Int("llama-gpu-layers", 99, "Number of GPU layers for llama-server")
	flag.Parse()

	// Load configuration
	cfg, err := harness.LoadConfig(*configPath)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Initialize tool manager (locate tools directory)
	log.Println("Initializing tools...")

	// Find tools directory (relative to executable or config)
	toolsDir := os.Getenv("GUIDO_TOOLS_DIR")
	if toolsDir == "" {
		// Try relative to current directory
		toolsDir = "lib/cli/bin/llama-cpp-tools"
		if _, err := os.Stat(toolsDir); os.IsNotExist(err) {
			// Try relative to executable
			exePath, err := os.Executable()
			if err == nil {
				toolsDir = filepath.Join(filepath.Dir(exePath), "llama-cpp-tools")
			}
		}
	}

	toolMgr, err := tools.NewManagerFromDir(toolsDir)
	if err != nil {
		log.Fatalf("Failed to initialize tools: %v", err)
	}
	defer toolMgr.Close()

	log.Printf("Tools extracted to: %s", toolMgr.ToolsDir())

	// Initialize harness
	h := harness.NewHarness(cfg)

	// Register backends based on configuration.
	// Each key in cfg.Backends is the provider name used for routing.
	// The optional "type" field overrides the backend type; otherwise the key name is used
	// for backward compatibility ("llamacpp", "openai", "anthropic", "mock", "huggingface").
	providers := make(map[string]harness.LLMProvider)
	usedPorts := make(map[int]bool)

	for name, bcfg := range cfg.Backends {
		// Resolve backend type: explicit field wins, then key name.
		typ := bcfg.Type
		if typ == "" {
			switch name {
			case "openai", "anthropic", "llamacpp", "mock", "huggingface":
				typ = name
			}
		}

		switch typ {
		case "openai":
			if bcfg.APIKey == "" {
				continue
			}
			model := bcfg.Model
			if model == "" {
				model = "gpt-4"
			}
			providers[name] = backends.NewOpenAIBackend(bcfg.APIKey, model)
			h.RegisterProvider(name, providers[name])

		case "anthropic":
			if bcfg.APIKey == "" {
				continue
			}
			model := bcfg.Model
			if model == "" {
				model = "claude-3-sonnet"
			}
			providers[name] = backends.NewAnthropicBackend(bcfg.APIKey, model)
			h.RegisterProvider(name, providers[name])

		case "llamacpp":
			if bcfg.URL == "" && bcfg.ModelPath == "" {
				continue
			}
			model := bcfg.Model
			if model == "" {
				model = "llama"
			}

			llamacppURL := bcfg.URL
			if bcfg.URL == "" || bcfg.URL == "embedded" {
				// Determine port: config explicit → flag default → auto-assign from 8000.
				port := bcfg.Port
				if port == 0 {
					port = *llamaPort
					// If llamaPort is already taken by a previous backend, find next free port.
					for usedPorts[port] {
						port++
					}
				}
				usedPorts[port] = true

				if bcfg.ModelPath == "" {
					log.Fatalf("model_path must be specified for embedded llamacpp backend %q", name)
				}

				gpuLayers := bcfg.GPULayers
				if gpuLayers == 0 {
					gpuLayers = *llamaGPU
				}
				log.Printf("Starting embedded llama-server for %q on port %d...", name, port)
				modelPath := os.ExpandEnv(bcfg.ModelPath)
				_, err := toolMgr.StartLlamaServer(modelPath, port, gpuLayers, bcfg.ChatTemplate)
				if err != nil {
					log.Fatalf("Failed to start llama-server for %q: %v", name, err)
				}
				llamacppURL = fmt.Sprintf("http://localhost:%d", port)
				log.Printf("Embedded llama-server for %q ready at %s", name, llamacppURL)
			} else {
				log.Printf("Using external llama-server for %q at %s", name, llamacppURL)
			}

			providers[name] = backends.NewLlamaCppBackend(llamacppURL, model)
			h.RegisterProvider(name, providers[name])

		case "mock":
			model := bcfg.Model
			if model == "" {
				model = "test-model"
			}
			providers[name] = backends.NewMockBackend(model)
			h.RegisterProvider(name, providers[name])

		case "huggingface":
			if bcfg.Model == "" {
				continue
			}
			var cacheDir string
			if extra, ok := bcfg.Extra["cache_dir"].(string); ok {
				cacheDir = extra
			}
			providers[name] = backends.NewHuggingFaceBackend(bcfg.Model, cacheDir)
			h.RegisterProvider(name, providers[name])
		}
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
	r.Post("/v1/chat/completions", handler.HandleChat)
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
		log.Printf("Received signal: %v, shutting down...", sig)
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		server.Shutdown(ctx)

		// Clean up tools
		if err := toolMgr.Close(); err != nil {
			log.Printf("Error during tool cleanup: %v", err)
		}
		os.Exit(0)
	}()

	log.Printf("Starting server on %s", addr)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Server error: %v", err)
	}
}
