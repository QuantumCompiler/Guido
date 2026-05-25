package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"guido/lib/cli/src/backends"
	"guido/lib/cli/src/harness"
	"guido/lib/cli/src/httpserver"
	"guido/lib/cli/src/tools"
)

func main() {
	// ── Flags ─────────────────────────────────────────────────────────────────
	// Keep the flag package so existing scripts using -config / -llama-port
	// keep working.
	defaultConfig := "config.yaml"
	if home, err := os.UserHomeDir(); err == nil {
		defaultConfig = filepath.Join(home, ".guido", "config", "config.yaml")
	}

	var (
		configPath = defaultConfig
		llamaPort  = 8000
		llamaGPU   = 99
	)
	for i, a := range os.Args[1:] {
		switch a {
		case "-config", "--config":
			if i+2 < len(os.Args) {
				configPath = os.Args[i+2]
			}
		case "-llama-port", "--llama-port":
			if i+2 < len(os.Args) {
				fmt.Sscanf(os.Args[i+2], "%d", &llamaPort)
			}
		case "-llama-gpu-layers", "--llama-gpu-layers":
			if i+2 < len(os.Args) {
				fmt.Sscanf(os.Args[i+2], "%d", &llamaGPU)
			}
		}
	}

	// ── Config ────────────────────────────────────────────────────────────────
	cfg, err := harness.LoadConfig(configPath)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// ── Tool manager ──────────────────────────────────────────────────────────
	toolsDir := os.Getenv("GUIDO_TOOLS_DIR")
	if toolsDir == "" {
		toolsDir = "lib/cli/exec/bin/llama-cpp-tools"
		if _, err := os.Stat(toolsDir); os.IsNotExist(err) {
			if exe, err := os.Executable(); err == nil {
				toolsDir = filepath.Join(filepath.Dir(exe), "llama-cpp-tools")
			}
		}
	}
	toolMgr, err := tools.NewManagerFromDir(toolsDir)
	if err != nil {
		log.Fatalf("Failed to initialize tools: %v", err)
	}

	// ── Backends ──────────────────────────────────────────────────────────────
	h := harness.NewHarness(cfg)
	providers := make(map[string]harness.LLMProvider)
	usedPorts := make(map[int]bool)

	for name, bcfg := range cfg.Backends {
		typ := bcfg.Type
		if typ == "" {
			switch name {
			case "openai", "anthropic", "llamacpp", "ollama", "mock", "huggingface":
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
			providers[name] = backends.NewOpenAIBackend(bcfg.APIKey, model, bcfg.URL)

		case "anthropic":
			if bcfg.APIKey == "" {
				continue
			}
			model := bcfg.Model
			if model == "" {
				model = "claude-3-sonnet"
			}
			providers[name] = backends.NewAnthropicBackend(bcfg.APIKey, model)

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
				port := bcfg.Port
				if port == 0 {
					port = llamaPort
					for usedPorts[port] {
						port++
					}
				}
				usedPorts[port] = true
				if bcfg.ModelPath == "" {
					log.Fatalf("model_path required for embedded llamacpp backend %q", name)
				}
				gpuLayers := bcfg.GPULayers
				if gpuLayers == 0 {
					gpuLayers = llamaGPU
				}
				modelPath := os.ExpandEnv(bcfg.ModelPath)
				mmProjPath := os.ExpandEnv(bcfg.MmProjPath)
				llamacppURL = fmt.Sprintf("http://localhost:%d", port)
				idleTimeout := time.Duration(bcfg.IdleTimeoutSeconds) * time.Second

				// Use a lazy backend so the llama-server only starts on the
				// first request and can unload after idle_timeout_seconds.
				lb := backends.NewLazyLlamaCppBackend(
					toolMgr, modelPath, mmProjPath, llamacppURL, model,
					bcfg.ChatTemplate, port, gpuLayers, idleTimeout,
				)
				providers[name] = lb
				h.RegisterProvider(name, lb)
				continue // skip NewLlamaCppBackend + end-of-loop registration below
			}
			providers[name] = backends.NewLlamaCppBackend(llamacppURL, model)

		case "ollama":
			model := bcfg.Model
			if model == "" {
				model = "llama3.2"
			}
			providers[name] = backends.NewOllamaBackend(model, bcfg.URL, os.ExpandEnv(bcfg.ModelPath))

		case "mock":
			model := bcfg.Model
			if model == "" {
				model = "test-model"
			}
			providers[name] = backends.NewMockBackend(model)

		case "huggingface":
			if bcfg.Model == "" {
				continue
			}
			var cacheDir string
			if extra, ok := bcfg.Extra["cache_dir"].(string); ok {
				cacheDir = extra
			}
			providers[name] = backends.NewHuggingFaceBackend(bcfg.Model, cacheDir)
		}

		if p, ok := providers[name]; ok {
			h.RegisterProvider(name, p)
		}
	}

	if len(providers) == 0 {
		log.Fatal("No backends configured. Set up at least one backend in config.")
	}

	h.SetRouter(harness.NewSimpleRouter(cfg, providers))

	log.Printf("[guido] serve mode — models load on first request")

	// ── HTTP server ───────────────────────────────────────────────────────────
	if err := httpserver.Serve(context.Background(), cfg, h, nil, func() {
		if err := toolMgr.Close(); err != nil {
			log.Printf("Tool cleanup error: %v", err)
		}
	}); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}
