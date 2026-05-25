package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"guido/lib/cli/backends"
	"guido/lib/cli/harness"
	"guido/lib/cli/httpserver"
	"guido/lib/cli/tools"
)

func main() {
	// ── Flags ─────────────────────────────────────────────────────────────────
	// Keep the flag package so existing scripts using -config / -llama-port
	// keep working.
	var (
		configPath = "config.yaml"
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
		toolsDir = "lib/cli/bin/llama-cpp-tools"
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
				if _, err := toolMgr.StartLlamaServer(modelPath, port, gpuLayers, bcfg.ChatTemplate); err != nil {
					log.Fatalf("Failed to start llama-server for %q: %v", name, err)
				}
				llamacppURL = fmt.Sprintf("http://localhost:%d", port)
			}
			providers[name] = backends.NewLlamaCppBackend(llamacppURL, model)

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

	// ── HTTP server ───────────────────────────────────────────────────────────
	if err := httpserver.Serve(context.Background(), cfg, h, func() {
		if err := toolMgr.Close(); err != nil {
			log.Printf("Tool cleanup error: %v", err)
		}
	}); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}
