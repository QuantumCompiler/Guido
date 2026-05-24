package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/spf13/cobra"

	"guido/lib/cli/backends"
	"guido/lib/cli/harness"
	"guido/lib/cli/tools"
)

var (
	configPath string
	model      string
	temperature float32
	maxTokens   int
	toolMgr    *tools.Manager
)

var rootCmd = &cobra.Command{
	Use:   "guido",
	Short: "Guido - LLM Model Harness",
	Long:  `Guido is a unified interface for interacting with local and cloud LLM models.`,
	PersistentPreRun: func(cmd *cobra.Command, args []string) {
		// Initialize tool manager once for all commands
		toolsDir := os.Getenv("GUIDO_TOOLS_DIR")
		if toolsDir == "" {
			// Try relative to current directory
			toolsDir = "bin/llama-cpp-tools"
			if _, err := os.Stat(toolsDir); os.IsNotExist(err) {
				// Try relative to executable
				exePath, err := os.Executable()
				if err == nil {
					toolsDir = filepath.Join(filepath.Dir(exePath), "llama-cpp-tools")
				}
			}
		}

		var err error
		toolMgr, err = tools.NewManagerFromDir(toolsDir)
		if err != nil {
			log.Fatalf("Failed to initialize tools: %v", err)
		}
	},
	PersistentPostRun: func(cmd *cobra.Command, args []string) {
		// Note: We intentionally do NOT close the tool manager here,
		// as the llama-server is a long-lived process that should persist
		// across multiple CLI invocations for efficient reuse.
		// The server will be cleaned up when the user exits their shell session.
	},
}

var completeCmd = &cobra.Command{
	Use:   "complete <prompt>",
	Short: "Get a completion for a prompt",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		prompt := args[0]

		// Load configuration
		cfg, err := harness.LoadConfig(configPath)
		if err != nil {
			log.Fatalf("Failed to load config: %v", err)
		}

		// Initialize harness
		h := harness.NewHarness(cfg)

		// Register backends (pass tool manager for embedded llama-server support)
		providers := initializeBackends(h, cfg, toolMgr)

		if len(providers) == 0 {
			log.Fatal("No backends configured")
		}

		router := harness.NewSimpleRouter(cfg, providers)
		h.SetRouter(router)

		// Create completion request
		req := &harness.CompletionRequest{
			Prompt:      prompt,
			Model:       model,
			Temperature: temperature,
			MaxTokens:   maxTokens,
			StreamMode:  false,
		}

		if req.Model == "" {
			req.Model = cfg.Models.Default
		}

		// Get completion
		ctx := context.Background()

		resp, err := h.Complete(ctx, req)
		if err != nil {
			log.Fatalf("Completion error: %v", err)
		}

		fmt.Println(resp.Text)
	},
}

var chatCmd = &cobra.Command{
	Use:   "chat",
	Short: "Start an interactive chat session",
	Run: func(cmd *cobra.Command, args []string) {
		// Load configuration
		cfg, err := harness.LoadConfig(configPath)
		if err != nil {
			log.Fatalf("Failed to load config: %v", err)
		}

		// Initialize harness
		h := harness.NewHarness(cfg)

		// Register backends
		providers := initializeBackends(h, cfg, toolMgr)

		if len(providers) == 0 {
			log.Fatal("No backends configured")
		}

		router := harness.NewSimpleRouter(cfg, providers)
		h.SetRouter(router)

		fmt.Println("Starting chat (type 'exit' to quit)")

		// Placeholder for interactive chat
		// This would need readline or similar for a real implementation
		fmt.Println("Chat mode not yet fully implemented")
	},
}

var modelsCmd = &cobra.Command{
	Use:   "models",
	Short: "List available models",
	Run: func(cmd *cobra.Command, args []string) {
		// Load configuration
		cfg, err := harness.LoadConfig(configPath)
		if err != nil {
			log.Fatalf("Failed to load config: %v", err)
		}

		// Initialize harness
		h := harness.NewHarness(cfg)

		// Register backends
		providers := initializeBackends(h, cfg, toolMgr)

		if len(providers) == 0 {
			fmt.Println("No backends configured")
			return
		}

		router := harness.NewSimpleRouter(cfg, providers)
		h.SetRouter(router)

		// List models
		ctx := context.Background()
		models, err := h.ListAllModels(ctx)
		if err != nil {
			log.Fatalf("Failed to list models: %v", err)
		}

		fmt.Println("Available Models:")
		for _, m := range models {
			fmt.Printf("  - %s (provider: %s, type: %s)\n", m.Name, m.Provider, m.Type)
		}
	},
}

// backendType resolves the effective backend type for a named config entry.
// Explicit "type" field wins; otherwise the key name is used for backward compatibility
// (e.g. a key named "openai" is treated as type "openai").
func backendType(key string, cfg harness.BackendConfig) string {
	if cfg.Type != "" {
		return cfg.Type
	}
	// Backward-compat: key name IS the type for the original single-backend style
	switch key {
	case "openai", "anthropic", "llamacpp", "mock", "huggingface":
		return key
	}
	return ""
}

// nextEmbeddedPort finds the next available port starting from basePort.
func nextEmbeddedPort(used map[int]bool, basePort int) int {
	p := basePort
	for used[p] {
		p++
	}
	return p
}

func initializeBackends(h *harness.Harness, cfg *harness.Config, tm *tools.Manager) map[string]harness.LLMProvider {
	providers := make(map[string]harness.LLMProvider)
	usedPorts := make(map[int]bool)

	for name, bcfg := range cfg.Backends {
		typ := backendType(name, bcfg)

		switch typ {
		case "openai":
			if bcfg.APIKey == "" {
				continue
			}
			modelName := bcfg.Model
			if modelName == "" {
				modelName = "gpt-4"
			}
			providers[name] = backends.NewOpenAIBackend(bcfg.APIKey, modelName)
			h.RegisterProvider(name, providers[name])

		case "anthropic":
			if bcfg.APIKey == "" {
				continue
			}
			modelName := bcfg.Model
			if modelName == "" {
				modelName = "claude-3-sonnet"
			}
			providers[name] = backends.NewAnthropicBackend(bcfg.APIKey, modelName)
			h.RegisterProvider(name, providers[name])

		case "llamacpp":
			if bcfg.URL == "" && bcfg.ModelPath == "" {
				continue
			}
			modelName := bcfg.Model
			if modelName == "" {
				modelName = "llama"
			}

			llamacppURL := bcfg.URL
			if bcfg.URL == "embedded" || bcfg.URL == "" {
				// Determine which port this instance should use.
				// Config can specify an explicit port; otherwise auto-assign from 8000 up.
				port := bcfg.Port
				if port == 0 {
					port = nextEmbeddedPort(usedPorts, 8000)
				}
				usedPorts[port] = true
				llamacppURL = fmt.Sprintf("http://localhost:%d", port)

				// Check if a server is already running on this port.
				serverRunning := false
				for i := 0; i < 3; i++ {
					resp, err := http.Get(llamacppURL + "/health")
					if err == nil && (resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusServiceUnavailable) {
						serverRunning = true
						resp.Body.Close()
						break
					}
					if resp != nil {
						resp.Body.Close()
					}
					time.Sleep(100 * time.Millisecond)
				}

				if !serverRunning {
					if tm != nil && bcfg.ModelPath != "" {
						log.Printf("Starting embedded llama-server for %q on port %d...", name, port)
						expandedModelPath := os.ExpandEnv(bcfg.ModelPath)
						_, err := tm.StartLlamaServer(expandedModelPath, port, 99)
						if err != nil {
							log.Fatalf("Failed to start embedded llama-server for %q: %v\n", name, err)
						}
						log.Printf("Embedded llama-server for %q ready at %s", name, llamacppURL)
					}
				} else {
					log.Printf("Using existing llama-server for %q at %s", name, llamacppURL)
				}
			}

			providers[name] = backends.NewLlamaCppBackend(llamacppURL, modelName)
			h.RegisterProvider(name, providers[name])

		case "mock":
			modelName := bcfg.Model
			if modelName == "" {
				modelName = "test-model"
			}
			providers[name] = backends.NewMockBackend(modelName)
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

	return providers
}

func init() {
	rootCmd.PersistentFlags().StringVarP(&configPath, "config", "c", "config.yaml", "Path to config file")
	completeCmd.Flags().StringVarP(&model, "model", "m", "", "Model to use (default from config)")
	completeCmd.Flags().Float32VarP(&temperature, "temperature", "t", 0.7, "Temperature for sampling")
	completeCmd.Flags().IntVarP(&maxTokens, "max-tokens", "n", -1, "Maximum tokens to generate (-1 for unlimited)")

	rootCmd.AddCommand(completeCmd)
	rootCmd.AddCommand(chatCmd)
	rootCmd.AddCommand(modelsCmd)
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}
