package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/spf13/cobra"

	"guido/lib/cli/backends"
	"guido/lib/cli/harness"
)

var (
	configPath string
	model      string
	temperature float32
	maxTokens   int
)

var rootCmd = &cobra.Command{
	Use:   "guido",
	Short: "Guido - LLM Model Harness",
	Long:  `Guido is a unified interface for interacting with local and cloud LLM models.`,
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

		// Register backends
		providers := initializeBackends(h, cfg)

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
		providers := initializeBackends(h, cfg)

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
		providers := initializeBackends(h, cfg)

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

func initializeBackends(h *harness.Harness, cfg *harness.Config) map[string]harness.LLMProvider {
	providers := make(map[string]harness.LLMProvider)

	if openaiCfg, ok := cfg.Backends["openai"]; ok && openaiCfg.APIKey != "" {
		modelName := openaiCfg.Model
		if modelName == "" {
			modelName = "gpt-4"
		}
		providers["openai"] = backends.NewOpenAIBackend(openaiCfg.APIKey, modelName)
		h.RegisterProvider("openai", providers["openai"])
	}

	if anthropicCfg, ok := cfg.Backends["anthropic"]; ok && anthropicCfg.APIKey != "" {
		modelName := anthropicCfg.Model
		if modelName == "" {
			modelName = "claude-3-sonnet"
		}
		providers["anthropic"] = backends.NewAnthropicBackend(anthropicCfg.APIKey, modelName)
		h.RegisterProvider("anthropic", providers["anthropic"])
	}

	if llamacppCfg, ok := cfg.Backends["llamacpp"]; ok && llamacppCfg.URL != "" {
		modelName := llamacppCfg.Model
		if modelName == "" {
			modelName = "llama"
		}
		providers["llamacpp"] = backends.NewLlamaCppBackend(llamacppCfg.URL, modelName)
		h.RegisterProvider("llamacpp", providers["llamacpp"])
	}

	if _, ok := cfg.Backends["mock"]; ok {
		providers["mock"] = backends.NewMockBackend("test-model")
		h.RegisterProvider("mock", providers["mock"])
	}

	if hfCfg, ok := cfg.Backends["huggingface"]; ok && hfCfg.Model != "" {
		var cacheDir string
		if extra, ok := hfCfg.Extra["cache_dir"].(string); ok {
			cacheDir = extra
		}
		providers["huggingface"] = backends.NewHuggingFaceBackend(hfCfg.Model, cacheDir)
		h.RegisterProvider("huggingface", providers["huggingface"])
	}

	return providers
}

func init() {
	rootCmd.PersistentFlags().StringVarP(&configPath, "config", "c", "config.yaml", "Path to config file")
	completeCmd.Flags().StringVarP(&model, "model", "m", "", "Model to use (default from config)")
	completeCmd.Flags().Float32VarP(&temperature, "temperature", "t", 0.7, "Temperature for sampling")
	completeCmd.Flags().IntVarP(&maxTokens, "max-tokens", "n", 256, "Maximum tokens to generate")

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
