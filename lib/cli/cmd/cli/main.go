package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"guido/lib/cli/backends"
	"guido/lib/cli/harness"
	"guido/lib/cli/httpserver"
	"guido/lib/cli/tools"
)

var (
	configPath   string
	model        string
	temperature  float32
	maxTokens    int
	systemPrompt string
	enableSearch bool // --search: give the model web_search + fetch_url tools
	toolMgr      *tools.Manager
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

		// Register backends (eager — complete command exits after one response)
		providers := initializeBackends(h, cfg, toolMgr, false)

		if len(providers) == 0 {
			log.Fatal("No backends configured")
		}

		// Kill any llama-server we started when this command exits.
		// If the server was already running we didn't add it to toolMgr.launched,
		// so Close() is a no-op in that case.
		defer toolMgr.Close()

		router := harness.NewSimpleRouter(cfg, providers)
		h.SetRouter(router)

		// Resolve model name
		chatModel := model
		if chatModel == "" {
			chatModel = cfg.Models.Default
		}

		// Use the chat endpoint so instruction-tuned models receive a properly
		// formatted prompt rather than bare text (which they tend to ignore).
		ctx := context.Background()
		req := &harness.ChatRequest{
			Messages: []harness.ChatMessage{
				{Role: "user", Content: prompt},
			},
			Model:       chatModel,
			Temperature: temperature,
			MaxTokens:   maxTokens,
			Stream:      true,
		}

		resp, err := h.StreamChat(ctx, req)
		if err != nil {
			log.Fatalf("Completion error: %v", err)
		}
		for token := range resp {
			fmt.Print(stripStopTokens(token))
		}
		fmt.Println()
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
		providers := initializeBackends(h, cfg, toolMgr, false)
		if len(providers) == 0 {
			log.Fatal("No backends configured")
		}

		// Kill any llama-server we started when the session ends.
		defer toolMgr.Close()

		router := harness.NewSimpleRouter(cfg, providers)
		h.SetRouter(router)

		// Resolve model name
		chatModel := model
		if chatModel == "" {
			chatModel = cfg.Models.Default
		}

		// Build conversation history
		var history []harness.ChatMessage
		if systemPrompt != "" {
			history = append(history, harness.ChatMessage{Role: "system", Content: systemPrompt})
		}

		// Handle Ctrl+C gracefully — finish the current response then exit
		ctx, cancel := context.WithCancel(context.Background())
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
		go func() {
			<-sigChan
			fmt.Println("\n\nInterrupted. Goodbye!")
			cancel()
			os.Exit(0)
		}()

		fmt.Printf("Guido (%s)  — type 'exit' to quit, Ctrl+C to interrupt\n", chatModel)
		if systemPrompt != "" {
			fmt.Printf("System: %s\n", systemPrompt)
		}
		if enableSearch {
			fmt.Println("Web search enabled (web_search + fetch_url)")
		}
		fmt.Println(strings.Repeat("─", 50))

		scanner := bufio.NewScanner(os.Stdin)
		for {
			fmt.Print("\nYou: ")
			if !scanner.Scan() {
				break // EOF (e.g. piped input finished)
			}
			input := strings.TrimSpace(scanner.Text())
			if input == "" {
				continue
			}
			if input == "exit" || input == "quit" || input == "/exit" || input == "/quit" {
				fmt.Println("Goodbye!")
				break
			}

			// Add user message to history
			history = append(history, harness.ChatMessage{Role: "user", Content: input})

			if enableSearch {
				// ── Agentic tool-calling loop ──────────────────────────────
				// Non-streaming so we can inspect tool_calls before printing.
				// Loops until the model gives a final answer with no tool calls.
				replied := false
				for {
					req := &harness.ChatRequest{
						Messages:    history,
						Model:       chatModel,
						Temperature: temperature,
						MaxTokens:   maxTokens,
						Tools:       tools.WebTools(),
					}

					resp, err := h.Chat(ctx, req)
					if err != nil {
						log.Printf("Error: %v", err)
						history = history[:len(history)-1] // undo user message
						break
					}

					// Record the assistant turn (may contain tool_calls).
					history = append(history, resp.Message)

					if len(resp.Message.ToolCalls) == 0 {
						// Final answer — print and break.
						text := strings.TrimSpace(stripStopTokens(resp.Message.Content))
						fmt.Printf("\nGuido: %s\n", text)
						replied = true
						break
					}

					// Execute each tool call the model requested.
					for _, tc := range resp.Message.ToolCalls {
						// Show the user what's happening.
						var args map[string]interface{}
						_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
						switch tc.Function.Name {
						case "web_search":
							q, _ := args["query"].(string)
							fmt.Printf("\n[searching] %q...\n", q)
						case "fetch_url":
							u, _ := args["url"].(string)
							fmt.Printf("\n[fetching] %s...\n", u)
						default:
							fmt.Printf("\n[tool] %s %v\n", tc.Function.Name, args)
						}

						result, execErr := tools.ExecuteTool(tc.Function.Name, tc.Function.Arguments)
						if execErr != nil {
							result = "Error: " + execErr.Error()
						}

						history = append(history, harness.ChatMessage{
							Role:       "tool",
							ToolCallID: tc.ID,
							Content:    result,
						})
					}
					// Loop — send tool results back to the model.
				}
				if !replied {
					// Error path — already undone above, nothing to add.
				}
			} else {
				// ── Normal streaming path (no tools) ──────────────────────
				req := &harness.ChatRequest{
					Messages:    history,
					Model:       chatModel,
					Temperature: temperature,
					MaxTokens:   maxTokens,
					Stream:      true,
				}

				fmt.Print("\nGuido: ")

				tokenChan, err := h.StreamChat(ctx, req)
				if err != nil {
					log.Printf("Error: %v", err)
					history = history[:len(history)-1]
					continue
				}

				var response strings.Builder
				for token := range tokenChan {
					cleaned := stripStopTokens(token)
					if cleaned != "" {
						fmt.Print(cleaned)
						response.WriteString(cleaned)
					}
				}
				fmt.Println()

				assistantText := strings.TrimSpace(response.String())
				if assistantText != "" {
					history = append(history, harness.ChatMessage{
						Role:    "assistant",
						Content: assistantText,
					})
				}
			}
		}
	},
}

// serveCmd starts the OpenAI-compatible HTTP harness server.
// It is equivalent to running guido-harness but is embedded directly in the
// CLI binary so you only need one executable.
var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the OpenAI-compatible HTTP harness server",
	Long: `Starts an HTTP server on the port configured in config.yaml (default 8080).
The server exposes OpenAI-compatible endpoints:
  POST /v1/completions
  POST /v1/chat/completions
  GET  /v1/models
  GET  /health

Press Ctrl+C to stop — any llama-server processes started by this session
will be terminated automatically.`,
	Run: func(cmd *cobra.Command, args []string) {
		cfg, err := harness.LoadConfig(configPath)
		if err != nil {
			log.Fatalf("Failed to load config: %v", err)
		}

		h := harness.NewHarness(cfg)
		// Lazy mode: embedded llamacpp backends start on the first request.
		// idle_timeout_minutes in config controls automatic VRAM unloading.
		providers := initializeBackends(h, cfg, toolMgr, true)
		if len(providers) == 0 {
			log.Fatal("No backends configured")
		}
		h.SetRouter(harness.NewSimpleRouter(cfg, providers))

		log.Printf("[guido] serve mode — models load on first request")

		if err := httpserver.Serve(context.Background(), cfg, h, func() {
			if err := toolMgr.Close(); err != nil {
				log.Printf("Backend cleanup error: %v", err)
			}
		}); err != nil {
			log.Fatalf("Server error: %v", err)
		}
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
		providers := initializeBackends(h, cfg, toolMgr, false)

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

// initializeBackends registers all configured backends with the harness.
//
// When lazy is true (serve mode), embedded llamacpp backends are wrapped in a
// LazyLlamaCppBackend so the llama-server process only starts on the first
// request and can optionally unload after an idle timeout.
//
// When lazy is false (complete / chat / models commands), the server is started
// eagerly so the command can run immediately and be cleaned up on exit.
func initializeBackends(h *harness.Harness, cfg *harness.Config, tm *tools.Manager, lazy bool) map[string]harness.LLMProvider {
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
				port := bcfg.Port
				if port == 0 {
					port = nextEmbeddedPort(usedPorts, 8000)
				}
				usedPorts[port] = true
				llamacppURL = fmt.Sprintf("http://localhost:%d", port)
				expandedModelPath := os.ExpandEnv(bcfg.ModelPath)

				if lazy {
					// ── Lazy path (serve mode) ────────────────────────────────
					// The LazyLlamaCppBackend manages the llama-server lifecycle
					// internally: it starts on the first request and optionally
					// unloads after idleTimeout minutes of inactivity.
					gpuLayers := bcfg.GPULayers
					if gpuLayers == 0 {
						gpuLayers = 99
					}
					idleTimeout := time.Duration(bcfg.IdleTimeoutSeconds) * time.Second
					lb := backends.NewLazyLlamaCppBackend(
						tm, expandedModelPath, llamacppURL, modelName,
						bcfg.ChatTemplate, port, gpuLayers, idleTimeout,
					)
					providers[name] = lb
					h.RegisterProvider(name, lb)
					continue // skip the eager startup + NewLlamaCppBackend below
				}

				// ── Eager path (complete / chat / models commands) ────────────
				// Check if a server is already running on this port.
				status := llamaServerStatus(llamacppURL, expandedModelPath)
				if status == serverLoading {
					log.Printf("Waiting for llama-server on %s to finish loading...", llamacppURL)
					status = waitForServer(llamacppURL, expandedModelPath, 5*time.Minute)
				}
				switch status {
				case serverReady:
					log.Printf("Using existing llama-server for %q at %s", name, llamacppURL)
				case serverWrongModel:
					log.Fatalf(
						"A llama-server is already running on %s but serves a different model.\n"+
							"Kill it first, then retry:\n\n  pkill -f 'llama-server.*%d'\n",
						llamacppURL, port,
					)
				case serverNotRunning, serverLoading:
					if tm != nil && expandedModelPath != "" {
						gpuLayers := bcfg.GPULayers
						if gpuLayers == 0 {
							gpuLayers = 99
						}
						_, err := tm.StartLlamaServer(expandedModelPath, port, gpuLayers, bcfg.ChatTemplate)
						if err != nil {
							log.Fatalf("Failed to start llama-server for %q: %v\n", name, err)
						}
					}
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

// defaultConfigPath returns ~/.guido/config/config.yaml, falling back to
// "config.yaml" in the current directory if the home dir can't be determined.
func defaultConfigPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return "config.yaml"
	}
	return filepath.Join(home, ".guido", "config", "config.yaml")
}

func init() {
	rootCmd.PersistentFlags().StringVar(&configPath, "config", defaultConfigPath(), "Path to config file")

	completeCmd.Flags().StringVarP(&model, "model", "m", "", "Model to use (default from config)")
	completeCmd.Flags().Float32VarP(&temperature, "temperature", "t", 0.7, "Temperature for sampling")
	completeCmd.Flags().IntVarP(&maxTokens, "max-tokens", "n", -1, "Maximum tokens to generate (-1 for unlimited)")

	chatCmd.Flags().StringVarP(&model, "model", "m", "", "Model to use (default from config)")
	chatCmd.Flags().Float32VarP(&temperature, "temperature", "t", 0.7, "Temperature for sampling")
	chatCmd.Flags().IntVarP(&maxTokens, "max-tokens", "n", -1, "Maximum tokens to generate (-1 for unlimited)")
	chatCmd.Flags().StringVarP(&systemPrompt, "system", "s", "", "System prompt to set the assistant's behavior")
	chatCmd.Flags().BoolVar(&enableSearch, "search", false, "Give the model web_search and fetch_url tools (internet access)")

	rootCmd.AddCommand(completeCmd)
	rootCmd.AddCommand(chatCmd)
	rootCmd.AddCommand(serveCmd)
	rootCmd.AddCommand(modelsCmd)
}

type llamaStatus int

const (
	serverNotRunning llamaStatus = iota
	serverReady
	serverLoading   // alive but model not yet fully loaded (503)
	serverWrongModel
)

// llamaServerStatus checks whether a llama-server on baseURL is alive and
// serving the expected model.
//
// States returned:
//   - serverNotRunning  — nothing on the port
//   - serverLoading     — server is up but model is still loading (HTTP 503)
//   - serverReady       — server is up, model loaded, path confirmed
//   - serverWrongModel  — server is up but serving a different model
func llamaServerStatus(baseURL, expectedModelPath string) llamaStatus {
	client := &http.Client{Timeout: 2 * time.Second}

	// Is anything alive on this port?
	healthResp, err := client.Get(baseURL + "/health")
	if err != nil {
		return serverNotRunning
	}
	healthResp.Body.Close()

	switch healthResp.StatusCode {
	case http.StatusServiceUnavailable: // 503 = model loading
		// Can't verify model path yet — assume it's ours and wait
		log.Printf("llama-server on %s is still loading the model (503)...", baseURL)
		return serverLoading
	case http.StatusOK:
		// ready — fall through to model verification
	default:
		return serverNotRunning
	}

	// If no model path configured (external server, user manages it), accept as-is.
	if expectedModelPath == "" {
		return serverReady
	}

	// Verify the loaded model via /props → {"model_path": "..."}
	propsResp, err := client.Get(baseURL + "/props")
	if err != nil {
		// /props unreachable on a healthy server — accept it (external server without /props)
		log.Printf("Warning: could not reach /props on %s — assuming correct model", baseURL)
		return serverReady
	}
	defer propsResp.Body.Close()

	var props struct {
		ModelPath string `json:"model_path"`
	}
	if err := json.NewDecoder(propsResp.Body).Decode(&props); err != nil || props.ModelPath == "" {
		// Can't read path — accept it
		log.Printf("Warning: could not parse /props on %s — assuming correct model", baseURL)
		return serverReady
	}

	// Compare by FULL absolute path, falling back to filename if one side is relative.
	runningAbs, _ := filepath.Abs(props.ModelPath)
	expectedAbs, _ := filepath.Abs(expectedModelPath)
	if runningAbs != expectedAbs {
		// Try filename-only as a last resort (handles relative vs absolute discrepancies).
		if filepath.Base(props.ModelPath) != filepath.Base(expectedModelPath) {
			log.Printf("Port conflict: server on %s is running %q but config expects %q",
				baseURL, props.ModelPath, expectedModelPath)
			return serverWrongModel
		}
	}

	return serverReady
}

// waitForServer polls llamaServerStatus until the server is ready or the
// timeout elapses. Returns the final status.
func waitForServer(baseURL, expectedModelPath string, timeout time.Duration) llamaStatus {
	deadline := time.Now().Add(timeout)
	tick := 3 * time.Second
	for time.Now().Before(deadline) {
		time.Sleep(tick)
		status := llamaServerStatus(baseURL, expectedModelPath)
		if status != serverLoading {
			return status
		}
		log.Printf("Still loading... (%s remaining)", time.Until(deadline).Round(time.Second))
	}
	return serverLoading
}

// stopTokens are end-of-turn markers that some models leak into their output
// even when llama-server is supposed to stop at them.
var stopTokens = []string{
	"<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end|>",
	"<|im_start|>user", "<|im_start|>assistant",
	"\nUser:", "\nHuman:",
}

// stripStopTokens removes any leaked end-of-turn tokens from a streamed chunk.
func stripStopTokens(token string) string {
	for _, stop := range stopTokens {
		if idx := strings.Index(token, stop); idx >= 0 {
			return token[:idx]
		}
	}
	return token
}

func main() {
	// Normalize single-dash -config to --config so the CLI accepts the same
	// flag style as the harness (which uses the standard flag package).
	for i, arg := range os.Args[1:] {
		if arg == "-config" {
			os.Args[i+1] = "--config"
		}
	}

	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}
