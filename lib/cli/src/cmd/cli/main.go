package main

import (
	"bufio"
	"context"
	"encoding/base64"
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

	"guido/lib/cli/src/backends"
	"guido/lib/cli/src/embeddedtools"
	"guido/lib/cli/src/harness"
	"guido/lib/cli/src/httpserver"
	"guido/lib/cli/src/logger"
	"guido/lib/cli/src/mcp"
	"guido/lib/cli/src/tools"
)

var (
	configPath   string
	model        string
	temperature  float32
	maxTokens    int
	systemPrompt string
	// Tool mode flags — at most one may be set (enforced by MarkFlagsMutuallyExclusive).
	// No flag → all available tools (web search + MCP from config).
	flagSearch bool // --search : web search only
	flagMCP    bool // --mcp    : MCP tools only
	flagNative bool // --native : no tools
	toolMgr    *tools.Manager

	flagResume   string // --resume   : resume a saved chat by id (from logs/chats/)
	flagContinue bool   // --continue : resume the most recent saved chat
)

var (
	contextStrings []string // --context
	contextFiles   []string // --file
	contextImages  []string // --image
)

var appLogger *logger.Logger // global logger; nil if initialization failed

var (
	allBackends      bool   // --all-backends: initialize every configured backend (complete/chat)
	serveModel       string // --model: which backend to serve (default: config default)
	serveAllBackends bool   // --all-backends: serve every configured backend

	// Default fallback values for embedded llamacpp backends that don't specify
	// port / gpu_layers in their config entry. Used by initializeBackends and
	// overridable from the harness subcommand flags.
	llamaPort int = 8000 // starting port for auto-assigned embedded backends
	llamaGPU  int = 99   // GPU layers when backend config omits gpu_layers
)

var rootCmd = &cobra.Command{
	Use:   "guido",
	Short: "Guido - LLM Model Harness",
	Long:  `Guido is a unified interface for interacting with local and cloud LLM models.`,
	PersistentPreRun: func(cmd *cobra.Command, args []string) {
		if lg, err := logger.New(""); err == nil {
			appLogger = lg
		} else {
			log.Printf("Warning: logging disabled: %v", err)
		}

		// Resolve tools directory — checked in priority order:
		// 1. $GUIDO_TOOLS_DIR env var (explicit override)
		// 2. exec/bin/guido-cpp-tools relative to CWD (dev/project layout)
		// 3. guido-cpp-tools adjacent to the installed binary
		// 4. Embedded tools baked into the binary (embed_tools build tag)
		var err error
		if toolsDir := os.Getenv("GUIDO_TOOLS_DIR"); toolsDir != "" {
			toolMgr, err = tools.NewManagerFromDir(toolsDir)
		} else if _, statErr := os.Stat("exec/bin/guido-cpp-tools"); statErr == nil {
			toolMgr, err = tools.NewManagerFromDir("exec/bin/guido-cpp-tools")
		} else if exePath, exeErr := os.Executable(); exeErr == nil {
			// Resolve symlinks so /usr/local/bin/guido → .../exec/bin/guido
			// before looking for guido-cpp-tools in the same directory.
			if resolved, resErr := filepath.EvalSymlinks(exePath); resErr == nil {
				exePath = resolved
			}
			adj := filepath.Join(filepath.Dir(exePath), "guido-cpp-tools")
			if _, statErr := os.Stat(adj); statErr == nil {
				toolMgr, err = tools.NewManagerFromDir(adj)
			}
		}

		if toolMgr == nil && err == nil {
			// Fall back to tools embedded in the binary (no-op in stub builds).
			extractDir := filepath.Join(os.Getenv("HOME"), ".guido", "tools")
			toolMgr, err = tools.ExtractEmbedded(embeddedtools.ToolsFS, extractDir)
		}

		if toolMgr == nil {
			if err != nil {
				log.Fatalf("Failed to initialize tools: %v", err)
			}
			log.Fatalf("Tools not found. Set $GUIDO_TOOLS_DIR or run 'make build'.")
		}
	},
	PersistentPostRun: func(cmd *cobra.Command, args []string) {
		// Note: We intentionally do NOT close the tool manager here,
		// as the guido-server is a long-lived process that should persist
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

		// Narrow to the requested backend (default: models.default from config).
		// --all-backends opts out of filtering so every configured backend is available.
		filterBackends(cfg, model, allBackends)

		// Initialize harness
		h := harness.NewHarness(cfg)

		// Register backends (eager — complete command exits after one response)
		providers := initializeBackends(h, cfg, toolMgr, false)

		if len(providers) == 0 {
			log.Fatal("No backends configured")
		}

		// Kill any guido-server we started when this command exits.
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
		content, err := buildMessageContent(prompt, contextStrings, contextFiles, contextImages)
		if err != nil {
			log.Fatalf("Failed to build message: %v", err)
		}

		useWeb, useMCP := resolveToolMode()
		activeTools, mcpReg := setupTools(ctx, cfg, useWeb, useMCP)
		if mcpReg != nil {
			defer mcpReg.Close()
		}

		var toolNames []string
		for _, t := range activeTools {
			toolNames = append(toolNames, t.Function.Name)
		}

		var completion *logger.ChatSession
		if appLogger != nil {
			appLogger.CompleteCall(chatModel, "cli")
			completion = appLogger.NewCompletionSession(logger.NewChatID(chatModel), chatModel, toolNames, true)
			completion.MarkEstimated()
		}

		history := []harness.ChatMessage{{Role: "user", Content: content}}
		promptEst := estimatePromptTokens(history)

		if len(activeTools) > 0 {
			// Agentic loop — run until the model gives a final answer.
			// Pass "" as finalPrefix: the loop prints the answer directly with
			// no label (complete is a one-shot command, no "Guido:" needed).
			answer, err := runAgenticLoop(ctx, h, &history, chatModel, temperature, maxTokens, activeTools, mcpReg, true, "")
			if err != nil {
				if completion != nil {
					completion.Finish("error")
				}
				log.Fatalf("Completion error: %v", err)
			}
			if completion != nil {
				// invoked_tools not surfaced from runAgenticLoop; record nil
				// (consistent with HTTP and the no-tools path).
				completion.RecordTurn(prompt, answer, promptEst, logger.EstimateTokens(answer), nil)
				completion.Finish("stop")
			}
		} else {
			// No tools — stream the response directly.
			req := &harness.ChatRequest{
				Messages:    history,
				Model:       chatModel,
				Temperature: temperature,
				MaxTokens:   maxTokens,
				Stream:      true,
			}
			resp, err := h.StreamChat(ctx, req)
			if err != nil {
				if completion != nil {
					completion.Finish("error")
				}
				log.Fatalf("Completion error: %v", err)
			}
			var out strings.Builder
			for token := range resp {
				cleaned := stripStopTokens(token)
				out.WriteString(cleaned)
				fmt.Print(cleaned)
			}
			fmt.Println()
			if completion != nil {
				completion.RecordTurn(prompt, out.String(), promptEst, logger.EstimateTokens(out.String()), nil)
				completion.Finish("stop")
			}
		}
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

		// Narrow to the requested backend (default: models.default from config).
		// --all-backends opts out of filtering so every configured backend is available.
		filterBackends(cfg, model, allBackends)

		// Initialize harness
		h := harness.NewHarness(cfg)
		providers := initializeBackends(h, cfg, toolMgr, false)
		if len(providers) == 0 {
			log.Fatal("No backends configured")
		}

		// Kill any guido-server we started when the session ends.
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
			history = append(history, harness.ChatMessage{Role: "system", Content: harness.Text(systemPrompt)})
		}

		// Resume a previous chat into context: rebuild its turns as history so the
		// model continues where it left off. --resume <id> loads a specific chat;
		// --continue (-c) loads the most recent saved chat. resumedChat stays in
		// scope so the session below appends new turns to the same JSON file.
		var resumedChat *logger.ChatMetrics
		if appLogger != nil && (flagResume != "" || flagContinue) {
			var rerr error
			if flagResume != "" {
				resumedChat, rerr = appLogger.LoadChat(flagResume)
			} else {
				resumedChat, rerr = appLogger.LatestChat()
			}
			if rerr != nil {
				log.Fatalf("Error resuming chat: %v", rerr)
			}
			for _, t := range resumedChat.Turns {
				if t.UserInput != "" {
					history = append(history, harness.ChatMessage{Role: "user", Content: harness.Text(t.UserInput)})
				}
				if t.ModelOutput != "" {
					history = append(history, harness.ChatMessage{Role: "assistant", Content: harness.Text(t.ModelOutput)})
				}
			}
			fmt.Printf("Resumed chat %s — %d prior turn(s) loaded into context.\n", resumedChat.ChatID, len(resumedChat.Turns))
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

		// Decide which tool sets to activate based on flags.
		useWeb, useMCP := resolveToolMode()
		activeTools, mcpReg := setupTools(ctx, cfg, useWeb, useMCP)
		if mcpReg != nil {
			defer mcpReg.Close()
		}

		fmt.Printf("Guido (%s)  — type 'exit' to quit, Ctrl+C to interrupt\n", chatModel)
		if systemPrompt != "" {
			fmt.Printf("System: %s\n", systemPrompt)
		}
		// Summarise active tools on startup.
		var toolSummary []string
		if useWeb {
			toolSummary = append(toolSummary, "web search")
		}
		if mcpReg != nil && len(mcpReg.Tools()) > 0 {
			toolSummary = append(toolSummary, fmt.Sprintf("MCP (%d tools)", len(mcpReg.Tools())))
		}
		if flagNative {
			fmt.Println("Tools: disabled")
		} else if len(toolSummary) > 0 {
			fmt.Printf("Tools: %s\n", strings.Join(toolSummary, " + "))
		} else if useMCP && mcpReg == nil {
			fmt.Println("Tools: MCP (no servers configured)")
		}
		fmt.Println(strings.Repeat("─", 50))

		if len(contextFiles) > 0 || len(contextImages) > 0 || len(contextStrings) > 0 {
			fmt.Print("Attached: ")
			var labels []string
			for _, f := range contextFiles {
				labels = append(labels, filepath.Base(f))
			}
			for _, img := range contextImages {
				labels = append(labels, filepath.Base(img)+" (image)")
			}
			if len(contextStrings) > 0 {
				labels = append(labels, fmt.Sprintf("%d context string(s)", len(contextStrings)))
			}
			fmt.Println(strings.Join(labels, ", "))
		}

		// One interactive session = one chat file. Create it once; each user
		// message below is recorded as a turn. The file is persisted after every
		// turn, so a Ctrl+C exit still leaves a complete record.
		var chatSession *logger.ChatSession
		if appLogger != nil {
			var toolNames []string
			for _, t := range activeTools {
				toolNames = append(toolNames, t.Function.Name)
			}
			if resumedChat != nil {
				// Continue the resumed chat: new turns append to its existing file.
				chatSession = appLogger.ResumeChatSession(resumedChat, toolNames, true)
			} else {
				chatSession = appLogger.NewChatSession(logger.NewChatID(chatModel), chatModel, toolNames, true)
			}
			chatSession.MarkEstimated()
			defer chatSession.Finish("stop")
		}

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
			content, err := buildMessageContent(input, contextStrings, contextFiles, contextImages)
			if err != nil {
				log.Printf("Error building message: %v", err)
				continue
			}
			// Attachments apply to the first message only; clear after use.
			contextStrings, contextFiles, contextImages = nil, nil, nil
			history = append(history, harness.ChatMessage{Role: "user", Content: content})

			if chatSession != nil {
				var toolNames []string
				for _, t := range activeTools {
					toolNames = append(toolNames, t.Function.Name)
				}
				appLogger.ChatSubmitted(chatSession.ChatID(), chatModel, toolNames)
				chatSession.BeginTurn()
			}

			// Prompt tokens are estimated from the history sent to the model.
			promptEst := estimatePromptTokens(history)

			if len(activeTools) > 0 {
				// ── Agentic tool-calling loop ──────────────────────────────
				// finalPrefix="\nGuido: " is printed once, right before the
				// final answer — whether it streams token-by-token or arrives
				// as a single block. No re-print needed after the call returns.
				answer, err := runAgenticLoop(ctx, h, &history, chatModel, temperature, maxTokens, activeTools, mcpReg, true, "\nGuido: ")
				if err != nil {
					log.Printf("Error: %v", err)
					history = history[:len(history)-1] // undo user message
				} else if chatSession != nil {
					// invoked_tools is not surfaced from runAgenticLoop; record nil
					// (consistent with the no-tools path and HTTP streaming paths
					// which track tools per-turn internally).
					chatSession.RecordTurn(input, answer, promptEst, logger.EstimateTokens(answer), nil)
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
					if chatSession != nil {
						chatSession.RecordTurn(input, "", promptEst, 0, nil)
					}
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
						Content: harness.Text(assistantText),
					})
				}
				if chatSession != nil {
					chatSession.RecordTurn(input, assistantText, promptEst, logger.EstimateTokens(assistantText), nil)
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

Press Ctrl+C to stop — any guido-server processes started by this session
will be terminated automatically.`,
	Run: func(cmd *cobra.Command, args []string) {
		cfg, err := harness.LoadConfig(configPath)
		if err != nil {
			log.Fatalf("Failed to load config: %v", err)
		}

		// Determine which backend(s) to expose.
		// Default: only the backend named in models.default — so that adding
		// extra models to config.yaml doesn't accidentally serve all of them.
		// --model <name>   : serve a specific backend instead of the default
		// --all-backends   : serve every configured backend (opt-in)
		filterBackends(cfg, serveModel, serveAllBackends)

		h := harness.NewHarness(cfg)
		// Lazy mode: embedded llamacpp backends start on the first request.
		// idle_timeout_seconds in config controls automatic VRAM unloading.
		providers := initializeBackends(h, cfg, toolMgr, true)
		if len(providers) == 0 {
			log.Fatal("No backends configured")
		}
		h.SetRouter(harness.NewSimpleRouter(cfg, providers))

		// Build the server-side tool set from flags (same logic as chat/complete).
		ctx := context.Background()
		useWeb, useMCP := resolveToolMode()
		activeTools, mcpReg := setupTools(ctx, cfg, useWeb, useMCP)
		if mcpReg != nil {
			defer mcpReg.Close()
		}

		// Summarise active tools in the startup log.
		var toolParts []string
		if useWeb {
			toolParts = append(toolParts, "web search")
		}
		if mcpReg != nil && len(mcpReg.Tools()) > 0 {
			toolParts = append(toolParts, fmt.Sprintf("MCP (%d tools)", len(mcpReg.Tools())))
		}
		if flagNative || len(toolParts) == 0 {
			log.Printf("[guido] serving %q — model loads on first request (no tools)", cfg.Models.Default)
		} else {
			log.Printf("[guido] serving %q — model loads on first request | tools: %s",
				cfg.Models.Default, strings.Join(toolParts, " + "))
		}

		// Build tool config for the HTTP handler (nil when no tools active).
		var tc *httpserver.ToolConfig
		if len(activeTools) > 0 {
			captured := mcpReg // capture for closure
			tc = &httpserver.ToolConfig{
				Tools: activeTools,
				ExecTool: func(rctx context.Context, name, argsJSON string) (string, error) {
					return dispatchTool(rctx, harness.ToolCall{
						Function: harness.ToolCallFunction{Name: name, Arguments: argsJSON},
					}, captured)
				},
			}
		}

		if err := httpserver.Serve(ctx, cfg, h, tc, func() {
			if err := toolMgr.Close(); err != nil {
				log.Printf("Backend cleanup error: %v", err)
			}
		}); err != nil {
			log.Fatalf("Server error: %v", err)
		}
	},
}

// harnessCmd is the embedded replacement for the former guido-harness binary.
// It starts a bare OpenAI-compatible HTTP server with ALL configured backends
// and no tool injection — intended for GUI applications that embed guido and
// manage the process lifetime themselves.
//
// Accepts --llama-port and --llama-gpu-layers to override the defaults used for
// embedded llamacpp backends that don't specify port / gpu_layers in config.
var harnessCmd = &cobra.Command{
	Use:   "harness",
	Short: "Start the bare HTTP harness server (all backends, no tool injection)",
	Long: `Starts an OpenAI-compatible HTTP server that exposes every backend
configured in config.yaml. All embedded llamacpp backends use lazy loading —
the guido-server process starts on the first request and can optionally unload
after the configured idle_timeout_seconds.

Unlike 'serve', this command does not inject tools into the model — it is
intended as an embedding target for GUI applications that wrap guido and handle
tool calling at a higher level.

Endpoints:
  POST /v1/completions
  POST /v1/chat/completions
  GET  /v1/models
  GET  /v1/model/status
  GET  /health`,
	Run: func(cmd *cobra.Command, args []string) {
		cfg, err := harness.LoadConfig(configPath)
		if err != nil {
			log.Fatalf("Failed to load config: %v", err)
		}

		h := harness.NewHarness(cfg)
		// Always lazy: guido-server starts on the first request.
		providers := initializeBackends(h, cfg, toolMgr, true)
		if len(providers) == 0 {
			log.Fatal("No backends configured. Set up at least one backend in config.")
		}
		h.SetRouter(harness.NewSimpleRouter(cfg, providers))

		log.Printf("[guido] harness mode — %d backend(s), models load on first request", len(providers))

		if err := httpserver.Serve(context.Background(), cfg, h, nil, func() {
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

		// Register backends — lazy=true so no guido-server is spawned just to list models.
		providers := initializeBackends(h, cfg, toolMgr, true)

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
			if m.Name != "" && m.Name != m.ID {
				fmt.Printf("  - %s  (%s — %s)\n", m.ID, m.Name, m.Provider)
			} else {
				fmt.Printf("  - %s  (%s)\n", m.ID, m.Provider)
			}
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
	case "openai", "anthropic", "llamacpp", "ollama", "mock", "huggingface":
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

// filterBackends restricts cfg.Backends to only the target backend and updates
// cfg.Models.Default to match. When all is true the map is left intact.
// resolveToolMode translates the mutually-exclusive tool flags into two booleans.
//
//	(no flag) → useWeb=true,  useMCP=true   — all available tools
//	--search  → useWeb=true,  useMCP=false  — web search only
//	--mcp     → useWeb=false, useMCP=true   — MCP only
//	--native  → useWeb=false, useMCP=false  — no tools
func resolveToolMode() (useWeb, useMCP bool) {
	switch {
	case flagNative:
		return false, false
	case flagSearch:
		return true, false
	case flagMCP:
		return false, true
	default:
		return true, true
	}
}

// setupTools connects to MCP servers (if useMCP and any are configured) and
// returns the combined active tool list plus the registry (may be nil).
// The caller is responsible for calling mcpReg.Close() when done.
func setupTools(ctx context.Context, cfg *harness.Config, useWeb, useMCP bool) (activeTools []harness.Tool, mcpReg *mcp.Registry) {
	if useMCP && len(cfg.MCPServers) > 0 {
		var regErr error
		mcpReg, regErr = mcp.NewRegistry(ctx, cfg.MCPServers)
		if regErr != nil {
			log.Printf("[mcp] registry init error (non-fatal): %v", regErr)
			mcpReg = nil
		}
	}
	if useWeb {
		activeTools = append(activeTools, tools.WebTools()...)
	}
	if mcpReg != nil {
		activeTools = append(activeTools, mcpReg.Tools()...)
	}
	return activeTools, mcpReg
}

// estimatePromptTokens approximates the prompt size of a message list for
// logging when a streamed response carries no usage block.
func estimatePromptTokens(msgs []harness.ChatMessage) int {
	var b strings.Builder
	for _, m := range msgs {
		b.WriteString(m.Content.PlainText())
		b.WriteByte('\n')
	}
	return logger.EstimateTokens(b.String())
}

// dispatchTool executes a single tool call, trying MCP first then built-ins.
func dispatchTool(ctx context.Context, tc harness.ToolCall, mcpReg *mcp.Registry) (string, error) {
	if mcpReg != nil {
		result, handled, err := mcpReg.ExecuteTool(ctx, tc.Function.Name, tc.Function.Arguments)
		if handled {
			return result, err
		}
	}
	return tools.ExecuteTool(tc.Function.Name, tc.Function.Arguments)
}

// runAgenticLoop drives the model→tool→model cycle until the model stops
// requesting tool calls. It appends each turn to *history in place and
// returns the final assistant text.
//
// When the backend uses in-text tool calls (llamacpp system-prompt injection),
// the loop uses a streaming path with a short lookahead buffer so the final
// answer streams to the terminal token-by-token. Other backends use the
// non-streaming Chat() path unchanged.
//
// finalPrefix is printed immediately before the final answer text (e.g.
// "\nGuido: " in the chat command, "" in complete). It is printed exactly once,
// whether the answer is streamed or returned as a single block.
func runAgenticLoop(
	ctx context.Context,
	h *harness.Harness,
	history *[]harness.ChatMessage,
	chatModel string,
	temperature float32,
	maxTokens int,
	activeTools []harness.Tool,
	mcpReg *mcp.Registry,
	printProgress bool,
	finalPrefix string,
) (string, error) {
	useStreaming := len(activeTools) > 0 && h.UsesInTextToolCalls(chatModel)

	for {
		req := &harness.ChatRequest{
			Messages:    *history,
			Model:       chatModel,
			Temperature: temperature,
			MaxTokens:   maxTokens,
			Tools:       activeTools,
		}

		if useStreaming {
			text, toolCalls, err := streamingAgenticTurn(ctx, h, req, printProgress, finalPrefix)
			if err != nil {
				return "", err
			}

			if len(toolCalls) > 0 {
				*history = append(*history, harness.ChatMessage{
					Role:      "assistant",
					Content:   harness.Text(""),
					ToolCalls: toolCalls,
				})
				if err := dispatchToolCalls(ctx, toolCalls, history, mcpReg, printProgress); err != nil {
					return "", err
				}
				continue
			}

			final := strings.TrimSpace(stripStopTokens(text))
			*history = append(*history, harness.ChatMessage{
				Role:    "assistant",
				Content: harness.Text(final),
			})
			return final, nil
		}

		// Non-streaming path (OpenAI, Anthropic, etc.).
		resp, err := h.Chat(ctx, req)
		if err != nil {
			return "", err
		}
		*history = append(*history, resp.Message)

		if len(resp.Message.ToolCalls) == 0 {
			text := strings.TrimSpace(stripStopTokens(resp.Message.Content.PlainText()))
			if printProgress {
				fmt.Print(finalPrefix)
				fmt.Println(text)
			}
			return text, nil
		}

		if err := dispatchToolCalls(ctx, resp.Message.ToolCalls, history, mcpReg, printProgress); err != nil {
			return "", err
		}
	}
}

// streamingAgenticTurn calls h.StreamChat and uses a lookahead buffer to
// distinguish tool-call turns from final-answer turns without waiting for the
// full response.
//
// Decision rule (for llamacpp system-prompt injection):
//   - If the first ≥10 non-whitespace characters start with "TOOL_CALL:" →
//     the entire response is a tool call. Buffer silently; return tool calls.
//   - Otherwise → flush the buffer to the terminal, then stream remaining
//     tokens live. Return the full accumulated text with no tool calls.
//
// The lookahead adds at most ~3 token delays before printing begins (negligible).
func streamingAgenticTurn(
	ctx context.Context,
	h *harness.Harness,
	req *harness.ChatRequest,
	printProgress bool,
	prefix string,
) (text string, toolCalls []harness.ToolCall, err error) {
	const lookahead = len("TOOL_CALL:") // 10 — enough to distinguish

	stream, err := h.StreamChat(ctx, req)
	if err != nil {
		return "", nil, err
	}

	var buf strings.Builder
	decided := false
	isContent := false // true once we've determined this is a content turn

	for token := range stream {
		buf.WriteString(token)

		if !decided {
			peeked := strings.TrimSpace(buf.String())

			// Can we make a decision yet?
			// Yes when: we have ≥ lookahead chars, OR we can already rule out
			// "TOOL_CALL:" because the prefix doesn't match.
			canDecide := len(peeked) >= lookahead ||
				(len(peeked) > 0 && !strings.HasPrefix("TOOL_CALL:", peeked))

			if canDecide {
				decided = true
				isContent = !strings.HasPrefix(peeked, "TOOL_CALL:")
				if isContent && printProgress {
					fmt.Print(prefix)       // e.g. "\nGuido: " in chat, "" in complete
					fmt.Print(buf.String()) // flush buffered content
				}
			}
			// else: keep buffering — still ambiguous
		} else if isContent && printProgress {
			fmt.Print(token) // stream live
		}
	}

	// Handle streams that ended before we accumulated enough to decide
	// (very short responses — treat as content).
	if !decided {
		peeked := strings.TrimSpace(buf.String())
		isContent = !strings.HasPrefix(peeked, "TOOL_CALL:")
		if isContent && printProgress {
			fmt.Print(prefix)
			fmt.Print(buf.String())
		}
	}

	if isContent && printProgress {
		fmt.Println()
	}

	fullText := buf.String()
	tcs := backends.ParseToolCalls(fullText)
	return fullText, tcs, nil
}

// dispatchToolCalls executes every tool call, logs progress, and appends tool
// result messages to *history.
func dispatchToolCalls(
	ctx context.Context,
	calls []harness.ToolCall,
	history *[]harness.ChatMessage,
	mcpReg *mcp.Registry,
	printProgress bool,
) error {
	for _, tc := range calls {
		if printProgress {
			var toolArgs map[string]interface{}
			_ = json.Unmarshal([]byte(tc.Function.Arguments), &toolArgs)
			switch tc.Function.Name {
			case "web_search":
				q, _ := toolArgs["query"].(string)
				fmt.Printf("\n[searching] %q...\n", q)
			case "fetch_url":
				u, _ := toolArgs["url"].(string)
				fmt.Printf("\n[fetching] %s...\n", u)
			default:
				fmt.Printf("\n[tool] %s %v\n", tc.Function.Name, toolArgs)
			}
		}
		result, execErr := dispatchTool(ctx, tc, mcpReg)
		if execErr != nil {
			result = "Error: " + execErr.Error()
		}
		*history = append(*history, harness.ChatMessage{
			Role:       "tool",
			ToolCallID: tc.ID,
			Name:       tc.Function.Name,
			Content:    harness.Text(result),
		})
	}
	return nil
}

// target="" falls through to cfg.Models.Default.
// Logs a fatal error if the target isn't found in the config.
func filterBackends(cfg *harness.Config, target string, all bool) {
	if all {
		return
	}
	if target == "" {
		target = cfg.Models.Default
	}
	if target == "" {
		log.Fatal("No default model set in config and --model not specified")
	}
	bcfg, ok := cfg.Backends[target]
	if !ok {
		var names []string
		for k := range cfg.Backends {
			names = append(names, k)
		}
		log.Fatalf("Backend %q not found in config. Available: %v", target, names)
	}
	cfg.Backends = map[string]harness.BackendConfig{target: bcfg}
	cfg.Models.Default = target
}

// initializeBackends registers all configured backends with the harness.
//
// When lazy is true (serve mode), embedded llamacpp backends are wrapped in a
// LazyLlamaCppBackend so the guido-server process only starts on the first
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
			providers[name] = backends.NewOpenAIBackend(bcfg.APIKey, modelName, bcfg.URL)
			h.RegisterProvider(name, providers[name])
			if appLogger != nil {
				appLogger.ModelLoaded(name, modelName)
			}

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
			if appLogger != nil {
				appLogger.ModelLoaded(name, modelName)
			}

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
					port = nextEmbeddedPort(usedPorts, llamaPort)
				}
				usedPorts[port] = true
				llamacppURL = fmt.Sprintf("http://localhost:%d", port)
				expandedModelPath := os.ExpandEnv(bcfg.ModelPath)

				expandedMmProjPath := os.ExpandEnv(bcfg.MmProjPath)

				if lazy {
					// ── Lazy path (serve mode) ────────────────────────────────
					// The LazyLlamaCppBackend manages the guido-server lifecycle
					// internally: it starts on the first request and optionally
					// unloads after idleTimeout seconds of inactivity.
					gpuLayers := bcfg.GPULayers
					if gpuLayers == 0 {
						gpuLayers = llamaGPU
					}
					idleTimeout := time.Duration(bcfg.IdleTimeoutSeconds) * time.Second
					lb := backends.NewLazyLlamaCppBackend(
						tm, name, expandedModelPath, expandedMmProjPath, llamacppURL, modelName,
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
					log.Printf("Waiting for guido-server on %s to finish loading...", llamacppURL)
					status = waitForServer(llamacppURL, expandedModelPath, 5*time.Minute)
				}
				switch status {
				case serverReady:
					log.Printf("Using existing guido-server for %q at %s", name, llamacppURL)
				case serverWrongModel:
					log.Fatalf(
						"A guido-server is already running on %s but serves a different model.\n"+
							"Kill it first, then retry:\n\n  pkill -f 'guido-server.*%d'\n",
						llamacppURL, port,
					)
				case serverNotRunning, serverLoading:
					if tm != nil && expandedModelPath != "" {
						gpuLayers := bcfg.GPULayers
						if gpuLayers == 0 {
							gpuLayers = llamaGPU
						}
						_, err := tm.StartLlamaServer(expandedModelPath, port, gpuLayers, bcfg.ChatTemplate, expandedMmProjPath)
						if err != nil {
							log.Fatalf("Failed to start guido-server for %q: %v\n", name, err)
						}
					}
				}
			}

			providers[name] = backends.NewLlamaCppBackend(llamacppURL, modelName)
			h.RegisterProvider(name, providers[name])
			if appLogger != nil {
				appLogger.ModelLoaded(name, modelName)
			}

		case "ollama":
			modelName := bcfg.Model
			if modelName == "" {
				modelName = "llama3.2"
			}
			providers[name] = backends.NewOllamaBackend(modelName, bcfg.URL, os.ExpandEnv(bcfg.ModelPath))
			h.RegisterProvider(name, providers[name])
			if appLogger != nil {
				appLogger.ModelLoaded(name, modelName)
			}

		case "mock":
			modelName := bcfg.Model
			if modelName == "" {
				modelName = "test-model"
			}
			providers[name] = backends.NewMockBackend(modelName)
			h.RegisterProvider(name, providers[name])
			if appLogger != nil {
				appLogger.ModelLoaded(name, modelName)
			}

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
			if appLogger != nil {
				appLogger.ModelLoaded(name, bcfg.Model)
			}
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
	chatCmd.Flags().StringVar(&flagResume, "resume", "", "Resume a saved chat by id (from logs/chats/)")
	chatCmd.Flags().BoolVarP(&flagContinue, "continue", "c", false, "Resume the most recent saved chat")

	completeCmd.Flags().StringArrayVar(&contextStrings, "context", nil, "Raw string to inject as context before the prompt")
	completeCmd.Flags().StringArrayVar(&contextFiles, "file", nil, "File to attach (text files injected as text, images base64-encoded)")
	completeCmd.Flags().StringArrayVar(&contextImages, "image", nil, "Image file to attach (base64-encoded)")
	completeCmd.Flags().BoolVar(&allBackends, "all-backends", false, "Initialize every configured backend instead of just the target model's backend")

	chatCmd.Flags().StringArrayVar(&contextStrings, "context", nil, "Raw string to inject as context in the first message")
	chatCmd.Flags().StringArrayVar(&contextFiles, "file", nil, "File to attach to the first message")
	chatCmd.Flags().StringArrayVar(&contextImages, "image", nil, "Image to attach to the first message")
	chatCmd.Flags().BoolVar(&allBackends, "all-backends", false, "Initialize every configured backend instead of just the target model's backend")

	serveCmd.Flags().StringVarP(&serveModel, "model", "m", "", "Backend to serve (default: models.default from config)")
	serveCmd.Flags().BoolVar(&serveAllBackends, "all-backends", false, "Serve every configured backend instead of just the default")

	harnessCmd.Flags().IntVar(&llamaPort, "llama-port", 8000, "Starting port for embedded llamacpp backends with no explicit port in config")
	harnessCmd.Flags().IntVar(&llamaGPU, "llama-gpu-layers", 99, "Default GPU layers for embedded llamacpp backends with no explicit gpu_layers in config")

	// Tool mode flags — identical on chat, complete, and serve.
	// At most one may be set per invocation (Cobra enforces the mutual exclusion).
	// No flag → all available tools are active (web search + any configured MCP servers).
	for _, cmd := range []*cobra.Command{chatCmd, completeCmd, serveCmd} {
		cmd.Flags().BoolVar(&flagSearch, "search", false, "Web search tools only (MCP disabled)")
		cmd.Flags().BoolVar(&flagMCP, "mcp", false, "MCP tools only (web search disabled)")
		cmd.Flags().BoolVar(&flagNative, "native", false, "No tools — native model capabilities only")
		cmd.MarkFlagsMutuallyExclusive("search", "mcp", "native")
	}

	rootCmd.AddCommand(completeCmd)
	rootCmd.AddCommand(chatCmd)
	rootCmd.AddCommand(serveCmd)
	rootCmd.AddCommand(harnessCmd)
	rootCmd.AddCommand(modelsCmd)
}

type llamaStatus int

const (
	serverNotRunning llamaStatus = iota
	serverReady
	serverLoading // alive but model not yet fully loaded (503)
	serverWrongModel
)

// llamaServerStatus checks whether a guido-server on baseURL is alive and
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
		log.Printf("guido-server on %s is still loading the model (503)...", baseURL)
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
// even when guido-server is supposed to stop at them.
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

// buildMessageContent assembles a MessageContent from the user's text plus any
// attached context strings, files, and images passed via CLI flags.
// Returns plain-text MessageContent when no attachments are present (common case).
func buildMessageContent(text string, contexts, files, images []string) (harness.MessageContent, error) {
	if len(contexts) == 0 && len(files) == 0 && len(images) == 0 {
		return harness.Text(text), nil
	}

	var parts []harness.ContentPart

	for _, ctx := range contexts {
		parts = append(parts, harness.TextPart(ctx))
	}

	for _, path := range files {
		data, err := os.ReadFile(path)
		if err != nil {
			return harness.MessageContent{}, fmt.Errorf("reading %s: %w", path, err)
		}
		if isImageExt(filepath.Ext(path)) {
			mime := mimeForExt(filepath.Ext(path))
			parts = append(parts, harness.ImageURLPart(
				"data:"+mime+";base64,"+base64.StdEncoding.EncodeToString(data),
			))
		} else {
			// Text file — label it with the filename so the model knows the source
			parts = append(parts, harness.TextPart(
				fmt.Sprintf("=== %s ===\n%s", filepath.Base(path), string(data)),
			))
		}
	}

	for _, path := range images {
		data, err := os.ReadFile(path)
		if err != nil {
			return harness.MessageContent{}, fmt.Errorf("reading image %s: %w", path, err)
		}
		mime := mimeForExt(filepath.Ext(path))
		parts = append(parts, harness.ImageURLPart(
			"data:"+mime+";base64,"+base64.StdEncoding.EncodeToString(data),
		))
	}

	// User's message text goes last so context precedes the question
	if text != "" {
		parts = append(parts, harness.TextPart(text))
	}

	return harness.Parts(parts...), nil
}

func isImageExt(ext string) bool {
	switch strings.ToLower(ext) {
	case ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp":
		return true
	}
	return false
}

func mimeForExt(ext string) string {
	switch strings.ToLower(ext) {
	case ".jpg", ".jpeg":
		return "image/jpeg"
	case ".png":
		return "image/png"
	case ".gif":
		return "image/gif"
	case ".webp":
		return "image/webp"
	case ".bmp":
		return "image/bmp"
	default:
		return "application/octet-stream"
	}
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
