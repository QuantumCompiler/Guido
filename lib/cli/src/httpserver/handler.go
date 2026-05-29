package httpserver

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"guido/lib/cli/src/backends"
	"guido/lib/cli/src/harness"
	"guido/lib/cli/src/logger"
)

// ToolConfig holds the server-side tool set and executor used when the serve
// command is started with --search, --mcp, or no tool flag (all tools).
// When non-nil, the handler runs the agentic loop internally and returns only
// the final text response — clients see a plain chat completion with no tool
// calls in the output.
type ToolConfig struct {
	Tools    []harness.Tool
	ExecTool func(ctx context.Context, name, argsJSON string) (string, error)
}

// stopTokens are end-of-turn markers that some models leak into output.
var stopTokens = []string{
	"<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end|>",
	"<|im_start|>user", "<|im_start|>assistant",
	"\nUser:", "\nHuman:",
}

// lastUserInput returns the plain text of the most recent user message, used
// to record what the user submitted in the per-chat log.
func lastUserInput(msgs []harness.ChatMessage) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "user" {
			return msgs[i].Content.PlainText()
		}
	}
	return ""
}

// estimatePromptTokens approximates the prompt size of a message list by
// estimating tokens over the concatenated plain text of every message. Used for
// streaming responses where the backend doesn't report a usage block.
func estimatePromptTokens(msgs []harness.ChatMessage) int {
	var b strings.Builder
	for _, m := range msgs {
		b.WriteString(m.Content.PlainText())
		b.WriteByte('\n')
	}
	return logger.EstimateTokens(b.String())
}

func stripStopTokens(s string) string {
	for _, tok := range stopTokens {
		if idx := strings.Index(s, tok); idx >= 0 {
			return s[:idx]
		}
	}
	return s
}

// Handler handles HTTP requests backed by a harness instance.
type Handler struct {
	h       *harness.Harness
	toolCfg *ToolConfig  // nil when no server-side tools are configured
	log     *logger.Logger // nil → logging disabled
}

// NewHandler creates a Handler wrapping the given harness.
// tc may be nil (no server-side tool injection).
// lg may be nil (no logging).
func NewHandler(h *harness.Harness, tc *ToolConfig, lg *logger.Logger) *Handler {
	return &Handler{h: h, toolCfg: tc, log: lg}
}

// toolsForMode returns the subset of tc.Tools appropriate for the requested
// mode. An empty/unrecognised mode returns all tools.
//
//	""      / "all"    → all configured tools (default)
//	"search"           → web_search and fetch_url only
//	"mcp"              → mcp__* tools only
//	"none"             → nil (skip tool loop entirely)
func toolsForMode(tc *ToolConfig, mode string) []harness.Tool {
	if tc == nil {
		return nil
	}
	switch strings.ToLower(mode) {
	case "none":
		return nil
	case "search":
		var out []harness.Tool
		for _, t := range tc.Tools {
			n := t.Function.Name
			if n == "web_search" || n == "fetch_url" {
				out = append(out, t)
			}
		}
		return out
	case "mcp":
		var out []harness.Tool
		for _, t := range tc.Tools {
			if strings.HasPrefix(t.Function.Name, "mcp__") {
				out = append(out, t)
			}
		}
		return out
	default: // "" or "all"
		return tc.Tools
	}
}

// HandleCompletion handles POST /v1/completions (OpenAI text-completion format).
func (hnd *Handler) HandleCompletion(w http.ResponseWriter, r *http.Request) {
	var req harness.CompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}
	if req.Prompt == "" {
		http.Error(w, "prompt is required", http.StatusBadRequest)
		return
	}
	if req.MaxTokens == 0 {
		req.MaxTokens = -1 // unlimited
	}
	if req.StreamMode {
		hnd.handleCompletionStream(w, r, &req)
		return
	}

	if hnd.log != nil {
		hnd.log.CompleteCall(req.Model, "http")
	}
	resp, err := hnd.h.Complete(r.Context(), &req)
	if err != nil {
		http.Error(w, fmt.Sprintf("completion error: %v", err), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"object":  "text_completion",
		"model":   resp.Model,
		"created": time.Now().Unix(),
		"choices": []map[string]interface{}{
			{"text": resp.Text, "index": 0, "finish_reason": resp.FinishReason},
		},
		"usage": map[string]interface{}{
			"completion_tokens": resp.TokensUsed,
			"total_tokens":      resp.TokensUsed,
		},
	})
}

// handleCompletionStream streams a text completion as OpenAI SSE.
func (hnd *Handler) handleCompletionStream(w http.ResponseWriter, r *http.Request, req *harness.CompletionRequest) {
	tokenChan, err := hnd.h.StreamTokens(r.Context(), req)
	if err != nil {
		http.Error(w, fmt.Sprintf("streaming error: %v", err), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}
	for token := range tokenChan {
		cleaned := stripStopTokens(token)
		if cleaned == "" {
			break
		}
		chunk := map[string]interface{}{
			"object": "text_completion",
			"model":  req.Model,
			"choices": []map[string]interface{}{
				{"text": cleaned, "index": 0, "finish_reason": nil},
			},
		}
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// HandleChat handles POST /v1/chat/completions (OpenAI-compatible).
//
// Per-request tool mode override via the "tool_mode" field:
//
//	""  / "all"   → all server-configured tools (default)
//	"search"      → web_search and fetch_url only
//	"mcp"         → mcp__* tools only
//	"none"        → no tools — passes straight to the model
//
// When a ToolConfig is set (and effective tools > 0) the handler runs the full
// agentic loop internally. Clients receive only the final answer; tool turns
// are invisible in the response.
//
// Streaming behaviour:
//   - llamacpp backends: tokens stream live via SSE; tool turns are silent.
//   - Other backends with tools: tool turns run synchronously then final answer streams.
//   - Other backends without tools: pass-through streaming.
//
// Tool transparency:
//   - X-Guido-Tools-Used response header: comma-separated list of invoked tool names.
//   - Non-streaming tool responses include a "guido_tool_calls" array in the body.
func (hnd *Handler) HandleChat(w http.ResponseWriter, r *http.Request) {
	// Use an inline struct so HTTP-only fields don't pollute harness.ChatRequest.
	var body struct {
		harness.ChatRequest
		ToolMode string `json:"tool_mode,omitempty"`
		// System is a convenience shorthand: if set, it is automatically prepended
		// as a system message (or merged with an existing system message at index 0).
		// Equivalent to adding {"role":"system","content":"..."} as messages[0].
		System string `json:"system,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}
	if len(body.Messages) == 0 {
		http.Error(w, "messages array is required", http.StatusBadRequest)
		return
	}
	if body.MaxTokens == 0 {
		body.MaxTokens = -1
	}

	// Resolve system prompt shorthand.
	if body.System != "" {
		if len(body.Messages) > 0 && body.Messages[0].Role == "system" {
			// Merge with existing system message (shorthand prepended).
			merged := body.System + "\n\n" + body.Messages[0].Content.PlainText()
			body.Messages[0].Content = harness.Text(merged)
		} else {
			// Prepend a new system message.
			body.Messages = append(
				[]harness.ChatMessage{{Role: "system", Content: harness.Text(body.System)}},
				body.Messages...,
			)
		}
	}

	req := &body.ChatRequest

	// ── Logging ───────────────────────────────────────────────────────────────
	var chatSession *logger.ChatSession
	if hnd.log != nil {
		chatID := logger.NewChatID()
		var toolNames []string
		if hnd.toolCfg != nil {
			for _, t := range hnd.toolCfg.Tools {
				toolNames = append(toolNames, t.Function.Name)
			}
		}
		hnd.log.ChatSubmitted(chatID, req.Model, toolNames)
		chatSession = hnd.log.NewHTTPSession(chatID, req.Model, toolNames, req.Stream)
	}

	// Resolve effective tools for this request.
	effectiveTools := toolsForMode(hnd.toolCfg, body.ToolMode)
	var effectiveTC *ToolConfig
	if len(effectiveTools) > 0 {
		effectiveTC = &ToolConfig{
			Tools:    effectiveTools,
			ExecTool: hnd.toolCfg.ExecTool,
		}
	}

	if effectiveTC != nil {
		// ── Agentic tool loop ─────────────────────────────────────────────────
		if req.Stream {
			if hnd.h.UsesInTextToolCalls(req.Model) {
				// llamacpp: stream tokens live with lookahead detection.
				hnd.handleStreamingToolLoop(w, r, req, effectiveTC, chatSession)
			} else {
				// OpenAI/Anthropic etc.: run tool turns synchronously, stream final answer.
				hnd.handleStreamingNonLlamacpp(w, r, req, effectiveTC, chatSession)
			}
			return
		}

		// Non-streaming: run full loop synchronously.
		text, model, invokedTools, err := hnd.runToolLoop(r.Context(), req, effectiveTC)
		if err != nil {
			if chatSession != nil {
				chatSession.Finish("error")
			}
			http.Error(w, fmt.Sprintf("chat error: %v", err), http.StatusInternalServerError)
			return
		}
		if chatSession != nil {
			// Token counts aren't surfaced by the synchronous tool loop, so estimate.
			chatSession.MarkEstimated()
			chatSession.RecordTurn(
				lastUserInput(req.Messages), text,
				estimatePromptTokens(req.Messages), logger.EstimateTokens(text),
				invokedTools,
			)
			chatSession.Finish("stop")
		}
		if len(invokedTools) > 0 {
			w.Header().Set("X-Guido-Tools-Used", strings.Join(invokedTools, ","))
		}
		w.Header().Set("Content-Type", "application/json")
		respBody := map[string]interface{}{
			"object":  "chat.completion",
			"model":   model,
			"created": time.Now().Unix(),
			"choices": []map[string]interface{}{
				{"index": 0, "message": map[string]string{"role": "assistant", "content": text}, "finish_reason": "stop"},
			},
		}
		if len(invokedTools) > 0 {
			respBody["guido_tool_calls"] = invokedTools
		}
		json.NewEncoder(w).Encode(respBody)
		return
	}

	// ── No tools: pass straight through ──────────────────────────────────────
	if req.Stream {
		hnd.handleChatStream(w, r, req, chatSession)
		return
	}
	resp, err := hnd.h.Chat(r.Context(), req)
	if err != nil {
		if chatSession != nil {
			chatSession.Finish("error")
		}
		http.Error(w, fmt.Sprintf("chat error: %v", err), http.StatusInternalServerError)
		return
	}
	if chatSession != nil {
		chatSession.RecordTurn(
			lastUserInput(req.Messages), resp.Message.Content.PlainText(),
			resp.PromptTokens, resp.TokensUsed, nil,
		)
		chatSession.Finish(resp.FinishReason)
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"object":  "chat.completion",
		"model":   resp.Model,
		"created": time.Now().Unix(),
		"choices": []map[string]interface{}{
			{
				"index":         0,
				"message":       resp.Message,
				"finish_reason": resp.FinishReason,
			},
		},
		"usage": map[string]interface{}{
			"prompt_tokens":     resp.PromptTokens,
			"completion_tokens": resp.TokensUsed,
			"total_tokens":      resp.PromptTokens + resp.TokensUsed,
		},
	})
}

// runToolLoop drives the synchronous model→tool→model cycle.
// Returns the final text, model name, all invoked tool names, and any error.
func (hnd *Handler) runToolLoop(ctx context.Context, req *harness.ChatRequest, tc *ToolConfig) (text, model string, toolNames []string, err error) {
	history := make([]harness.ChatMessage, len(req.Messages))
	copy(history, req.Messages)

	for {
		chatReq := &harness.ChatRequest{
			Messages:    history,
			Model:       req.Model,
			Temperature: req.Temperature,
			MaxTokens:   req.MaxTokens,
			Tools:       tc.Tools,
		}
		resp, err := hnd.h.Chat(ctx, chatReq)
		if err != nil {
			return "", "", toolNames, err
		}
		history = append(history, resp.Message)

		if len(resp.Message.ToolCalls) == 0 {
			return strings.TrimSpace(stripStopTokens(resp.Message.Content.PlainText())), resp.Model, toolNames, nil
		}

		for _, call := range resp.Message.ToolCalls {
			toolNames = append(toolNames, call.Function.Name)
			result, execErr := tc.ExecTool(ctx, call.Function.Name, call.Function.Arguments)
			if execErr != nil {
				result = "Error: " + execErr.Error()
			}
			history = append(history, harness.ChatMessage{
				Role:       "tool",
				ToolCallID: call.ID,
				Name:       call.Function.Name,
				Content:    harness.Text(result),
			})
		}
	}
}

// handleStreamingNonLlamacpp runs the agentic loop for non-llamacpp backends
// (OpenAI, Anthropic, etc.) with streaming enabled. Tool turns are run
// synchronously via Chat(); the final answer is streamed via StreamChat().
//
// When no tools are actually invoked, the direct Chat() answer is returned as
// a single-chunk SSE to avoid paying an extra model call.
func (hnd *Handler) handleStreamingNonLlamacpp(w http.ResponseWriter, r *http.Request, req *harness.ChatRequest, tc *ToolConfig, sess *logger.ChatSession) {
	ctx := r.Context()
	history := make([]harness.ChatMessage, len(req.Messages))
	copy(history, req.Messages)

	var toolsUsed []string
	anyToolsInvoked := false

	// ── Run tool detection turns with Chat() ──────────────────────────────────
	for {
		chatReq := &harness.ChatRequest{
			Messages:    history,
			Model:       req.Model,
			Temperature: req.Temperature,
			MaxTokens:   req.MaxTokens,
			Tools:       tc.Tools,
		}
		resp, err := hnd.h.Chat(ctx, chatReq)
		if err != nil {
			http.Error(w, fmt.Sprintf("chat error: %v", err), http.StatusInternalServerError)
			return
		}

		if len(resp.Message.ToolCalls) == 0 {
			if !anyToolsInvoked {
				// Model answered directly without using any tools — return as-is.
				// No extra model call: use the response we already have.
				text := strings.TrimSpace(stripStopTokens(resp.Message.Content.PlainText()))
				if sess != nil {
					sess.RecordTurn(lastUserInput(req.Messages), text, resp.PromptTokens, resp.TokensUsed, nil)
					sess.Finish(resp.FinishReason)
				}
				hnd.writeSSEText(w, resp.Model, text)
				return
			}
			// All tool turns done; break to stream the final answer.
			break
		}

		history = append(history, resp.Message)
		anyToolsInvoked = true
		for _, call := range resp.Message.ToolCalls {
			toolsUsed = append(toolsUsed, call.Function.Name)
			result, execErr := tc.ExecTool(ctx, call.Function.Name, call.Function.Arguments)
			if execErr != nil {
				result = "Error: " + execErr.Error()
			}
			history = append(history, harness.ChatMessage{
				Role:       "tool",
				ToolCallID: call.ID,
				Name:       call.Function.Name,
				Content:    harness.Text(result),
			})
		}
	}

	// ── Stream the final answer (tool turns already done) ────────────────────
	// Set transparency header before any body write.
	if len(toolsUsed) > 0 {
		w.Header().Set("X-Guido-Tools-Used", strings.Join(toolsUsed, ","))
	}

	streamReq := &harness.ChatRequest{
		Messages:    history,
		Model:       req.Model,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		// No Tools on the final answer turn — avoids re-triggering tool logic.
	}
	tokenChan, err := hnd.h.StreamChat(ctx, streamReq)
	if err != nil {
		http.Error(w, fmt.Sprintf("streaming error: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	flusher, _ := w.(http.Flusher)
	var output strings.Builder
	for token := range tokenChan {
		cleaned := stripStopTokens(token)
		if cleaned == "" {
			break
		}
		output.WriteString(cleaned)
		hnd.writeSSEChunk(w, req.Model, cleaned)
		if flusher != nil {
			flusher.Flush()
		}
	}
	if sess != nil {
		sess.MarkEstimated()
		sess.RecordTurn(
			lastUserInput(req.Messages), output.String(),
			estimatePromptTokens(history), logger.EstimateTokens(output.String()),
			toolsUsed,
		)
		sess.Finish("stop")
	}
	fmt.Fprintf(w, "data: [DONE]\n\n")
	if flusher != nil {
		flusher.Flush()
	}
}

// handleStreamingToolLoop runs the agentic loop for llamacpp backends and
// streams the final answer token-by-token via SSE. Tool-call turns are handled
// silently — the client only sees content tokens and [DONE].
//
// The X-Guido-Tools-Used header is set with all invoked tool names before the
// first content token is written (headers are still writable at that point since
// tool turns produce no body output).
func (hnd *Handler) handleStreamingToolLoop(w http.ResponseWriter, r *http.Request, req *harness.ChatRequest, tc *ToolConfig, sess *logger.ChatSession) {
	const lookahead = len("TOOL_CALL:") // 10 chars

	ctx := r.Context()
	history := make([]harness.ChatMessage, len(req.Messages))
	copy(history, req.Messages)

	// Collect tool names across all turns; header set before first body write.
	var toolsUsed []string
	headerSent := false

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	// maybeSetToolHeader sets X-Guido-Tools-Used exactly once, before the first
	// body write. Must be called right before writeSSEChunk/Fprintf to w.
	maybeSetToolHeader := func() {
		if !headerSent {
			if len(toolsUsed) > 0 {
				w.Header().Set("X-Guido-Tools-Used", strings.Join(toolsUsed, ","))
			}
			headerSent = true
		}
	}

	for {
		chatReq := &harness.ChatRequest{
			Messages:    history,
			Model:       req.Model,
			Temperature: req.Temperature,
			MaxTokens:   req.MaxTokens,
			Tools:       tc.Tools,
		}

		tokenChan, err := hnd.h.StreamChat(ctx, chatReq)
		if err != nil {
			maybeSetToolHeader()
			fmt.Fprintf(w, "data: {\"error\": %q}\n\n", err.Error())
			flusher.Flush()
			return
		}

		var buf strings.Builder
		decided := false
		isContent := false

		for token := range tokenChan {
			buf.WriteString(token)

			if !decided {
				peeked := strings.TrimSpace(buf.String())
				canDecide := len(peeked) >= lookahead ||
					(len(peeked) > 0 && !strings.HasPrefix("TOOL_CALL:", peeked))
				if canDecide {
					decided = true
					isContent = !strings.HasPrefix(peeked, "TOOL_CALL:")
					if isContent {
						maybeSetToolHeader()
						hnd.writeSSEChunk(w, req.Model, buf.String())
						flusher.Flush()
					}
				}
			} else if isContent {
				hnd.writeSSEChunk(w, req.Model, token)
				flusher.Flush()
			}
		}

		if !decided {
			peeked := strings.TrimSpace(buf.String())
			isContent = !strings.HasPrefix(peeked, "TOOL_CALL:")
			if isContent {
				maybeSetToolHeader()
				hnd.writeSSEChunk(w, req.Model, buf.String())
				flusher.Flush()
			}
		}

		fullText := buf.String()
		toolCalls := backends.ParseToolCalls(fullText)

		if len(toolCalls) == 0 {
			final := strings.TrimSpace(stripStopTokens(fullText))
			if sess != nil {
				sess.MarkEstimated()
				sess.RecordTurn(
					lastUserInput(req.Messages), final,
					estimatePromptTokens(history), logger.EstimateTokens(final),
					toolsUsed,
				)
				sess.Finish("stop")
			}
			history = append(history, harness.ChatMessage{
				Role:    "assistant",
				Content: harness.Text(final),
			})
			maybeSetToolHeader()
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			return
		}

		history = append(history, harness.ChatMessage{
			Role:      "assistant",
			Content:   harness.Text(""),
			ToolCalls: toolCalls,
		})
		for _, call := range toolCalls {
			toolsUsed = append(toolsUsed, call.Function.Name)
			result, execErr := tc.ExecTool(ctx, call.Function.Name, call.Function.Arguments)
			if execErr != nil {
				result = "Error: " + execErr.Error()
			}
			history = append(history, harness.ChatMessage{
				Role:       "tool",
				ToolCallID: call.ID,
				Name:       call.Function.Name,
				Content:    harness.Text(result),
			})
		}
	}
}

// writeSSEChunk emits a single OpenAI-compatible SSE content chunk.
func (hnd *Handler) writeSSEChunk(w http.ResponseWriter, model, content string) {
	chunk := map[string]interface{}{
		"object": "chat.completion.chunk",
		"model":  model,
		"choices": []map[string]interface{}{
			{"index": 0, "delta": map[string]string{"content": content}},
		},
	}
	data, _ := json.Marshal(chunk)
	fmt.Fprintf(w, "data: %s\n\n", data)
}

// writeSSEText writes a complete response as a minimal SSE stream —
// one content chunk followed by [DONE]. Used when a streaming client
// request is answered by the synchronous tool loop (non-llamacpp backends).
func (hnd *Handler) writeSSEText(w http.ResponseWriter, model, text string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	chunk := map[string]interface{}{
		"object": "chat.completion.chunk",
		"model":  model,
		"choices": []map[string]interface{}{
			{"index": 0, "delta": map[string]string{"content": text}},
		},
	}
	data, _ := json.Marshal(chunk)
	fmt.Fprintf(w, "data: %s\n\n", data)
	fmt.Fprintf(w, "data: [DONE]\n\n")
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

func (hnd *Handler) handleChatStream(w http.ResponseWriter, r *http.Request, req *harness.ChatRequest, sess *logger.ChatSession) {
	tokenChan, err := hnd.h.StreamChat(r.Context(), req)
	if err != nil {
		if sess != nil {
			sess.Finish("error")
		}
		http.Error(w, fmt.Sprintf("streaming error: %v", err), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	flusher, ok := w.(http.Flusher)
	if !ok {
		if sess != nil {
			sess.Finish("error")
		}
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}
	var output strings.Builder
	for token := range tokenChan {
		cleaned := stripStopTokens(token)
		if cleaned == "" {
			break
		}
		output.WriteString(cleaned)
		chunk := map[string]interface{}{
			"object": "chat.completion.chunk",
			"choices": []map[string]interface{}{
				{"index": 0, "delta": map[string]string{"content": cleaned}},
			},
		}
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}
	if sess != nil {
		sess.MarkEstimated()
		sess.RecordTurn(
			lastUserInput(req.Messages), output.String(),
			estimatePromptTokens(req.Messages), logger.EstimateTokens(output.String()),
			nil,
		)
		sess.Finish("stop")
	}
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// HandleListModels handles GET /v1/models — OpenAI-compatible format.
//
// Each entry includes a "default" boolean field (extension to the OpenAI spec)
// so clients can discover which model is used when "model" is omitted from a
// chat/completion request.
func (hnd *Handler) HandleListModels(w http.ResponseWriter, r *http.Request) {
	models, err := hnd.h.ListAllModels(r.Context())
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to list models: %v", err), http.StatusInternalServerError)
		return
	}

	defaultModel := hnd.h.DefaultModel()
	data := make([]map[string]interface{}, 0, len(models))
	for _, m := range models {
		data = append(data, map[string]interface{}{
			"id":       m.ID,
			"object":   "model",
			"created":  0,
			"owned_by": m.Provider,
			"default":  m.ID == defaultModel,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"object": "list",
		"data":   data,
	})
}

// HandleHealth handles GET /health
func (hnd *Handler) HandleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// HandleModelStatus handles GET /v1/model/status
//
// Without query params: returns status for all lazy-loading backends.
// With ?backend=<name>: returns status for that specific backend only.
func (hnd *Handler) HandleModelStatus(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if name := r.URL.Query().Get("backend"); name != "" {
		info, ok := hnd.h.ModelStatus(name)
		if !ok {
			http.Error(w,
				fmt.Sprintf("backend %q not found or does not support status reporting", name),
				http.StatusNotFound)
			return
		}
		json.NewEncoder(w).Encode(info)
		return
	}

	json.NewEncoder(w).Encode(map[string]interface{}{
		"backends": hnd.h.AllModelStatuses(),
	})
}
