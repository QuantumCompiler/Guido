package httpserver

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"guido/lib/cli/src/harness"
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
	toolCfg *ToolConfig // nil when no server-side tools are configured
}

// NewHandler creates a Handler wrapping the given harness.
// tc may be nil (no server-side tool injection).
func NewHandler(h *harness.Harness, tc *ToolConfig) *Handler {
	return &Handler{h: h, toolCfg: tc}
}

// HandleCompletion handles POST /v1/completions
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
		req.MaxTokens = 256
	}
	if req.StreamMode {
		hnd.handleStream(w, r, &req)
		return
	}
	resp, err := hnd.h.Complete(r.Context(), &req)
	if err != nil {
		http.Error(w, fmt.Sprintf("completion error: %v", err), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (hnd *Handler) handleStream(w http.ResponseWriter, r *http.Request, req *harness.CompletionRequest) {
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
		json.NewEncoder(w).Encode(map[string]interface{}{"token": cleaned})
		flusher.Flush()
	}
	json.NewEncoder(w).Encode(map[string]interface{}{"done": true})
	flusher.Flush()
}

// HandleChat handles POST /v1/chat/completions (OpenAI-compatible).
// When the handler has a ToolConfig, it runs the full agentic loop internally
// and returns only the final text response — clients see a plain chat
// completion with no tool_calls in the output.
func (hnd *Handler) HandleChat(w http.ResponseWriter, r *http.Request) {
	var req harness.ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}
	if len(req.Messages) == 0 {
		http.Error(w, "messages array is required", http.StatusBadRequest)
		return
	}
	if req.MaxTokens == 0 {
		req.MaxTokens = -1
	}

	// When server-side tools are configured, run the agentic loop internally.
	// The client receives only the final answer — no tool_call turns leak out.
	if hnd.toolCfg != nil && len(hnd.toolCfg.Tools) > 0 {
		text, model, err := hnd.runToolLoop(r.Context(), &req)
		if err != nil {
			http.Error(w, fmt.Sprintf("chat error: %v", err), http.StatusInternalServerError)
			return
		}
		if req.Stream {
			hnd.writeSSEText(w, model, text)
		} else {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"object": "chat.completion",
				"model":  model,
				"choices": []map[string]interface{}{
					{"index": 0, "message": map[string]string{"role": "assistant", "content": text}, "finish_reason": "stop"},
				},
			})
		}
		return
	}

	if req.Stream {
		hnd.handleChatStream(w, r, &req)
		return
	}
	resp, err := hnd.h.Chat(r.Context(), &req)
	if err != nil {
		http.Error(w, fmt.Sprintf("chat error: %v", err), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"object": "chat.completion",
		"model":  resp.Model,
		"choices": []map[string]interface{}{
			{
				"index":         0,
				"message":       resp.Message,
				"finish_reason": resp.FinishReason,
			},
		},
		"usage": map[string]interface{}{
			"completion_tokens": resp.TokensUsed,
		},
	})
}

// runToolLoop drives the model→tool→model cycle until the model produces a
// non-tool response, then returns the final text and model name.
func (hnd *Handler) runToolLoop(ctx context.Context, req *harness.ChatRequest) (text, model string, err error) {
	history := make([]harness.ChatMessage, len(req.Messages))
	copy(history, req.Messages)

	for {
		chatReq := &harness.ChatRequest{
			Messages:    history,
			Model:       req.Model,
			Temperature: req.Temperature,
			MaxTokens:   req.MaxTokens,
			Tools:       hnd.toolCfg.Tools,
		}
		resp, err := hnd.h.Chat(ctx, chatReq)
		if err != nil {
			return "", "", err
		}
		history = append(history, resp.Message)

		if len(resp.Message.ToolCalls) == 0 {
			return strings.TrimSpace(stripStopTokens(resp.Message.Content.PlainText())), resp.Model, nil
		}

		for _, tc := range resp.Message.ToolCalls {
			result, execErr := hnd.toolCfg.ExecTool(ctx, tc.Function.Name, tc.Function.Arguments)
			if execErr != nil {
				result = "Error: " + execErr.Error()
			}
			history = append(history, harness.ChatMessage{
				Role:       "tool",
				ToolCallID: tc.ID,
				Name:       tc.Function.Name,
				Content:    harness.Text(result),
			})
		}
	}
}

// writeSSEText writes a complete response as a minimal SSE stream —
// one content chunk followed by [DONE]. Used when a streaming client
// request is answered by the internal tool loop (which is non-streaming).
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

func (hnd *Handler) handleChatStream(w http.ResponseWriter, r *http.Request, req *harness.ChatRequest) {
	tokenChan, err := hnd.h.StreamChat(r.Context(), req)
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
			"object": "chat.completion.chunk",
			"choices": []map[string]interface{}{
				{"index": 0, "delta": map[string]string{"content": cleaned}},
			},
		}
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// HandleListModels handles GET /v1/models
func (hnd *Handler) HandleListModels(w http.ResponseWriter, r *http.Request) {
	models, err := hnd.h.ListAllModels(r.Context())
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to list models: %v", err), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"models": models})
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
//
// Example responses:
//
//	GET /v1/model/status
//	{"backends":{"gemma4-q4km":{"model":"gemma4:31b-q4km","status":"ready","idle_seconds":42}}}
//
//	GET /v1/model/status?backend=gemma4-q4km
//	{"model":"gemma4:31b-q4km","status":"loading"}
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
