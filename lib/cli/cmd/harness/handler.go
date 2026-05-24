package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"guido/lib/cli/harness"
)

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

// Handler handles HTTP requests for the harness
type Handler struct {
	harness *harness.Harness
}

// NewHandler creates a new request handler
func NewHandler(h *harness.Harness) *Handler {
	return &Handler{harness: h}
}

// HandleCompletion handles POST /v1/completions requests
func (h *Handler) HandleCompletion(w http.ResponseWriter, r *http.Request) {
	var req harness.CompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.Prompt == "" {
		http.Error(w, "prompt is required", http.StatusBadRequest)
		return
	}

	// Default to non-streaming
	if req.MaxTokens == 0 {
		req.MaxTokens = 256
	}

	ctx := r.Context()

	if req.StreamMode {
		h.handleStream(w, r, &req)
	} else {
		resp, err := h.harness.Complete(ctx, &req)
		if err != nil {
			http.Error(w, fmt.Sprintf("completion error: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}
}

// handleStream handles streaming responses using Server-Sent Events
func (h *Handler) handleStream(w http.ResponseWriter, r *http.Request, req *harness.CompletionRequest) {
	tokenChan, err := h.harness.StreamTokens(r.Context(), req)
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
			break // hit a stop token — end the stream
		}
		data := map[string]interface{}{"token": cleaned}
		if err := json.NewEncoder(w).Encode(data); err != nil {
			return // client disconnected
		}
		flusher.Flush()
	}

	json.NewEncoder(w).Encode(map[string]interface{}{"done": true})
	flusher.Flush()
}

// HandleChat handles POST /v1/chat/completions requests (OpenAI-compatible)
func (h *Handler) HandleChat(w http.ResponseWriter, r *http.Request) {
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
		req.MaxTokens = -1 // unlimited by default for chat
	}

	ctx := r.Context()

	if req.Stream {
		h.handleChatStream(w, r, &req)
		return
	}

	resp, err := h.harness.Chat(ctx, &req)
	if err != nil {
		http.Error(w, fmt.Sprintf("chat error: %v", err), http.StatusInternalServerError)
		return
	}

	// Return OpenAI-compatible response shape
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

// handleChatStream handles streaming /v1/chat/completions using SSE
func (h *Handler) handleChatStream(w http.ResponseWriter, r *http.Request, req *harness.ChatRequest) {
	tokenChan, err := h.harness.StreamChat(r.Context(), req)
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
		// Emit OpenAI-compatible SSE chunk
		chunk := map[string]interface{}{
			"object": "chat.completion.chunk",
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]string{"content": cleaned},
				},
			},
		}
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// HandleListModels handles GET /v1/models requests
func (h *Handler) HandleListModels(w http.ResponseWriter, r *http.Request) {
	models, err := h.harness.ListAllModels(r.Context())
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to list models: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	response := map[string]interface{}{
		"models": models,
	}
	json.NewEncoder(w).Encode(response)
}

// HandleHealth is a simple health check endpoint
func (h *Handler) HandleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}
