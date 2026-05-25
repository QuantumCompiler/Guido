package httpserver

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

// Handler handles HTTP requests backed by a harness instance.
type Handler struct {
	h *harness.Harness
}

// NewHandler creates a Handler wrapping the given harness.
func NewHandler(h *harness.Harness) *Handler {
	return &Handler{h: h}
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

// HandleChat handles POST /v1/chat/completions (OpenAI-compatible)
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
