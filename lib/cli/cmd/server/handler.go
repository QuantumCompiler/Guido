package main

import (
	"encoding/json"
	"fmt"
	"net/http"

	"guido/lib/cli/harness"
)

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
		data := map[string]interface{}{
			"token": token,
		}
		if err := json.NewEncoder(w).Encode(data); err != nil {
			// Client disconnected
			return
		}
		flusher.Flush()
	}

	// Send final event
	finalEvent := map[string]interface{}{
		"done": true,
	}
	json.NewEncoder(w).Encode(finalEvent)
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
