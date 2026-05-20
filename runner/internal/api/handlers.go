package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/taylor/guido/runner/internal/llm"
	"github.com/taylor/guido/runner/internal/registry"
)

// ---------------------------------------------------------------------------
// GET /
// ---------------------------------------------------------------------------

func (s *Server) handleRoot(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "minollama is running")
}

// ---------------------------------------------------------------------------
// GET /api/version
// ---------------------------------------------------------------------------

func (s *Server) handleVersion(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, VersionResponse{Version: version})
}

// ---------------------------------------------------------------------------
// GET /api/tags  — list locally available models
// ---------------------------------------------------------------------------

func (s *Server) handleTags(w http.ResponseWriter, r *http.Request) {
	models, err := s.reg.List()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	var infos []ModelInfo
	for _, m := range models {
		infos = append(infos, toModelInfo(m))
	}
	if infos == nil {
		infos = []ModelInfo{} // return [] not null
	}

	writeJSON(w, http.StatusOK, ListResponse{Models: infos})
}

// ---------------------------------------------------------------------------
// POST /api/show  — metadata for a specific model
// ---------------------------------------------------------------------------

func (s *Server) handleShow(w http.ResponseWriter, r *http.Request) {
	var req ShowRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if req.Name == "" {
		writeError(w, http.StatusBadRequest, "name is required")
		return
	}

	m, err := s.reg.Resolve(req.Name)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}

	writeJSON(w, http.StatusOK, ShowResponse{
		ModelInfo: toModelInfo(*m),
		Modelfile: fmt.Sprintf("# auto-generated\nFROM %s\n", m.Path),
	})
}

// ---------------------------------------------------------------------------
// POST /api/generate  — raw text completion
// ---------------------------------------------------------------------------

func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	var req GenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "model is required")
		return
	}
	if req.Prompt == "" {
		writeError(w, http.StatusBadRequest, "prompt is required")
		return
	}

	// Resolve model → load if necessary
	model, err := s.reg.Resolve(req.Model)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}

	loadStart := time.Now()
	if err := s.runner.EnsureModel(model.Path); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to load model: "+err.Error())
		return
	}
	loadDuration := time.Since(loadStart).Nanoseconds()

	// Build prompt (prepend system if provided)
	prompt := req.Prompt
	if req.System != "" {
		prompt = req.System + "\n\n" + prompt
	}

	// Extract inference options
	temp, topP, topK, nPredict := extractOptions(req.Options)

	llamaReq := llm.CompletionReq{
		Prompt:      prompt,
		NPredict:    nPredict,
		Temperature: temp,
		TopP:        topP,
		TopK:        topK,
		Stream:      req.ShouldStream(),
		CachePrompt: true,
	}

	genStart := time.Now()
	ch, err := s.runner.Complete(r.Context(), llamaReq)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	// Set up streaming headers
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Transfer-Encoding", "chunked")

	flusher, canFlush := w.(http.Flusher)
	if !canFlush {
		writeError(w, http.StatusInternalServerError, "streaming not supported by server")
		return
	}

	var totalTokens, promptTokens int

	for chunk := range ch {
		promptTokens = chunk.TokensEvaluated

		resp := GenerateResponse{
			Model:     req.Model,
			CreatedAt: time.Now(),
			Response:  chunk.Content,
			Done:      chunk.Stop,
		}

		if chunk.Stop {
			totalTokens = chunk.TokensPredicted
			resp.TotalDuration = time.Since(genStart).Nanoseconds()
			resp.LoadDuration = loadDuration
			resp.EvalCount = totalTokens
			resp.EvalDuration = time.Since(genStart).Nanoseconds()
			resp.PromptEvalCount = promptTokens
			resp.PromptEvalDuration = loadDuration
		}

		if err := writeNDJSON(w, flusher, resp); err != nil {
			return // client disconnected
		}
	}
}

// ---------------------------------------------------------------------------
// POST /api/chat  — multi-turn conversation
// ---------------------------------------------------------------------------

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "model is required")
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages must not be empty")
		return
	}

	model, err := s.reg.Resolve(req.Model)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}

	loadStart := time.Now()
	if err := s.runner.EnsureModel(model.Path); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to load model: "+err.Error())
		return
	}
	loadDuration := time.Since(loadStart).Nanoseconds()

	temp, _, _, nPredict := extractOptions(req.Options)

	// Convert Ollama messages → llm.ChatMessage
	var msgs []llm.ChatMessage
	for _, m := range req.Messages {
		msgs = append(msgs, llm.ChatMessage{Role: m.Role, Content: m.Content})
	}

	chatReq := llm.ChatReq{
		Messages:    msgs,
		MaxTokens:   nPredict,
		Temperature: temp,
		Stream:      req.ShouldStream(),
	}

	genStart := time.Now()
	ch, err := s.runner.Chat(r.Context(), chatReq)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Transfer-Encoding", "chunked")

	flusher, canFlush := w.(http.Flusher)
	if !canFlush {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	var fullContent strings.Builder

	for chunk := range ch {
		fullContent.WriteString(chunk.Content)

		resp := ChatResponse{
			Model:     req.Model,
			CreatedAt: time.Now(),
			Message: Message{
				Role:    "assistant",
				Content: chunk.Content,
			},
			Done: chunk.Done,
		}

		if chunk.Done {
			resp.TotalDuration = time.Since(genStart).Nanoseconds()
			resp.LoadDuration = loadDuration
			resp.EvalCount = chunk.CompTokens
			resp.PromptEvalCount = chunk.PromptTokens
		}

		if err := writeNDJSON(w, flusher, resp); err != nil {
			return
		}
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func toModelInfo(m registry.Model) ModelInfo {
	// Try to infer family from the model name
	name := strings.ToLower(m.Name)
	family := "unknown"
	for _, f := range []string{"llama", "mistral", "gemma", "phi", "qwen", "deepseek", "falcon"} {
		if strings.Contains(name, f) {
			family = f
			break
		}
	}

	// Infer quant level from name tag
	quantLevel := ""
	if idx := strings.Index(m.Name, ":"); idx >= 0 {
		quantLevel = m.Name[idx+1:]
	}

	return ModelInfo{
		Name:       m.Name,
		ModifiedAt: m.ModifiedAt,
		Size:       m.Size,
		Digest:     m.Digest,
		Details: ModelDetails{
			Format:            "gguf",
			Family:            family,
			QuantizationLevel: quantLevel,
		},
	}
}

// extractOptions pulls common inference params from the options map.
// Falls back to sensible defaults if not set.
func extractOptions(opts map[string]interface{}) (temp, topP float64, topK, nPredict int) {
	temp = 0.8
	topP = 0.9
	topK = 40
	nPredict = 512

	if opts == nil {
		return
	}

	if v, ok := opts["temperature"]; ok {
		if f, ok := toFloat(v); ok {
			temp = f
		}
	}
	if v, ok := opts["top_p"]; ok {
		if f, ok := toFloat(v); ok {
			topP = f
		}
	}
	if v, ok := opts["top_k"]; ok {
		if f, ok := toFloat(v); ok {
			topK = int(f)
		}
	}
	if v, ok := opts["num_predict"]; ok {
		if f, ok := toFloat(v); ok {
			nPredict = int(f)
		}
	}
	return
}

func toFloat(v interface{}) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	}
	return 0, false
}
