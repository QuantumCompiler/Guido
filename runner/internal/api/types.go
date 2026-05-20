package api

import "time"

// ---------------------------------------------------------------------------
// Ollama-compatible API types (what clients send us / we send back)
// ---------------------------------------------------------------------------

// GenerateRequest mirrors POST /api/generate
type GenerateRequest struct {
	Model   string                 `json:"model"`
	Prompt  string                 `json:"prompt"`
	System  string                 `json:"system,omitempty"`
	Stream  *bool                  `json:"stream,omitempty"` // nil = true (stream by default)
	Options map[string]interface{} `json:"options,omitempty"`
}

func (r *GenerateRequest) ShouldStream() bool {
	return r.Stream == nil || *r.Stream
}

// GenerateResponse is one NDJSON line for /api/generate
type GenerateResponse struct {
	Model     string    `json:"model"`
	CreatedAt time.Time `json:"created_at"`
	Response  string    `json:"response"`
	Done      bool      `json:"done"`

	// These fields are only populated in the final (done=true) message
	TotalDuration      int64 `json:"total_duration,omitempty"`
	LoadDuration       int64 `json:"load_duration,omitempty"`
	EvalCount          int   `json:"eval_count,omitempty"`
	EvalDuration       int64 `json:"eval_duration,omitempty"`
	PromptEvalCount    int   `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64 `json:"prompt_eval_duration,omitempty"`
}

// ChatRequest mirrors POST /api/chat
type ChatRequest struct {
	Model    string                 `json:"model"`
	Messages []Message              `json:"messages"`
	Stream   *bool                  `json:"stream,omitempty"`
	Options  map[string]interface{} `json:"options,omitempty"`
}

func (r *ChatRequest) ShouldStream() bool {
	return r.Stream == nil || *r.Stream
}

// ChatResponse is one NDJSON line for /api/chat
type ChatResponse struct {
	Model     string    `json:"model"`
	CreatedAt time.Time `json:"created_at"`
	Message   Message   `json:"message"`
	Done      bool      `json:"done"`

	TotalDuration      int64 `json:"total_duration,omitempty"`
	LoadDuration       int64 `json:"load_duration,omitempty"`
	EvalCount          int   `json:"eval_count,omitempty"`
	EvalDuration       int64 `json:"eval_duration,omitempty"`
	PromptEvalCount    int   `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64 `json:"prompt_eval_duration,omitempty"`
}

// Message is used in chat requests and responses
type Message struct {
	Role    string `json:"role"`    // "system" | "user" | "assistant"
	Content string `json:"content"`
}

// ModelInfo describes a locally available model
type ModelInfo struct {
	Name       string    `json:"name"`
	ModifiedAt time.Time `json:"modified_at"`
	Size       int64     `json:"size"`
	Digest     string    `json:"digest"`
	Details    ModelDetails `json:"details"`
}

type ModelDetails struct {
	Format            string `json:"format"`
	Family            string `json:"family"`
	ParameterSize     string `json:"parameter_size"`
	QuantizationLevel string `json:"quantization_level"`
}

// ListResponse is returned by GET /api/tags
type ListResponse struct {
	Models []ModelInfo `json:"models"`
}

// ShowRequest / ShowResponse for POST /api/show
type ShowRequest struct {
	Name string `json:"name"`
}

type ShowResponse struct {
	ModelInfo
	Modelfile string `json:"modelfile"`
}

// ErrorResponse wraps API errors
type ErrorResponse struct {
	Error string `json:"error"`
}

// VersionResponse for GET /api/version
type VersionResponse struct {
	Version string `json:"version"`
}

// ---------------------------------------------------------------------------
// llama-server wire types (what we send to / receive from the subprocess)
// These match llama.cpp's REST API format.
// ---------------------------------------------------------------------------

// LlamaCompletionReq is sent to llama-server POST /completion
type LlamaCompletionReq struct {
	Prompt      string   `json:"prompt"`
	NPredict    int      `json:"n_predict"`
	Temperature float64  `json:"temperature"`
	TopP        float64  `json:"top_p"`
	TopK        int      `json:"top_k"`
	Stream      bool     `json:"stream"`
	Stop        []string `json:"stop,omitempty"`
	CachePrompt bool     `json:"cache_prompt"`
}

// LlamaCompletionChunk is one SSE chunk (or the full response when stream=false)
type LlamaCompletionChunk struct {
	Content         string `json:"content"`
	Stop            bool   `json:"stop"`
	TokensEvaluated int    `json:"tokens_evaluated"`
	TokensPredicted int    `json:"tokens_predicted"`
	Timings         *LlamaTimings `json:"timings,omitempty"`
}

type LlamaTimings struct {
	PredictedMs   float64 `json:"predicted_ms"`
	PromptMs      float64 `json:"prompt_ms"`
	PredictedN    int     `json:"predicted_n"`
	PromptN       int     `json:"prompt_n"`
}

// LlamaChatReq is sent to llama-server POST /v1/chat/completions (OpenAI-compat)
type LlamaChatReq struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	MaxTokens   int       `json:"max_tokens"`
	Temperature float64   `json:"temperature"`
	Stream      bool      `json:"stream"`
}

// LlamaChatChunk is an OpenAI-format SSE chunk
type LlamaChatChunk struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
			Role    string `json:"role,omitempty"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage,omitempty"`
}
