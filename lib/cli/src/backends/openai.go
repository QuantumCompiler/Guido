package backends

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"guido/lib/cli/src/harness"
)

const defaultOpenAIBase = "https://api.openai.com"

// OpenAIBackend implements harness.LLMProvider for OpenAI-compatible APIs.
// Set baseURL to a custom endpoint (e.g. "http://localhost:11434") to talk to
// Ollama or any other OpenAI-compatible server.
type OpenAIBackend struct {
	apiKey  string
	model   string
	baseURL string
	client  *http.Client
}

// ── Wire types ────────────────────────────────────────────────────────────────

type openaiRequest struct {
	Model       string           `json:"model"`
	Messages    []openaiMessage  `json:"messages"`
	Temperature float32          `json:"temperature"`
	MaxTokens   int              `json:"max_tokens,omitempty"`
	Stream      bool             `json:"stream"`
	Tools       []harness.Tool   `json:"tools,omitempty"`       // nil → omitted (no tool mode)
	ToolChoice  string           `json:"tool_choice,omitempty"` // "auto" when tools present
}

// openaiMessage is used for both request serialisation and response parsing.
type openaiMessage struct {
	Role       string                 `json:"role"`
	Content    harness.MessageContent `json:"content"`
	ToolCalls  []harness.ToolCall     `json:"tool_calls,omitempty"`
	ToolCallID string                 `json:"tool_call_id,omitempty"`
}

type openaiResponse struct {
	Choices []openaiChoice `json:"choices"`
	Usage   openaiUsage    `json:"usage"`
}

type openaiChoice struct {
	Message      openaiMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type openaiUsage struct {
	CompletionTokens int `json:"completion_tokens"`
	PromptTokens     int `json:"prompt_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ── Constructor ───────────────────────────────────────────────────────────────

// NewOpenAIBackend creates a backend for OpenAI-compatible APIs.
// baseURL selects the server — pass "" to use the OpenAI default.
// For Ollama: baseURL = "http://localhost:11434", apiKey = "ollama".
func NewOpenAIBackend(apiKey, model, baseURL string) *OpenAIBackend {
	if baseURL == "" {
		baseURL = defaultOpenAIBase
	}
	return &OpenAIBackend{
		apiKey:  apiKey,
		model:   model,
		baseURL: strings.TrimRight(baseURL, "/"),
		client:  &http.Client{Timeout: 0},
	}
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// toWireMessages converts harness messages (which carry ToolCalls and
// ToolCallID) into the openaiMessage slice sent over the wire.
func toWireMessages(msgs []harness.ChatMessage) []openaiMessage {
	out := make([]openaiMessage, len(msgs))
	for i, m := range msgs {
		out[i] = openaiMessage{
			Role:       m.Role,
			Content:    m.Content,
			ToolCalls:  m.ToolCalls,
			ToolCallID: m.ToolCallID,
		}
	}
	return out
}

// fromWireMessage converts a response openaiMessage back to a harness.ChatMessage.
func fromWireMessage(m openaiMessage) harness.ChatMessage {
	return harness.ChatMessage{
		Role:      m.Role,
		Content:   m.Content,
		ToolCalls: m.ToolCalls,
	}
}

// post sends a JSON request to path and returns the raw HTTP response.
func (ob *OpenAIBackend) post(ctx context.Context, path string, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, "POST", ob.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if ob.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+ob.apiKey)
	}
	return ob.client.Do(req)
}

// ── LLMProvider ───────────────────────────────────────────────────────────────

// Complete implements harness.LLMProvider (single-turn, non-streaming).
func (ob *OpenAIBackend) Complete(ctx context.Context, req *harness.CompletionRequest) (*harness.CompletionResponse, error) {
	payload, err := json.Marshal(openaiRequest{
		Model:       ob.model,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		Messages:    []openaiMessage{{Role: "user", Content: harness.Text(req.Prompt)}},
		Stream:      false,
	})
	if err != nil {
		return nil, fmt.Errorf("marshal: %w", err)
	}

	resp, err := ob.post(ctx, "/v1/chat/completions", payload)
	if err != nil {
		return nil, fmt.Errorf("request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("openai status %d: %s", resp.StatusCode, b)
	}

	var r openaiResponse
	if err := json.NewDecoder(resp.Body).Decode(&r); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	if len(r.Choices) == 0 {
		return nil, fmt.Errorf("openai returned no choices")
	}

	return &harness.CompletionResponse{
		Text:         r.Choices[0].Message.Content.PlainText(),
		FinishReason: r.Choices[0].FinishReason,
		TokensUsed:   r.Usage.CompletionTokens,
		Model:        ob.model,
	}, nil
}

// StreamTokens implements harness.LLMProvider (single-turn, streaming).
func (ob *OpenAIBackend) StreamTokens(ctx context.Context, req *harness.CompletionRequest) (<-chan string, error) {
	payload, err := json.Marshal(openaiRequest{
		Model:       ob.model,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		Messages:    []openaiMessage{{Role: "user", Content: harness.Text(req.Prompt)}},
		Stream:      true,
	})
	if err != nil {
		return nil, fmt.Errorf("marshal: %w", err)
	}

	resp, err := ob.post(ctx, "/v1/chat/completions", payload)
	if err != nil {
		return nil, fmt.Errorf("request: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("openai status %d: %s", resp.StatusCode, b)
	}

	ch := make(chan string)
	go func() {
		defer resp.Body.Close()
		defer close(ch)
		streamTextTokens(ctx, resp.Body, ch)
	}()
	return ch, nil
}

// Chat implements harness.LLMProvider (multi-turn, non-streaming, with tool support).
func (ob *OpenAIBackend) Chat(ctx context.Context, req *harness.ChatRequest) (*harness.ChatResponse, error) {
	oreq := openaiRequest{
		Model:       ob.model,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		Messages:    toWireMessages(req.Messages),
		Stream:      false,
		Tools:       req.Tools,
	}
	if len(req.Tools) > 0 {
		oreq.ToolChoice = "auto"
	}

	payload, err := json.Marshal(oreq)
	if err != nil {
		return nil, fmt.Errorf("marshal: %w", err)
	}

	resp, err := ob.post(ctx, "/v1/chat/completions", payload)
	if err != nil {
		return nil, fmt.Errorf("request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("openai status %d: %s", resp.StatusCode, b)
	}

	var r openaiResponse
	if err := json.NewDecoder(resp.Body).Decode(&r); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	if len(r.Choices) == 0 {
		return nil, fmt.Errorf("openai returned no choices")
	}

	return &harness.ChatResponse{
		Message:      fromWireMessage(r.Choices[0].Message),
		FinishReason: r.Choices[0].FinishReason,
		TokensUsed:   r.Usage.CompletionTokens,
		Model:        ob.model,
	}, nil
}

// StreamChat implements harness.LLMProvider (multi-turn, streaming, text only).
// Tool calls are not supported in streaming mode — use Chat when tools are active.
func (ob *OpenAIBackend) StreamChat(ctx context.Context, req *harness.ChatRequest) (<-chan string, error) {
	payload, err := json.Marshal(openaiRequest{
		Model:       ob.model,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		Messages:    toWireMessages(req.Messages),
		Stream:      true,
		// Tools intentionally omitted: the agentic loop uses non-streaming Chat.
	})
	if err != nil {
		return nil, fmt.Errorf("marshal: %w", err)
	}

	resp, err := ob.post(ctx, "/v1/chat/completions", payload)
	if err != nil {
		return nil, fmt.Errorf("request: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("openai status %d: %s", resp.StatusCode, b)
	}

	ch := make(chan string)
	go func() {
		defer resp.Body.Close()
		defer close(ch)
		streamTextTokens(ctx, resp.Body, ch)
	}()
	return ch, nil
}

// ListModels implements harness.LLMProvider.
func (ob *OpenAIBackend) ListModels(_ context.Context) ([]harness.ModelInfo, error) {
	return []harness.ModelInfo{{
		ID:       ob.model,
		Name:     ob.model,
		Provider: "openai",
		Type:     "chat",
	}}, nil
}

// ── SSE streaming helper ──────────────────────────────────────────────────────

// streamTextTokens reads an OpenAI-style SSE stream and sends each text
// delta to ch. Stops on [DONE], context cancellation, or read error.
func streamTextTokens(ctx context.Context, body io.Reader, ch chan<- string) {
	scanner := bufio.NewScanner(body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			return
		}

		var chunk sseChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		if content := chunk.Choices[0].Delta.Content; content != "" {
			select {
			case ch <- content:
			case <-ctx.Done():
				return
			}
		}
	}
}
