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

	"guido/lib/cli/harness"
)

// AnthropicBackend implements harness.LLMProvider for Anthropic API via HTTP
type AnthropicBackend struct {
	apiKey string
	model  string
	client *http.Client
}

// anthropicRequest is the request format for Anthropic API
type anthropicRequest struct {
	Model       string                    `json:"model"`
	MaxTokens   int                       `json:"max_tokens"`
	Temperature float32                   `json:"temperature"`
	Messages    []anthropicMessage        `json:"messages"`
	Stream      bool                      `json:"stream"`
}

type anthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// anthropicResponse is the response format from Anthropic API
type anthropicResponse struct {
	Content []anthropicContent `json:"content"`
	Usage   anthropicUsage     `json:"usage"`
}

type anthropicContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// NewAnthropicBackend creates a new Anthropic backend
func NewAnthropicBackend(apiKey, model string) *AnthropicBackend {
	return &AnthropicBackend{
		apiKey: apiKey,
		model:  model,
		client: &http.Client{
			Timeout: 0, // No timeout for long-running completions
		},
	}
}

// Complete implements harness.LLMProvider
func (ab *AnthropicBackend) Complete(ctx context.Context, req *harness.CompletionRequest) (*harness.CompletionResponse, error) {
	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = 1024
	}

	anthropicReq := anthropicRequest{
		Model:       ab.model,
		MaxTokens:   maxTokens,
		Temperature: req.Temperature,
		Messages: []anthropicMessage{
			{
				Role:    "user",
				Content: req.Prompt,
			},
		},
		Stream: false,
	}

	body, err := json.Marshal(anthropicReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", ab.apiKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

	resp, err := ab.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("anthropic returned status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var anthropicResp anthropicResponse
	if err := json.NewDecoder(resp.Body).Decode(&anthropicResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	text := ""
	if len(anthropicResp.Content) > 0 {
		text = anthropicResp.Content[0].Text
	}

	return &harness.CompletionResponse{
		Text:       text,
		TokensUsed: anthropicResp.Usage.OutputTokens,
		Model:      ab.model,
	}, nil
}

// StreamTokens implements harness.LLMProvider
func (ab *AnthropicBackend) StreamTokens(ctx context.Context, req *harness.CompletionRequest) (<-chan string, error) {
	tokenChan := make(chan string)

	go func() {
		defer close(tokenChan)

		maxTokens := req.MaxTokens
		if maxTokens == 0 {
			maxTokens = 1024
		}

		anthropicReq := anthropicRequest{
			Model:       ab.model,
			MaxTokens:   maxTokens,
			Temperature: req.Temperature,
			Messages: []anthropicMessage{
				{
					Role:    "user",
					Content: req.Prompt,
				},
			},
			Stream: true,
		}

		body, err := json.Marshal(anthropicReq)
		if err != nil {
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, "POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(body))
		if err != nil {
			return
		}

		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("x-api-key", ab.apiKey)
		httpReq.Header.Set("anthropic-version", "2023-06-01")

		resp, err := ab.client.Do(httpReq)
		if err != nil {
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			return
		}

		decoder := json.NewDecoder(resp.Body)
		for {
			var streamEvent map[string]interface{}
			if err := decoder.Decode(&streamEvent); err != nil {
				if err == io.EOF {
					break
				}
				return
			}

			// Extract content from content_block_delta events
			if eventType, ok := streamEvent["type"].(string); ok {
				if eventType == "content_block_delta" {
					if deltaObj, ok := streamEvent["delta"].(map[string]interface{}); ok {
						if text, ok := deltaObj["text"].(string); ok && text != "" {
							select {
							case tokenChan <- text:
							case <-ctx.Done():
								return
							}
						}
					}
				}
			}
		}
	}()

	return tokenChan, nil
}

// anthropicChatRequest is the Anthropic messages API format
type anthropicChatRequest struct {
	Model       string             `json:"model"`
	MaxTokens   int                `json:"max_tokens"`
	Temperature float32            `json:"temperature"`
	System      string             `json:"system,omitempty"`
	Messages    []anthropicMessage `json:"messages"`
	Stream      bool               `json:"stream"`
}

// Chat implements harness.LLMProvider — non-streaming multi-turn chat
func (ab *AnthropicBackend) Chat(ctx context.Context, req *harness.ChatRequest) (*harness.ChatResponse, error) {
	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = 1024
	}

	chatReq, systemPrompt := buildAnthropicChatRequest(ab.model, req, maxTokens, false)

	body, err := json.Marshal(chatReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	_ = systemPrompt

	httpReq, err := http.NewRequestWithContext(ctx, "POST",
		"https://api.anthropic.com/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", ab.apiKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

	resp, err := ab.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("anthropic returned status %d: %s", resp.StatusCode, b)
	}

	var anthropicResp anthropicResponse
	if err := json.NewDecoder(resp.Body).Decode(&anthropicResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	text := ""
	if len(anthropicResp.Content) > 0 {
		text = anthropicResp.Content[0].Text
	}

	return &harness.ChatResponse{
		Message:      harness.ChatMessage{Role: "assistant", Content: text},
		TokensUsed:   anthropicResp.Usage.OutputTokens,
		Model:        ab.model,
	}, nil
}

// StreamChat implements harness.LLMProvider — streaming multi-turn chat via SSE
func (ab *AnthropicBackend) StreamChat(ctx context.Context, req *harness.ChatRequest) (<-chan string, error) {
	tokenChan := make(chan string)

	go func() {
		defer close(tokenChan)

		maxTokens := req.MaxTokens
		if maxTokens == 0 {
			maxTokens = 1024
		}

		chatReq, _ := buildAnthropicChatRequest(ab.model, req, maxTokens, true)

		body, err := json.Marshal(chatReq)
		if err != nil {
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, "POST",
			"https://api.anthropic.com/v1/messages", bytes.NewReader(body))
		if err != nil {
			return
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("x-api-key", ab.apiKey)
		httpReq.Header.Set("anthropic-version", "2023-06-01")

		resp, err := ab.client.Do(httpReq)
		if err != nil {
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			return
		}

		// Anthropic SSE: each line is "data: <json>" with type field
		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			payload := strings.TrimPrefix(line, "data: ")

			var event struct {
				Type  string `json:"type"`
				Delta struct {
					Type string `json:"type"`
					Text string `json:"text"`
				} `json:"delta"`
			}
			if err := json.Unmarshal([]byte(payload), &event); err != nil {
				continue
			}

			if event.Type == "content_block_delta" && event.Delta.Text != "" {
				select {
				case tokenChan <- event.Delta.Text:
				case <-ctx.Done():
					return
				}
			}
		}
	}()

	return tokenChan, nil
}

// buildAnthropicChatRequest converts a ChatRequest into the Anthropic API format.
// Anthropic separates the system prompt from conversation messages.
func buildAnthropicChatRequest(model string, req *harness.ChatRequest, maxTokens int, stream bool) (anthropicChatRequest, string) {
	var systemPrompt string
	var msgs []anthropicMessage

	for _, m := range req.Messages {
		if m.Role == "system" {
			systemPrompt = m.Content
			continue
		}
		msgs = append(msgs, anthropicMessage{Role: m.Role, Content: m.Content})
	}

	return anthropicChatRequest{
		Model:       model,
		MaxTokens:   maxTokens,
		Temperature: req.Temperature,
		System:      systemPrompt,
		Messages:    msgs,
		Stream:      stream,
	}, systemPrompt
}

// ListModels implements harness.LLMProvider
func (ab *AnthropicBackend) ListModels(ctx context.Context) ([]harness.ModelInfo, error) {
	return []harness.ModelInfo{
		{
			ID:       ab.model,
			Name:     ab.model,
			Provider: "anthropic",
			Type:     "chat",
		},
	}, nil
}
