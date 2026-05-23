package backends

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"guido/lib/cli/harness"
)

// OpenAIBackend implements harness.LLMProvider for OpenAI API via HTTP
type OpenAIBackend struct {
	apiKey string
	model  string
	client *http.Client
}

// openaiRequest is the request format for OpenAI API
type openaiRequest struct {
	Model       string           `json:"model"`
	MaxTokens   int              `json:"max_tokens"`
	Temperature float32          `json:"temperature"`
	Messages    []openaiMessage  `json:"messages"`
	Stream      bool             `json:"stream"`
}

type openaiMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// openaiResponse is the response format from OpenAI API
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

// NewOpenAIBackend creates a new OpenAI backend
func NewOpenAIBackend(apiKey, model string) *OpenAIBackend {
	return &OpenAIBackend{
		apiKey: apiKey,
		model:  model,
		client: &http.Client{
			Timeout: 0, // No timeout for long-running completions
		},
	}
}

// Complete implements harness.LLMProvider
func (ob *OpenAIBackend) Complete(ctx context.Context, req *harness.CompletionRequest) (*harness.CompletionResponse, error) {
	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = 256
	}

	openaiReq := openaiRequest{
		Model:       ob.model,
		MaxTokens:   maxTokens,
		Temperature: req.Temperature,
		Messages: []openaiMessage{
			{
				Role:    "user",
				Content: req.Prompt,
			},
		},
		Stream: false,
	}

	body, err := json.Marshal(openaiReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", "https://api.openai.com/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+ob.apiKey)

	resp, err := ob.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("openai returned status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var openaiResp openaiResponse
	if err := json.NewDecoder(resp.Body).Decode(&openaiResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(openaiResp.Choices) == 0 {
		return nil, fmt.Errorf("openai returned no choices")
	}

	choice := openaiResp.Choices[0]

	return &harness.CompletionResponse{
		Text:         choice.Message.Content,
		FinishReason: choice.FinishReason,
		TokensUsed:   openaiResp.Usage.CompletionTokens,
		Model:        ob.model,
	}, nil
}

// StreamTokens implements harness.LLMProvider
func (ob *OpenAIBackend) StreamTokens(ctx context.Context, req *harness.CompletionRequest) (<-chan string, error) {
	tokenChan := make(chan string)

	go func() {
		defer close(tokenChan)

		maxTokens := req.MaxTokens
		if maxTokens == 0 {
			maxTokens = 256
		}

		openaiReq := openaiRequest{
			Model:       ob.model,
			MaxTokens:   maxTokens,
			Temperature: req.Temperature,
			Messages: []openaiMessage{
				{
					Role:    "user",
					Content: req.Prompt,
				},
			},
			Stream: true,
		}

		body, err := json.Marshal(openaiReq)
		if err != nil {
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, "POST", "https://api.openai.com/v1/chat/completions", bytes.NewReader(body))
		if err != nil {
			return
		}

		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Authorization", "Bearer "+ob.apiKey)

		resp, err := ob.client.Do(httpReq)
		if err != nil {
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			return
		}

		// Read streaming response (Server-Sent Events format)
		decoder := json.NewDecoder(resp.Body)
		for {
			var streamEvent map[string]interface{}
			if err := decoder.Decode(&streamEvent); err != nil {
				if err == io.EOF {
					break
				}
				return
			}

			// Parse SSE format data line
			// Each line starts with "data: " followed by JSON
			// Extract choices and delta.content
			if choices, ok := streamEvent["choices"].([]interface{}); ok && len(choices) > 0 {
				if choice, ok := choices[0].(map[string]interface{}); ok {
					if delta, ok := choice["delta"].(map[string]interface{}); ok {
						if content, ok := delta["content"].(string); ok && content != "" {
							select {
							case tokenChan <- content:
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

// ListModels implements harness.LLMProvider
func (ob *OpenAIBackend) ListModels(ctx context.Context) ([]harness.ModelInfo, error) {
	return []harness.ModelInfo{
		{
			ID:       ob.model,
			Name:     ob.model,
			Provider: "openai",
			Type:     "chat",
		},
	}, nil
}
