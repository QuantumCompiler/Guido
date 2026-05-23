package backends

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"guido/lib/cli/harness"
)

// LlamaCppBackend implements harness.LLMProvider for llama.cpp HTTP server
type LlamaCppBackend struct {
	baseURL string
	model   string
	client  *http.Client
}

// llamaCppCompletionRequest is the request format for llama.cpp API
type llamaCppCompletionRequest struct {
	Prompt      string  `json:"prompt"`
	Temperature float32 `json:"temperature"`
	NPredict    int     `json:"n_predict"`
	Stream      bool    `json:"stream"`
}

// llamaCppCompletionResponse is the response format from llama.cpp API
type llamaCppCompletionResponse struct {
	Content    string `json:"content"`
	Stop       bool   `json:"stop"`
	StopReason string `json:"stop_reason"`
	Tokens     int    `json:"tokens_evaluated"`
}

// NewLlamaCppBackend creates a new llama.cpp backend
func NewLlamaCppBackend(baseURL, model string) *LlamaCppBackend {
	return &LlamaCppBackend{
		baseURL: strings.TrimSuffix(baseURL, "/"),
		model:   model,
		client: &http.Client{
			Timeout: 0, // No timeout for long-running completions
		},
	}
}

// Complete implements harness.LLMProvider
func (lcb *LlamaCppBackend) Complete(ctx context.Context, req *harness.CompletionRequest) (*harness.CompletionResponse, error) {
	llmReq := llamaCppCompletionRequest{
		Prompt:      req.Prompt,
		Temperature: req.Temperature,
		NPredict:    req.MaxTokens,
		Stream:      false,
	}

	body, err := json.Marshal(llmReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", fmt.Sprintf("%s/completion", lcb.baseURL), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := lcb.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("llama.cpp returned status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var llmResp llamaCppCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&llmResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &harness.CompletionResponse{
		Text:         llmResp.Content,
		FinishReason: llmResp.StopReason,
		TokensUsed:   llmResp.Tokens,
		Model:        lcb.model,
	}, nil
}

// StreamTokens implements harness.LLMProvider
func (lcb *LlamaCppBackend) StreamTokens(ctx context.Context, req *harness.CompletionRequest) (<-chan string, error) {
	tokenChan := make(chan string)

	go func() {
		defer close(tokenChan)

		llmReq := llamaCppCompletionRequest{
			Prompt:      req.Prompt,
			Temperature: req.Temperature,
			NPredict:    req.MaxTokens,
			Stream:      true,
		}

		body, err := json.Marshal(llmReq)
		if err != nil {
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, "POST", fmt.Sprintf("%s/completion", lcb.baseURL), bytes.NewReader(body))
		if err != nil {
			return
		}

		httpReq.Header.Set("Content-Type", "application/json")

		resp, err := lcb.client.Do(httpReq)
		if err != nil {
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			return
		}

		decoder := json.NewDecoder(resp.Body)
		for {
			var llmResp llamaCppCompletionResponse
			if err := decoder.Decode(&llmResp); err != nil {
				if err == io.EOF {
					break
				}
				return
			}

			if llmResp.Content != "" {
				select {
				case tokenChan <- llmResp.Content:
				case <-ctx.Done():
					return
				}
			}

			if llmResp.Stop {
				break
			}
		}
	}()

	return tokenChan, nil
}

// ListModels implements harness.LLMProvider
func (lcb *LlamaCppBackend) ListModels(ctx context.Context) ([]harness.ModelInfo, error) {
	// llama.cpp doesn't have a models endpoint, so we just return the configured model
	return []harness.ModelInfo{
		{
			ID:       lcb.model,
			Name:     lcb.model,
			Provider: "llamacpp",
			Type:     "completion",
		},
	}, nil
}
