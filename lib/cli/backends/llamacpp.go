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

// LlamaCppBackend implements harness.LLMProvider for llama.cpp HTTP server
type LlamaCppBackend struct {
	baseURL string
	model   string
	client  *http.Client
}

// llamaCppCompletionRequest is the request format for llama.cpp API
type llamaCppCompletionRequest struct {
	Prompt      string   `json:"prompt"`
	Temperature float32  `json:"temperature"`
	NPredict    int      `json:"n_predict"`
	Stream      bool     `json:"stream"`
	Stop        []string `json:"stop,omitempty"`
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
		Stop:        defaultStopSequences,
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
			Stop:        defaultStopSequences,
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

// llamaCppChatRequest is the OpenAI-compatible chat request format for llama-server
type llamaCppChatRequest struct {
	Messages    []harness.ChatMessage `json:"messages"`
	Temperature float32               `json:"temperature"`
	MaxTokens   int                   `json:"max_tokens"`
	Stream      bool                  `json:"stream"`
	// Stop sequences cover the most common chat-template end-of-turn tokens:
	//   <|im_end|>      ChatML (most instruction-tuned models)
	//   <|eot_id|>      Llama 3
	//   <end_of_turn>   Gemma native template
	//   <|end|>         Phi-3
	//   </s>            Legacy SentencePiece models
	Stop []string `json:"stop,omitempty"`
}

// defaultStopSequences lists end-of-turn tokens that cover the most common chat
// templates used by instruction-tuned GGUF models. llama-server will stop
// generation when any of these tokens is produced.
var defaultStopSequences = []string{
	"<|im_end|>",    // ChatML — Mistral-instruct, Qwen, many others
	"<|eot_id|>",   // Llama 3
	"<end_of_turn>", // Gemma native
	"<|end|>",       // Phi-3
	"</s>",          // Legacy SentencePiece (Llama 1/2, Mistral v0.1)
	"\nUser:",        // Q&A fallback format
	"\nHuman:",       // Some RLHF formats
}

// Chat implements harness.LLMProvider — non-streaming multi-turn chat
func (lcb *LlamaCppBackend) Chat(ctx context.Context, req *harness.ChatRequest) (*harness.ChatResponse, error) {
	chatReq := llamaCppChatRequest{
		Messages:    req.Messages,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		Stream:      false,
		Stop:        defaultStopSequences,
	}

	body, err := json.Marshal(chatReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal chat request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST",
		fmt.Sprintf("%s/v1/chat/completions", lcb.baseURL), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create chat request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := lcb.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("chat request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("llama.cpp returned status %d: %s", resp.StatusCode, b)
	}

	var result struct {
		Choices []struct {
			Message      harness.ChatMessage `json:"message"`
			FinishReason string              `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode chat response: %w", err)
	}
	if len(result.Choices) == 0 {
		return nil, fmt.Errorf("llama.cpp returned no choices")
	}

	return &harness.ChatResponse{
		Message:      result.Choices[0].Message,
		FinishReason: result.Choices[0].FinishReason,
		TokensUsed:   result.Usage.CompletionTokens,
		Model:        lcb.model,
	}, nil
}

// StreamChat implements harness.LLMProvider — streaming multi-turn chat via SSE
func (lcb *LlamaCppBackend) StreamChat(ctx context.Context, req *harness.ChatRequest) (<-chan string, error) {
	tokenChan := make(chan string)

	go func() {
		defer close(tokenChan)

		chatReq := llamaCppChatRequest{
			Messages:    req.Messages,
			Temperature: req.Temperature,
			MaxTokens:   req.MaxTokens,
			Stream:      true,
			Stop:        defaultStopSequences,
		}

		body, err := json.Marshal(chatReq)
		if err != nil {
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, "POST",
			fmt.Sprintf("%s/v1/chat/completions", lcb.baseURL), bytes.NewReader(body))
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

		// Parse SSE stream: each line is "data: <json>" or "data: [DONE]"
		scanner := bufio.NewScanner(resp.Body)
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

			if len(chunk.Choices) > 0 {
				if content := chunk.Choices[0].Delta.Content; content != "" {
					select {
					case tokenChan <- content:
					case <-ctx.Done():
						return
					}
				}
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
