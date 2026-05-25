package backends

import (
	"context"
	"fmt"
	"strings"
	"time"

	"guido/lib/cli/harness"
)

// MockBackend is a test backend that returns canned responses
// Useful for testing the harness without external dependencies
type MockBackend struct {
	model      string
	response   string
	tokenDelay time.Duration
}

// NewMockBackend creates a new mock backend
func NewMockBackend(model string) *MockBackend {
	return &MockBackend{
		model:      model,
		response:   "This is a mock response from the test backend. Perfect for testing without external API calls or model loading.",
		tokenDelay: 10 * time.Millisecond,
	}
}

// SetResponse allows customizing the response for specific tests
func (mb *MockBackend) SetResponse(response string) {
	mb.response = response
}

// SetTokenDelay allows customizing the delay between tokens for streaming tests
func (mb *MockBackend) SetTokenDelay(delay time.Duration) {
	mb.tokenDelay = delay
}

// Complete implements harness.LLMProvider
func (mb *MockBackend) Complete(ctx context.Context, req *harness.CompletionRequest) (*harness.CompletionResponse, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	return &harness.CompletionResponse{
		Text:         mb.response,
		FinishReason: "stop",
		TokensUsed:   len(strings.Fields(mb.response)),
		Model:        mb.model,
	}, nil
}

// StreamTokens implements harness.LLMProvider
// Streams the response one word at a time with configurable delays
func (mb *MockBackend) StreamTokens(ctx context.Context, req *harness.CompletionRequest) (<-chan string, error) {
	tokenChan := make(chan string)

	go func() {
		defer close(tokenChan)

		words := strings.Fields(mb.response)
		for _, word := range words {
			select {
			case <-ctx.Done():
				return
			case tokenChan <- word + " ":
				time.Sleep(mb.tokenDelay)
			}
		}
	}()

	return tokenChan, nil
}

// Chat implements harness.LLMProvider
func (mb *MockBackend) Chat(ctx context.Context, req *harness.ChatRequest) (*harness.ChatResponse, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Echo the last user message for testing
	lastMsg := ""
	for _, m := range req.Messages {
		if m.Role == "user" {
			lastMsg = m.Content.PlainText()
		}
	}
	text := fmt.Sprintf("[mock] You said: %q — %s", lastMsg, mb.response)

	return &harness.ChatResponse{
		Message: harness.ChatMessage{
			Role:    "assistant",
			Content: harness.Text(text),
		},
		FinishReason: "stop",
		TokensUsed:   len(strings.Fields(text)),
		Model:        mb.model,
	}, nil
}

// StreamChat implements harness.LLMProvider
func (mb *MockBackend) StreamChat(ctx context.Context, req *harness.ChatRequest) (<-chan string, error) {
	tokenChan := make(chan string)

	go func() {
		defer close(tokenChan)

		lastMsg := ""
		for _, m := range req.Messages {
			if m.Role == "user" {
				lastMsg = m.Content.PlainText()
			}
		}
		text := fmt.Sprintf("[mock] You said: %q — %s", lastMsg, mb.response)

		words := strings.Fields(text)
		for _, word := range words {
			select {
			case <-ctx.Done():
				return
			case tokenChan <- word + " ":
				time.Sleep(mb.tokenDelay)
			}
		}
	}()

	return tokenChan, nil
}

// ListModels implements harness.LLMProvider
func (mb *MockBackend) ListModels(ctx context.Context) ([]harness.ModelInfo, error) {
	return []harness.ModelInfo{
		{
			ID:       mb.model,
			Name:     mb.model,
			Provider: "mock",
			Type:     "test",
		},
	}, nil
}
