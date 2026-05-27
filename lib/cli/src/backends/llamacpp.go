package backends

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"regexp"
	"strings"

	"guido/lib/cli/src/harness"
)

// LlamaCppBackend implements harness.LLMProvider for llama.cpp HTTP server
type LlamaCppBackend struct {
	baseURL string
	model   string
	client  *http.Client
}

// UsesInTextToolCalls implements harness.InTextToolCaller.
// llamacpp uses system-prompt injection: tool calls appear as TOOL_CALL: lines
// in the text stream, making streaming with lookahead detection possible.
func (lcb *LlamaCppBackend) UsesInTextToolCalls() bool { return true }

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

// Complete implements harness.LLMProvider.
// Wraps the prompt as a user message and delegates to /v1/chat/completions —
// the stable OpenAI-compatible endpoint. The legacy /completion endpoint was
// removed or made unreliable in newer llama-server builds.
func (lcb *LlamaCppBackend) Complete(ctx context.Context, req *harness.CompletionRequest) (*harness.CompletionResponse, error) {
	chatResp, err := lcb.Chat(ctx, &harness.ChatRequest{
		Messages: []harness.ChatMessage{
			{Role: "user", Content: harness.Text(req.Prompt)},
		},
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		Model:       lcb.model,
	})
	if err != nil {
		return nil, err
	}
	return &harness.CompletionResponse{
		Text:         chatResp.Message.Content.PlainText(),
		FinishReason: chatResp.FinishReason,
		TokensUsed:   chatResp.TokensUsed,
		Model:        chatResp.Model,
	}, nil
}

// StreamTokens implements harness.LLMProvider.
// Wraps the prompt as a user message and delegates to StreamChat so both
// streaming paths share the same /v1/chat/completions SSE logic.
func (lcb *LlamaCppBackend) StreamTokens(ctx context.Context, req *harness.CompletionRequest) (<-chan string, error) {
	return lcb.StreamChat(ctx, &harness.ChatRequest{
		Messages: []harness.ChatMessage{
			{Role: "user", Content: harness.Text(req.Prompt)},
		},
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		Model:       lcb.model,
	})
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
	// Tools is included for future use when llama-server's native tool-call
	// serialization is stable. Currently Chat() uses system-prompt injection
	// (chatWithToolPrompt) instead of this field.
	Tools []harness.Tool `json:"tools,omitempty"`
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

// Chat implements harness.LLMProvider — non-streaming multi-turn chat.
// When req.Tools is non-empty, tool calling is handled via system-prompt
// injection rather than the native API (llama-server has a serialization bug
// with tool calls in some versions that returns HTTP 500).
func (lcb *LlamaCppBackend) Chat(ctx context.Context, req *harness.ChatRequest) (*harness.ChatResponse, error) {
	if len(req.Tools) > 0 {
		return lcb.chatWithToolPrompt(ctx, req)
	}

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
			Message struct {
				Role    string                 `json:"role"`
				Content harness.MessageContent `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
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
		Message: harness.ChatMessage{
			Role:    result.Choices[0].Message.Role,
			Content: result.Choices[0].Message.Content,
		},
		FinishReason: result.Choices[0].FinishReason,
		PromptTokens: result.Usage.PromptTokens,
		TokensUsed:   result.Usage.CompletionTokens,
		Model:        lcb.model,
	}, nil
}

// ── Tool calling via system-prompt injection ───────────────────────────────────

// toolCallLine matches a TOOL_CALL: line anywhere in the model's response.
// The captured group is the raw JSON object: {"name": "...", "arguments": {...}}
var toolCallLine = regexp.MustCompile(`(?m)^TOOL_CALL:\s*(\{.+\})\s*$`)

// toolSystemPrompt returns a system-message block describing the available tools
// and the TOOL_CALL: output format the model should use.
func toolSystemPrompt(tools []harness.Tool) string {
	var sb strings.Builder
	sb.WriteString("You have access to the following tools. When you need to call a tool, output EXACTLY ONE line in this format and nothing else before or after:\n")
	sb.WriteString("TOOL_CALL: {\"name\": \"<tool_name>\", \"arguments\": <arguments_json_object>}\n\n")
	sb.WriteString("Available tools:\n")
	for _, t := range tools {
		sb.WriteString(fmt.Sprintf("- %s: %s\n", t.Function.Name, t.Function.Description))
		if len(t.Function.Parameters) > 0 && string(t.Function.Parameters) != "null" {
			sb.WriteString(fmt.Sprintf("  Parameters schema: %s\n", string(t.Function.Parameters)))
		}
	}
	sb.WriteString("\nWhen you have enough information to answer without a tool, reply normally.")
	return sb.String()
}

// rewriteMessagesForTools rewrites the message list so llama-server can process
// it without native tool-call support:
//   - role="tool" → role="user" with "Tool result for <name>: <content>"
//   - role="assistant" with tool_calls → plain text with TOOL_CALL: lines
//   - all other messages pass through unchanged
func rewriteMessagesForTools(msgs []harness.ChatMessage) []harness.ChatMessage {
	out := make([]harness.ChatMessage, 0, len(msgs))
	for _, m := range msgs {
		switch m.Role {
		case "tool":
			name := m.Name
			if name == "" {
				name = "tool"
			}
			out = append(out, harness.ChatMessage{
				Role:    "user",
				Content: harness.Text(fmt.Sprintf("Tool result for %s: %s", name, m.Content.PlainText())),
			})
		case "assistant":
			if len(m.ToolCalls) > 0 {
				var sb strings.Builder
				for _, tc := range m.ToolCalls {
					sb.WriteString(fmt.Sprintf("TOOL_CALL: {\"name\": %q, \"arguments\": %s}\n",
						tc.Function.Name, tc.Function.Arguments))
				}
				out = append(out, harness.ChatMessage{
					Role:    "assistant",
					Content: harness.Text(strings.TrimRight(sb.String(), "\n")),
				})
			} else {
				out = append(out, m)
			}
		default:
			out = append(out, m)
		}
	}
	return out
}

// parseToolCalls scans text for TOOL_CALL: lines and returns ToolCall structs.
func parseToolCalls(text string) []harness.ToolCall {
	matches := toolCallLine.FindAllStringSubmatch(text, -1)
	if len(matches) == 0 {
		return nil
	}
	calls := make([]harness.ToolCall, 0, len(matches))
	for i, m := range matches {
		var raw struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal([]byte(m[1]), &raw); err != nil {
			log.Printf("[llamacpp] failed to parse TOOL_CALL JSON: %v — raw: %s", err, m[1])
			continue
		}
		argsStr := string(raw.Arguments)
		if argsStr == "" || argsStr == "null" {
			argsStr = "{}"
		}
		calls = append(calls, harness.ToolCall{
			ID:   fmt.Sprintf("call_%d", i),
			Type: "function",
			Function: harness.ToolCallFunction{
				Name:      raw.Name,
				Arguments: argsStr,
			},
		})
	}
	return calls
}

// chatWithToolPrompt implements tool calling via system-prompt injection.
// It rewrites the message list to avoid native tool-call API fields, injects
// a system prompt describing the available tools, then parses any TOOL_CALL:
// lines from the model's response.
func (lcb *LlamaCppBackend) chatWithToolPrompt(ctx context.Context, req *harness.ChatRequest) (*harness.ChatResponse, error) {
	msgs := rewriteMessagesForTools(req.Messages)

	// Inject the tool-description system prompt.
	prompt := toolSystemPrompt(req.Tools)
	if len(msgs) > 0 && msgs[0].Role == "system" {
		msgs[0].Content = harness.Text(prompt + "\n\n" + msgs[0].Content.PlainText())
	} else {
		msgs = append([]harness.ChatMessage{{Role: "system", Content: harness.Text(prompt)}}, msgs...)
	}

	chatReq := llamaCppChatRequest{
		Messages:    msgs,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		Stream:      false,
		Stop:        defaultStopSequences,
		// Tools intentionally omitted — we're using prompt injection
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
			Message struct {
				Role    string                 `json:"role"`
				Content harness.MessageContent `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode chat response: %w", err)
	}
	if len(result.Choices) == 0 {
		return nil, fmt.Errorf("llama.cpp returned no choices")
	}

	text := result.Choices[0].Message.Content.PlainText()
	if toolCalls := parseToolCalls(text); len(toolCalls) > 0 {
		return &harness.ChatResponse{
			Message: harness.ChatMessage{
				Role:      "assistant",
				Content:   harness.Text(""),
				ToolCalls: toolCalls,
			},
			FinishReason: "tool_calls",
			PromptTokens: result.Usage.PromptTokens,
			TokensUsed:   result.Usage.CompletionTokens,
			Model:        lcb.model,
		}, nil
	}

	return &harness.ChatResponse{
		Message: harness.ChatMessage{
			Role:    result.Choices[0].Message.Role,
			Content: result.Choices[0].Message.Content,
		},
		FinishReason: result.Choices[0].FinishReason,
		PromptTokens: result.Usage.PromptTokens,
		TokensUsed:   result.Usage.CompletionTokens,
		Model:        lcb.model,
	}, nil
}

// streamChatWithToolPrompt is the streaming counterpart of chatWithToolPrompt.
// It rewrites messages for system-prompt injection (same as the non-streaming
// version) but calls llama-server's SSE API so tokens arrive as they are
// generated. The raw token stream is returned — TOOL_CALL: lines, if any,
// flow through as ordinary text so the caller can apply lookahead detection.
func (lcb *LlamaCppBackend) streamChatWithToolPrompt(ctx context.Context, req *harness.ChatRequest) (<-chan string, error) {
	msgs := rewriteMessagesForTools(req.Messages)

	prompt := toolSystemPrompt(req.Tools)
	if len(msgs) > 0 && msgs[0].Role == "system" {
		msgs[0].Content = harness.Text(prompt + "\n\n" + msgs[0].Content.PlainText())
	} else {
		msgs = append([]harness.ChatMessage{{Role: "system", Content: harness.Text(prompt)}}, msgs...)
	}

	chatReq := llamaCppChatRequest{
		Messages:    msgs,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		Stream:      true,
		Stop:        defaultStopSequences,
		// Tools omitted — using system-prompt injection
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
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("llama.cpp returned status %d: %s", resp.StatusCode, b)
	}

	tokenChan := make(chan string)
	go func() {
		defer close(tokenChan)
		defer resp.Body.Close()

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

// ParseToolCalls scans text for TOOL_CALL: lines and returns the parsed ToolCall
// structs. Exported so the CLI's streaming agentic loop can detect tool calls
// from a lookahead-buffered token stream.
func ParseToolCalls(text string) []harness.ToolCall { return parseToolCalls(text) }

// StreamChat implements harness.LLMProvider — streaming multi-turn chat via SSE.
// When req.Tools is non-empty it delegates to streamChatWithToolPrompt so that
// tool calls are handled via system-prompt injection on the streaming path too.
func (lcb *LlamaCppBackend) StreamChat(ctx context.Context, req *harness.ChatRequest) (<-chan string, error) {
	if len(req.Tools) > 0 {
		return lcb.streamChatWithToolPrompt(ctx, req)
	}

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
			b, _ := io.ReadAll(resp.Body)
			log.Printf("llama.cpp stream error (HTTP %d): %s", resp.StatusCode, b)
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
