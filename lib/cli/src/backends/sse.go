package backends

// sseChunk is the shared struct for OpenAI-compatible SSE chat completion chunks.
// Both llama-server and OpenAI return data in this format.
//
//	data: {"choices":[{"delta":{"content":"token"},"finish_reason":null}]}
//	data: [DONE]
type sseChunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
}
