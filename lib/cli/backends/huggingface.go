package backends

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"guido/lib/cli/harness"
)

// HuggingFaceBackend implements harness.LLMProvider using Python's transformers library
// It runs inference via Python subprocess to leverage the transformers ecosystem
type HuggingFaceBackend struct {
	model    string
	cacheDir string
}

// NewHuggingFaceBackend creates a new HuggingFace backend
// model: HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b-hf")
// cacheDir: optional cache directory for HF models (uses ~/.cache/huggingface if empty)
func NewHuggingFaceBackend(model, cacheDir string) *HuggingFaceBackend {
	if cacheDir == "" {
		homeDir, _ := os.UserHomeDir()
		cacheDir = fmt.Sprintf("%s/.cache/huggingface", homeDir)
	}
	return &HuggingFaceBackend{
		model:    model,
		cacheDir: cacheDir,
	}
}

// Complete implements harness.LLMProvider
func (hb *HuggingFaceBackend) Complete(ctx context.Context, req *harness.CompletionRequest) (*harness.CompletionResponse, error) {
	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = 256
	}

	pythonCode := hb.buildCompletionScript(req.Prompt, maxTokens, req.Temperature)

	cmd := exec.CommandContext(ctx, "python3", "-c", pythonCode)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		if stderr.Len() > 0 {
			return nil, fmt.Errorf("huggingface error: %s", stderr.String())
		}
		return nil, fmt.Errorf("failed to run inference: %w", err)
	}

	var resp harness.CompletionResponse
	if err := json.Unmarshal(stdout.Bytes(), &resp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &resp, nil
}

// StreamTokens implements harness.LLMProvider
// Streams tokens by running inference with streaming output
func (hb *HuggingFaceBackend) StreamTokens(ctx context.Context, req *harness.CompletionRequest) (<-chan string, error) {
	tokenChan := make(chan string)
	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = 256
	}

	go func() {
		defer close(tokenChan)

		pythonCode := hb.buildStreamScript(req.Prompt, maxTokens, req.Temperature)

		cmd := exec.CommandContext(ctx, "python3", "-c", pythonCode)
		stdout, err := cmd.StdoutPipe()
		if err != nil {
			return
		}

		if err := cmd.Start(); err != nil {
			return
		}

		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			line := scanner.Text()
			var data map[string]interface{}
			if err := json.Unmarshal([]byte(line), &data); err != nil {
				continue
			}

			if token, ok := data["token"].(string); ok {
				select {
				case <-ctx.Done():
					cmd.Process.Kill()
					return
				case tokenChan <- token:
				}
			}
		}
		cmd.Wait()
	}()

	return tokenChan, nil
}

// ListModels implements harness.LLMProvider
func (hb *HuggingFaceBackend) ListModels(ctx context.Context) ([]harness.ModelInfo, error) {
	return []harness.ModelInfo{
		{
			ID:       hb.model,
			Name:     hb.model,
			Provider: "huggingface",
			Type:     "text-generation",
		},
	}, nil
}

// buildCompletionScript generates Python code for non-streaming completion
func (hb *HuggingFaceBackend) buildCompletionScript(prompt string, maxTokens int, temperature float32) string {
	escapedPrompt := strings.NewReplacer(
		`\`, `\\`,
		`"`, `\"`,
		"\n", `\n`,
		"\r", `\r`,
		"\t", `\t`,
	).Replace(prompt)

	return fmt.Sprintf(`
import os
import json
os.environ['HF_HOME'] = '%s'

try:
    from transformers import pipeline

    pipe = pipeline("text-generation", model="%s", device_map="auto", trust_remote_code=True)
    result = pipe("%s", max_new_tokens=%d, temperature=%.2f, do_sample=True)

    generated = result[0]['generated_text']
    if generated.startswith("%s"):
        generated = generated[%d:].strip()

    print(json.dumps({
        "text": generated,
        "finish_reason": "length",
        "tokens_used": %d,
        "model": "%s"
    }))
except Exception as e:
    print(json.dumps({"error": str(e)}), file=__import__('sys').stderr)
    exit(1)
`, hb.cacheDir, hb.model, escapedPrompt, maxTokens, temperature, escapedPrompt, len(prompt), maxTokens/4, hb.model)
}

// buildStreamScript generates Python code for streaming completion
func (hb *HuggingFaceBackend) buildStreamScript(prompt string, maxTokens int, temperature float32) string {
	escapedPrompt := strings.NewReplacer(
		`\`, `\\`,
		`"`, `\"`,
		"\n", `\n`,
		"\r", `\r`,
		"\t", `\t`,
	).Replace(prompt)

	return fmt.Sprintf(`
import os
import json
os.environ['HF_HOME'] = '%s'

try:
    from transformers import TextIteratorStreamer, pipeline
    from threading import Thread

    pipe = pipeline("text-generation", model="%s", device_map="auto", trust_remote_code=True)
    streamer = TextIteratorStreamer(pipe.tokenizer)

    kwargs = dict(
        text_inputs="%s",
        max_new_tokens=%d,
        temperature=%.2f,
        do_sample=True,
        streamer=streamer,
    )

    thread = Thread(target=pipe, kwargs=kwargs)
    thread.start()

    for text in streamer:
        if text.strip():
            print(json.dumps({"token": text}))

    thread.join()
except Exception as e:
    print(json.dumps({"error": str(e)}), file=__import__('sys').stderr)
    exit(1)
`, hb.cacheDir, hb.model, escapedPrompt, maxTokens, temperature)
}
