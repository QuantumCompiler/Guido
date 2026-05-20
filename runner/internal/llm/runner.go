package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Types (mirrored here so llm package is self-contained)
// ---------------------------------------------------------------------------

type CompletionReq struct {
	Prompt      string   `json:"prompt"`
	NPredict    int      `json:"n_predict"`
	Temperature float64  `json:"temperature"`
	TopP        float64  `json:"top_p"`
	TopK        int      `json:"top_k"`
	Stream      bool     `json:"stream"`
	Stop        []string `json:"stop,omitempty"`
	CachePrompt bool     `json:"cache_prompt"`
}

type CompletionChunk struct {
	Content         string  `json:"content"`
	Stop            bool    `json:"stop"`
	TokensEvaluated int     `json:"tokens_evaluated"`
	TokensPredicted int     `json:"tokens_predicted"`
	GenerationMs    float64 // populated after the stream ends
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatReq struct {
	Messages    []ChatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens"`
	Temperature float64       `json:"temperature"`
	Stream      bool          `json:"stream"`
}

type ChatChunk struct {
	Content      string
	Done         bool
	PromptTokens int
	CompTokens   int
}

// ---------------------------------------------------------------------------
// Runner — manages one llama-server child process
// ---------------------------------------------------------------------------

// Runner owns a single llama-server process.
// Only one model can be loaded at a time; swapping the model kills and
// restarts the subprocess (same as the simplest Ollama behaviour).
type Runner struct {
	mu           sync.Mutex
	llamaBin     string
	verbose      bool
	cmd          *exec.Cmd
	port         int
	baseURL      string
	currentModel string // absolute path of the currently loaded model
}

func NewRunner(llamaBin string, verbose bool) *Runner {
	return &Runner{llamaBin: llamaBin, verbose: verbose}
}

// EnsureModel loads modelPath if it isn't already running.
// Thread-safe; callers can call this before every request.
func (r *Runner) EnsureModel(modelPath string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.currentModel == modelPath && r.isAlive() {
		return nil
	}

	return r.start(modelPath)
}

// CurrentModel returns the path of the currently loaded model (or "").
func (r *Runner) CurrentModel() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.currentModel
}

// Shutdown kills the llama-server process cleanly.
func (r *Runner) Shutdown() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.stop()
}

// ---------------------------------------------------------------------------
// Internal process lifecycle
// ---------------------------------------------------------------------------

func (r *Runner) isAlive() bool {
	if r.cmd == nil || r.cmd.Process == nil {
		return false
	}
	// Process.Signal(0) returns nil if the process is still alive.
	return r.cmd.Process.Signal(os.Signal(nil)) == nil
}

func (r *Runner) stop() {
	if r.cmd != nil && r.cmd.Process != nil {
		_ = r.cmd.Process.Kill()
		_ = r.cmd.Wait()
	}
	r.cmd = nil
	r.currentModel = ""
}

func (r *Runner) start(modelPath string) error {
	r.stop() // kill any previous process

	// Pick a random high port to avoid conflicts
	r.port = 20000 + rand.Intn(10000)
	r.baseURL = fmt.Sprintf("http://127.0.0.1:%d", r.port)

	args := []string{
		"--model", modelPath,
		"--port", strconv.Itoa(r.port),
		"--host", "127.0.0.1",
		"--ctx-size", "4096",
		"--n-gpu-layers", "99",  // use all GPU layers if available; falls back to CPU
		"--parallel", "1",       // one request at a time (simplest)
		"--no-mmap",             // safer across platforms
	}

	log.Printf("[llm] starting: %s %s", r.llamaBin, strings.Join(args, " "))

	r.cmd = exec.Command(r.llamaBin, args...)

	if r.verbose {
		r.cmd.Stdout = os.Stdout
		r.cmd.Stderr = os.Stderr
	} else {
		r.cmd.Stdout = io.Discard
		r.cmd.Stderr = io.Discard
	}

	if err := r.cmd.Start(); err != nil {
		return fmt.Errorf("could not start llama-server: %w\n"+
			"  Make sure llama-server is installed and on your PATH.\n"+
			"  Build from: https://github.com/ggml-org/llama.cpp", err)
	}

	r.currentModel = modelPath

	log.Printf("[llm] waiting for llama-server on port %d ...", r.port)
	if err := r.waitReady(45 * time.Second); err != nil {
		r.stop()
		return err
	}

	log.Printf("[llm] model ready: %s", modelPath)
	return nil
}

func (r *Runner) waitReady(timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	client := &http.Client{Timeout: 2 * time.Second}

	for time.Now().Before(deadline) {
		resp, err := client.Get(r.baseURL + "/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		// Also bail early if the process already died
		if !r.isAlive() {
			return fmt.Errorf("llama-server exited prematurely")
		}
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("llama-server did not become ready within %s", timeout)
}

// ---------------------------------------------------------------------------
// Inference — text completion
// ---------------------------------------------------------------------------

// Complete runs a /completion request against the running llama-server.
// Returns a channel that yields successive chunks; the channel is closed
// when generation is complete or the context is cancelled.
func (r *Runner) Complete(ctx context.Context, req CompletionReq) (<-chan CompletionChunk, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		r.baseURL+"/completion", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := (&http.Client{}).Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("llama-server request failed: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("llama-server returned HTTP %d", resp.StatusCode)
	}

	ch := make(chan CompletionChunk, 16)

	go func() {
		defer close(ch)
		defer resp.Body.Close()

		if !req.Stream {
			// Non-streaming: one JSON blob
			var chunk CompletionChunk
			if err := json.NewDecoder(resp.Body).Decode(&chunk); err != nil {
				log.Printf("[llm] decode error: %v", err)
				return
			}
			chunk.Stop = true
			ch <- chunk
			return
		}

		// Streaming: SSE  →  data: {...}\n\n
		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 64*1024), 64*1024)

		for scanner.Scan() {
			line := scanner.Text()
			data, ok := strings.CutPrefix(line, "data: ")
			if !ok {
				continue
			}
			if data == "[DONE]" {
				break
			}
			var chunk CompletionChunk
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				log.Printf("[llm] chunk parse error: %v (data=%s)", err, data)
				continue
			}
			ch <- chunk
			if chunk.Stop {
				break
			}
		}
		if err := scanner.Err(); err != nil && err != context.Canceled {
			log.Printf("[llm] scanner error: %v", err)
		}
	}()

	return ch, nil
}

// ---------------------------------------------------------------------------
// Inference — chat (OpenAI-compatible endpoint)
// ---------------------------------------------------------------------------

// Chat runs a /v1/chat/completions request and streams ChatChunks.
func (r *Runner) Chat(ctx context.Context, req ChatReq) (<-chan ChatChunk, error) {
	// llama-server uses the OpenAI format here
	payload := map[string]interface{}{
		"messages":    req.Messages,
		"max_tokens":  req.MaxTokens,
		"temperature": req.Temperature,
		"stream":      req.Stream,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		r.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := (&http.Client{}).Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("llama-server chat request failed: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("llama-server returned HTTP %d", resp.StatusCode)
	}

	ch := make(chan ChatChunk, 16)

	go func() {
		defer close(ch)
		defer resp.Body.Close()

		if !req.Stream {
			// Full OpenAI response object
			var full struct {
				Choices []struct {
					Message struct {
						Content string `json:"content"`
					} `json:"message"`
				} `json:"choices"`
				Usage struct {
					PromptTokens     int `json:"prompt_tokens"`
					CompletionTokens int `json:"completion_tokens"`
				} `json:"usage"`
			}
			if err := json.NewDecoder(resp.Body).Decode(&full); err != nil {
				log.Printf("[llm] chat decode error: %v", err)
				return
			}
			content := ""
			if len(full.Choices) > 0 {
				content = full.Choices[0].Message.Content
			}
			ch <- ChatChunk{
				Content:      content,
				Done:         true,
				PromptTokens: full.Usage.PromptTokens,
				CompTokens:   full.Usage.CompletionTokens,
			}
			return
		}

		// SSE streaming
		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 64*1024), 64*1024)

		for scanner.Scan() {
			line := scanner.Text()
			data, ok := strings.CutPrefix(line, "data: ")
			if !ok {
				continue
			}
			if data == "[DONE]" {
				ch <- ChatChunk{Done: true}
				break
			}

			var chunk struct {
				Choices []struct {
					Delta struct {
						Content string `json:"content"`
					} `json:"delta"`
					FinishReason *string `json:"finish_reason"`
				} `json:"choices"`
			}
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				continue
			}
			if len(chunk.Choices) == 0 {
				continue
			}
			delta := chunk.Choices[0].Delta.Content
			done := chunk.Choices[0].FinishReason != nil
			ch <- ChatChunk{Content: delta, Done: done}
			if done {
				break
			}
		}
	}()

	return ch, nil
}
