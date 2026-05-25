package backends

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"path/filepath"
	"sync"
	"time"

	"guido/lib/cli/src/harness"
	"guido/lib/cli/src/tools"
)

// lazyState tracks the lifecycle of the embedded llama-server process.
type lazyState int

const (
	lazyUnloaded lazyState = iota
	lazyLoading
	lazyReady
	lazyErrored
)

// LazyLlamaCppBackend wraps LlamaCppBackend with on-demand loading and optional
// idle-timeout unloading. It implements harness.LLMProvider and harness.StatusReporter.
//
// State machine:
//
//	unloaded ──► loading ──► ready ──► unloaded (idle timeout)
//	                    └──► errored ──► loading (next request retries)
type LazyLlamaCppBackend struct {
	mu           sync.Mutex
	state        lazyState
	inner        *LlamaCppBackend // nil when not ready
	loadingCh    chan struct{}     // closed when load() finishes (success or failure)
	loadErr      error
	lastActivity time.Time
	idleTimeout  time.Duration // 0 = never unload

	// immutable startup config
	toolMgr      *tools.Manager
	modelPath    string
	mmProjPath   string // optional multimodal projector (vision models)
	baseURL      string
	model        string
	chatTemplate string
	port         int
	gpuLayers    int
}

// NewLazyLlamaCppBackend creates a lazy-loading llama.cpp backend.
// idleTimeout of 0 means "never unload once loaded".
// mmProjPath is optional — set it to enable vision/multimodal support.
func NewLazyLlamaCppBackend(
	tm *tools.Manager,
	modelPath, mmProjPath, baseURL, model, chatTemplate string,
	port, gpuLayers int,
	idleTimeout time.Duration,
) *LazyLlamaCppBackend {
	return &LazyLlamaCppBackend{
		toolMgr:      tm,
		modelPath:    modelPath,
		mmProjPath:   mmProjPath,
		baseURL:      baseURL,
		model:        model,
		chatTemplate: chatTemplate,
		port:         port,
		gpuLayers:    gpuLayers,
		idleTimeout:  idleTimeout,
		state:        lazyUnloaded,
	}
}

// ── StatusReporter ────────────────────────────────────────────────────────────

// ModelStatus implements harness.StatusReporter.
func (lb *LazyLlamaCppBackend) ModelStatus() harness.ModelStatusInfo {
	lb.mu.Lock()
	defer lb.mu.Unlock()

	info := harness.ModelStatusInfo{Model: lb.model}
	switch lb.state {
	case lazyUnloaded, lazyErrored:
		info.Status = harness.ModelStatusUnloaded
	case lazyLoading:
		info.Status = harness.ModelStatusLoading
	case lazyReady:
		info.Status = harness.ModelStatusReady
		if !lb.lastActivity.IsZero() {
			info.IdleSeconds = int64(time.Since(lb.lastActivity).Seconds())
		}
	}
	return info
}

// ── Load / unload ─────────────────────────────────────────────────────────────

// EnsureLoaded blocks until the backend is ready or ctx is cancelled.
// If the backend is already ready it returns immediately (fast path).
func (lb *LazyLlamaCppBackend) EnsureLoaded(ctx context.Context) error {
	lb.mu.Lock()

	switch lb.state {
	case lazyReady:
		lb.lastActivity = time.Now()
		lb.mu.Unlock()
		return nil

	case lazyLoading:
		// Already loading — just wait for it to finish.
		ch := lb.loadingCh
		lb.mu.Unlock()
		select {
		case <-ch:
		case <-ctx.Done():
			return ctx.Err()
		}
		lb.mu.Lock()
		err := lb.loadErr
		if err == nil {
			lb.lastActivity = time.Now()
		}
		lb.mu.Unlock()
		if err != nil {
			return fmt.Errorf("model load failed: %w", err)
		}
		return nil

	default: // lazyUnloaded or lazyErrored — kick off a new load
		lb.state = lazyLoading
		lb.loadingCh = make(chan struct{})
		ch := lb.loadingCh
		lb.mu.Unlock()

		go lb.load(ch)

		select {
		case <-ch:
		case <-ctx.Done():
			return ctx.Err()
		}
		lb.mu.Lock()
		err := lb.loadErr
		if err == nil {
			lb.lastActivity = time.Now()
		}
		lb.mu.Unlock()
		if err != nil {
			return fmt.Errorf("model load failed: %w", err)
		}
		return nil
	}
}

// load runs in a goroutine and performs the actual server startup.
// It closes done when it finishes (whether successful or not).
func (lb *LazyLlamaCppBackend) load(done chan struct{}) {
	defer close(done)

	log.Printf("[guido] lazy-loading model %q...", lb.model)

	// Check if a llama-server is already running on this port.
	status := lb.checkServerStatus()

	var err error
	switch status {
	case lazyServerReady:
		log.Printf("[guido] reusing existing llama-server for %q at %s", lb.model, lb.baseURL)

	case lazyServerLoading:
		log.Printf("[guido] waiting for existing llama-server at %s to finish loading...", lb.baseURL)
		if lb.waitForServerReady(5*time.Minute) != lazyServerReady {
			err = fmt.Errorf("llama-server at %s did not become ready in time", lb.baseURL)
		}

	case lazyServerWrongModel:
		err = fmt.Errorf(
			"a llama-server is already running on %s but serves a different model — "+
				"kill it first (pkill -f 'llama-server.*%d') then retry",
			lb.baseURL, lb.port,
		)

	case lazyServerNotRunning:
		_, err = lb.toolMgr.StartLlamaServer(lb.modelPath, lb.port, lb.gpuLayers, lb.chatTemplate, lb.mmProjPath)
	}

	lb.mu.Lock()
	defer lb.mu.Unlock()

	if err != nil {
		lb.state = lazyErrored
		lb.loadErr = err
		log.Printf("[guido] model %q failed to load: %v", lb.model, err)
		return
	}

	lb.inner = NewLlamaCppBackend(lb.baseURL, lb.model)
	lb.state = lazyReady
	lb.loadErr = nil
	log.Printf("[guido] model %q is ready", lb.model)

	if lb.idleTimeout > 0 {
		go lb.watchIdle()
	}
}

// unload stops the server and resets state. Caller must hold lb.mu.
func (lb *LazyLlamaCppBackend) unload() {
	log.Printf("[guido] unloading model %q after %s idle", lb.model, lb.idleTimeout)
	if lb.toolMgr != nil {
		if err := lb.toolMgr.StopLlamaServer(); err != nil {
			log.Printf("[guido] error stopping llama-server: %v", err)
		}
	}
	lb.inner = nil
	lb.state = lazyUnloaded
}

// watchIdle polls every 30 s and calls unload when the idle threshold is hit.
func (lb *LazyLlamaCppBackend) watchIdle() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		lb.mu.Lock()
		if lb.state != lazyReady {
			lb.mu.Unlock()
			return
		}
		if time.Since(lb.lastActivity) >= lb.idleTimeout {
			lb.unload()
			lb.mu.Unlock()
			return
		}
		lb.mu.Unlock()
	}
}

// recordActivity stamps the last-used time (called after every successful call).
func (lb *LazyLlamaCppBackend) recordActivity() {
	lb.mu.Lock()
	lb.lastActivity = time.Now()
	lb.mu.Unlock()
}

// ── Server status helpers ─────────────────────────────────────────────────────

type lazyServerStatus int

const (
	lazyServerNotRunning lazyServerStatus = iota
	lazyServerLoading                     // 503 — bound but model still loading
	lazyServerReady                       // 200 OK, correct model
	lazyServerWrongModel                  // 200 OK, different model file
)

func (lb *LazyLlamaCppBackend) checkServerStatus() lazyServerStatus {
	client := &http.Client{Timeout: 2 * time.Second}

	healthResp, err := client.Get(lb.baseURL + "/health")
	if err != nil {
		return lazyServerNotRunning
	}
	healthResp.Body.Close()

	switch healthResp.StatusCode {
	case http.StatusServiceUnavailable:
		return lazyServerLoading
	case http.StatusOK:
		// fall through to model verification
	default:
		return lazyServerNotRunning
	}

	// No model path to verify against (external server) — accept it.
	if lb.modelPath == "" {
		return lazyServerReady
	}

	propsResp, err := client.Get(lb.baseURL + "/props")
	if err != nil {
		return lazyServerReady // /props unavailable — external server, accept
	}
	defer propsResp.Body.Close()

	var props struct {
		ModelPath string `json:"model_path"`
	}
	if err := json.NewDecoder(propsResp.Body).Decode(&props); err != nil || props.ModelPath == "" {
		return lazyServerReady // can't verify, accept
	}

	runningAbs, _ := filepath.Abs(props.ModelPath)
	expectedAbs, _ := filepath.Abs(lb.modelPath)
	if runningAbs == expectedAbs {
		return lazyServerReady
	}
	// Filename-only fallback (handles relative vs absolute path discrepancies).
	if filepath.Base(props.ModelPath) == filepath.Base(lb.modelPath) {
		return lazyServerReady
	}
	return lazyServerWrongModel
}

func (lb *LazyLlamaCppBackend) waitForServerReady(timeout time.Duration) lazyServerStatus {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		time.Sleep(3 * time.Second)
		if s := lb.checkServerStatus(); s != lazyServerLoading {
			return s
		}
	}
	return lazyServerLoading
}

// ── LLMProvider interface ─────────────────────────────────────────────────────

func (lb *LazyLlamaCppBackend) Complete(ctx context.Context, req *harness.CompletionRequest) (*harness.CompletionResponse, error) {
	if err := lb.EnsureLoaded(ctx); err != nil {
		return nil, err
	}
	defer lb.recordActivity()
	lb.mu.Lock()
	inner := lb.inner
	lb.mu.Unlock()
	return inner.Complete(ctx, req)
}

func (lb *LazyLlamaCppBackend) StreamTokens(ctx context.Context, req *harness.CompletionRequest) (<-chan string, error) {
	if err := lb.EnsureLoaded(ctx); err != nil {
		return nil, err
	}
	lb.mu.Lock()
	inner := lb.inner
	lb.mu.Unlock()

	ch, err := inner.StreamTokens(ctx, req)
	if err != nil {
		return nil, err
	}

	// Wrap the channel so we record activity once the stream drains.
	wrapped := make(chan string)
	go func() {
		defer lb.recordActivity()
		defer close(wrapped)
		for token := range ch {
			wrapped <- token
		}
	}()
	return wrapped, nil
}

func (lb *LazyLlamaCppBackend) Chat(ctx context.Context, req *harness.ChatRequest) (*harness.ChatResponse, error) {
	if err := lb.EnsureLoaded(ctx); err != nil {
		return nil, err
	}
	defer lb.recordActivity()
	lb.mu.Lock()
	inner := lb.inner
	lb.mu.Unlock()
	return inner.Chat(ctx, req)
}

func (lb *LazyLlamaCppBackend) StreamChat(ctx context.Context, req *harness.ChatRequest) (<-chan string, error) {
	if err := lb.EnsureLoaded(ctx); err != nil {
		return nil, err
	}
	lb.mu.Lock()
	inner := lb.inner
	lb.mu.Unlock()

	ch, err := inner.StreamChat(ctx, req)
	if err != nil {
		return nil, err
	}

	wrapped := make(chan string)
	go func() {
		defer lb.recordActivity()
		defer close(wrapped)
		for token := range ch {
			wrapped <- token
		}
	}()
	return wrapped, nil
}

// ListModels returns static metadata without forcing a load — the GUI can call
// this to discover available models even while they're unloaded.
func (lb *LazyLlamaCppBackend) ListModels(_ context.Context) ([]harness.ModelInfo, error) {
	return []harness.ModelInfo{
		{
			ID:       lb.model,
			Name:     lb.model,
			Provider: "llamacpp",
			Type:     "chat",
		},
	}, nil
}
