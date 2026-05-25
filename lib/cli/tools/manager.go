package tools

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"time"
)

// Manager handles lifecycle of tools
type Manager struct {
	toolsDir string     // Directory where tools are located
	launched map[string]*ManagedProcess
}

// ManagedProcess represents a running tool process/
type ManagedProcess struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout io.ReadCloser
	stderr io.ReadCloser
}

// NewManagerFromDir creates a new tool manager using an existing tools directory
func NewManagerFromDir(toolsDir string) (*Manager, error) {
	// Verify tools directory exists
	if _, err := os.Stat(toolsDir); os.IsNotExist(err) {
		return nil, fmt.Errorf("tools directory not found: %s", toolsDir)
	}

	return &Manager{
		toolsDir: toolsDir,
		launched: make(map[string]*ManagedProcess),
	}, nil
}

// GetToolPath returns the absolute path to a tool executable
func (m *Manager) GetToolPath(toolName string) (string, error) {
	path := filepath.Join(m.toolsDir, toolName)

	// Add .exe on Windows
	if runtime.GOOS == "windows" && filepath.Ext(toolName) != ".exe" {
		path += ".exe"
	}

	// Verify tool exists
	if _, err := os.Stat(path); err != nil {
		return "", fmt.Errorf("tool not found: %s", toolName)
	}

	return path, nil
}

// StartLlamaServer starts llama-server with the given model and waits until it
// is actually accepting connections (or times out).
// chatTemplate and mmProjPath are both optional (pass "" to omit).
// mmProjPath is the path to a multimodal projector file — required for vision
// models such as Gemma 4, LLaVA, Qwen-VL, etc.
func (m *Manager) StartLlamaServer(modelPath string, port int, nGPULayers int, chatTemplate, mmProjPath string) (int, error) {
	toolPath, err := m.GetToolPath("llama-server")
	if err != nil {
		return 0, err
	}

	args := []string{
		"-m", modelPath,
		"--port", fmt.Sprintf("%d", port),
		"--gpu-layers", fmt.Sprintf("%d", nGPULayers),
		// Use the model's embedded Jinja template for chat formatting.
		// This is required for proper function-calling / tool-use support
		// (e.g. Gemma 4, Llama 3.1, Qwen 2.5).  Without --jinja the server
		// falls back to a simplified built-in template that silently ignores
		// the `tools` field in /v1/chat/completions requests.
		"--jinja",
	}
	if mmProjPath != "" {
		args = append(args, "--mmproj", mmProjPath)
	}
	if chatTemplate != "" {
		args = append(args, "--chat-template", chatTemplate)
	}

	// Suppress llama-server's own verbose output — we emit clean progress lines.
	cmd := exec.Command(toolPath, args...)
	cmd.Stdout = io.Discard
	cmd.Stderr = io.Discard

	fmt.Printf("[guido] starting llama-server on port %d...\n", port)

	if err := cmd.Start(); err != nil {
		return 0, fmt.Errorf("failed to start llama-server: %w", err)
	}

	// Watch for early exit in a goroutine. cmd.ProcessState is only populated
	// after cmd.Wait() returns, so we must call Wait() to detect crashes.
	exitErr := make(chan error, 1)
	go func() {
		exitErr <- cmd.Wait()
	}()

	// Poll until /health returns 200 OK (model fully loaded and ready).
	// 503 means the port is bound but the model is still being loaded into
	// VRAM — keep waiting.  Large models (31B+) can take several minutes.
	healthURL := fmt.Sprintf("http://localhost:%d/health", port)
	client := &http.Client{Timeout: 1 * time.Second}
	deadline := time.Now().Add(10 * time.Minute)
	startTime := time.Now()
	lastLog := time.Now()
	var ready bool

	fmt.Printf("[guido] loading model (this may take a minute)...\n")

	for time.Now().Before(deadline) {
		// Did the process crash before becoming ready?
		select {
		case err := <-exitErr:
			if err != nil {
				return 0, fmt.Errorf("llama-server exited during startup: %w", err)
			}
			return 0, fmt.Errorf("llama-server exited during startup (no error)")
		default:
		}

		resp, err := client.Get(healthURL)
		if err == nil {
			resp.Body.Close()
			switch resp.StatusCode {
			case http.StatusOK:
				// Model is fully loaded and accepting requests.
				ready = true
			case http.StatusServiceUnavailable:
				// Server is up but still loading — log progress every 15 s.
				if time.Since(lastLog) >= 15*time.Second {
					fmt.Printf("[guido] still loading... (%s elapsed)\n",
						time.Since(startTime).Round(time.Second))
					lastLog = time.Now()
				}
			}
		}

		if ready {
			break
		}
		time.Sleep(500 * time.Millisecond)
	}

	if !ready {
		cmd.Process.Kill() //nolint:errcheck
		return 0, fmt.Errorf("llama-server did not become ready within %s", time.Since(startTime).Round(time.Second))
	}

	fmt.Printf("[guido] llama-server ready on port %d (%s)\n", port, time.Since(startTime).Round(time.Second))

	// Store process reference
	m.launched["llama-server"] = &ManagedProcess{
		cmd: cmd,
	}

	return port, nil
}

// StopLlamaServer stops the running llama-server instance
func (m *Manager) StopLlamaServer() error {
	proc, ok := m.launched["llama-server"]
	if !ok {
		return errors.New("llama-server not running")
	}

	if proc.cmd.Process != nil {
		if err := proc.cmd.Process.Kill(); err != nil {
			return fmt.Errorf("failed to kill llama-server: %w", err)
		}
	}

	delete(m.launched, "llama-server")
	return nil
}

// Close cleans up all managed processes
func (m *Manager) Close() error {
	var errs []error

	for range m.launched {
		if err := m.StopLlamaServer(); err != nil {
			errs = append(errs, err)
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("errors during cleanup: %v", errs)
	}

	return nil
}

// ToolsDir returns the directory where tools are extracted
func (m *Manager) ToolsDir() string {
	return m.toolsDir
}
