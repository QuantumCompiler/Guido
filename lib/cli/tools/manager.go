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
// is actually accepting connections (or times out). chatTemplate is optional.
func (m *Manager) StartLlamaServer(modelPath string, port int, nGPULayers int, chatTemplate string) (int, error) {
	toolPath, err := m.GetToolPath("llama-server")
	if err != nil {
		return 0, err
	}

	args := []string{
		"-m", modelPath,
		"--port", fmt.Sprintf("%d", port),
		"--gpu-layers", fmt.Sprintf("%d", nGPULayers),
	}
	if chatTemplate != "" {
		args = append(args, "--chat-template", chatTemplate)
	}

	// Log the exact command so it's visible in the terminal for debugging.
	fmt.Printf("[guido] starting llama-server: %s %v\n", toolPath, args)

	cmd := exec.Command(toolPath, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return 0, fmt.Errorf("failed to start llama-server: %w", err)
	}

	// Watch for early exit in a goroutine. cmd.ProcessState is only populated
	// after cmd.Wait() returns, so we must call Wait() to detect crashes.
	exitErr := make(chan error, 1)
	go func() {
		exitErr <- cmd.Wait()
	}()

	// Poll until the server is accepting HTTP connections (200 or 503 = loading).
	// Give up after 30 seconds — if it hasn't bound the port by then, something
	// is wrong (bad flag, immediate crash, etc.).
	healthURL := fmt.Sprintf("http://localhost:%d/health", port)
	client := &http.Client{Timeout: 1 * time.Second}
	deadline := time.Now().Add(30 * time.Second)

	for time.Now().Before(deadline) {
		// Did the process exit already?
		select {
		case err := <-exitErr:
			if err != nil {
				return 0, fmt.Errorf("llama-server exited during startup: %w", err)
			}
			return 0, fmt.Errorf("llama-server exited during startup (no error)")
		default:
		}

		// Is it listening yet?
		resp, err := client.Get(healthURL)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusServiceUnavailable {
				// 200 = ready, 503 = model still loading but server is up
				fmt.Printf("[guido] llama-server is listening on port %d (model may still be loading)\n", port)
				// Put the exit watcher back so the process still gets reaped.
				go func() { <-exitErr }()
				break
			}
		}
		time.Sleep(500 * time.Millisecond)
	}

	if cmd.Process == nil {
		return 0, errors.New("llama-server process not running")
	}

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
