package tools

import (
	"errors"
	"fmt"
	"io"
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

// ManagedProcess represents a running tool process
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

// StartLlamaServer starts llama-server with the given model and returns the port
func (m *Manager) StartLlamaServer(modelPath string, port int, nGPULayers int) (int, error) {
	toolPath, err := m.GetToolPath("llama-server")
	if err != nil {
		return 0, err
	}

	// Build command
	args := []string{
		"-m", modelPath,
		"--port", fmt.Sprintf("%d", port),
		"--gpu-layers", fmt.Sprintf("%d", nGPULayers),
	}

	cmd := exec.Command(toolPath, args...)

	// Capture output for debugging
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return 0, fmt.Errorf("failed to start llama-server: %w", err)
	}

	// Wait for server to be ready and check if process is still running
	for i := 0; i < 5; i++ {
		time.Sleep(500 * time.Millisecond)

		// Check if process exited
		if cmd.ProcessState != nil && cmd.ProcessState.Exited() {
			return 0, fmt.Errorf("llama-server exited immediately (port may be in use)")
		}
	}

	// Final verification
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
