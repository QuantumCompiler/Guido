package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
)

// Transport is the low-level send/receive layer for JSON-RPC 2.0 messages.
// Phase 1 has one implementation: StdioTransport (subprocess over stdin/stdout).
// Phase 2 will add SSETransport (HTTP server-sent events) without touching this interface.
type Transport interface {
	// Send encodes and writes one JSON-RPC message. Thread-safe.
	Send(msg interface{}) error
	// Recv reads the next raw JSON message. Blocks until a message arrives or
	// the transport is closed. Only one goroutine should call Recv at a time.
	Recv() (json.RawMessage, error)
	// Close shuts down the transport and its underlying process (if any).
	Close() error
}

// StdioTransport spawns a subprocess and communicates via newline-delimited
// JSON (JSON-Lines / NDJSON) over its stdin/stdout.
type StdioTransport struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
	mu     sync.Mutex // guards stdin writes only; reads happen from one goroutine
}

// NewStdioTransport spawns command with args, sets up JSON-RPC over its
// stdin/stdout, and forwards stderr to the host process's stderr so MCP server
// log output is visible without contaminating the JSON stream.
func NewStdioTransport(ctx context.Context, command string, args []string, env []string) (*StdioTransport, error) {
	cmd := exec.CommandContext(ctx, command, args...)

	// Inherit the full parent environment, then overlay server-specific vars.
	// MCP servers commonly need $HOME, $PATH, and tool-specific env like DATABASE_URL.
	cmd.Env = append(os.Environ(), env...)
	cmd.Stderr = os.Stderr

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("mcp stdio: stdin pipe: %w", err)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("mcp stdio: stdout pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("mcp stdio: start %q: %w", command, err)
	}

	return &StdioTransport{
		cmd:    cmd,
		stdin:  stdin,
		stdout: bufio.NewReaderSize(stdout, 1<<20), // 1 MB — handles large tools/list responses
	}, nil
}

// Send encodes msg as JSON and writes it followed by a newline.
func (t *StdioTransport) Send(msg interface{}) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("mcp stdio: marshal: %w", err)
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if _, err := t.stdin.Write(append(data, '\n')); err != nil {
		return fmt.Errorf("mcp stdio: write: %w", err)
	}
	return nil
}

// Recv reads the next newline-delimited JSON message from stdout.
func (t *StdioTransport) Recv() (json.RawMessage, error) {
	line, err := t.stdout.ReadBytes('\n')
	if err != nil {
		return nil, fmt.Errorf("mcp stdio: read: %w", err)
	}
	return json.RawMessage(line), nil
}

// Close kills the subprocess and waits for it to exit.
func (t *StdioTransport) Close() error {
	_ = t.stdin.Close()
	if t.cmd.Process != nil {
		_ = t.cmd.Process.Kill()
		_ = t.cmd.Wait()
	}
	return nil
}
