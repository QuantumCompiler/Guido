package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
)

// ErrMethodNotAllowed is returned by NewSSETransport when the server responds
// with HTTP 405 to the GET /sse request, indicating it uses the newer
// Streamable HTTP transport (MCP spec 2025-03-26) instead.
// NewRegistry detects this sentinel and falls back to NewStreamableTransport.
var ErrMethodNotAllowed = errors.New("mcp: server does not support HTTP+SSE transport")

// SSETransport implements Transport for remote MCP servers using the HTTP+SSE
// transport defined by the MCP specification (protocol version 2024-11-05).
//
// Two HTTP connections are used:
//   - A persistent GET <url>/sse receives JSON-RPC messages (server→client).
//   - Individual POST requests to a session endpoint send messages (client→server).
//
// The POST session URL is discovered dynamically from the first "endpoint" SSE
// event that the server emits on the GET stream.
//
// Usage:
//
//	t, err := NewSSETransport(ctx, "https://mcp.example.com", headers)
//	client, err := NewClient(ctx, "myserver", t)
type SSETransport struct {
	baseURL string
	postURL string            // discovered from "endpoint" SSE event
	headers map[string]string // extra headers on every request (auth, etc.)
	client  *http.Client

	recvCh chan json.RawMessage
	done   chan struct{}
	cancel context.CancelFunc
	once   sync.Once
}

// NewSSETransport connects to sseURL, waits for the server to announce the
// session POST endpoint, then returns a ready-to-use transport.
//
// If sseURL does not already end with "/sse", the suffix is appended.
// headers are added to every outgoing HTTP request (GET and POST).
func NewSSETransport(ctx context.Context, sseURL string, headers map[string]string) (*SSETransport, error) {
	base := strings.TrimRight(sseURL, "/")
	connectURL := base
	if !strings.HasSuffix(base, "/sse") {
		connectURL = base + "/sse"
	}

	// Use a child context so Close() can cancel the persistent GET connection.
	sseCtx, cancel := context.WithCancel(ctx)

	t := &SSETransport{
		baseURL: base,
		headers: headers,
		client:  &http.Client{},
		recvCh:  make(chan json.RawMessage, 64),
		done:    make(chan struct{}),
		cancel:  cancel,
	}

	req, err := http.NewRequestWithContext(sseCtx, "GET", connectURL, nil)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("mcp sse: build GET request: %w", err)
	}
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Cache-Control", "no-cache")
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := t.client.Do(req)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("mcp sse: connect %s: %w", connectURL, err)
	}
	if resp.StatusCode == http.StatusMethodNotAllowed {
		resp.Body.Close()
		cancel()
		return nil, fmt.Errorf("%w (HTTP 405 from %s)", ErrMethodNotAllowed, connectURL)
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		cancel()
		return nil, fmt.Errorf("mcp sse: server returned HTTP %d from %s", resp.StatusCode, connectURL)
	}

	// Start the background reader. It signals the POST endpoint via endpointCh
	// and then forwards all subsequent message events to t.recvCh.
	endpointCh := make(chan string, 1)
	go t.readSSE(resp.Body, endpointCh)

	// Wait until the server tells us where to POST.
	select {
	case rawEndpoint := <-endpointCh:
		resolved, err := resolveEndpoint(base, rawEndpoint)
		if err != nil {
			t.Close()
			return nil, fmt.Errorf("mcp sse: bad endpoint URL %q: %w", rawEndpoint, err)
		}
		t.postURL = resolved
		return t, nil
	case <-sseCtx.Done():
		// Either caller cancelled or connection dropped before endpoint arrived.
		return nil, fmt.Errorf("mcp sse: cancelled before endpoint event was received from %s", connectURL)
	}
}

// Send POSTs a JSON-encoded JSON-RPC message to the session endpoint.
func (t *SSETransport) Send(msg interface{}) error {
	select {
	case <-t.done:
		return fmt.Errorf("mcp sse: transport is closed")
	default:
	}

	data, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("mcp sse: marshal: %w", err)
	}

	req, err := http.NewRequest("POST", t.postURL, bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("mcp sse: build POST: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range t.headers {
		req.Header.Set(k, v)
	}

	resp, err := t.client.Do(req)
	if err != nil {
		return fmt.Errorf("mcp sse: POST: %w", err)
	}
	defer resp.Body.Close()
	// Drain the body so the connection can be reused.
	io.Copy(io.Discard, resp.Body) //nolint:errcheck
	if resp.StatusCode >= 300 {
		return fmt.Errorf("mcp sse: POST returned HTTP %d", resp.StatusCode)
	}
	return nil
}

// Recv returns the next JSON-RPC message received from the SSE stream.
// Blocks until a message arrives or the transport is closed.
func (t *SSETransport) Recv() (json.RawMessage, error) {
	select {
	case msg, ok := <-t.recvCh:
		if !ok {
			return nil, fmt.Errorf("mcp sse: transport closed")
		}
		return msg, nil
	case <-t.done:
		return nil, fmt.Errorf("mcp sse: transport closed")
	}
}

// Close shuts down the SSE transport. Safe to call multiple times.
func (t *SSETransport) Close() error {
	t.once.Do(func() {
		t.cancel()    // cancels the persistent GET connection → unblocks readSSE
		close(t.done) // unblocks any Recv call
	})
	return nil
}

// ── internal ──────────────────────────────────────────────────────────────────

// readSSE reads SSE events from body until EOF or Close.
//
// The first "endpoint" event's data is sent to endpointCh (buffer=1, non-blocking
// after the first send). All subsequent events whose data looks like JSON are
// forwarded to t.recvCh.
func (t *SSETransport) readSSE(body io.ReadCloser, endpointCh chan<- string) {
	defer body.Close()
	defer close(t.recvCh)

	scanner := bufio.NewScanner(body)
	var eventType, data string
	gotEndpoint := false

	for scanner.Scan() {
		line := scanner.Text()

		if line == "" {
			// Blank line — dispatch the accumulated event.
			if data == "" {
				eventType = ""
				continue
			}

			trimmed := strings.TrimSpace(data)

			if !gotEndpoint {
				// The first non-empty event is always the endpoint announcement.
				// Some servers set event:endpoint; others omit the event field.
				select {
				case endpointCh <- trimmed:
				case <-t.done:
					return
				}
				gotEndpoint = true
			} else if strings.HasPrefix(trimmed, "{") {
				// Looks like JSON — it's a JSON-RPC response.
				select {
				case t.recvCh <- json.RawMessage(trimmed):
				case <-t.done:
					return
				}
			}
			// Non-JSON data after the endpoint (ping frames, comments) is ignored.

			eventType = ""
			data = ""
			continue
		}

		// Accumulate SSE fields.
		switch {
		case strings.HasPrefix(line, "event:"):
			eventType = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		case strings.HasPrefix(line, "data:"):
			d := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if data != "" {
				data += "\n"
			}
			data += d
		case strings.HasPrefix(line, ":"):
			// SSE comment — ignore (used as keep-alive by many servers).
		}
		_ = eventType // used implicitly when dispatching above
	}
	// scanner.Err() returns nil on clean EOF, or the context-cancel error when
	// Close() was called — both are expected and don't need separate handling.
}

// resolveEndpoint resolves a possibly-relative endpoint path against base.
// Absolute URLs (http:// / https://) are returned unchanged.
func resolveEndpoint(base, endpoint string) (string, error) {
	if strings.HasPrefix(endpoint, "http://") || strings.HasPrefix(endpoint, "https://") {
		return endpoint, nil
	}
	baseU, err := url.Parse(base)
	if err != nil {
		return "", fmt.Errorf("parse base URL: %w", err)
	}
	endU, err := url.Parse(endpoint)
	if err != nil {
		return "", fmt.Errorf("parse endpoint: %w", err)
	}
	return baseU.ResolveReference(endU).String(), nil
}
