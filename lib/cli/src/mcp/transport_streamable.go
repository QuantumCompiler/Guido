package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// StreamableTransport implements Transport for the MCP Streamable HTTP protocol
// (spec version 2025-03-26). Unlike the older HTTP+SSE transport, this protocol
// uses a single POST endpoint for all traffic:
//
//   - Client sends JSON-RPC requests via POST to the MCP URL.
//   - Server responds with either Content-Type: application/json (simple response)
//     or Content-Type: text/event-stream (streaming response).
//   - No persistent GET connection is required.
//
// Session affinity is maintained by capturing the Mcp-Session-Id header from
// the server's initialize response and including it in all subsequent requests.
//
// Auto-detection: NewRegistry tries SSETransport first; if the server returns
// HTTP 405, it falls back to this transport automatically.
type StreamableTransport struct {
	url       string
	headers   map[string]string
	sessionID string     // captured from Mcp-Session-Id response header
	mu        sync.Mutex // protects sessionID

	client *http.Client
	recvCh chan json.RawMessage
	done   chan struct{}
	once   sync.Once
}

// NewStreamableTransport creates a Streamable HTTP transport for url.
// No upfront connection is made — the first Send establishes the session.
//
// url should be the base MCP endpoint (e.g. "https://gitmcp.io/owner/repo").
// Any trailing "/sse" suffix is stripped automatically in case the caller
// copied it from an SSE-configured entry.
func NewStreamableTransport(_ context.Context, rawURL string, headers map[string]string) (*StreamableTransport, error) {
	// Strip /sse suffix — Streamable HTTP uses the base URL directly.
	u := strings.TrimRight(rawURL, "/")
	if strings.HasSuffix(u, "/sse") {
		u = strings.TrimSuffix(u, "/sse")
	}

	return &StreamableTransport{
		url:     u,
		headers: headers,
		client:  &http.Client{Timeout: 30 * time.Second},
		recvCh:  make(chan json.RawMessage, 64),
		done:    make(chan struct{}),
	}, nil
}

// Send POSTs a JSON-RPC message to the MCP endpoint and reads the response
// in a background goroutine. The response is forwarded to recvCh so that
// the Client's readLoop picks it up via Recv.
//
// Notifications (no ID, server returns 202 with empty body) produce no recvCh
// entry — the client correctly does not wait for a response to those.
func (t *StreamableTransport) Send(msg interface{}) error {
	select {
	case <-t.done:
		return fmt.Errorf("mcp streamable: transport closed")
	default:
	}

	data, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("mcp streamable: marshal: %w", err)
	}

	req, err := http.NewRequest("POST", t.url, bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("mcp streamable: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	for k, v := range t.headers {
		req.Header.Set(k, v)
	}

	// Include session ID once established.
	t.mu.Lock()
	if t.sessionID != "" {
		req.Header.Set("Mcp-Session-Id", t.sessionID)
	}
	t.mu.Unlock()

	resp, err := t.client.Do(req)
	if err != nil {
		return fmt.Errorf("mcp streamable: POST: %w", err)
	}

	// Capture session affinity header.
	if sid := resp.Header.Get("Mcp-Session-Id"); sid != "" {
		t.mu.Lock()
		t.sessionID = sid
		t.mu.Unlock()
	}

	if resp.StatusCode >= 300 && resp.StatusCode != 202 {
		resp.Body.Close()
		return fmt.Errorf("mcp streamable: POST returned HTTP %d", resp.StatusCode)
	}

	// Read response body in a goroutine so Send returns immediately.
	go t.readResponse(resp)
	return nil
}

// Recv blocks until the next JSON-RPC response arrives or the transport closes.
func (t *StreamableTransport) Recv() (json.RawMessage, error) {
	select {
	case msg, ok := <-t.recvCh:
		if !ok {
			return nil, fmt.Errorf("mcp streamable: transport closed")
		}
		return msg, nil
	case <-t.done:
		return nil, fmt.Errorf("mcp streamable: transport closed")
	}
}

// Close shuts down the transport. Safe to call multiple times.
func (t *StreamableTransport) Close() error {
	t.once.Do(func() { close(t.done) })
	return nil
}

// ── internal ──────────────────────────────────────────────────────────────────

// readResponse reads one HTTP response and forwards any JSON-RPC messages to
// recvCh. It handles both application/json and text/event-stream responses.
func (t *StreamableTransport) readResponse(resp *http.Response) {
	defer resp.Body.Close()

	ct := resp.Header.Get("Content-Type")

	if strings.Contains(ct, "text/event-stream") {
		t.readSSEBody(resp.Body)
		return
	}

	// application/json (or unset) — read full body as one JSON message.
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return
	}
	trimmed := strings.TrimSpace(string(body))
	if trimmed == "" {
		// 202 Accepted with no body — notification acknowledged, no response.
		return
	}
	if strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, "[") {
		select {
		case t.recvCh <- json.RawMessage(trimmed):
		case <-t.done:
		}
	}
}

// readSSEBody parses an SSE body (text/event-stream) from a POST response and
// forwards each JSON data event to recvCh. Used when the server streams a
// long-running tool result.
func (t *StreamableTransport) readSSEBody(body io.Reader) {
	scanner := bufio.NewScanner(body)
	var dataLine string

	for scanner.Scan() {
		line := scanner.Text()

		if line == "" {
			// Blank line — dispatch accumulated event.
			trimmed := strings.TrimSpace(dataLine)
			if trimmed != "" && strings.HasPrefix(trimmed, "{") {
				select {
				case t.recvCh <- json.RawMessage(trimmed):
				case <-t.done:
					return
				}
			}
			dataLine = ""
			continue
		}

		if strings.HasPrefix(line, "data:") {
			d := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if dataLine != "" {
				dataLine += "\n"
			}
			dataLine += d
		}
		// Ignore "event:", "id:", and ":" (comments/keep-alives).
	}

	// Flush final event if no trailing blank line.
	trimmed := strings.TrimSpace(dataLine)
	if trimmed != "" && strings.HasPrefix(trimmed, "{") {
		select {
		case t.recvCh <- json.RawMessage(trimmed):
		case <-t.done:
		}
	}
}
