package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"sync"
	"sync/atomic"

	"guido/lib/cli/src/harness"
)

// Client manages a single MCP server connection.
// A background read goroutine demultiplexes incoming messages and routes
// responses to their waiting callers via a pending-call table.
type Client struct {
	name      string
	transport Transport

	nextID  atomic.Int64
	mu      sync.Mutex
	pending map[int64]chan Response

	tools []harness.Tool // cached after initialize + tools/list
	done  chan struct{}
}

// NewClient dials the transport, performs the MCP handshake, and caches the
// server's tool list. Call Close when done.
func NewClient(ctx context.Context, name string, transport Transport) (*Client, error) {
	c := &Client{
		name:      name,
		transport: transport,
		pending:   make(map[int64]chan Response),
		done:      make(chan struct{}),
	}
	go c.readLoop()

	if err := c.initialize(ctx); err != nil {
		c.Close()
		return nil, fmt.Errorf("initialize: %w", err)
	}
	if err := c.loadTools(ctx); err != nil {
		c.Close()
		return nil, fmt.Errorf("tools/list: %w", err)
	}
	return c, nil
}

// Tools returns the namespaced harness.Tool slice for this server.
// Safe after NewClient returns; does not re-contact the server.
func (c *Client) Tools() []harness.Tool { return c.tools }

// CallTool executes a tool by its bare (un-namespaced) name with JSON-encoded
// arguments and returns the concatenated text content of the result.
func (c *Client) CallTool(ctx context.Context, toolName, argsJSON string) (string, error) {
	params := ToolCallParams{
		Name:      toolName,
		Arguments: json.RawMessage(argsJSON),
	}
	rawParams, _ := json.Marshal(params)

	var result ToolCallResult
	if err := c.call(ctx, "tools/call", rawParams, &result); err != nil {
		return "", err
	}
	if result.IsError {
		return "", fmt.Errorf("mcp tool %q: %s", toolName, collectText(result.Content))
	}
	return collectText(result.Content), nil
}

// Close terminates the connection and stops the background read goroutine.
func (c *Client) Close() error {
	select {
	case <-c.done:
	default:
		close(c.done)
	}
	return c.transport.Close()
}

// ─── internal ─────────────────────────────────────────────────────────────────

// call sends a JSON-RPC request and blocks until the response arrives or ctx is cancelled.
func (c *Client) call(ctx context.Context, method string, params json.RawMessage, out interface{}) error {
	id := c.nextID.Add(1)
	ch := make(chan Response, 1)

	c.mu.Lock()
	c.pending[id] = ch
	c.mu.Unlock()

	defer func() {
		c.mu.Lock()
		delete(c.pending, id)
		c.mu.Unlock()
	}()

	req := Request{
		JSONRPC: "2.0",
		ID:      id,
		Method:  method,
		Params:  params,
	}
	if err := c.transport.Send(req); err != nil {
		return err
	}

	select {
	case resp := <-ch:
		if resp.Error != nil {
			return resp.Error
		}
		if out != nil {
			return json.Unmarshal(resp.Result, out)
		}
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-c.done:
		return fmt.Errorf("mcp %s: connection closed", c.name)
	}
}

// readLoop receives messages from the transport and dispatches them.
// Runs in its own goroutine until Close is called or the transport errors.
func (c *Client) readLoop() {
	for {
		select {
		case <-c.done:
			return
		default:
		}

		raw, err := c.transport.Recv()
		if err != nil {
			select {
			case <-c.done:
				return // expected close
			default:
				log.Printf("[mcp] %s: read error: %v", c.name, err)
				return
			}
		}

		var msg IncomingMessage
		if err := json.Unmarshal(raw, &msg); err != nil {
			log.Printf("[mcp] %s: bad message: %v", c.name, err)
			continue
		}

		if msg.ID == nil {
			// Server-initiated notification (e.g. notifications/tools/list_changed).
			// Logged but not acted on in phase 1; a future iteration can trigger tool refresh.
			log.Printf("[mcp] %s: notification: %s", c.name, msg.Method)
			continue
		}

		c.mu.Lock()
		ch, ok := c.pending[*msg.ID]
		c.mu.Unlock()

		if !ok {
			log.Printf("[mcp] %s: unexpected response id %d", c.name, *msg.ID)
			continue
		}

		ch <- Response{ID: *msg.ID, Result: msg.Result, Error: msg.Error}
	}
}

// initialize sends the MCP handshake and the required initialized notification.
func (c *Client) initialize(ctx context.Context) error {
	params := InitializeParams{
		ProtocolVersion: "2024-11-05",
		Capabilities:    ClientCapabilities{Tools: &struct{}{}},
		ClientInfo:      ClientInfo{Name: "guido", Version: "0.1.0"},
	}
	rawParams, _ := json.Marshal(params)

	var result InitializeResult
	if err := c.call(ctx, "initialize", rawParams, &result); err != nil {
		return err
	}

	// Send initialized notification — required by the MCP spec (no response expected).
	return c.transport.Send(Notification{
		JSONRPC: "2.0",
		Method:  "notifications/initialized",
	})
}

// loadTools fetches tools/list and converts them to namespaced harness.Tool values.
func (c *Client) loadTools(ctx context.Context) error {
	var result ToolsListResult
	if err := c.call(ctx, "tools/list", json.RawMessage(`{}`), &result); err != nil {
		return err
	}

	c.tools = make([]harness.Tool, 0, len(result.Tools))
	for _, t := range result.Tools {
		c.tools = append(c.tools, harness.Tool{
			Type: "function",
			Function: harness.ToolFunction{
				Name:        namespacedName(c.name, t.Name),
				Description: t.Description,
				Parameters:  t.InputSchema,
			},
		})
	}
	return nil
}

// namespacedName returns "mcp__<server>__<tool>", matching Claude Code conventions.
func namespacedName(server, tool string) string {
	return "mcp__" + server + "__" + tool
}

// collectText joins all text-type content blocks into a single string.
func collectText(content []ToolContent) string {
	var sb strings.Builder
	for _, c := range content {
		if c.Type == "text" && c.Text != "" {
			sb.WriteString(c.Text)
		}
	}
	return sb.String()
}
