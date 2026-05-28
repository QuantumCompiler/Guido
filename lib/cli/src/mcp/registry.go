package mcp

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"guido/lib/cli/src/harness"
)

// Registry holds all active MCP server connections and provides the two
// integration points needed by the CLI: aggregated tool definitions and tool dispatch.
type Registry struct {
	clients map[string]*Client // server name → client
}

// NewRegistry connects to all enabled MCP servers.
// Transport is selected automatically:
//   - url set   → HTTP+SSE (spec 2024-11-05); auto-falls-back to Streamable HTTP (spec 2025-03-26) on HTTP 405
//   - command set → stdio subprocess
//   - both set  → url takes precedence
//
// Failed connections are logged and skipped so a broken server doesn't
// prevent Guido from starting.
func NewRegistry(ctx context.Context, servers []harness.MCPServerConfig) (*Registry, error) {
	r := &Registry{clients: make(map[string]*Client)}

	for _, srv := range servers {
		if !srv.Enabled {
			continue
		}

		var transport Transport
		var err error

		switch {
		case srv.URL != "":
			// Remote MCP server — try HTTP+SSE (spec 2024-11-05) first.
			// If the server returns HTTP 405 it speaks Streamable HTTP (spec 2025-03-26) instead;
			// fall back automatically so either protocol works with the same config.
			transport, err = NewSSETransport(ctx, srv.URL, srv.Headers)
			if errors.Is(err, ErrMethodNotAllowed) {
				fmt.Printf("[mcp] %s: HTTP+SSE not supported, trying Streamable HTTP\n", srv.Name)
				transport, err = NewStreamableTransport(ctx, srv.URL, srv.Headers)
			}
			if err != nil {
				fmt.Printf("[mcp] warning: %s: connect failed: %v\n", srv.Name, err)
				continue
			}
		case srv.Command != "":
			// Local MCP server — stdio subprocess transport.
			transport, err = NewStdioTransport(ctx, srv.Command, srv.Args, srv.Env)
			if err != nil {
				fmt.Printf("[mcp] warning: %s: failed to start: %v\n", srv.Name, err)
				continue
			}
		default:
			fmt.Printf("[mcp] warning: %s: skipped — set either url (remote) or command (local)\n", srv.Name)
			continue
		}

		client, err := NewClient(ctx, srv.Name, transport)
		if err != nil {
			fmt.Printf("[mcp] warning: %s: handshake failed: %v\n", srv.Name, err)
			continue
		}

		r.clients[srv.Name] = client
		fmt.Printf("[mcp] connected: %s (%d tools)\n", srv.Name, len(client.Tools()))
	}

	return r, nil
}

// Tools returns the combined harness.Tool slice from all connected servers.
// Merge this into the tool list passed to ChatRequest.Tools.
func (r *Registry) Tools() []harness.Tool {
	var all []harness.Tool
	for _, c := range r.clients {
		all = append(all, c.Tools()...)
	}
	return all
}

// ExecuteTool dispatches a namespaced MCP tool call.
//
// Returns (result, true, nil) when name matches an MCP tool.
// Returns ("", false, nil) when name is not an MCP tool, so the caller
// can fall through to the built-in tools.ExecuteTool.
func (r *Registry) ExecuteTool(ctx context.Context, name, argsJSON string) (string, bool, error) {
	serverName, toolName, ok := parseNamespacedName(name)
	if !ok {
		return "", false, nil
	}

	client, ok := r.clients[serverName]
	if !ok {
		return "", true, fmt.Errorf("mcp: no connected server %q for tool %q", serverName, name)
	}

	result, err := client.CallTool(ctx, toolName, argsJSON)
	return result, true, err
}

// Close shuts down all server connections.
func (r *Registry) Close() error {
	for _, c := range r.clients {
		_ = c.Close()
	}
	return nil
}

// parseNamespacedName splits "mcp__<server>__<tool>" into server and tool name.
// Returns ok=false for names that don't follow the mcp__ prefix convention.
// Uses strings.Index to find the first __ after the prefix so tool names with
// single underscores (e.g. "read_file") are handled correctly.
func parseNamespacedName(name string) (server, tool string, ok bool) {
	if !strings.HasPrefix(name, "mcp__") {
		return "", "", false
	}
	rest := strings.TrimPrefix(name, "mcp__")
	idx := strings.Index(rest, "__")
	if idx < 0 {
		return "", "", false
	}
	return rest[:idx], rest[idx+2:], true
}
