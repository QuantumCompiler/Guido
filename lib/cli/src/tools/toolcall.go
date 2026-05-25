package tools

import (
	"encoding/json"
	"fmt"
	"strings"

	"guido/lib/cli/src/harness"
)

// WebTools returns the tool definitions for web_search and fetch_url.
// Pass these in ChatRequest.Tools to give the model internet access.
func WebTools() []harness.Tool {
	return []harness.Tool{
		{
			Type: "function",
			Function: harness.ToolFunction{
				Name: "web_search",
				Description: "Search the web for current information, recent events, or facts you don't know. " +
					"Returns result titles, URLs, and short snippets. " +
					"Use fetch_url afterwards to read the full content of a promising result.",
				Parameters: json.RawMessage(`{
					"type": "object",
					"properties": {
						"query": {
							"type": "string",
							"description": "The search query"
						}
					},
					"required": ["query"]
				}`),
			},
		},
		{
			Type: "function",
			Function: harness.ToolFunction{
				Name: "fetch_url",
				Description: "Fetch and read the plain-text content of a specific web page. " +
					"Use this to read an article or page in full after finding its URL via web_search.",
				Parameters: json.RawMessage(`{
					"type": "object",
					"properties": {
						"url": {
							"type": "string",
							"description": "The full URL of the page to fetch"
						}
					},
					"required": ["url"]
				}`),
			},
		},
	}
}

// ExecuteTool dispatches a tool call by name, runs it, and returns the result
// as a plain string suitable for feeding back to the model as a "tool" message.
func ExecuteTool(name, argsJSON string) (string, error) {
	var args map[string]interface{}
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("bad tool arguments for %q: %w", name, err)
	}

	switch name {
	case "web_search":
		query, _ := args["query"].(string)
		if query == "" {
			return "", fmt.Errorf("web_search: missing 'query' argument")
		}

		results, err := SearchDDG(query, 5)
		if err != nil {
			// Return error as content so the model can react gracefully.
			return fmt.Sprintf("Search failed: %v", err), nil
		}
		if len(results) == 0 {
			return fmt.Sprintf("No results found for: %q", query), nil
		}

		var sb strings.Builder
		fmt.Fprintf(&sb, "Search results for %q:\n\n", query)
		for i, r := range results {
			fmt.Fprintf(&sb, "%d. %s\n", i+1, r.Title)
			if r.URL != "" {
				fmt.Fprintf(&sb, "   URL: %s\n", r.URL)
			}
			if r.Snippet != "" {
				fmt.Fprintf(&sb, "   %s\n", r.Snippet)
			}
			fmt.Fprintln(&sb)
		}
		return sb.String(), nil

	case "fetch_url":
		rawURL, _ := args["url"].(string)
		if rawURL == "" {
			return "", fmt.Errorf("fetch_url: missing 'url' argument")
		}

		content, err := FetchURL(rawURL, 8000)
		if err != nil {
			return fmt.Sprintf("Failed to fetch %s: %v", rawURL, err), nil
		}
		return fmt.Sprintf("Content from %s:\n\n%s", rawURL, content), nil

	default:
		return "", fmt.Errorf("unknown tool: %q", name)
	}
}
