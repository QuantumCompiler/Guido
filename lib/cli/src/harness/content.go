package harness

import (
	"encoding/json"
	"fmt"
	"strings"
)

// MessageContent holds the content of a chat message.
// Serialises to a plain JSON string when no parts are set, or to a JSON array
// of OpenAI-compatible content parts when rich content is present.
// Both forms are accepted on input — existing clients sending plain strings work unchanged.
type MessageContent struct {
	text  string
	parts []ContentPart
}

// Text creates a plain-text MessageContent.
func Text(s string) MessageContent { return MessageContent{text: s} }

// Parts creates a multi-part MessageContent (text blocks, images, etc.)
func Parts(parts ...ContentPart) MessageContent { return MessageContent{parts: parts} }

// PlainText returns a string representation. For rich content, text parts are
// concatenated; non-text parts are omitted. Safe fallback for backends that
// don't support images.
func (c MessageContent) PlainText() string {
	if len(c.parts) == 0 {
		return c.text
	}
	var sb strings.Builder
	for _, p := range c.parts {
		if p.Type == "text" {
			sb.WriteString(p.Text)
		}
	}
	return sb.String()
}

// ContentParts returns the rich parts slice, or nil for plain-text content.
func (c MessageContent) ContentParts() []ContentPart { return c.parts }

// IsRich reports whether any non-text parts are present.
func (c MessageContent) IsRich() bool {
	for _, p := range c.parts {
		if p.Type != "text" {
			return true
		}
	}
	return false
}

// String implements fmt.Stringer — returns PlainText for logging/display.
func (c MessageContent) String() string { return c.PlainText() }

// MarshalJSON serialises as a bare string when there are no parts (common case,
// keeps payloads small), or as a JSON array of content parts otherwise.
func (c MessageContent) MarshalJSON() ([]byte, error) {
	if len(c.parts) > 0 {
		return json.Marshal(c.parts)
	}
	return json.Marshal(c.text)
}

// UnmarshalJSON accepts both a bare JSON string and an OpenAI-style content-part
// array, keeping the HTTP API backward-compatible with plain-string clients.
func (c *MessageContent) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		c.text, c.parts = s, nil
		return nil
	}
	var parts []ContentPart
	if err := json.Unmarshal(data, &parts); err != nil {
		return fmt.Errorf("message content must be a string or content-part array: %w", err)
	}
	c.parts, c.text = parts, ""
	return nil
}

// ── Content part types ────────────────────────────────────────────────────────

// ContentPart is one element of a multi-part message.
// Mirrors the OpenAI content-part schema for wire compatibility with llama-server,
// OpenAI, and any OpenAI-compatible API.
type ContentPart struct {
	Type     string    `json:"type"`               // "text" | "image_url"
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL holds the URL or base64 data URI for an image content part.
type ImageURL struct {
	URL    string `json:"url"`              // https://… or data:image/jpeg;base64,…
	Detail string `json:"detail,omitempty"` // "low" | "high" | "auto"
}

// TextPart creates a text ContentPart.
func TextPart(text string) ContentPart {
	return ContentPart{Type: "text", Text: text}
}

// ImageURLPart creates an image ContentPart from a URL or base64 data URI.
func ImageURLPart(url string) ContentPart {
	return ContentPart{Type: "image_url", ImageURL: &ImageURL{URL: url}}
}
