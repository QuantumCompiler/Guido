//go:build !embed_tools

package embeddedtools

import "embed"

// ToolsFS is empty in regular (non-embedded) builds.
// The CLI falls back to finding tools on the filesystem instead.
var ToolsFS embed.FS
