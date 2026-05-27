//go:build embed_tools

package embeddedtools

import "embed"

// ToolsFS holds the compiled llama.cpp tools baked in at build time.
// Populated only when built with -tags embed_tools (via make build-embedded).
//
// Layout inside the FS:
//
//	data/bin/   – executables  (llama-server, llama-cli, llama-quantize, …)
//	data/lib/   – dylibs       (libllama, libggml, … — macOS only)
//	data/.version – stamp written by make stage-embed; used to skip re-extraction
//
//go:embed all:data
var ToolsFS embed.FS
