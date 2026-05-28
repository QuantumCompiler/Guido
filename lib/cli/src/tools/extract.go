package tools

import (
	"embed"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// ExtractEmbedded extracts the embedded llama.cpp tools from embeddedFS into
// targetDir (e.g. ~/.guido/tools), and returns a ready-to-use *Manager.
//
// The function is version-aware: if targetDir/.version already contains the
// same stamp as data/.version inside the FS, extraction is skipped and the
// existing directory is reused.  This means upgrades re-extract automatically
// while normal startups pay only a single stat syscall.
//
// Returns (nil, nil) when embeddedFS is empty (non-embedded build); the caller
// should fall back to finding tools on the filesystem.
func ExtractEmbedded(embeddedFS embed.FS, targetDir string) (*Manager, error) {
	// In stub builds ToolsFS is zero-value — nothing to extract.
	entries, err := embeddedFS.ReadDir("data")
	if err != nil || len(entries) == 0 {
		return nil, nil //nolint:nilnil // intentional: signals "not embedded"
	}

	// Read the version stamp baked in at build time.
	stampBytes, err := embeddedFS.ReadFile("data/.version")
	if err != nil {
		return nil, fmt.Errorf("embedded tools: missing data/.version stamp: %w", err)
	}
	stamp := strings.TrimSpace(string(stampBytes))

	// Check whether the on-disk extraction is already current.
	versionFile := filepath.Join(targetDir, ".version")
	if existing, err := os.ReadFile(versionFile); err == nil {
		if strings.TrimSpace(string(existing)) == stamp {
			// Already extracted and up-to-date.
			return newManagerFromExtracted(targetDir)
		}
	}

	// Extract fresh.
	fmt.Printf("[guido] extracting embedded tools to %s ...\n", targetDir)
	if err := os.MkdirAll(targetDir, 0o755); err != nil {
		return nil, fmt.Errorf("embedded tools: mkdir %s: %w", targetDir, err)
	}

	if err := extractFS(embeddedFS, "data", targetDir); err != nil {
		return nil, fmt.Errorf("embedded tools: extract: %w", err)
	}

	// Write version stamp so next run skips extraction.
	if err := os.WriteFile(versionFile, []byte(stamp+"\n"), 0o644); err != nil {
		return nil, fmt.Errorf("embedded tools: write version file: %w", err)
	}

	fmt.Printf("[guido] tools extracted ✓\n")
	return newManagerFromExtracted(targetDir)
}

// newManagerFromExtracted builds a Manager pointing at the extracted bin/ directory
// and, if a lib/ directory is present, records it for DYLD/LD_LIBRARY_PATH injection.
func newManagerFromExtracted(targetDir string) (*Manager, error) {
	binDir := filepath.Join(targetDir, "bin")
	libDir := filepath.Join(targetDir, "lib")

	// lib/ is macOS-only; silently absent on Linux (static build).
	if _, err := os.Stat(libDir); os.IsNotExist(err) {
		libDir = ""
	}

	m, err := NewManagerFromDir(binDir)
	if err != nil {
		return nil, err
	}
	m.libDir = libDir
	return m, nil
}

// extractFS walks srcFS rooted at srcRoot and writes every file into dstRoot,
// recreating the directory tree.  Execute bits are preserved for files under bin/.
func extractFS(srcFS embed.FS, srcRoot, dstRoot string) error {
	return fs.WalkDir(srcFS, srcRoot, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		// Compute destination path by stripping the srcRoot prefix.
		rel, err := filepath.Rel(srcRoot, path)
		if err != nil {
			return err
		}
		dst := filepath.Join(dstRoot, rel)

		if d.IsDir() {
			return os.MkdirAll(dst, 0o755)
		}

		// Determine file permissions: executables under bin/ get 0o755.
		perm := fs.FileMode(0o644)
		if strings.HasPrefix(rel, "bin"+string(filepath.Separator)) {
			perm = 0o755
		}

		srcFile, err := srcFS.Open(path)
		if err != nil {
			return err
		}
		defer srcFile.Close()

		dstFile, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, perm)
		if err != nil {
			return err
		}
		defer dstFile.Close()

		_, err = io.Copy(dstFile, srcFile)
		return err
	})
}
