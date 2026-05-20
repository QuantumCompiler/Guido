package registry

import (
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Model holds everything we know about a locally stored model file.
type Model struct {
	Name       string    // human-friendly name, e.g. "mistral:7b-q4"
	Path       string    // absolute path to the .gguf file
	Size       int64     // file size in bytes
	ModifiedAt time.Time // last modified time
	Digest     string    // sha256 of first 4 MB (fast, not full hash)
}

// Registry manages the directory of local model files.
type Registry struct {
	dir string
}

// New creates a Registry backed by dir.
func New(dir string) *Registry {
	return &Registry{dir: dir}
}

// List returns all models found in the models directory.
func (r *Registry) List() ([]Model, error) {
	entries, err := os.ReadDir(r.dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	var models []Model
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if !strings.HasSuffix(strings.ToLower(name), ".gguf") {
			continue
		}

		info, err := e.Info()
		if err != nil {
			continue
		}

		fullPath := filepath.Join(r.dir, name)
		digest := partialDigest(fullPath)

		models = append(models, Model{
			Name:       fileNameToModelName(name),
			Path:       fullPath,
			Size:       info.Size(),
			ModifiedAt: info.ModTime(),
			Digest:     digest,
		})
	}
	return models, nil
}

// Resolve finds the model matching name (case-insensitive, .gguf optional).
// Returns an error if not found.
func (r *Registry) Resolve(name string) (*Model, error) {
	models, err := r.List()
	if err != nil {
		return nil, err
	}

	// Normalize: strip tag ":" suffix if present, lowercase
	want := strings.ToLower(strings.TrimSuffix(name, ".gguf"))

	for _, m := range models {
		candidate := strings.ToLower(m.Name)
		if candidate == want {
			return &m, nil
		}
		// Also try matching by bare filename without extension
		base := strings.ToLower(strings.TrimSuffix(filepath.Base(m.Path), ".gguf"))
		if base == want {
			return &m, nil
		}
	}

	return nil, fmt.Errorf("model %q not found in %s\n"+
		"  Drop a .gguf file there and try again.\n"+
		"  Download models from: https://huggingface.co/models?search=gguf",
		name, r.dir)
}

// Dir returns the models directory path.
func (r *Registry) Dir() string {
	return r.dir
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// fileNameToModelName converts "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
// into a friendlier "mistral-7b-instruct-v0.2:Q4_K_M"
func fileNameToModelName(filename string) string {
	// Strip .gguf
	name := strings.TrimSuffix(filename, ".gguf")

	// Detect quantization suffix (Q4_K_M, Q8_0, F16, …) and turn it into a tag
	parts := strings.Split(name, ".")
	if len(parts) > 1 {
		last := parts[len(parts)-1]
		if looksLikeQuant(last) {
			base := strings.Join(parts[:len(parts)-1], ".")
			return base + ":" + last
		}
	}
	return name
}

func looksLikeQuant(s string) bool {
	upper := strings.ToUpper(s)
	prefixes := []string{"Q2", "Q3", "Q4", "Q5", "Q6", "Q8", "F16", "F32", "BF16", "IQ"}
	for _, p := range prefixes {
		if strings.HasPrefix(upper, p) {
			return true
		}
	}
	return false
}

// partialDigest hashes the first 4 MB of the file for a cheap identity check.
func partialDigest(path string) string {
	f, err := os.Open(path)
	if err != nil {
		return "unknown"
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.CopyN(h, f, 4*1024*1024); err != nil && err != io.EOF {
		return "unknown"
	}
	return fmt.Sprintf("sha256:%x", h.Sum(nil))
}
