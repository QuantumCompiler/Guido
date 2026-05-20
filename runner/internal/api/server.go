package api

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/taylor/guido/runner/internal/llm"
	"github.com/taylor/guido/runner/internal/registry"
)

const version = "0.1.0"

// Server is the main HTTP server.
type Server struct {
	port   int
	reg    *registry.Registry
	runner *llm.Runner
	mux    *http.ServeMux
}

// NewServer wires everything together.
func NewServer(port int, reg *registry.Registry, runner *llm.Runner) *Server {
	s := &Server{
		port:   port,
		reg:    reg,
		runner: runner,
		mux:    http.NewServeMux(),
	}
	s.routes()
	return s
}

// Start begins listening.
func (s *Server) Start() error {
	addr := fmt.Sprintf(":%d", s.port)
	srv := &http.Server{
		Addr:         addr,
		Handler:      logging(s.mux),
		ReadTimeout:  5 * time.Second,
		// WriteTimeout intentionally left long for streaming responses
		IdleTimeout:  120 * time.Second,
	}
	return srv.ListenAndServe()
}

// routes registers all API endpoints.
func (s *Server) routes() {
	// Model management
	s.mux.HandleFunc("GET /api/version", s.handleVersion)
	s.mux.HandleFunc("GET /api/tags", s.handleTags)
	s.mux.HandleFunc("POST /api/show", s.handleShow)

	// Inference
	s.mux.HandleFunc("POST /api/generate", s.handleGenerate)
	s.mux.HandleFunc("POST /api/chat", s.handleChat)

	// Health / compatibility
	s.mux.HandleFunc("GET /", s.handleRoot)
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, ErrorResponse{Error: msg})
}

// writeNDJSON writes one NDJSON line and flushes.
func writeNDJSON(w http.ResponseWriter, flusher http.Flusher, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	if _, err := w.Write(b); err != nil {
		return err
	}
	flusher.Flush()
	return nil
}

// logging middleware prints method, path, and duration.
func logging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start).Round(time.Millisecond))
	})
}
