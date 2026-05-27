package httpserver

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"

	"guido/lib/cli/src/harness"
)

// corsMiddleware adds permissive CORS headers to every response.
// Guido is a local-inference server typically accessed from localhost; the
// open CORS policy lets browser-based frontends and Electron UIs call it
// without proxy workarounds.
func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		w.Header().Set("Access-Control-Expose-Headers", "X-Guido-Tools-Used, X-Guido-Resolved-Model")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// Serve starts the OpenAI-compatible HTTP server on the port in cfg, registers
// all routes, and blocks until SIGINT/SIGTERM or ctx is cancelled.
// tc may be nil — when non-nil the handler runs the agentic tool loop
// internally for every chat request and returns only the final answer.
// onShutdown is called (if non-nil) after the HTTP server has stopped — use it
// to clean up backends (e.g. kill the llama-server process).
func Serve(ctx context.Context, cfg *harness.Config, h *harness.Harness, tc *ToolConfig, onShutdown func()) error {
	r := chi.NewRouter()
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)
	r.Use(corsMiddleware)

	hnd := NewHandler(h, tc)
	r.Post("/v1/completions", hnd.HandleCompletion)
	r.Post("/v1/chat/completions", hnd.HandleChat)
	r.Get("/v1/models", hnd.HandleListModels)
	r.Get("/v1/model/status", hnd.HandleModelStatus)
	r.Get("/health", hnd.HandleHealth)

	addr := fmt.Sprintf(":%d", cfg.Server.Port)
	srv := &http.Server{
		Addr:    addr,
		Handler: r,
	}

	// Catch OS signals and ctx cancellation on a single channel.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	// Graceful-shutdown goroutine.
	shutdownDone := make(chan struct{})
	go func() {
		defer close(shutdownDone)
		select {
		case sig := <-sigCh:
			log.Printf("Received signal %v, shutting down…", sig)
		case <-ctx.Done():
			log.Println("Context cancelled, shutting down…")
		}
		sdCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := srv.Shutdown(sdCtx); err != nil {
			log.Printf("HTTP shutdown error: %v", err)
		}
		if onShutdown != nil {
			onShutdown()
		}
	}()

	log.Printf("Guido harness listening on %s", addr)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		return fmt.Errorf("server error: %w", err)
	}
	<-shutdownDone
	return nil
}
