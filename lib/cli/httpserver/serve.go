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

	"guido/lib/cli/harness"
)

// Serve starts the OpenAI-compatible HTTP server on the port in cfg, registers
// all routes, and blocks until SIGINT/SIGTERM or ctx is cancelled.
// onShutdown is called (if non-nil) after the HTTP server has stopped — use it
// to clean up backends (e.g. kill the llama-server process).
func Serve(ctx context.Context, cfg *harness.Config, h *harness.Harness, onShutdown func()) error {
	r := chi.NewRouter()
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)

	hnd := NewHandler(h)
	r.Post("/v1/completions", hnd.HandleCompletion)
	r.Post("/v1/chat/completions", hnd.HandleChat)
	r.Get("/v1/models", hnd.HandleListModels)
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
