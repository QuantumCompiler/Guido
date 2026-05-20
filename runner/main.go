package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/taylor/guido/runner/internal/api"
	"github.com/taylor/guido/runner/internal/llm"
	"github.com/taylor/guido/runner/internal/registry"
)

func main() {
	port := flag.Int("port", 11434, "Port to listen on")
	modelsDir := flag.String("models", defaultModelsDir(), "Directory containing .gguf model files")
	llamaBin := flag.String("llama-server", "llama-server", "Path to the llama-server binary (from llama.cpp)")
	verbose := flag.Bool("v", false, "Verbose llama-server output")
	flag.Parse()

	// Ensure models directory exists
	if err := os.MkdirAll(*modelsDir, 0755); err != nil {
		log.Fatalf("cannot create models dir: %v", err)
	}

	fmt.Printf("minollama\n")
	fmt.Printf("  models dir : %s\n", *modelsDir)
	fmt.Printf("  backend    : %s\n", *llamaBin)
	fmt.Printf("  listening  : http://localhost:%d\n\n", *port)

	reg := registry.New(*modelsDir)
	runner := llm.NewRunner(*llamaBin, *verbose)
	srv := api.NewServer(*port, reg, runner)

	log.Fatal(srv.Start())
}

func defaultModelsDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return "./models"
	}
	return filepath.Join(home, ".minollama", "models")
}
