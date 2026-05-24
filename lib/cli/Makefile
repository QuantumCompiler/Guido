.PHONY: help build build-llama build-go clean install test

# Detect OS and architecture
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
	OS := darwin
	ifeq ($(UNAME_M),arm64)
		ARCH := arm64
	else
		ARCH := amd64
	endif
else ifeq ($(UNAME_S),Linux)
	OS := linux
	ifeq ($(UNAME_M),x86_64)
		ARCH := amd64
	else ifeq ($(UNAME_M),aarch64)
		ARCH := arm64
	else
		ARCH := $(UNAME_M)
	endif
else
	OS := windows
	ARCH := amd64
endif

VERSION ?= $(shell git describe --tags --always 2>/dev/null || echo "v0.1.0")
BIN_DIR := bin
LLAMA_BUILD_DIR := llama.cpp/build
LLAMA_TOOLS_DIR := $(BIN_DIR)/llama-cpp-tools

help:
	@echo "Guido - LLM Model Harness Build System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  build           Build everything (llama.cpp tools + Go binaries)"
	@echo "  build-llama     Build only llama.cpp tools"
	@echo "  build-go        Build only Go binaries (requires build-llama first)"
	@echo "  clean           Clean build artifacts"
	@echo "  install         Build and install binaries to ./bin"
	@echo "  test            Run tests"
	@echo ""
	@echo "Build info: $(OS)-$(ARCH) v$(VERSION)"

# Main build target
build: build-llama build-go

# Build llama.cpp tools
build-llama:
	@echo "Building llama.cpp tools for $(OS)-$(ARCH)..."
	@./scripts/build-llama.sh "$(LLAMA_BUILD_DIR)" "$(LLAMA_TOOLS_DIR)"
	@echo "✓ llama.cpp tools built: $(LLAMA_TOOLS_DIR)/"

# Build Go binaries
build-go:
	@echo "Building Go binaries..."
	@mkdir -p bin
	@cd . && go build -o bin/guido-harness ./cmd/harness
	@cd . && go build -o bin/guido-cli ./cmd/cli
	@echo "✓ Go binaries built: bin/guido-{harness,cli}"

# Install binaries (just reports where they are)
install: build
	@echo ""
	@echo "✓ Installation complete!"
	@echo ""
	@echo "Binaries available at:"
	@ls -lh bin/guido-harness bin/guido-cli 2>/dev/null || echo "  (run 'make build' first)"
	@echo ""
	@echo "To use:"
	@echo "  ./bin/guido-harness -config config.yaml"
	@echo "  ./bin/guido-cli complete \"Your prompt\""

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf bin $(LLAMA_BUILD_DIR)
	@go clean
	@echo "✓ Clean complete"

# Run tests
test:
	@echo "Running tests..."
	@go test ./...

# Development targets
dev-build:
	@go build -o bin/guido-harness ./cmd/harness
	@go build -o bin/guido-cli ./cmd/cli

dev-watch:
	@echo "Watching for changes... (requires fswatch or similar)"
	@while true; do \
		find . -name "*.go" | fswatch -e build -e bin | xargs -I {} make dev-build; \
	done
