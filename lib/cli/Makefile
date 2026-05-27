.PHONY: help build build-llama build-go build-embedded stage-embed clean install symlinks test dev-build

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

VERSION     ?= $(shell git describe --tags --always 2>/dev/null || echo "v0.1.0")
BIN_DIR     := exec/bin
SCRIPTS_DIR := exec/scripts
MODULES_DIR := modules
CONFIG_DIR  := $(HOME)/.guido/config
INSTALL_DIR := $(HOME)/bin
SYMLINK_DIR := /usr/local/bin

LLAMA_BUILD_DIR := $(MODULES_DIR)/llama.cpp/build
LLAMA_TOOLS_DIR := $(BIN_DIR)/llama-cpp-tools

EMBED_DIR       := src/embeddedtools/data
EMBED_BIN_DIR   := $(EMBED_DIR)/bin
EMBED_LIB_DIR   := $(EMBED_DIR)/lib

help:
	@echo "Guido - LLM Model Harness Build System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  build           Build everything — llama.cpp tools + self-contained embedded binary"
	@echo "  build-llama     Build only llama.cpp tools"
	@echo "  build-go        Build only Go binary (requires build-llama first)"
	@echo "  stage-embed     Copy compiled tools into src/embeddedtools/data/ for embedding"
	@echo "  build-embedded  stage-embed + build self-contained binary (-tags embed_tools)"
	@echo "  clean           Clean build artifacts"
	@echo "  install         Build + copy config + create symlinks in $(SYMLINK_DIR)"
	@echo "  symlinks        (Re)create /usr/local/bin symlinks without rebuilding"
	@echo "  test            Run tests"
	@echo ""
	@echo "Build info: $(OS)-$(ARCH) v$(VERSION)"

# Main build target — compiles llama.cpp tools, stages them for embedding,
# then builds self-contained Go binaries with all tools baked in.
build: build-llama build-embedded

# Build llama.cpp tools
build-llama:
	@echo "Building llama.cpp tools for $(OS)-$(ARCH)..."
	@./$(SCRIPTS_DIR)/build-llama.sh "$(LLAMA_BUILD_DIR)" "$(LLAMA_TOOLS_DIR)"
	@echo "✓ llama.cpp tools built: $(LLAMA_TOOLS_DIR)/"

# Build Go binaries
build-go:
	@echo "Building Go binaries..."
	@mkdir -p $(BIN_DIR)
	@go build -o $(BIN_DIR)/guido ./src/cmd/cli
	@echo "✓ Go binaries built: $(BIN_DIR)/guido"

# Copy compiled llama.cpp tools into the embed staging directory.
# Run this after build-llama; it populates src/embeddedtools/data/ which is
# compiled into the binary by build-embedded.
stage-embed:
	@echo "Staging tools for embedding..."
	@mkdir -p $(EMBED_BIN_DIR) $(EMBED_LIB_DIR)
	@# Copy key executables (skip Python scripts which need a runtime anyway)
	@for tool in llama-server llama-cli llama-quantize llama-gguf llama-imatrix llama-bench; do \
		src="$(LLAMA_TOOLS_DIR)/$$tool"; \
		if [ -f "$$src" ]; then \
			cp "$$src" "$(EMBED_BIN_DIR)/$$tool"; \
			echo "  bin/$$tool"; \
		fi; \
	done
	@# Copy shared libraries (macOS dylibs / Linux .so files)
	@find "$(LLAMA_BUILD_DIR)/bin" -maxdepth 1 \( -name "*.dylib" -o -name "*.so*" \) | while read lib; do \
		name=$$(basename "$$lib"); \
		cp "$$lib" "$(EMBED_LIB_DIR)/$$name"; \
		echo "  lib/$$name"; \
	done
	@# Write version stamp so the runtime can skip re-extraction on identical builds
	@echo "$(VERSION)-$(OS)-$(ARCH)" > $(EMBED_DIR)/.version
	@echo "✓ Staged to $(EMBED_DIR)/ (version: $(VERSION)-$(OS)-$(ARCH))"

# Build self-contained binary with all llama.cpp tools baked in.
# Requires build-llama (or an existing LLAMA_TOOLS_DIR) to have run first.
build-embedded: stage-embed
	@echo "Building self-contained embedded binary..."
	@mkdir -p $(BIN_DIR)
	@go build -tags embed_tools -o $(BIN_DIR)/guido ./src/cmd/cli
	@echo "✓ Self-contained binary built: $(BIN_DIR)/guido"
	@echo "  (llama.cpp tools are embedded — no external dependencies needed)"

# Install binary, config, and /usr/local/bin symlinks
install: build
	@echo "── Config ───────────────────────────────────────────────"
	@mkdir -p "$(CONFIG_DIR)"
	@if [ -f "$(CONFIG_DIR)/config.yaml" ]; then \
		echo "  config already exists at $(CONFIG_DIR)/config.yaml — skipping"; \
	else \
		cp config.yaml "$(CONFIG_DIR)/config.yaml"; \
		echo "  ✓ config → $(CONFIG_DIR)/config.yaml"; \
	fi
	@$(MAKE) --no-print-directory symlinks

# Create (or refresh) /usr/local/bin symlinks for every executable in exec/bin/.
# Runs 'sudo ln -sf' so existing links are updated in place.
# Requires sudo — you will be prompted for your password.
symlinks:
	@echo "── Symlinks → $(SYMLINK_DIR) ─────────────────────────────"
	@echo "  Guido binary:"
	@sudo ln -sf "$(CURDIR)/$(BIN_DIR)/guido" "$(SYMLINK_DIR)/guido"
	@echo "    guido"
	@echo "  llama.cpp tools:"
	@for tool in "$(CURDIR)/$(LLAMA_TOOLS_DIR)"/*; do \
		name=$$(basename "$$tool"); \
		sudo ln -sf "$$tool" "$(SYMLINK_DIR)/$$name"; \
		echo "    $$name"; \
	done
	@echo ""
	@echo "✓ Installation complete!"
	@echo ""
	@echo "Config:    $(CONFIG_DIR)/config.yaml"
	@echo "Symlinks:  $(SYMLINK_DIR)/guido, llama-server, ..."
	@echo ""
	@echo "To use:"
	@echo "  guido serve"
	@echo "  guido harness"
	@echo "  guido chat"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BIN_DIR) $(LLAMA_BUILD_DIR)
	@go clean
	@echo "✓ Clean complete"

# Run tests
test:
	@echo "Running tests..."
	@go test ./src/...

# Fast rebuild of Go binary only (skips llama.cpp)
dev-build:
	@go build -o $(BIN_DIR)/guido ./src/cmd/cli
