.PHONY: help build build-llama build-go clean install symlinks test dev-build

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
	@echo "  install         Build + copy config + create symlinks in $(SYMLINK_DIR)"
	@echo "  symlinks        (Re)create /usr/local/bin symlinks without rebuilding"
	@echo "  test            Run tests"
	@echo ""
	@echo "Build info: $(OS)-$(ARCH) v$(VERSION)"

# Main build target
build: build-llama build-go

# Build llama.cpp tools
build-llama:
	@echo "Building llama.cpp tools for $(OS)-$(ARCH)..."
	@./$(SCRIPTS_DIR)/build-llama.sh "$(LLAMA_BUILD_DIR)" "$(LLAMA_TOOLS_DIR)"
	@echo "✓ llama.cpp tools built: $(LLAMA_TOOLS_DIR)/"

# Build Go binaries
build-go:
	@echo "Building Go binaries..."
	@mkdir -p $(BIN_DIR)
	@go build -o $(BIN_DIR)/guido-harness ./src/cmd/harness
	@go build -o $(BIN_DIR)/guido         ./src/cmd/cli
	@echo "✓ Go binaries built: $(BIN_DIR)/guido-{harness,} "

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
	@echo "  Guido binaries:"
	@sudo ln -sf "$(CURDIR)/$(BIN_DIR)/guido"         "$(SYMLINK_DIR)/guido"
	@echo "    guido"
	@sudo ln -sf "$(CURDIR)/$(BIN_DIR)/guido-harness" "$(SYMLINK_DIR)/guido-harness"
	@echo "    guido-harness"
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
	@echo "Symlinks:  $(SYMLINK_DIR)/guido, guido-harness, llama-server, ..."
	@echo ""
	@echo "To use:"
	@echo "  guido serve"
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

# Fast rebuild of Go binaries only (skips llama.cpp)
dev-build:
	@go build -o $(BIN_DIR)/guido-harness ./src/cmd/harness
	@go build -o $(BIN_DIR)/guido         ./src/cmd/cli
