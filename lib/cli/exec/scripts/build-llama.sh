#!/bin/bash
set -e

# Build llama.cpp and extract compiled tools
# Usage: ./build-llama.sh <build_dir> <output_dir>

# Get absolute paths - SAVE EARLY before any directory changes
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script lives at exec/scripts/ — two levels up to reach lib/cli root
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LLAMA_SRC_DIR="$BASE_DIR/modules/llama.cpp"
PY_WRAPPER_SCRIPT="$SCRIPT_DIR/create-py-wrappers.sh"

# Build dir and output dir - convert to absolute paths
if [[ "$1" = /* ]]; then
    LLAMA_BUILD_DIR="$1"
else
    LLAMA_BUILD_DIR="$BASE_DIR/$1"
fi

if [[ "$2" = /* ]]; then
    OUTPUT_DIR="$2"
else
    OUTPUT_DIR="$BASE_DIR/$2"
fi

echo "Building llama.cpp from: $LLAMA_SRC_DIR"
echo "Build directory: $LLAMA_BUILD_DIR"
echo "Output directory: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Configure and build llama.cpp
cd "$LLAMA_SRC_DIR"

if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir -p build
fi

cd "$LLAMA_BUILD_DIR"

# Detect OS and architecture
UNAME_S=$(uname -s)
UNAME_M=$(uname -m)

echo "Configuring CMake for $UNAME_S $UNAME_M..."

# Common cmake options
CMAKE_COMMON_OPTS="-DCMAKE_BUILD_TYPE=Release"

# Platform-specific options
if [ "$UNAME_S" = "Darwin" ]; then
    # macOS: use Metal for GPU acceleration
    CMAKE_OPTS="$CMAKE_COMMON_OPTS -DLLAMA_METAL=ON"
    # Use all CPU cores
    MAKE_JOBS=$(sysctl -n hw.ncpu)
elif [ "$UNAME_S" = "Linux" ]; then
    # Linux: check for CUDA
    if command -v nvidia-smi &> /dev/null; then
        CMAKE_OPTS="$CMAKE_COMMON_OPTS -DLLAMA_CUDA=ON"
    else
        CMAKE_OPTS="$CMAKE_COMMON_OPTS"
    fi
    MAKE_JOBS=$(nproc)
else
    # Windows or other
    CMAKE_OPTS="$CMAKE_COMMON_OPTS"
    MAKE_JOBS=4
fi

# Run CMake configuration (skip if already configured)
if [ ! -f "CMakeCache.txt" ]; then
    cmake .. $CMAKE_OPTS
fi

# Build
echo "Building llama.cpp with $MAKE_JOBS jobs..."
cmake --build . --config Release -j $MAKE_JOBS

# Define which binaries to extract
TOOLS=(
    "bin/llama-server"
    "bin/llama-cli"
    "bin/llama-quantize"
    "bin/llama-convert"
    "bin/llama-bench"
    "bin/llama-imatrix"
    "bin/llama-gguf"
    "bin/llama-export-lora"
    "bin/llama-perplexity"
)

# On Windows, binaries have .exe extension
if [ "$UNAME_S" = "MINGW64_NT" ] || [ "$UNAME_S" = "MSYS" ]; then
    TOOLS=("${TOOLS[@]/.exe}")
    TOOLS=("${TOOLS[@]/%/\.exe}")
fi

echo "Extracting compiled tools to $OUTPUT_DIR..."

# Copy found tools, renaming llama-* → guido-* so Guido's bundled server is
# distinguishable from any system-installed llama-server the user may have.
FOUND_COUNT=0
for TOOL in "${TOOLS[@]}"; do
    if [ -f "$TOOL" ]; then
        ORIGINAL_NAME="$(basename "$TOOL")"
        RENAMED="${ORIGINAL_NAME/llama-/guido-}"
        cp "$TOOL" "$OUTPUT_DIR/$RENAMED"
        echo "  ✓ $ORIGINAL_NAME → $RENAMED"
        ((FOUND_COUNT++))
    fi
done

if [ $FOUND_COUNT -eq 0 ]; then
    echo "WARNING: No tools found in build output"
    echo "Build output directory contents:"
    ls -la bin/ 2>/dev/null || echo "  (bin directory not found)"
    exit 1
fi

echo ""
echo "✓ Build complete! Extracted $FOUND_COUNT C++ tools"
echo ""

# Create Python script wrappers
echo "Creating Python script wrappers..."
"$PY_WRAPPER_SCRIPT" "$LLAMA_SRC_DIR" "$OUTPUT_DIR"

echo ""
echo "Available tools:"
ls -1 "$OUTPUT_DIR" | sed 's/^/  /'
