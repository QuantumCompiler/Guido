#!/bin/bash
set -e

# Create executable wrappers for Python scripts
# Usage: ./create-py-wrappers.sh <llama_cpp_dir> <output_dir>

LLAMA_CPP_DIR="${1:-./llama.cpp}"
OUTPUT_DIR="${2:-./bin/llama-cpp-tools}"

echo "Creating Python script wrappers..."
echo "  Source: $LLAMA_CPP_DIR"
echo "  Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# List of Python scripts to wrap
SCRIPTS=(
    "convert_hf_to_gguf.py:convert-hf-to-gguf"
    "convert_lora_to_gguf.py:convert-lora-to-gguf"
    "convert_llama_ggml_to_gguf.py:convert-llama-ggml-to-gguf"
    "convert_hf_to_gguf_update.py:convert-hf-to-gguf-update"
)

for SCRIPT_PAIR in "${SCRIPTS[@]}"; do
    SCRIPT_NAME="${SCRIPT_PAIR%:*}"
    EXECUTABLE_NAME="${SCRIPT_PAIR#*:}"
    SCRIPT_PATH="$LLAMA_CPP_DIR/$SCRIPT_NAME"
    WRAPPER_PATH="$OUTPUT_DIR/$EXECUTABLE_NAME"

    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "  ⚠ Skipping $SCRIPT_NAME (not found)"
        continue
    fi

    # Create wrapper script
    cat > "$WRAPPER_PATH" << 'WRAPPER_TEMPLATE'
#!/bin/bash
# Auto-generated wrapper for llama.cpp Python script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$(cd "$SCRIPT_DIR/../../llama.cpp" && pwd)"
SCRIPT_NAME="PLACEHOLDER_SCRIPT"

exec python3 "$LLAMA_DIR/$SCRIPT_NAME" "$@"
WRAPPER_TEMPLATE

    # Replace placeholder with actual script name
    sed -i.bak "s|PLACEHOLDER_SCRIPT|$SCRIPT_NAME|g" "$WRAPPER_PATH"
    rm -f "$WRAPPER_PATH.bak"

    # Make executable
    chmod +x "$WRAPPER_PATH"

    echo "  ✓ $EXECUTABLE_NAME"
done

echo ""
echo "✓ Python wrappers created in $OUTPUT_DIR/"
