#!/bin/bash
# Compare AQ maps between Rust jpegli-rs and C++ cjpegli
# Usage: ./compare_aq_maps.sh <input.png> [quality]

set -e

INPUT="${1:-internal/jpegli-cpp/testdata/jxl/flower/flower.png}"
QUALITY="${2:-75}"

CPP_AQ="/tmp/cpp_aq.bin"
RUST_AQ="/tmp/rust_aq.bin"
CPP_OUT="/tmp/cpp_out.jpg"
RUST_OUT="/tmp/rust_out.jpg"

echo "=== AQ Map Comparison ==="
echo "Input: $INPUT"
echo "Quality: $QUALITY"
echo

# Encode with C++ jpegli (dumps AQ map)
echo "Encoding with C++ cjpegli..."
DUMP_AQ_MAP="$CPP_AQ" ./internal/jpegli-cpp/build/tools/cjpegli \
    "$INPUT" "$CPP_OUT" -q "$QUALITY" 2>&1 | grep -v "^$" || true

# Encode with Rust jpegli-rs (dumps AQ map)
echo "Encoding with Rust jpegli-rs..."
DUMP_AQ_MAP="$RUST_AQ" cargo run --release -p jpegli-rs --example encode_simple -- \
    "$INPUT" "$RUST_OUT" "$QUALITY" 2>&1 | grep -v "^$" || true

echo
echo "=== Comparing AQ maps ==="

# Compare file sizes
if [ -f "$CPP_AQ" ] && [ -f "$RUST_AQ" ]; then
    CPP_SIZE=$(stat -c%s "$CPP_AQ")
    RUST_SIZE=$(stat -c%s "$RUST_AQ")
    echo "C++ AQ file: $CPP_SIZE bytes"
    echo "Rust AQ file: $RUST_SIZE bytes"

    # Read headers
    echo
    echo "Headers (width_blocks, height_blocks):"
    echo -n "C++:  "
    xxd -l 8 -e "$CPP_AQ" | awk '{print $2, $3}'
    echo -n "Rust: "
    xxd -l 8 -e "$RUST_AQ" | awk '{print $2, $3}'

    # Compare with od/hexdump
    echo
    echo "First 20 AQ values (after 8-byte header):"
    echo -n "C++:  "
    od -A n -t f4 -j 8 -N 80 "$CPP_AQ" | head -1
    echo -n "Rust: "
    od -A n -t f4 -j 8 -N 80 "$RUST_AQ" | head -1
else
    echo "ERROR: AQ map files not found!"
    [ ! -f "$CPP_AQ" ] && echo "  Missing: $CPP_AQ"
    [ ! -f "$RUST_AQ" ] && echo "  Missing: $RUST_AQ"
fi

# Compare JPEG sizes
echo
echo "=== JPEG Output Sizes ==="
if [ -f "$CPP_OUT" ]; then
    CPP_JPEG_SIZE=$(stat -c%s "$CPP_OUT")
    echo "C++ JPEG: $CPP_JPEG_SIZE bytes"
fi
if [ -f "$RUST_OUT" ]; then
    RUST_JPEG_SIZE=$(stat -c%s "$RUST_OUT")
    echo "Rust JPEG: $RUST_JPEG_SIZE bytes"
fi
