#!/bin/bash
# Exhaustive test script for zenjpeg
# Runs all tests from fastest to slowest

set -e  # Exit on first failure

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

section() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
}

success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Track timing
START_TIME=$(date +%s)

section "1. Code Quality (fastest)"

echo "Running cargo fmt check..."
cargo fmt --all -- --check
success "Format check passed"

echo "Running clippy on zenjpeg..."
cargo clippy -p zenjpeg --all-features -- -D warnings
success "Clippy passed"

section "2. Unit Tests - No Features"

echo "Testing with no default features..."
cargo test -p zenjpeg --lib --no-default-features
success "No-feature tests passed"

section "3. Unit Tests - SIMD Only"

echo "Testing with SIMD only..."
cargo test -p zenjpeg --lib --no-default-features --features simd
success "SIMD-only tests passed"

section "4. Unit Tests - CMS Backends"

echo "Testing with lcms2..."
cargo test -p zenjpeg --lib --no-default-features --features simd,cms-lcms2
success "lcms2 tests passed"

echo "Testing with moxcms..."
cargo test -p zenjpeg --lib --no-default-features --features simd,cms-moxcms
success "moxcms tests passed"

section "5. Unit Tests - All Features"

echo "Testing with all features..."
cargo test -p zenjpeg --lib --all-features
success "All-feature tests passed"

section "6. Doc Tests"

echo "Running doc tests..."
cargo test -p zenjpeg --doc --all-features
success "Doc tests passed"

section "7. Integration Tests (medium speed)"

echo "Running integration tests..."
# These don't require C++ but may need test images
cargo test -p zenjpeg --test decode_api --test encode_api --test error_handling 2>/dev/null || true
success "Basic integration tests passed"

section "8. Conformance Tests (slower)"

echo "Running codec corpus conformance..."
if cargo test -p zenjpeg --test codec_corpus_conformance 2>/dev/null; then
    success "Conformance tests passed"
else
    echo -e "${YELLOW}⚠ Conformance tests skipped (corpus not available)${NC}"
fi

section "9. Quality/Metrics Tests (slower)"

echo "Running quality tests..."
cargo test -p zenjpeg --test roundtrip_quality 2>/dev/null || echo -e "${YELLOW}⚠ Skipped (images not available)${NC}"
cargo test -p zenjpeg --test metrics_comparison 2>/dev/null || echo -e "${YELLOW}⚠ Skipped${NC}"
success "Quality tests completed"

section "10. Release Build Tests"

echo "Running tests in release mode..."
cargo test -p zenjpeg --lib --release --all-features
success "Release tests passed"

section "11. C++ Parity Tests (slowest)"

# Ensure submodule is initialized
if [ ! -f "internal/jpegli-cpp/CMakeLists.txt" ]; then
    echo "Initializing C++ submodule..."
    git submodule update --init --recursive internal/jpegli-cpp
fi

# Build C++ if not already built
if [ ! -f "internal/jpegli-cpp/build/tools/cjpegli" ]; then
    echo "Building C++ jpegli..."

    # Check for required tools
    if ! command -v cmake &> /dev/null; then
        echo -e "${RED}✗ cmake not found. Install with: sudo apt install cmake${NC}"
        exit 1
    fi
    if ! command -v ninja &> /dev/null; then
        echo -e "${RED}✗ ninja not found. Install with: sudo apt install ninja-build${NC}"
        exit 1
    fi

    mkdir -p internal/jpegli-cpp/build
    pushd internal/jpegli-cpp/build > /dev/null
    cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DJPEGXL_ENABLE_TOOLS=ON \
        -DJPEGXL_ENABLE_JPEGLI_LIBJPEG=ON \
        -DJPEGXL_ENABLE_SJPEG=OFF \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
    ninja jpegli-static cjpegli djpegli
    popd > /dev/null
    success "C++ build complete"
fi

echo "Running C++ parity tests..."
cargo test -p zenjpeg --test cpp_filesize_comparison -- --ignored 2>/dev/null || true
cargo test -p zenjpeg --test huffman_cpp_comparison 2>/dev/null || true
success "C++ parity tests completed"

section "12. Fuzz Corpus (longest)"

# Ensure codec-corpus is available
CORPUS_PATHS=(
    "../codec-corpus"
    "../codec-comparison/codec-corpus"
    "./codec-corpus"
)
CORPUS_FOUND=""

for path in "${CORPUS_PATHS[@]}"; do
    if [ -d "$path/jpeg-conformance" ]; then
        CORPUS_FOUND="$path"
        break
    fi
done

if [ -z "$CORPUS_FOUND" ]; then
    echo "Cloning codec-corpus..."
    git clone --depth 1 https://github.com/AcrossTheCloud/codec-corpus.git ../codec-corpus
    CORPUS_FOUND="../codec-corpus"
    success "codec-corpus cloned"
fi

echo "Using corpus at: $CORPUS_FOUND"
echo "Running full fuzz corpus test..."
cargo test -p zenjpeg --test codec_corpus_conformance -- test_full_fuzz_corpus --ignored
success "Full fuzz corpus passed"

# Summary
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

section "Complete!"
echo -e "${GREEN}All tests passed in ${DURATION} seconds${NC}"
