#!/bin/bash
#
# Setup script for zenjpeg FFI tests
#
# This script:
#   1. Initializes the jpegli-cpp git submodule
#   2. Builds the C++ jpegli library
#   3. Enables the ffi-tests feature in Cargo.toml
#   4. Runs the FFI comparison tests
#
# Usage:
#   ./internal/setup-ffi-tests.sh [--build-only] [--test-only]
#
# Options:
#   --build-only    Only build C++, don't modify Cargo.toml or run tests
#   --test-only     Skip C++ build, just run tests (assumes already built)
#   --clean         Clean C++ build directory and rebuild
#   --jobs N        Number of parallel build jobs (default: auto)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
JPEGLI_CPP="$SCRIPT_DIR/jpegli-cpp"
BUILD_DIR="$JPEGLI_CPP/build"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Parse arguments
BUILD_ONLY=false
TEST_ONLY=false
CLEAN=false
JOBS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only) BUILD_ONLY=true; shift ;;
        --test-only) TEST_ONLY=true; shift ;;
        --clean) CLEAN=true; shift ;;
        --jobs) JOBS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--build-only] [--test-only] [--clean] [--jobs N]"
            exit 0
            ;;
        *) error "Unknown option: $1" ;;
    esac
done

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."

    command -v git >/dev/null 2>&1 || error "git is required"
    command -v cmake >/dev/null 2>&1 || error "cmake is required"
    command -v cargo >/dev/null 2>&1 || error "cargo is required"

    # Check for ninja or make
    if command -v ninja >/dev/null 2>&1; then
        CMAKE_GENERATOR="Ninja"
    elif command -v make >/dev/null 2>&1; then
        CMAKE_GENERATOR="Unix Makefiles"
    else
        error "ninja or make is required"
    fi

    info "Using CMake generator: $CMAKE_GENERATOR"
}

# Initialize git submodule
init_submodule() {
    info "Initializing jpegli-cpp submodule..."

    cd "$REPO_ROOT"

    if [ ! -f "$JPEGLI_CPP/CMakeLists.txt" ]; then
        git submodule update --init --recursive internal/jpegli-cpp
    else
        info "Submodule already initialized"
    fi

    if [ ! -f "$JPEGLI_CPP/CMakeLists.txt" ]; then
        error "Failed to initialize submodule. Check your git configuration."
    fi
}

# Build C++ jpegli
build_cpp() {
    info "Building C++ jpegli..."

    if [ "$CLEAN" = true ] && [ -d "$BUILD_DIR" ]; then
        info "Cleaning build directory..."
        rm -rf "$BUILD_DIR"
    fi

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # Configure
    info "Configuring CMake..."
    cmake -G "$CMAKE_GENERATOR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DJPEGXL_ENABLE_TOOLS=ON \
        -DJPEGXL_ENABLE_JPEGLI_LIBJPEG=ON \
        -DJPEGXL_ENABLE_SJPEG=OFF \
        -DJPEGXL_ENABLE_OPENEXR=OFF \
        -DJPEGXL_ENABLE_SKCMS=OFF \
        -DJPEGXL_STATIC=ON \
        ..

    # Build
    info "Building (this may take 5-10 minutes)..."
    BUILD_CMD="cmake --build . --target jpegli-static --target cjpegli"
    if [ -n "$JOBS" ]; then
        BUILD_CMD="$BUILD_CMD --parallel $JOBS"
    else
        BUILD_CMD="$BUILD_CMD --parallel"
    fi
    eval $BUILD_CMD

    # Verify build
    if [ ! -f "lib/libjpegli-static.a" ] && [ ! -f "lib/jpegli-static.lib" ]; then
        error "Build failed - libjpegli-static not found"
    fi

    info "C++ build complete!"
}

# Enable ffi-tests feature
enable_ffi_feature() {
    info "Enabling ffi-tests feature..."

    CARGO_TOML="$REPO_ROOT/zenjpeg/Cargo.toml"

    # Check if already enabled
    if grep -q '^jpegli-internals-sys = { path' "$CARGO_TOML"; then
        info "ffi-tests feature already enabled"
        return
    fi

    # Uncomment the dependency
    sed -i.bak 's|^# jpegli-internals-sys = { path|jpegli-internals-sys = { path|' "$CARGO_TOML"

    # Update the feature
    sed -i.bak 's|^ffi-tests = \[\]|ffi-tests = ["dep:jpegli-internals-sys"]|' "$CARGO_TOML"

    # Clean up backup
    rm -f "$CARGO_TOML.bak"

    info "ffi-tests feature enabled in Cargo.toml"
    warn "Remember to revert before committing if you don't want local changes!"
}

# Run FFI tests
run_tests() {
    info "Running FFI comparison tests..."

    cd "$REPO_ROOT"

    # Build with ffi-tests feature
    cargo build --features ffi-tests -p zenjpeg

    # Run tests
    cargo test --features ffi-tests -p zenjpeg -- --nocapture

    info "FFI tests complete!"
}

# Main
main() {
    echo "============================================"
    echo "  zenjpeg FFI Test Setup"
    echo "============================================"
    echo

    check_prerequisites

    if [ "$TEST_ONLY" = false ]; then
        init_submodule
        build_cpp
    fi

    if [ "$BUILD_ONLY" = false ]; then
        enable_ffi_feature
        run_tests
    fi

    echo
    echo "============================================"
    info "Setup complete!"
    echo
    echo "To run FFI tests again:"
    echo "  cargo test --features ffi-tests -p zenjpeg"
    echo
    echo "To generate C++ test data:"
    echo "  cd internal/jpegli-cpp/build"
    echo "  GENERATE_RUST_TEST_DATA=1 ./tools/cjpegli input.png output.jpg"
    echo "============================================"
}

main "$@"
