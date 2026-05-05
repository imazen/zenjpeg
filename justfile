# zenjpeg development commands

# Run all fuzz targets for 60 seconds each
fuzz SECONDS="60":
    cd zenjpeg && cargo +nightly fuzz run fuzz_decode -- -max_total_time={{SECONDS}} -dict=fuzz/jpeg.dict
    cd zenjpeg && cargo +nightly fuzz run fuzz_read_info -- -max_total_time={{SECONDS}} -dict=fuzz/jpeg.dict
    cd zenjpeg && cargo +nightly fuzz run fuzz_decode_limits -- -max_total_time={{SECONDS}} -dict=fuzz/jpeg.dict
    cd zenjpeg && cargo +nightly fuzz run fuzz_decode_paths -- -max_total_time={{SECONDS}} -dict=fuzz/jpeg.dict
    cd zenjpeg && cargo +nightly fuzz run fuzz_roundtrip -- -max_total_time={{SECONDS}} -dict=fuzz/jpeg.dict
    cd zenjpeg && cargo +nightly fuzz run fuzz_encode -- -max_total_time={{SECONDS}} -dict=fuzz/jpeg.dict
    cd zenjpeg && cargo +nightly fuzz run fuzz_differential -- -max_total_time={{SECONDS}} -dict=fuzz/jpeg.dict

# Run all lib tests
test:
    cargo test -p zenjpeg --lib

# Pre-commit: fmt + clippy + test
ci:
    cargo fmt -p zenjpeg -- --check && cargo clippy -p zenjpeg -- -D warnings && just test

# Default test image for profiling
TEST_IMAGE := env_var_or_default("TEST_IMAGE", "~/work/codec-eval/codec-corpus/CID22/CID22-512/validation/1025469.png")

# Profile with cjpegli-compatible CLI (progressive, d1.0, 444, 50 iterations)
profile IMAGE=TEST_IMAGE:
    cargo run --release -p zenjpeg --example cjpegli_rs_profile -- {{IMAGE}} --num_reps 50

# Profile with flamegraph (progressive, d1.0, 444, 50 iterations)
flamegraph IMAGE=TEST_IMAGE:
    cargo flamegraph --release -p zenjpeg --example cjpegli_rs_profile -- {{IMAGE}} --num_reps 50
    perf report --stdio --no-children -g none --percent-limit 0.5 2>/dev/null

# Profile 4K synthetic (sequential, q90, 4:2:0, 10 iterations) - old style
profile-4k:
    cargo flamegraph --release -p zenjpeg --example flamegraph_profile -- 4k
    perf report --stdio --no-children -g none --percent-limit 0.5 2>/dev/null

# Profile 8K synthetic (sequential, q90, 4:2:0, 3 iterations) - old style
profile-8k:
    cargo flamegraph --release -p zenjpeg --example flamegraph_profile -- 8k
    perf report --stdio --no-children -g none --percent-limit 0.5 2>/dev/null

# Profile with just perf report (no flamegraph regeneration)
profile-report:
    perf report --stdio --no-children -g none --percent-limit 0.5 2>/dev/null

# Run aq_simd benchmark (wide vs archmage comparison)
bench-aq:
    cargo bench -p zenjpeg --bench aq_simd --features "archmage-simd,test-utils"

# Run C++ comparison benchmark
bench-cpp:
    cargo bench -p zenjpeg --bench cpp_comparison

# Quick parity check (10 images x 50 quality levels)
parity:
    cargo test --release -p zenjpeg --test comprehensive_cpp_comparison -- --nocapture --ignored

# Default image for visual comparisons
COMPARE_IMAGE := env_var_or_default("COMPARE_IMAGE", "~/work/codec-eval/codec-corpus/kodak/1.png")

# XYB per-channel debug (compare Rust vs C++ per-row RGB differences)
xyb-debug IMAGE=COMPARE_IMAGE:
    cargo run --release --example xyb_debug -- {{IMAGE}}

# XYB visual diff (C++ | Rust | R diff | G diff | B diff)
xyb-diff IMAGE=COMPARE_IMAGE:
    cargo run --release --example xyb_debug -- {{IMAGE}}
    @# Extract per-channel differences (amplified 10x, shown as grayscale)
    convert /tmp/xyb_debug_cpp_decoded.png /tmp/xyb_debug_rust_decoded.png \
        -compose difference -composite -channel R -separate -evaluate multiply 10 /tmp/xyb_diff_r.png
    convert /tmp/xyb_debug_cpp_decoded.png /tmp/xyb_debug_rust_decoded.png \
        -compose difference -composite -channel G -separate -evaluate multiply 10 /tmp/xyb_diff_g.png
    convert /tmp/xyb_debug_cpp_decoded.png /tmp/xyb_debug_rust_decoded.png \
        -compose difference -composite -channel B -separate -evaluate multiply 10 /tmp/xyb_diff_b.png
    montage /tmp/xyb_debug_cpp_decoded.png /tmp/xyb_debug_rust_decoded.png \
        /tmp/xyb_diff_r.png /tmp/xyb_diff_g.png /tmp/xyb_diff_b.png \
        -geometry +2+2 -tile 5x1 -font Helvetica -pointsize 12 \
        -label "C++" -label "Rust" -label "ΔR ×10" -label "ΔG ×10" -label "ΔB ×10" \
        /tmp/xyb_compare.png
    display /tmp/xyb_compare.png &

# YCbCr visual diff (C++ | Rust | R diff | G diff | B diff)
ycbcr-diff IMAGE=COMPARE_IMAGE:
    cargo run --release --example ycbcr_debug -- {{IMAGE}}
    @# Extract per-channel differences (amplified 10x, shown as grayscale)
    convert /tmp/ycbcr_debug_cpp_decoded.png /tmp/ycbcr_debug_rust_decoded.png \
        -compose difference -composite -channel R -separate -evaluate multiply 10 /tmp/ycbcr_diff_r.png
    convert /tmp/ycbcr_debug_cpp_decoded.png /tmp/ycbcr_debug_rust_decoded.png \
        -compose difference -composite -channel G -separate -evaluate multiply 10 /tmp/ycbcr_diff_g.png
    convert /tmp/ycbcr_debug_cpp_decoded.png /tmp/ycbcr_debug_rust_decoded.png \
        -compose difference -composite -channel B -separate -evaluate multiply 10 /tmp/ycbcr_diff_b.png
    montage /tmp/ycbcr_debug_cpp_decoded.png /tmp/ycbcr_debug_rust_decoded.png \
        /tmp/ycbcr_diff_r.png /tmp/ycbcr_diff_g.png /tmp/ycbcr_diff_b.png \
        -geometry +2+2 -tile 5x1 -font Helvetica -pointsize 12 \
        -label "C++" -label "Rust" -label "ΔR ×10" -label "ΔG ×10" -label "ΔB ×10" \
        /tmp/ycbcr_compare.png
    display /tmp/ycbcr_compare.png &

# XYB parity test (size and DSSIM comparison across quality levels)
xyb-parity:
    cargo run --release --example xyb_parity_test

# WASM SIMD128 benchmark
wasm-bench-simd:
    CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --wasm simd" \
    RUSTFLAGS="-C target-feature=+simd128" \
    cargo run --release -p zenjpeg --example wasm_bench \
        --target wasm32-wasip1 --no-default-features --features "std,decoder"

# WASM scalar (no SIMD) benchmark
wasm-bench-scalar:
    CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime" \
    cargo run --release -p zenjpeg --example wasm_bench \
        --target wasm32-wasip1 --no-default-features --features "std,decoder"

# Run both WASM benchmarks for comparison
wasm-bench:
    @echo "=== WASM SIMD128 ===" && just wasm-bench-simd
    @echo ""
    @echo "=== WASM Scalar ===" && just wasm-bench-scalar

# WASM transpose benchmark (SIMD intrinsics vs wide crate)
wasm-transpose:
    CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --wasm simd" \
    RUSTFLAGS="-C target-feature=+simd128" \
    cargo run --release -p zenjpeg --example wasm_simd_transpose \
        --target wasm32-wasip1 --no-default-features --features std

# WASM DCT benchmark
wasm-dct:
    CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --wasm simd" \
    RUSTFLAGS="-C target-feature=+simd128" \
    cargo run --release -p zenjpeg --example wasm_dct_bench \
        --target wasm32-wasip1 --no-default-features --features std

# SSIM2 Pareto sweep: find zenjpeg configs that beat C++ jpegli 444 per RDKnee angular bucket
ssim2-pareto *ARGS:
    cargo run --release -p zenjpeg --example ssim2_pareto_sweep -- {{ARGS}}

# Gather Huffman frequencies from codec-corpus (all modes × all qualities)
gather-huffman-freq:
    cargo run --release -p zenjpeg --example gather_corpus_frequencies

# WASM magetypes vs wide benchmark
wasm-magetypes:
    CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --wasm simd" \
    RUSTFLAGS="-C target-feature=+simd128" \
    cargo run --release -p zenjpeg --example wasm_magetypes_bench \
        --target wasm32-wasip1 --no-default-features --features "std,magetypes-simd"

# Check all locally-resolvable feature permutations
feature-check:
    cargo test -p zenjpeg --release --no-default-features --features std
    cargo test -p zenjpeg --release --features decoder
    cargo test -p zenjpeg --release --features "decoder,parallel"
    cargo test -p zenjpeg --release --features "decoder,sharp-yuv"
    cargo test -p zenjpeg --release --features "decoder,trellis"
    cargo test -p zenjpeg --release --features "decoder,mozjpeg-tables"
    cargo test -p zenjpeg --release --features "decoder,optimized-tables"
    cargo test -p zenjpeg --release --features "decoder,parallel,trellis"
    cargo check -p zenjpeg --no-default-features --features "std,zencodec"
    cargo check -p zenjpeg --no-default-features --features "std,ultrahdr"
    cargo check -p zenjpeg --no-default-features --features "std,layout"

# Cross-compile and test via QEMU user-mode emulation.
# Strips jpegli-internals-sys (C++ FFI) from the workspace since it can't cross-compile,
# then restores Cargo.toml after the test. Uses cargo directly with qemu runners from
# ~/.cargo/config.toml (no Docker container, so sibling path deps resolve normally).

# Helper: strip C++ FFI members/deps that can't cross-compile
_strip-cpp-ffi:
    #!/usr/bin/env bash
    # From workspace Cargo.toml: remove jpegli-internals-sys and zjpeg members
    python3 -c "
    import re
    t = open('Cargo.toml').read()
    t = re.sub(r'\s*\"internal/jpegli-internals-sys\",?', '', t)
    t = re.sub(r'\s*\"zjpeg\",?', '', t)
    open('Cargo.toml', 'w').write(t)
    "
    # From zenjpeg/Cargo.toml: remove jpegli-internals-sys dep and cjpegli-ffi feature ref
    python3 -c "
    import re
    t = open('zenjpeg/Cargo.toml').read()
    t = re.sub(r'jpegli-internals-sys[^\n]*\n', '', t)
    t = re.sub(r', features = \[\"cjpegli-ffi\"\]', '', t)
    open('zenjpeg/Cargo.toml', 'w').write(t)
    "

# Test on i686 (32-bit x86) — catches pointer-width bugs, WASM-relevant
test-i686:
    #!/usr/bin/env bash
    set -euo pipefail
    cp Cargo.toml Cargo.toml.bak
    cp zenjpeg/Cargo.toml zenjpeg/Cargo.toml.bak
    trap 'mv Cargo.toml.bak Cargo.toml; mv zenjpeg/Cargo.toml.bak zenjpeg/Cargo.toml' EXIT
    just _strip-cpp-ffi
    cargo test -p zenjpeg --target i686-unknown-linux-gnu --lib --tests

# Test on aarch64 (ARM64) — catches NEON codegen, locked value divergence
test-aarch64:
    #!/usr/bin/env bash
    set -euo pipefail
    cp Cargo.toml Cargo.toml.bak
    cp zenjpeg/Cargo.toml zenjpeg/Cargo.toml.bak
    trap 'mv Cargo.toml.bak Cargo.toml; mv zenjpeg/Cargo.toml.bak zenjpeg/Cargo.toml' EXIT
    just _strip-cpp-ffi
    cargo test -p zenjpeg --target aarch64-unknown-linux-gnu --lib --tests

# Run all cross-compilation targets
test-cross:
    just test-i686
    just test-aarch64

# ─────────────────────────────────────────────────────────────────────
# zenjpeg-diagnostics-viewer (UNSTABLE __diagnostics surface + viewer)
# ─────────────────────────────────────────────────────────────────────

# Default chain: rust unit + integration tests, wasm build, e2e suite.
diagnostics-all: diagnostics-test diagnostics-wasm diagnostics-e2e

# Rust unit + integration tests for the __diagnostics feature.
diagnostics-test:
    cargo test -p zenjpeg --features __diagnostics,trellis --test diagnostics_smoke
    cargo test -p zenjpeg --features __diagnostics --lib encode::diagnostics
    cargo test -p zenjpeg-diagnostics-wasm

# Build the wasm bindings via wasm-pack for the web demo.
diagnostics-wasm:
    cd zenjpeg-diagnostics-viewer/wasm && wasm-pack build --target web --out-dir ../web/wasm-pkg --release

# Build the demo viewer once.
diagnostics-viewer-build: diagnostics-wasm
    cd zenjpeg-diagnostics-viewer/web && npm install && npx vite build

# Serve the demo viewer locally (vite preview, port 3173).
diagnostics-viewer: diagnostics-viewer-build
    cd zenjpeg-diagnostics-viewer/web && npx vite preview --port 3173 --strictPort

# Run the Playwright E2E suite (headless chromium).
diagnostics-e2e: diagnostics-viewer-build
    cd zenjpeg-diagnostics-viewer/web && npx playwright install chromium
    cd zenjpeg-diagnostics-viewer/web && npx playwright test

# Typecheck the web app without running it.
diagnostics-typecheck:
    cd zenjpeg-diagnostics-viewer/web && npm install && npx tsc --noEmit

# Build everything (rust + wasm + viewer) without running tests.
diagnostics-build:
    cargo build -p zenjpeg --features __diagnostics,trellis
    cargo build -p zenjpeg-diagnostics-wasm
    cd zenjpeg-diagnostics-viewer/wasm && wasm-pack build --target web --out-dir ../web/wasm-pkg --release
    cd zenjpeg-diagnostics-viewer/web && npm install && npx vite build
