# zenjpeg development commands

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

# XYB per-channel debug (compare Rust vs C++ per-row RGB differences)
xyb-debug:
    cargo run --release --example xyb_debug

# XYB visual diff (side-by-side + difference image)
xyb-diff:
    cargo run --release --example xyb_debug
    convert /tmp/xyb_debug_cpp_decoded.png /tmp/xyb_debug_rust_decoded.png \
        -compose difference -composite -evaluate multiply 10 /tmp/xyb_diff.png
    montage /tmp/xyb_debug_cpp_decoded.png /tmp/xyb_debug_rust_decoded.png /tmp/xyb_diff.png \
        -geometry +4+4 -tile 3x1 -set label '%f' /tmp/xyb_compare.png
    display /tmp/xyb_compare.png &

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

# WASM magetypes vs wide benchmark
wasm-magetypes:
    CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --wasm simd" \
    RUSTFLAGS="-C target-feature=+simd128" \
    cargo run --release -p zenjpeg --example wasm_magetypes_bench \
        --target wasm32-wasip1 --no-default-features --features "std,magetypes-simd"
