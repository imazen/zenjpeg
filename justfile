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

# Benchmark fused vs batched pipeline (cache experiment)
bench-batched:
    cargo run --release -p zenjpeg --features test-utils --example bench_batched_pipeline

# Benchmark batched with custom PPM and iterations
bench-batched-ppm PPM ITERATIONS="10":
    cargo run --release -p zenjpeg --features test-utils --example bench_batched_pipeline -- --ppm {{PPM}} -n {{ITERATIONS}}

# Cachegrind comparison of fused vs batched pipeline
cachegrind-batched PPM BATCH="8":
    cargo build --release -p zenjpeg --features test-utils --example bench_batched_pipeline
    valgrind --tool=cachegrind ./target/release/examples/bench_batched_pipeline --ppm {{PPM}} --cachegrind --batch {{BATCH}}

# Profile fused pipeline with cachegrind
cachegrind-fused PPM:
    cargo build --release -p zenjpeg --features test-utils --example profile_pipeline
    valgrind --tool=cachegrind ./target/release/examples/profile_pipeline {{PPM}} 1

# Quick parity check (10 images x 50 quality levels)
parity:
    cargo test --release -p zenjpeg --test comprehensive_cpp_comparison -- --nocapture --ignored
