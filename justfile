# jpegli-rs development commands

# Default test image for profiling
TEST_IMAGE := env_var_or_default("TEST_IMAGE", "~/work/codec-eval/codec-corpus/cid22/artificial.png")

# Profile with cjpegli-compatible CLI (progressive, d1.0, 444, 50 iterations)
profile IMAGE=TEST_IMAGE:
    cargo run --release -p jpegli-rs --example profile_50 -- {{IMAGE}} --num_reps 50

# Profile with flamegraph (progressive, d1.0, 444, 50 iterations)
flamegraph IMAGE=TEST_IMAGE:
    cargo flamegraph --release -p jpegli-rs --example profile_50 -- {{IMAGE}} --num_reps 50
    perf report --stdio --no-children -g none --percent-limit 0.5 2>/dev/null

# Profile 4K synthetic (sequential, q90, 4:2:0, 10 iterations) - old style
profile-4k:
    cargo flamegraph --release -p jpegli-rs --example flamegraph_profile -- 4k
    perf report --stdio --no-children -g none --percent-limit 0.5 2>/dev/null

# Profile 8K synthetic (sequential, q90, 4:2:0, 3 iterations) - old style
profile-8k:
    cargo flamegraph --release -p jpegli-rs --example flamegraph_profile -- 8k
    perf report --stdio --no-children -g none --percent-limit 0.5 2>/dev/null

# Profile with just perf report (no flamegraph regeneration)
profile-report:
    perf report --stdio --no-children -g none --percent-limit 0.5 2>/dev/null

# Run aq_simd benchmark (wide vs archmage comparison)
bench-aq:
    cargo bench -p jpegli-rs --bench aq_simd --features "archmage-simd,test-utils"

# Run C++ comparison benchmark
bench-cpp:
    cargo bench -p jpegli-rs --bench cpp_comparison

# Quick parity check (10 images x 50 quality levels)
parity:
    cargo test --release -p jpegli-rs --test comprehensive_cpp_comparison -- --nocapture --ignored
