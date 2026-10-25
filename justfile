# zenjpeg development commands

# Run all 13 fuzz targets for 60 seconds each (needs nightly + cargo-fuzz).
# Keep this list in sync with zenjpeg/fuzz/Cargo.toml's [[bin]] entries —
# `cargo fuzz run` on a name with no [[bin]] fails, and the five container /
# push-decode targets were missing here until 2026-08-29.
fuzz SECONDS="60":
    cd zenjpeg && cargo +nightly fuzz run fuzz_decode -- -max_total_time={{SECONDS}} -dict=fuzz/jpeg.dict
    cd zenjpeg && cargo +nightly fuzz run fuzz_read_info -- -max_total_time={{SECONDS}} -dict=fuzz/jpeg.dict
    cd zenjpeg && cargo +nightly fuzz run fuzz_truncation -- -max_total_time={{SECONDS}} -dict=fuzz/jpeg.dict
    cd zenjpeg && cargo +nightly fuzz run fuzz_decode_limits -- -max_total_time={{SECONDS}} -dict=fuzz/jpeg.dict
    cd zenjpeg && cargo +nightly fuzz run fuzz_decode_paths -- -max_total_time={{SECONDS}} -dict=fuzz/jpeg.dict
    cd zenjpeg && cargo +nightly fuzz run fuzz_push_decode -- -max_total_time={{SECONDS}} -dict=fuzz/jpeg.dict
    cd zenjpeg && cargo +nightly fuzz run fuzz_roundtrip -- -max_total_time={{SECONDS}} -dict=fuzz/jpeg.dict
    cd zenjpeg && cargo +nightly fuzz run fuzz_encode -- -max_total_time={{SECONDS}} -dict=fuzz/jpeg.dict
    cd zenjpeg && cargo +nightly fuzz run fuzz_differential -- -max_total_time={{SECONDS}} -dict=fuzz/jpeg.dict
    cd zenjpeg && cargo +nightly fuzz run fuzz_container_marker -- -max_total_time={{SECONDS}} -dict=fuzz/jpeg.dict
    cd zenjpeg && cargo +nightly fuzz run fuzz_container_mpf -- -max_total_time={{SECONDS}} -dict=fuzz/jpeg.dict
    cd zenjpeg && cargo +nightly fuzz run fuzz_container_xmp -- -max_total_time={{SECONDS}} -dict=fuzz/jpeg.dict
    cd zenjpeg && cargo +nightly fuzz run fuzz_container_probe -- -max_total_time={{SECONDS}} -dict=fuzz/jpeg.dict

# Compile every fuzz target on stable — the same gate .github/workflows/fuzz.yml
# runs. `zenjpeg/fuzz` is its own workspace, so nothing at the repo root ever
# compiles it; this is what stops the targets bit-rotting between hand-run
# fuzzing sessions. Needs the ../zenanalyze sibling checked out.
fuzz-check:
    cd zenjpeg/fuzz && cargo check --all-targets

# Replay the committed crash seeds in zenjpeg/fuzz/regression/ on stable.
# Neither nightly nor cargo-fuzz required.
fuzz-regression:
    cargo test -p zenjpeg --test fuzz_regression -- --nocapture

# Colour-model oracles: libjpeg-turbo (Pillow), IJG djpeg and jpegli djpegli
# decode the colour-model fixtures; each must match its documented rule and
# zenjpeg's pixels (tests/colorspace_oracles.rs), including on the fused and
# threaded paths (`parallel`). Every tool is required.
colorspace-oracles DJPEGLI="internal/jpegli-cpp/build/tools/djpegli":
    ZENJPEG_DJPEGLI="$(realpath {{DJPEGLI}})" cargo test --release -p zenjpeg --features __oracle-tests,parallel --test colorspace_oracles -- --nocapture

# Path reach for tests/colorspace_paths.rs (scripts/colorspace_mutation_reach.py):
# copies the working copy's tracked files to $MUTANT_ROOT/<UTC stamp>/tree (kept,
# so every log's line numbers stay checkable), corrupts one conversion site at a
# time and writes logs, per-site diffs and results to .../out.
# `just colorspace-mutation-reach list` lists the sites; `START END` runs a range.
colorspace-mutation-reach *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    root="${MUTANT_ROOT:-$HOME/tmp/zenjpeg-mutation}/$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$root/tree/internal" "$root/out"
    if [ -d .jj ]; then jj file list; else git ls-files; fi \
        | grep -vx 'internal/jpegli-cpp' | rsync -a --files-from=- ./ "$root/tree/"
    ln -s "$(realpath internal/jpegli-cpp)" "$root/tree/internal/jpegli-cpp"
    if [ -d ../zenanalyze ]; then ln -s "$(realpath ../zenanalyze)" "$root/zenanalyze"; fi
    if [ -d .jj ]; then jj log -r @ --no-graph -T 'commit_id ++ "\n"'; else git rev-parse HEAD; fi > "$root/out/SOURCE"
    echo "tree: $root/tree  results: $root/out"
    python3 scripts/colorspace_mutation_reach.py "$root/tree" "$root/out" {{ARGS}}

# Run all lib tests
test:
    cargo test -p zenjpeg --lib

# Pre-commit: fmt + clippy + test + the two fuzz gates CI runs
ci:
    cargo fmt -p zenjpeg -- --check && cargo clippy -p zenjpeg -- -D warnings && just test && just fuzz-check && just fuzz-regression

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
    cargo test -p zenjpeg --release
    cargo test -p zenjpeg --release --features parallel
    cargo check -p zenjpeg --no-default-features --features "std,zencodec"
    cargo check -p zenjpeg --no-default-features --features "std,ultrahdr"
    cargo check -p zenjpeg --no-default-features --features "std,layout"
    cargo check -p zenjpeg --no-default-features --features "std,boundary-rd"
    cargo check -p zenjpeg --no-default-features --features "std,recompress"

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

# Pinned rustdoc-JSON nightly for the API snapshots (keep in sync with the
# public-api job in .github/workflows/ci.yml). An unpinned tracking nightly
# churns cross-crate path rendering (e.g. the 2026-07 core::io re-homing
# turned `std::io::Write` into `core::io::write::Write`), breaking the CI
# gate with zero repo changes. Bump deliberately, regenerating in the same
# commit.
apidoc_toolchain := "nightly-2026-07-13"

# Format + regenerate the public-API surface snapshots (docs/public-api/).
# The snapshot runner lives in the workspace-excluded apidoc/ package, so it
# is never built or run by plain `cargo test` or any CI job.
fmt:
    cargo fmt --all
    ZEN_API_DOC_TOOLCHAIN={{apidoc_toolchain}} cargo test --manifest-path apidoc/Cargo.toml

# Regenerate the public-API surface snapshots only
api-doc:
    ZEN_API_DOC_TOOLCHAIN={{apidoc_toolchain}} cargo test --manifest-path apidoc/Cargo.toml

# Verify the committed snapshots are current
api-doc-check:
    ZEN_API_DOC=check ZEN_API_DOC_TOOLCHAIN={{apidoc_toolchain}} cargo test --manifest-path apidoc/Cargo.toml

# Native ARM runtime-dispatch comparison; preserve full benchmark output.
arm-tiers-macos group="":
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "$HOME/tmp"
    group_args=()
    if [[ -n '{{group}}' ]]; then group_args+=("--group={{group}}"); fi
    # macOS cannot enforce a P-core process boundary; this is exploratory.
    # Local path patches can require a lock refresh before this bench builds.
    CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 TMPDIR="$HOME/tmp" nice -n 19 cargo bench -p zenjpeg --features _dev --bench tier_isolation -- "${group_args[@]}" --cpu-policy=unrestricted --format=llm > "$HOME/tmp/zenjpeg-arm-tiers.log" 2>&1

# JPEG framing and validated reassembly; existing fixtures and decoder paths.
metadata-check:
    cargo test -p zenjpeg --features extended-metadata --lib container::
    cargo test -p zenjpeg --features extended-metadata --test extended_metadata --test fuzz_regression
    cargo clippy -p zenjpeg --features extended-metadata --lib --test extended_metadata --test fuzz_regression -- -D warnings

metadata-oracle:
    cargo test -p zenjpeg --features __oracle-exiftool --test extended_metadata_exiftool -- --nocapture

metadata-inventory-unit:
    cargo test -p zenjpeg --features inventory --lib codec::inventory

metadata-inventory-check:
    cargo test -p zenjpeg --features zencodec --test inventory_feature
    cargo test -p zenjpeg --features inventory --test inventory_feature
    cargo test -p zenjpeg --features inventory --test bundled inventory_
    cargo clippy -p zenjpeg --features inventory,extended-metadata --lib --test inventory_feature --test bundled -- -D warnings

metadata-fuzz-check:
    cargo check --manifest-path zenjpeg/fuzz/Cargo.toml --bin fuzz_inventory --bin fuzz_container_metadata

metadata-inventory-oracle:
    INVENTORY_ORACLE_EXIFTOOL=exiftool cargo test -p zenjpeg --features inventory --test inventory_oracle -- --nocapture

# Reused PR210 filtering tests, independent of extended inspection and inventory.
metadata-filter-check:
    cargo test -p zenjpeg --no-default-features --lib container::metadata::filter::
    cargo test -p zenjpeg --features inventory --test inventory_feature
    cargo clippy -p zenjpeg --no-default-features --lib -- -D warnings

# Force a patch comparison: the local version has already been bumped to 0.9.
metadata-api-check:
    #!/usr/bin/env bash
    set -euo pipefail
    baseline=$(cargo read zenjpeg==0.8.4 --path-only)
    RUSTC_BOOTSTRAP=1 cargo rustdoc -p zenjpeg --lib --features inventory,extended-metadata,zencodec,ultrahdr -- -Zunstable-options --output-format json
    RUSTC_BOOTSTRAP=1 cargo rustdoc --locked --manifest-path "$baseline/Cargo.toml" --target-dir "$PWD/target/published-api/0.8.4" --lib --features zencodec,ultrahdr -- -Zunstable-options --output-format json
    cargo semver-checks check-release -p zenjpeg --current-rustdoc target/doc/zenjpeg.json --baseline-rustdoc target/published-api/0.8.4/doc/zenjpeg.json --release-type patch

metadata-integration-oracles:
    just metadata-oracle
    just metadata-inventory-oracle
    cargo test -p zenjpeg --features inventory,extended-metadata,ultrahdr --doc
    just api-doc

metadata-hdr-check:
    cargo test -p zenjpeg --features inventory,extended-metadata,ultrahdr --lib container::
    cargo test -p zenjpeg --features inventory,extended-metadata,ultrahdr --lib codec::inventory
    cargo test -p zenjpeg --features inventory,extended-metadata,ultrahdr --doc
    cargo clippy -p zenjpeg --features inventory,extended-metadata,ultrahdr --lib -- -D warnings
    just api-doc

metadata-hdr-feature-matrix:
    cargo clippy -p zenjpeg --lib --tests --benches --features "parallel,moxcms,ultrahdr,zencodec,__test-utils" -- -D warnings
    cargo clippy -p zenjpeg --lib --tests --features "inventory,extended-metadata,ultrahdr,zencodec,__test-utils" -- -D warnings

metadata-ci-lib:
    cargo test -p zenjpeg --lib --features "parallel,moxcms,ultrahdr,zencodec,__test-utils"
    cargo test -p zenjpeg --lib --features "inventory,extended-metadata,ultrahdr,zencodec,__test-utils"

# Populate corpus dependencies once before parallel metadata tests.
metadata-prefetch:
    cargo run -p zenjpeg --example metadata_corpus_prefetch

metadata-prefetch-check:
    just metadata-prefetch
    cargo test -p zenjpeg --features inventory,extended-metadata --test extended_metadata --test inventory_feature --test fuzz_regression

# Reconcile EXIF framing from the pending metadata source stack.
metadata-exif-check:
    cargo test -p zenjpeg --features zencodec --lib exif
    cargo test -p zenjpeg --features zencodec --test bundled metadata_integration
    cargo clippy -p zenjpeg --features zencodec --lib -- -D warnings

metadata-iso-check:
    cargo test -p zenjpeg --features inventory,extended-metadata,ultrahdr --lib ultrahdr::decode
    cargo test -p zenjpeg --features inventory,extended-metadata,ultrahdr --test bundled ultrahdr_roundtrip
    cargo test -p zenjpeg --features inventory,extended-metadata,ultrahdr --test bundled inventory_review
    cargo clippy -p zenjpeg --features inventory,extended-metadata,ultrahdr --lib -- -D warnings

metadata-rewrite-check:
    cargo test -p zenjpeg --no-default-features --features metadata-rewrite --test metadata_rewrite
    cargo clippy -p zenjpeg --no-default-features --features metadata-rewrite --lib --test metadata_rewrite -- -D warnings

metadata-rewrite-gainmap-check:
    cargo test -p zenjpeg --features metadata-rewrite,ultrahdr --test metadata_rewrite
    CARGO_TARGET_I686_UNKNOWN_LINUX_GNU_RUNNER=env cargo test -p zenjpeg --no-default-features --features metadata-rewrite --test metadata_rewrite --target i686-unknown-linux-gnu

metadata-rewrite-oracle:
    #!/usr/bin/env bash
    set -euo pipefail
    output="$HOME/tmp/zenmeta-jpeg-rewrite-oracle"
    JPEG_METADATA_OUTPUT="$output" cargo test -p zenjpeg --features metadata-rewrite,ultrahdr --test metadata_rewrite scrubs_both_components -- --nocapture
    for suffix in "" "-map"; do
        djpeg -outfile "$output/before$suffix.ppm" "$output/before$suffix.jpg"
        djpeg -outfile "$output/after$suffix.ppm" "$output/after$suffix.jpg"
        cmp "$output/before$suffix.ppm" "$output/after$suffix.ppm"
    done
    exiftool -a -validate -warning -error "$output/after.jpg"

metadata-limits-check:
    cargo test -p zenjpeg --test metadata_limits
    cargo test -p zenjpeg --features zencodec --test bundled metadata_integration
    cargo clippy -p zenjpeg --lib --features metadata-rewrite -- -D warnings

metadata-policy-check:
    cargo test -p zenjpeg --features ultrahdr --test bundled ultrahdr_roundtrip
    cargo clippy -p zenjpeg --features ultrahdr --lib -- -D warnings

metadata-dependency-check:
    cargo check -p zenjpeg --features ultrahdr --lib

metadata-gamut-check:
    cargo test -p zenjpeg --features ultrahdr --test bundled convenience_hdr_readers_retain_display_p3_primaries

metadata-lossless-check:
    cargo test -p zenjpeg --features ultrahdr --test lossless_mpf --test mpf_bounds --test lossless_coefficient_range --test lossless_edges
    cargo test -p zenjpeg --features ultrahdr --lib lossless::
    cargo clippy -p zenjpeg --features ultrahdr --lib --test lossless_mpf --test mpf_bounds -- -D warnings

metadata-lossless-oracle:
    ZENJPEG_ORACLE_TOOLS=1 cargo test -p zenjpeg --features ultrahdr --test lossless_oracle -- --test-threads=1

metadata-lossless-feature-check:
    cargo check -p zenjpeg --no-default-features
    cargo test -p zenjpeg --no-default-features --lib gain_map_preservation
    cargo test -p zenjpeg --no-default-features --lib gain_map_decode_inherits
    cargo test -p zenjpeg --features metadata-rewrite,ultrahdr --test metadata_rewrite

metadata-lossless-fuzz-check:
    cargo test -p zenjpeg --features metadata-rewrite,extended-metadata,inventory --test fuzz_regression
    cargo check --manifest-path zenjpeg/fuzz/Cargo.toml --bin fuzz_lossless --bin fuzz_metadata_rewrite

metadata-lossless-api-check:
    cargo fmt -p zenjpeg --check
    cargo clippy -p zenjpeg --no-default-features --lib -- -D warnings
    just api-doc

metadata-lossless-portability-check:
    CARGO_TARGET_I686_UNKNOWN_LINUX_GNU_RUNNER=env cargo test -p zenjpeg --target i686-unknown-linux-gnu --test lossless_mpf --test mpf_bounds --test lossless_coefficient_range --test lossless_edges
    actionlint .github/workflows/ci.yml

metadata-icc-parity-check:
    cargo test -p zenjpeg --features extended-metadata --test metadata_limits
    cargo test -p zenjpeg --features extended-metadata --test extended_metadata icc_
    cargo clippy -p zenjpeg --features extended-metadata --lib --test metadata_limits -- -D warnings
    cargo clippy -p zenjpeg --no-default-features --lib -- -D warnings

metadata-lossless-regression-check:
    just metadata-ci-lib
    cargo test -p zenjpeg --features layout,ultrahdr --test bundled layout_pipeline
    cargo test -p zenjpeg --features layout,ultrahdr --test lossless_edges

metadata-recompress-admission-check:
    cargo test -p zenjpeg --features recompress,__test-utils --test recompress_metadata --test recompress_api
    cargo clippy -p zenjpeg --features recompress,__test-utils --lib --test recompress_metadata -- -D warnings

# Approved gain-map alignment and explicit edge rejection contracts.
metadata-layout-contract-check:
    cargo test -p zenjpeg --features layout,ultrahdr --test bundled layout_pipeline
    cargo test -p zenjpeg --features layout,ultrahdr --test lossless_edges

metadata-progressive-corpus-check:
    cargo run -p zenjpeg --example metadata_corpus_prefetch
    cargo test -p zenjpeg --test bundled codec_corpus_conformance::test_image_rs_progressive -- --exact

# Default compatibility, strict rejection, bounded extraction, and rewrite refusal.
metadata-mpf-compat-check:
    cargo test -p zenjpeg --features inventory,ultrahdr --test mpf_bounds --test lossless_mpf
    cargo test -p zenjpeg --features inventory,ultrahdr --test bundled inventory_review
    cargo clippy -p zenjpeg --features inventory,ultrahdr --lib --test mpf_bounds --test bundled -- -D warnings

metadata-final-regression-check:
    just metadata-mpf-compat-check
    cargo test -p zenjpeg --features parallel,moxcms,ultrahdr,zencodec,inventory,extended-metadata,metadata-rewrite --lib --test bundled
    just metadata-lossless-oracle
    just metadata-lossless-portability-check
    just metadata-lossless-feature-check
    just metadata-icc-parity-check
    cargo fmt -p zenjpeg --check
    just api-doc

metadata-depth-bounds-check:
    cargo test -p zenjpeg --lib decode::depth::tests
    cargo test -p zenjpeg --test mpf_bounds
    CARGO_TARGET_I686_UNKNOWN_LINUX_GNU_RUNNER=env cargo test -p zenjpeg --target i686-unknown-linux-gnu --test mpf_bounds native_mpf_offsets_do_not_wrap
    cargo clippy -p zenjpeg --lib --test mpf_bounds -- -D warnings

metadata-admission-check:
    cargo test -p zenjpeg --features inventory,ultrahdr --lib codec::inventory
    cargo test -p zenjpeg --features inventory,ultrahdr --lib ultrahdr::decode
    cargo test -p zenjpeg --features inventory,ultrahdr --test bundled inventory_review
    cargo clippy -p zenjpeg --features inventory,ultrahdr --lib --test bundled -- -D warnings

metadata-orientation-check:
    just metadata-admission-check
    cargo test -p zenjpeg --features inventory,ultrahdr --test bundled orientation_descriptor
    cargo test -p zenjpeg --features inventory,ultrahdr --test bundled inventory_corpus
    cargo test -p zenjpeg --lib container::marker

metadata-extraction-policy-check:
    cargo test -p zenjpeg --features inventory,ultrahdr --lib metadata_policy_agrees
    just metadata-admission-check
    cargo test -p zenjpeg --features inventory,ultrahdr --test bundled orientation_descriptor

metadata-hdr-dispatch-check:
    cargo test -p zenjpeg --features inventory,ultrahdr --lib ultrahdr::decode::
    cargo test -p zenjpeg --features inventory,ultrahdr --test bundled codec_hdr_
    cargo test -p zenjpeg --features inventory,ultrahdr --test bundled inventory_review
    cargo test -p zenjpeg --features inventory,ultrahdr --test bundled orientation_descriptor
    cargo clippy -p zenjpeg --features inventory,ultrahdr --lib -- -D warnings

# Continue the final gate after the combined library/bundled suite.
metadata-final-oracle-check:
    just metadata-lossless-oracle
    just metadata-lossless-portability-check
    just metadata-lossless-feature-check
    just metadata-icc-parity-check
    cargo fmt -p zenjpeg --check
    just api-doc

metadata-legacy-icc-check:
    cargo test -p zenjpeg --features inventory --lib container::chunks::
    cargo test -p zenjpeg --features extended-metadata,metadata-rewrite,inventory --test metadata_limits
    just metadata-icc-parity-check
    cargo clippy -p zenjpeg --features extended-metadata,inventory --lib -- -D warnings

# Typed rewrite migration plus the gain-map carrier and EXIF compatibility paths.
shared-contract-check:
    cargo test -p zenjpeg --features zencodec,inventory,metadata-rewrite --test shared_contracts
    cargo test -p zenjpeg --features zencodec,metadata-rewrite --test shared_contracts
    cargo clippy -p zenjpeg --features zencodec,inventory,metadata-rewrite --test shared_contracts -- -D warnings
    just metadata-rewrite-check metadata-rewrite-gainmap-check
    cargo clippy -p zenjpeg --features metadata-rewrite,ultrahdr,zencodec --lib -- -D warnings

# Include native EXIF helper tests and the independent fuzz dependency graph.
shared-contract-ci-check:
    cargo test -p zenjpeg --lib encode::exif::tests
    cargo clippy -p zenjpeg --lib --tests --benches --features parallel,moxcms,ultrahdr,zencodec,__test-utils -- -D warnings
    cargo check --manifest-path zenjpeg/fuzz/Cargo.toml --bins

# Refresh the shared codec/metadata interface in both standalone lockfiles.
sync-interface-locks:
    cargo update -p zencodec
    cargo update --manifest-path zenjpeg/fuzz/Cargo.toml -p zencodec

# Check the consumer-facing feature set after an interface revision change.
check-interface-pin:
    cargo clippy --locked -p zenjpeg --lib --features parallel,moxcms,ultrahdr,zencodec,inventory,metadata-rewrite -- -D warnings

# Native settings must also survive the adapter with generic parallel defaults.
test-native-configuration:
    cargo test --locked -p zenjpeg --features parallel,moxcms,ultrahdr,zencodec,__test-utils --test streaming_encode

check-native-configuration:
    cargo clippy --locked -p zenjpeg --features parallel,moxcms,ultrahdr,zencodec --lib -- -D warnings
    cargo fmt -p zenjpeg --check

# Development-only causal profile and checked external-process worker.
coz-build:
    cargo build --release --locked -p coz-loop

coz-check:
    cargo fmt -p coz-loop -- --check
    cargo clippy --locked -p coz-loop --all-targets -- -D warnings

coz-worker +args:
    ./target/release/coz-loop {{args}}

# Optional output is caller-selected; the test always executes the full matrix.
coz-fixtures:
    test -n "$ZENJPEG_DC_FIXTURE_DIR"
    ZENJPEG_DC_BUILD_COMMIT="$(jj log -r "latest(ancestors(@) ~ empty())" --no-graph -T commit_id)" cargo test --locked -p zenjpeg --lib lossless::tests::dc_only_fixture_matrix -- --exact

# Independent readback; requires ImageMagick and never skips missing tools.
coz-fixtures-oracle:
    test -n "$ZENJPEG_DC_FIXTURE_DIR"
    python3 internal/coz-loop/check-fixtures.py "$ZENJPEG_DC_FIXTURE_DIR"

# Check compatibility after admitting the current archmage/magetypes beta.
beta-cohort-check:
    cargo fmt --check
    cargo clippy --locked --workspace --lib -- -D warnings
    cargo test --locked --workspace --lib --no-fail-fast

# Constructor spellings changed in examples and benches as well as the library.
beta-example-check:
    cargo check --locked --workspace --all-targets

# Header, native decode and inspection must agree on ICC chunk completeness.
metadata-header-check:
    cargo test -p zenjpeg --features zencodec,extended-metadata --test metadata_limits
    cargo clippy -p zenjpeg --features zencodec,extended-metadata --lib --test metadata_limits -- -D warnings
