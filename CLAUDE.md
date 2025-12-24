# jpegli-rs Project Handoff

## Quick Start (Ubuntu 22.04)

```bash
# Install Rust if needed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Navigate to the Rust project
cd jpegli-rs

# Build and test
cargo build
cargo test
cargo test --release  # Run with optimizations
```

## Repository Info

- **Main branch**: `main`
- **Current working branch**: `stepbystep2`
- **Rust project location**: `jpegli-rs/` subdirectory within the main jpegli repo
- **C++ source**: Root of repository (this is Google's libjxl/jpegli)

## Project Overview

This is a Rust port of **jpegli** - Google's improved JPEG encoder/decoder from the JPEG XL project. The goal is a complete, idiomatic Rust implementation with method-level accuracy parity to the C++ original.

## ⚠️ C++ cjpegli Default Settings (TARGET)

**IMPORTANT**: The `cjpegli` tool defaults differ from the low-level library defaults.

| Setting | cjpegli Tool Default | Library Default | Rust Current |
|---------|---------------------|-----------------|--------------|
| **Chroma subsampling** | **4:4:4** (no subsampling) | 4:2:0 | 4:4:4 ✓ |
| **Adaptive quantization** | **ON** | ON | Per-block ✓ |
| **Progressive level** | **2** (10+ scans) | 0 (sequential) | Not implemented ✗ |
| **Huffman optimization** | **ON** | OFF (fixed tables) | ON ✓ |
| **Quality** | **90** | 90 | 90 ✓ |

**Why this matters**: The tool overrides library defaults in `lib/extras/enc/jpegli.cc`:
```cpp
} else if (!jpeg_settings.xyb) {
  // Default is no chroma subsampling.
  cinfo.comp_info[0].h_samp_factor = 1;
  cinfo.comp_info[0].v_samp_factor = 1;
}
```

**Rust must match tool defaults, not library defaults**, since users compare against `cjpegli` output.

### Gap Analysis vs cjpegli Defaults

| Feature | Impact | Status |
|---------|--------|--------|
| Progressive level 2 | ~2-3% smaller files | ✗ Not implemented |
| Per-block adaptive quant | ~3-4% smaller files | ✓ Implemented (Rust ~5% smaller than C++) |
| Huffman optimization | ~3-4% smaller files | ✓ Implemented |
| 4:4:4 subsampling | Quality improvement | ✓ Implemented |

**Note**: With matching settings (4:4:4, AQ, sequential, fixed Huffman), Rust produces
~4.6% smaller files on average than C++ jpegli. This suggests our AQ is slightly more
aggressive, but quality (DSSIM) remains good.

## ⚠️ MANDATORY: Port Verification Rules

**These rules exist because previous work was marked "done" without actual integration or parity verification.**

### What "Ported" ACTUALLY Means

A feature is **NOT ported** until ALL of these are true:

1. **Code exists** - Rust implementation exists
2. **Matches C++ algorithm** - Not a "simplified version" or "skeleton"
3. **Integrated into encoder/decoder** - Actually USED, not just importable
4. **Produces matching output** - Verified via `cpp_filesize_comparison` test
5. **Has C++ reference test** - Uses instrumented C++ testdata for validation

### Anti-Patterns to AVOID

| ❌ DON'T | ✅ DO INSTEAD |
|----------|---------------|
| Create struct + unit tests = "done" | Verify encoder USES the struct |
| Port constants only = "done" | Port constants AND wire into pipeline |
| Make tests pass with loose thresholds | Use exact C++ values as reference |

## ⚠️ MANDATORY: Git Commit Discipline

**Uncommitted work is invisible work. It cannot be reviewed, reverted, or understood.**

### When to Commit

1. **Before trying something new** - Create a checkpoint before experiments
2. **After any working change** - Even small fixes should be committed
3. **After adding/modifying examples or tests** - These are valuable artifacts
4. **Before ending a session** - NEVER leave work uncommitted
5. **When investigation produces insights** - Commit notes/findings even if code unchanged

### cargo fmt Discipline

**ALWAYS run `cargo fmt` BEFORE making changes** - not after. This ensures:
- Formatting changes are separate from functional changes
- Reviewers can skip `style: cargo fmt` commits
- Functional changes are easy to review without noise

```bash
# At start of work session:
cargo fmt
git add -A && git commit -m "style: cargo fmt"

# Then make your functional changes
# ...

# Before committing functional changes:
cargo fmt  # Should be no-op if done correctly
git add -A && git commit -m "feat: actual change description"
```

### Commit Message Guidelines

```
<type>: <short description>

<what was done>
<what was found/learned>
<what still needs work>
```

Types: `fix`, `feat`, `refactor`, `test`, `docs`, `investigate`, `wip`

### Anti-Patterns

| ❌ DON'T | ✅ DO INSTEAD |
|----------|---------------|
| "I'll commit when it's done" | Commit incrementally as you work |
| Modify 10 files then commit once | Commit logical chunks as you go |
| Debug without committing attempts | Commit each approach tried |
| Leave examples uncommitted | Examples ARE deliverables, commit them |
| Forget to commit at end of session | Always `git status` before stopping |

## ⚠️ MANDATORY: C++ Instrumentation Preservation

**C++ instrumentation code must NEVER be removed.** It was previously lost during an upstream cherry-pick and had to be recovered from git history.

### Rules

1. **Never delete instrumentation** - Even if it seems unused
2. **Hide behind flags, don't remove** - Use `#if ENABLE_RUST_TEST_INSTRUMENTATION`
3. **Reference from Rust tests** - Every instrumented function should have a Rust test that parses its output
4. **Document in CLAUDE.md** - List all instrumented functions and their test files

### Instrumented C++ Functions

| C++ File | Function | Testdata File | Rust Test File |
|----------|----------|---------------|----------------|
| `adaptive_quantization.cc` | `PerBlockModulations()` | `PerBlockModulations.testdata` | ❌ TODO |
| `adaptive_quantization.cc` | `FuzzyErosion()` | `FuzzyErosion.testdata` | ❌ TODO |
| `adaptive_quantization.cc` | `ComputePreErosion()` | `ComputePreErosion.testdata` | ❌ TODO |
| `adaptive_quantization.cc` | `ComputeAdaptiveQuantField()` | `ComputeAdaptiveQuantField.testdata` | ❌ TODO |
| `quant.cc` | `SetQuantMatrices()` | `SetQuantMatrices.testdata` | ❌ TODO |
| `quant.cc` | `InitQuantizer()` | `InitQuantizer.testdata` | ❌ TODO |
| `huffman.cc` | `CreateHuffmanTree()` | `CreateHuffmanTree.testdata` | ✅ `tests/huffman_cpp_comparison.rs` |
| `encode.cc` | `PadInputBuffer()` | `PadInputBuffer.testdata` | ❌ TODO |

**Note:** Only Huffman test currently implemented. Others need Rust tests that parse testdata.

### Why This Matters

- Instrumentation generates `.testdata` files with intermediate values
- Rust tests parse these to verify algorithm correctness
- Without instrumentation, we can only compare final outputs (less precise)
- Recovering lost instrumentation from git history is error-prone

### If Upstream Conflicts

When cherry-picking upstream commits that touch instrumented files:
1. **Preserve instrumentation blocks** - Manually re-add if stripped
2. **Test that instrumentation still works** - `GENERATE_RUST_TEST_DATA=1 ./build/tools/cjpegli ...`
3. **Verify Rust tests still pass** - `cargo test --test huffman_cpp_comparison`

### Mandatory Verification Checklist

Before marking ANY feature as complete:

```bash
# 1. Generate C++ reference data
GENERATE_RUST_TEST_DATA=1 ./build/tools/cjpegli input.png output.jpg

# 2. Run parity test
cargo test --test cpp_filesize_comparison -- --ignored --nocapture

# 3. Verify file sizes within 1%
# If >1% difference, feature is NOT done

# 4. Check the feature is actually USED
grep -r "feature_name" jpegli-rs/jpegli/src/encode.rs
# Must find actual usage, not just imports
```

### Definition of Done

A task can ONLY be marked `[x]` when:

1. ✅ C++ testdata exists for the feature
2. ✅ Rust implementation matches C++ testdata exactly
3. ✅ Feature is called from encoder (not dead code)
4. ✅ `cpp_filesize_comparison` test passes with <1% difference
5. ✅ No "simplified", "skeleton", or "placeholder" comments in code
6. ✅ Default behavior matches C++ defaults

### Known Anti-Pattern Examples (From This Project)

These were marked "done" but were NOT actually complete:

| Claimed Done | Reality |
|--------------|---------|
| "Port adaptive quantization" | Skeleton with made-up thresholds, NOT USED by encoder |
| "Port zero-bias tables" | Tables exist, but `quantize_block_with_zero_bias()` not called |
| "Port Layer 4: Entropy coding" | Uses FIXED standard tables, no optimization |
| "Port Huffman" | Tree building works, but no table optimization |

## Project Structure

```
jpegli-rs/
├── Cargo.toml          # Workspace root
├── jpegli/             # Main library crate
│   ├── src/
│   │   ├── lib.rs           # Module exports
│   │   ├── consts.rs        # Constants, tables, matrices
│   │   ├── types.rs         # Core types (ColorSpace, PixelFormat, etc.)
│   │   ├── huffman.rs       # Huffman coding
│   │   ├── quant.rs         # Quantization
│   │   ├── dct.rs           # Forward DCT ✓
│   │   ├── idct.rs          # Inverse DCT ✓
│   │   ├── color.rs         # RGB/YCbCr conversion
│   │   ├── xyb.rs           # XYB perceptual color space ✓
│   │   ├── butteraugli.rs   # Butteraugli quality metric (skeleton)
│   │   ├── bitstream.rs     # Bitstream I/O
│   │   ├── entropy.rs       # Entropy coding
│   │   ├── encode.rs        # Encoder pipeline
│   │   ├── decode.rs        # Decoder pipeline
│   │   ├── adaptive_quant.rs # AQ placeholder (C++ matching - NOT YET IMPLEMENTED)
│   │   ├── simplified_quant.rs # Simplified AQ (arbitrary thresholds - NOT C++)
│   │   ├── icc.rs           # ICC profile extraction and CMS integration ✓
│   │   └── error.rs         # Error types
│   ├── tests/
│   │   ├── aq_locked_tests.rs    # ⚠️ LOCKED - AQ tests that MUST NEVER be disabled
│   │   ├── roundtrip_quality.rs  # DSSIM-based quality tests
│   │   ├── pareto_front.rs       # Pareto efficiency vs mozjpeg
│   │   ├── decode_external.rs    # Decode JPEGs from other tools
│   │   ├── metrics_comparison.rs # DSSIM vs SSIMULACRA2 tests
│   │   ├── xyb_roundtrip.rs      # XYB color space tests
│   │   └── quality_mapping.rs    # mozjpeg->jpegli quality mapping
│   └── examples/
│       ├── roundtrip_corpus.rs    # Batch corpus testing
│       ├── corpus_comparison.rs   # HTML chart: jpegli vs mozjpeg
│       ├── multi_codec_comparison.rs # Compare with CID22 dataset
│       └── compare_quality.rs     # Quality comparison tool
└── jpegli-sys/         # FFI bindings (for testing)
```

## Completed Tasks (Verified)

**These have been verified against C++ output:**

- [x] Create jpegli-rs project structure with Cargo workspace
- [x] Port Layer 0: Constants, types, tables (zigzag, quant matrices, XYB params)
- [x] Port Layer 3: Bitstream I/O
- [x] **Fix DCT/IDCT scaling** - 1/8 scaling factor for JPEG compatibility
- [x] **Quantization tables** - Match C++ exactly (verified via testdata)
- [x] **Huffman tree building** - 61/61 test cases match C++ exactly
- [x] **XYB color conversion** - sRGB↔linear↔XYB roundtrip working
- [x] **ICC profile embedding** - 720-byte XYB profile matches C++
- [x] **Port DistanceToScale()** - Per-frequency non-linear quality scaling
- [x] **Port zero-bias tables** - Constants match C++, wired into encoder
- [x] **ICC profile extraction** - Decoder extracts from APP2 markers
- [x] **cpp_filesize_comparison test** - Verifies parity with C++

## Partially Complete (NOT VERIFIED - DO NOT USE)

**These exist but are NOT integrated or don't match C++:**

- [⚠️] Port Layer 1: Huffman - Tree building ✓, but NO table optimization
- [⚠️] Port Layer 4: Entropy coding - Works, but uses FIXED tables only
- [⚠️] Port Layer 5-6: Encoder pipeline - Works, but missing features below
- [⚠️] Port adaptive quantization - **USING CONSTANT aq_strength=0.08** (calibrated from C++ testdata mean)
  - `simplified_quant.rs` - Made-up algorithm (NOT C++), not used
  - `adaptive_quant.rs` - Placeholder for C++ matching implementation
  - See `docs/ADAPTIVE_QUANTIZATION.md` for detailed analysis
  - See `tests/aq_locked_tests.rs` for invariant tests (NEVER disable these)
- [⚠️] Butteraugli metric - **SKELETON ONLY** (partial implementation)
- [⚠️] XYB encoding pipeline - Infrastructure exists but incomplete

## Test Infrastructure (Verified)

- [x] DSSIM quality testing - Integrated with mozjpeg comparison
- [x] SSIMULACRA2 metric - Added via ssimulacra2 crate
- [x] Quality mapping tests - Find equivalent Q values
- [x] Pareto front validation - Compare with mozjpeg
- [x] corpus_comparison caching - Versioned JPEG cache
- [x] C++ instrumentation - Generates .testdata files

## Pending Tasks

### Priority 1: Close the File Size Gap (~3-6% with matching settings)

**Current status**: With matching settings (4:4:4, no AQ, sequential, fixed Huffman), Rust produces ~3-6% larger files than C++. This is the MINIMUM gap to close.

**Root causes identified:**
1. ⚠️ **Adaptive Quantization** - USING CONSTANT aq_strength=0.08 (from C++ testdata mean)
   - C++ file: `lib/jpegli/adaptive_quantization.cc`
   - Functions to port: `PerBlockModulations()`, `FuzzyErosion()`, `ComputePreErosion()`
   - `adaptive_quant.rs` - Placeholder for C++ matching implementation
   - `simplified_quant.rs` - Made-up algorithm (NOT used)
   - Locked tests in `aq_locked_tests.rs` ensure AQ invariants hold

2. ✗ **Huffman Table Optimization** - NOT IMPLEMENTED
   - C++ uses optimized tables built from actual coefficient statistics
   - Rust uses fixed standard Huffman tables only
   - C++ file: `lib/jpegli/huffman.cc` - table optimization section

3. ✗ **Progressive JPEG** - Returns "not implemented" error
   - C++ file: `lib/jpegli/encode.cc` - multiple scan support
   - Rust has `ScanSpec` type but no implementation

4. ? **DCT/Entropy precision** - Unknown source of ~1-2% gap
   - May be floating-point precision differences
   - May be subtle entropy coding differences

**Verification test:** `cargo test --test parity_enforcement -- --ignored --nocapture`

### Priority 2: Complete XYB Mode
XYB color conversion is **complete** in `xyb.rs`:
- [x] `srgb_to_linear` / `linear_to_srgb` - gamma conversion
- [x] `linear_rgb_to_xyb` / `xyb_to_linear_rgb` - opsin matrix + cube root
- [x] `scale_xyb` / `unscale_xyb` - jpegli scaling for JPEG
- [x] ICC profile embedding (720-byte XYB profile)
- [x] `DistanceToScale()` with `FREQUENCY_EXPONENT[64]`

**Remaining work:**
- [ ] End-to-end XYB encode/decode quality validation against C++
- [ ] Verify scaled XYB values match C++ exactly (instrumentation needed)

Test coverage:
- `tests/xyb_roundtrip.rs` - 5 tests (roundtrip, gray, buffer, encode/decode)
- `tests/xyb_cpp_comparison.rs` - 3 tests (C++ comparison, ICC embedding, values)

### Priority 3: Add SIMD Toggle Feature Flag
Make toggling SIMD on/off easy to:
- Ensure SIMD and non-SIMD produce identical images
- Max difference should be ≤1 when decoded
- Add accuracy tests comparing SIMD vs scalar

### Priority 4: Fix 4:2:0 Subsampling
Currently only 4:4:4 is supported. 4:2:0 requires:
- MCU interleaving in decoder
- Chroma upsampling
- C++ file: `lib/jpegli/decode.cc`

### Priority 5: Performance Benchmarks
- [ ] Performance benchmarks (encode/decode timing vs C++)
- [ ] Full function-level accuracy validation using C++ test data
- [x] Stage-by-stage C++ instrumentation (recovered from commit fe1e841f)
- [x] Output comparison via `xyb_cpp_comparison.rs`, `parity_enforcement.rs`

### Future: Add Fuzz Testing
Replicate fuzzing coverage from other JPEG libraries:
- libjpeg-turbo fuzz targets
- jpeg-decoder fuzzing
- zune-jpeg fuzzing
- mozjpeg fuzzing
Critical for server-side untrusted input handling.

### Future: Security Examination and Red Teaming
Required for server-side deployment:
- Memory safety review (unsafe blocks)
- Integer overflow checks
- Input validation (malformed JPEGs)
- DoS resistance (large images, many components)
- Comparison with CVEs from other JPEG libs

### Future: Set Up Test Image Submodule
- Create separate git repo for test images (size conscious)
- Add as submodule to avoid bloating main repo
- Include: gradient, photo, graphic, edge case images

## C++ Instrumentation for Rust Validation

### History
C++ instrumentation was created (commit fe1e841f) to capture intermediate values
for validating Rust implementations. It was inadvertently removed when cherry-picking
upstream clang-tidy cleanup (c9c2be2d). Instrumentation has been restored.

### Instrumented Functions
| File | Function | Test Data Type |
|------|----------|---------------|
| `adaptive_quantization.cc` | `PerBlockModulations()` | `PerBlockModulationsTest` |
| `adaptive_quantization.cc` | `FuzzyErosion()` | `FuzzyErosionTest` |
| `adaptive_quantization.cc` | `ComputePreErosion()` | TBD |
| `quant.cc` | `SetQuantMatrices()` | Quant table outputs |
| `quant.cc` | `InitQuantizer()` | Quantizer state |
| `huffman.cc` | `CreateHuffmanTree()` | `CreateHuffmanTreeTest` |
| `encode.cc` | Various | Encoding pipeline state |

### Using Instrumentation

1. Build C++ with instrumentation (enabled by default):
   ```bash
   mkdir -p build && cd build
   cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
       -DJPEGXL_ENABLE_TOOLS=ON ..
   ninja cjpegli
   ```

2. Generate test data:
   ```bash
   GENERATE_RUST_TEST_DATA=1 ./build/tools/cjpegli input.png output.jpg
   ```

3. Test data written to working directory:
   - `PerBlockModulations.testdata`
   - `FuzzyErosion.testdata`
   - `SetQuantMatrices.testdata`
   - `CreateHuffmanTree.testdata`
   - etc.

### How It Works
- `ENABLE_RUST_TEST_INSTRUMENTATION` macro (default ON in `test_data_gen.h`)
- `GENERATE_RUST_TEST_DATA=1` env var enables runtime capture
- JSON output to `*.testdata` files (one JSON object per line)
- Thread-safe via mutex

### Strategy: Fork vs Upstream
This repo is now a **fork** of Google's JPEG XL, not a downstream cherry-picker.
C++ modifications for instrumentation will be maintained independently.

## Quality Metrics

### ⚠️ MANDATORY: Use DSSIM, NOT PSNR

**PSNR is banned for new tests.** It's statistically misleading and doesn't correlate well with perceptual quality.

| ❌ DON'T | ✅ DO INSTEAD |
|----------|---------------|
| `PSNR = 10 * log10(255^2 / MSE)` | `dssim::Dssim::new().compare()` |
| "PSNR is 41.8 dB" | "DSSIM is 0.00123" |
| Compare PSNR values | Compare DSSIM values |

### Available Metrics

| Metric | Crate | Description | Range | Use For |
|--------|-------|-------------|-------|---------|
| DSSIM | `dssim` | Structural dissimilarity | 0 = identical, lower = better | **Primary metric** |
| SSIMULACRA2 | `ssimulacra2` | Perceptual quality | 100 = identical, higher = better | Secondary metric |
| Butteraugli | `jpegli::butteraugli` | Psychovisual distance | < 1.0 = good, > 2.0 = bad | XYB mode validation |

### Using Metrics in Tests

```rust
use dssim::Dssim;
use ssimulacra2::{compute_frame_ssimulacra2, Rgb, ColorPrimaries, TransferCharacteristic};

// DSSIM
let attr = Dssim::new();
let orig = attr.create_image_rgba(&orig_rgba, width, height).unwrap();
let comp = attr.create_image_rgba(&comp_rgba, width, height).unwrap();
let (dssim, _) = attr.compare(&orig, comp);

// SSIMULACRA2
let rgb = Rgb::new(
    pixels.chunks(3).map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0]).collect(),
    width, height,
    TransferCharacteristic::SRGB,
    ColorPrimaries::BT709,
).unwrap();
let score = compute_frame_ssimulacra2(orig_rgb, dist_rgb).unwrap();
```

## Test Corpora

### CID22-512 (Recommended for quick tests)
512x512 center crops of CID22 images:
- Path: `/mnt/v/work/corpus/CID22-512`
- ~250 PNG files
- Good variety of content types

### CID22 Full Dataset
Cloudinary Image Dataset 2022 with pre-encoded files:
- Path: `/mnt/v/work/CID22/CID22`
- `original/` - Source PNGs
- `compressed/<image_id>/<codec>/` - Pre-encoded at various Q levels
- Codecs: mozjpeg, libjxl, aom, cld_avif, cld_heic, cld_jp2, cld_webp, vis_avif
- CSV with MCOS (human quality scores)

### Flower Test Images
Small test images for quick validation:
- Path: `/home/lilith/work/jpegli/testdata/jxl/flower/`
- `flower_small.rgb.png` - Small test image
- Various pre-encoded JPEGs for decode testing

### codec-corpus (Auto-discovered)
The corpus_comparison tools check for codec-corpus in these locations (in order):
1. `../codec-comparison/codec-corpus` ← **Currently available at this path**
2. `../codec-corpus`
3. `./codec-corpus`
4. `./codec-comparison`

If no corpus is found, tools will clone it locally:
```bash
git clone https://github.com/AcrossTheCloud/codec-corpus.git codec-corpus
```

Available test sets in codec-corpus:
- `kodak/` - 24 classic test images (512x768)
- `CID22/` - Cloudinary dataset
- `clic2025/` - CLIC challenge images
- `mozjpeg/` - mozjpeg test images
- `image-rs/` - Rust image crate test images
- `pngsuite/` - PNG test suite
- `zune/` - zune-jpeg test images

Test with:
```bash
cargo test --test cpp_filesize_comparison -- --ignored --nocapture
```

## Comparison Outputs

**Location**: `jpegli-rs/comparison_outputs/` (gitignored)

When running file size comparisons between C++ and Rust, always store outputs here:
- `rust_xyb_q{N}.jpg` - Rust XYB mode at quality N
- `rust_ycbcr_q{N}.jpg` - Rust YCbCr mode at quality N
- `cpp_xyb_q{N}.jpg` - C++ XYB mode at quality N
- `cpp_ycbcr_q{N}.jpg` - C++ YCbCr mode at quality N

This directory persists between sessions for easy visual inspection.

## Comparison Tools

### corpus_comparison.rs
Generates HTML chart comparing jpegli vs mozjpeg with DSSIM and SSIMULACRA2:
```bash
MAX_FILES=50 cargo run --release --example corpus_comparison -- \
    /mnt/v/work/corpus/CID22-512 /mnt/v/work/jpegli_data/comparison.html
```

Features:
- **Caching**: Encoded JPEGs cached in `jpeg_cache/` (version-tagged for invalidation)
- **Low-Q analysis**: Per-image breakdown at Q30 showing where each encoder excels
- **Dual metrics**: Both DSSIM and SSIMULACRA2 charts
- Env vars: `MAX_FILES=N` (limit images), `NO_CACHE=1` (disable caching)

### multi_codec_comparison.rs
Uses CID22 CSV data to compare all codecs:
```bash
cargo run --release --example multi_codec_comparison -- \
    /mnt/v/work/CID22/CID22 /mnt/v/work/jpegli_data/multi_codec.html
```

### quality_mapping test
Finds jpegli Q that matches mozjpeg DSSIM:
```bash
cargo test --test quality_mapping -- --nocapture
CORPUS_DIR=/mnt/v/work/corpus/CID22-512 cargo test --test quality_mapping test_quality_mapping_corpus -- --ignored --nocapture
```

## Key Test Results

### jpegli vs mozjpeg (4:4:4 subsampling)
- jpegli achieves **10-17% better DSSIM** at same Q value
- At Q90+, jpegli also produces **5-8% smaller files**
- jpegli wins on both quality AND size at high quality settings

### SSIMULACRA2 Results (30 images, CID22-512 corpus)
| Quality | jpegli bpp | jpegli SSIM2 | mozjpeg bpp | mozjpeg SSIM2 |
|---------|------------|--------------|-------------|---------------|
| Q30     | 0.77       | 59.5         | 0.57        | 50.8          |
| Q50     | 0.94       | 67.4         | 0.85        | 65.7          |
| Q70     | 1.28       | 76.4         | 1.22        | 74.5          |
| Q90     | 2.32       | 87.4         | 2.53        | 86.4          |

**Key finding**: jpegli produces higher SSIM2 scores (better quality) at similar or lower bitrates.

### Quality Mapping (to match DSSIM)
- mozjpeg Q60 → jpegli ~Q55 (jpegli is more efficient)
- mozjpeg Q90 → jpegli ~Q89

### XYB Mode Status
**Functional but not validated** - XYB color conversion is complete in `xyb.rs`:
- Full sRGB → linear → XYB → scaled XYB pipeline
- ICC profile embedding (720-byte XYB profile)
- Frequency-dependent `DistanceToScale()` with `FREQUENCY_EXPONENT[64]`
- Huffman algorithm matches C++ exactly (61/61 test cases pass)

### C++ vs Rust File Size Comparison

**Investigation Status** - Verified matching components:

| Component | Status | Verified By |
|-----------|--------|-------------|
| Quantization tables | ✓ Match exactly | `compare_cpp_quant.rs` example |
| Huffman tree building | ✓ Match exactly | 61/61 test cases in `huffman_cpp_comparison.rs` |
| Color conversion formula | ✓ Match exactly | BT.601/JFIF formulas verified |
| DCT precision | ✓ Good (round-trip error ~0.000008) | `test_dct_precision.rs` |
| Zero-biasing | ✓ Wired in | No benefit without AQ (aq_strength=0) |

**Not yet implemented** (contribute to default C++ advantage):
| Feature | Est. Impact | C++ File |
|---------|-------------|----------|
| Adaptive quantization | ~3-4% | `adaptive_quantization.cc` |
| Progressive encoding | ~2-3% | `encode.cc` |
| Huffman optimization | ~3-4% | `huffman.cc` |

**With matching settings** (4:4:4, no AQ, sequential, fixed Huffman):
| Image | C++ | Rust | Diff |
|-------|-----|------|------|
| flower_small | 61,476 | 63,906 | +4.0% |

**Remaining ~4% gap analysis** (with matching settings):
- Decoded images differ by max 8 pixel values → coefficients slightly different
- Scan data (entropy-coded coefficients) is 4% larger
- YCbCr now uses f32 precision throughout (matches C++ approach)
- DCT implementation verified correct via round-trip tests

**Likely causes of remaining gap**:
1. DCT implementation details (normalization, scaling, SIMD differences)
2. Floating-point accumulation differences (f32 vs C++ float)
3. Edge handling at image boundaries (padding strategy)

**Verification test**: `cargo test --test parity_enforcement -- --ignored --nocapture`

## Adaptive Quantization Debugging (Confirmed Bugs)

### Bug 1: LIMIT Scaling (FIXED)

**Location**: `adaptive_quant.rs:compute_pre_erosion_scalar`

**Bug**: `let limit = LIMIT / K_INPUT_SCALING` → limit = 51.0 instead of 0.2

**Problem**: `ratio_of_derivatives` returns values in range 1.6-1.9, which are ALL < 51.0. This caused every pixel to hit the `offset` branch, producing uniform pre_erosion output.

**Fix**: Use `let limit = LIMIT` directly (0.2). The ratio_of_derivatives already has K_INPUT_SCALING baked into its constants.

**Test confirmation**: After fix, pre_erosion values varied from 6.31-6.56 instead of uniform 4.845.

### Bug 2: FuzzyErosion Global Minimum (FIXED)

**Location**: `adaptive_quant.rs:fuzzy_erosion_scalar`

**Bug**: Original code tracked 4 smallest values across ENTIRE row/column using `update_min4`, never resetting or using a window.

**Problem**: By end of row, it found the GLOBAL minimum and assigned that to ALL cells, producing uniform output.

**Fix**: Rewrite to match C++ algorithm:
1. For each pixel, find 4 smallest in 3x3 LOCAL window
2. Weighted sum: `0.125*min0 + 0.075*min1 + 0.06*min2 + 0.05*min3`
3. Sum 2x2 blocks to get final values

**Expected**: C++ produces quant_field mean 6.75 from pre_erosion mean 5.64

### Bug 3: Missing 0.25 Scale Factor (FIXED)

**Location**: `adaptive_quant.rs:compute_pre_erosion_scalar`

**Bug**: After summing 4 adjacent x-values, C++ multiplies by 0.25:
```cpp
row_d_out[x] = (sum of 4) * 0.25f;
```

**Problem**: Without this, pre_erosion values were 4x too high (21-24 instead of 5-8).

**Fix**: Added `sum * 0.25` after the 4x downsampling sum.

### Bug 4: K_MASK_BASE Sign (FIXED)

**Location**: `adaptive_quant.rs:K_MASK_BASE`

**Bug**: `K_MASK_BASE = 0.6109318733215332` (POSITIVE)

**Problem**: C++ has `kBase = -0.74174993` (NEGATIVE). This completely changed the ComputeMask output, making it positive when it should be negative.

**Fix**: Changed to `K_MASK_BASE = -0.74174993`.

**Impact**: This was the biggest bug. ComputeMask(6.6) changed from +0.62 to -0.38, enabling the 2^x transform to produce quant_field in the correct 0.5-0.6 range.

### Bug 5: Wrong K_MUL Constants (FIXED)

**Location**: `adaptive_quant.rs:compute_mask_scalar`

**Bug**: Used `K_MUL4 = 0.039` instead of `K_MASK_MUL4 = 3.24` (100x difference!)

**Problem**: Both constant sets existed in the file with similar names:
- `K_MUL4 = 0.039` (wrong, ~100x smaller)
- `K_MASK_MUL4 = 3.24` (correct C++ value)

**Fix**: Changed function to use `K_MASK_MUL*` constants.

### Bug 6: PerBlockModulations Scaling (FIXED)

**Location**: `adaptive_quant.rs:per_block_modulations_scalar`

**Bug**: Used `input_scaled` (0-1 range) but C++ uses unscaled input (0-255 range).

**Problem**: HfModulation and GammaModulation constants assume 0-255 input range.

**Fix**: Pass original `y_plane` (unscaled) to per_block_modulations.

### Current Status

After all fixes:
- **Rust aq_strength**: min=0.00, max=0.12, mean=0.065
- **C++ aq_strength**: min=0.00, max=0.20, mean=0.081
- **Mean abs diff**: 0.016 (down from 0.08)

Remaining 20% difference likely due to:
1. Edge handling differences in FuzzyErosion
2. FastLog2f vs log2() approximation differences
   - **Note**: `moxcms/src/math/log2fs.rs` has a good FastLog2f implementation to try
3. Border padding handling

## Red Herrings (Investigated but NOT the cause)

Things we investigated that looked promising but turned out not to be the issue:

### DCT Scaling (1/8 vs 1/64)

**Hypothesis**: C++ jpegli comments in `dct-inl.h` mention 1/8 scaling applied per dimension (rows + columns = 1/64 total). Rust uses 1/8 total. Maybe changing to 1/64 would fix the file size gap.

**What we tried**:
```rust
// Changed dct.rs from:
let scale = 1.0 / 8.0;
// To:
let scale = 1.0 / 64.0;
```

**Result**: CATASTROPHIC FAILURE
- Decoded pixel values were ~8× too small (e.g., 137 instead of 200)
- DSSIM jumped from 0.002 to 0.46 (10× worse)
- jpeg-decoder couldn't interpret the coefficients correctly

**Root cause**: The 1/8 scaling is correct for compatibility with standard JPEG decoders like libjpeg/jpeg-decoder. The C++ jpegli must handle things differently internally but produces decoder-compatible output.

**Lesson**: Don't trust C++ comments about internal scaling without testing decoder compatibility.

**Examples created**: `trace_dct_scaling.rs`, `test_decoder_compat.rs`

### DC Bias Timing (Before vs After DCT)

**Hypothesis**: C++ subtracts 128 from DC coefficient AFTER DCT (see `dct-inl.h` line 253: `const float dc = (dct[0] - kDCBias) * qmc[0]`). Rust subtracts 128 from ALL pixels BEFORE DCT. Maybe this causes coefficient differences.

**What we found**:
- For uniform block of value V:
  - C++ approach: DCT(V) - 128 = V - 128 (with 1/64 scale, DC = V)
  - Rust approach: DCT(V - 128) = (V - 128) * 8 (with 1/8 scale)
- The final quantized values are different due to scaling, not timing

**Why it's NOT the issue**: The 1/8 scaling compensates. With 1/8 scale, level shift before DCT produces the same quantized coefficients as 1/64 scale with level shift after DCT.

**Test created**: `test_dc_bias.rs`

## C++ Build Instructions

### Ubuntu 22.04

```bash
sudo apt update
sudo apt install -y cmake build-essential ninja-build \
    libbrotli-dev libgif-dev libjpeg-dev libpng-dev \
    libwebp-dev pkg-config

cd /path/to/jpegli
mkdir -p build && cd build
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DJPEGXL_ENABLE_TOOLS=ON \
    -DJPEGXL_ENABLE_JPEGLI_LIBJPEG=ON \
    -DJPEGXL_ENABLE_SJPEG=OFF \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
ninja jpegli-static cjpegli djpegli
```

## Key C++ Source Files

| C++ File | Purpose | Rust Equivalent |
|----------|---------|-----------------|
| `lib/jpegli/dct-inl.h` | Forward DCT | `dct.rs` |
| `lib/jpegli/idct.cc` | Inverse DCT | `idct.rs` |
| `lib/jpegli/color_transform.cc` | Color conversion | `color.rs` |
| `lib/jpegli/encode.cc` | Encoder | `encode.rs` |
| `lib/jpegli/decode.cc` | Decoder | `decode.rs` |
| `lib/extras/butteraugli.cc` | Quality metric | `butteraugli.rs` |
| `lib/jxl/enc_xyb.cc` | XYB color space | `xyb.rs` |

## Dependencies

### Production
- `wide` - SIMD (Highway equivalent)
- `bytemuck` - Safe transmutes
- `arrayref` - Array references
- `rgb`, `imgref` - Image buffers

### Testing
- `dssim` - DSSIM quality metric
- `ssimulacra2` - SSIMULACRA2 quality metric
- `png` - Image I/O
- `mozjpeg` - Encoder comparison
- `jpeg-decoder` - Reference decoder

## Architecture Notes

### Layer Structure
```
Layer 0: consts, types (pure data)
Layer 1: huffman, quant (pure math)
Layer 2: dct, idct, color, xyb (transforms)
Layer 3: bitstream (I/O)
Layer 4: entropy (stateful)
Layer 5-6: encode, decode (pipelines)
Metrics: butteraugli (quality assessment)
```

### XYB Color Space
- XYB is pure math with fixed constants
- Opsin absorbance matrix and bias values in `consts.rs`
- Roundtrip error < 2 bits (confirmed by tests)

### SIMD Feature Flag
```toml
[features]
default = ["simd"]
simd = []
```

To disable SIMD: `cargo build --no-default-features`

## Running Tests

```bash
cargo test                              # All tests
cargo test --release                    # With optimizations
cargo test --test xyb_roundtrip         # XYB specific
cargo test --test metrics_comparison    # Quality metrics
cargo test -- --ignored                 # Ignored tests (need files)
```

## What Makes jpegli Special

1. **Adaptive Quantization**: Content-aware bit allocation
2. **XYB Color Space**: Perceptually optimized (from JPEG XL)
3. **Improved Quantization Tables**: Better than IJG libjpeg
4. **Float-based Pipeline**: Higher precision
5. **Smart Zero-Biasing**: Intelligent coefficient rounding

Output is **backward compatible** - standard JPEG readable by any decoder.

## Git Workflow

```bash
git status                    # Check current state
git add jpegli-rs/           # Stage changes
git commit -m "Description"  # Commit
# Stay on stepbystep2 branch for development
# PR to main when features are complete
```
