# Progressive JPEG Porting Plan

## Executive Summary

**Goal:** Port C++ jpegli's progressive JPEG encoding/decoding to Rust with full compatibility.

**Current Status:**
- ✓ Baseline YCbCr/XYB encoding/decoding works
- ✗ Progressive decoder fails (UnexpectedEof)
- ✗ Progressive XYB encoder produces 3× larger files than C++
- ✗ Rust defaults to Baseline, C++ defaults to Progressive level 2

**Strategy:** Fix decoder first (validate with C++ output), then optimize encoder, working with YCbCr before XYB.

## Current State Analysis

### What Works
- ✓ Baseline YCbCr encoding/decoding (all subsampling modes)
- ✓ Baseline XYB encoding/decoding
- ✓ Progressive YCbCr encoding (structure is correct)
- ✓ Huffman algorithm selection infrastructure
- ✓ Progressive scan script generation

### What's Broken
- ✗ Progressive decoder fails for both YCbCr and XYB (UnexpectedEof)
- ✗ Progressive XYB encoder produces 140KB vs C++ 49KB (3× larger)
- ✗ Can't decode C++ progressive JPEGs
- ✗ Default mode is Baseline instead of Progressive level 2

### Root Causes

#### Progressive Decoder Issues
1. **AC refinement bit encoding/decoding mismatch**
   - Fixed in previous session but may have remaining issues
   - Bit order in refinement scans critical
   - Need to verify against C++ and zune-jpeg

2. **Bit buffer management in successive approximation**
   - EOB runs may not be handled correctly
   - Bit position tracking across scans
   - Zero runs in refinement scans

3. **Spectral selection boundaries**
   - May not handle Ss/Se transitions correctly
   - DC/AC scan separation
   - Coefficient range validation

#### Progressive XYB Encoder Issues
1. **No optimized Huffman tables**
   - Using standard tables instead of per-scan optimized
   - C++ builds new tables for each scan type
   - Major contributor to 3× size difference

2. **MCU block ordering with 2x2,2x2,1x1 subsampling**
   - R/X: 4 blocks per MCU
   - G/Y: 4 blocks per MCU
   - B: 1 block per MCU
   - Block count calculation may be wrong

3. **Scan structure differences**
   - Fixed: now uses non-interleaved DC scans
   - Fixed: correct component IDs (82,71,66)
   - May need different scan script than YCbCr

## Reference Implementations

### C++ jpegli (Ground Truth)
- **Location:** `internal/jpegli-cpp/`
- **Use for:** Final validation, instrumentation, algorithm reference
- **Key files:**
  - `lib/jpegli/encode.cc` - Progressive encoder
  - `lib/jpegli/decode.cc` - Progressive decoder
  - `lib/jpegli/huffman.cc` - Huffman optimization

### zune-jpeg (Pure Rust Decoder)
- **Dependency:** Already in Cargo.toml
- **Use for:** Progressive decoder validation, algorithm comparison
- **Advantages:**
  - Pure Rust (easier to read/debug)
  - Known to work with various progressive JPEGs
  - Can decode C++ jpegli output

### mozjpeg (Encoder Reference)
- **Dependency:** Already in Cargo.toml (via mozjpeg crate)
- **Use for:** Huffman optimization reference, trellis quantization
- **Limitations:** C FFI, different API style

## Phased Porting Strategy

### Phase 1: Fix Progressive Decoder (YCbCr only)
**Goal:** Decode C++ progressive YCbCr JPEGs correctly

**Approach:**
1. Create minimal test case:
   ```bash
   cjpegli flower.png cpp_prog_ycbcr.jpg -q 90
   ```

2. Compare three decoders:
   ```rust
   // Decode with jpegli-rs
   let rust_result = jpegli::Decoder::new().decode(&jpeg)?;

   // Decode with zune-jpeg
   let zune_result = zune_jpeg::JpegDecoder::new(&jpeg).decode()?;

   // Decode with jpeg-decoder (pure Rust)
   let jpeg_decoder_result = jpeg_decoder::Decoder::new(&jpeg[..]).decode()?;
   ```

3. Compare outputs:
   - If zune-jpeg succeeds but jpegli-rs fails → algorithm bug
   - If both fail → might be C++ encoding issue
   - Compare pixel values where both succeed

4. Add detailed logging to Rust decoder:
   ```rust
   eprintln!("Scan: ss={} se={} ah={} al={} comp={}", ...);
   eprintln!("Block {}: DC={} AC=[...]", ...);
   ```

5. Instrument C++ decoder to match logging format

6. Fix identified issues:
   - AC refinement bit order
   - EOB run handling
   - DC prediction across scans
   - Bit buffer management

**Validation:**
```bash
# Test with C++ progressive YCbCr
cargo run --example test_progressive_decode -- cpp_prog_ycbcr.jpg

# Should output:
# ✓ jpegli-rs decoded successfully
# ✓ zune-jpeg decoded successfully
# ✓ Max pixel difference: 0 (exact match)
```

**Success Criteria:**
- [ ] Decodes C++ progressive YCbCr without errors
- [ ] Pixel-perfect match with zune-jpeg output
- [ ] Handles all progressive levels (0, 1, 2)

### Phase 2: Test YCbCr Progressive Roundtrip
**Goal:** Rust progressive encoding → Rust progressive decoding works

**Approach:**
1. Encode with Rust progressive:
   ```rust
   let jpeg = Encoder::new()
       .mode(JpegMode::Progressive)
       .quality(Quality::from_quality(90.0))
       .encode(rgb)?;
   ```

2. Decode with all three decoders:
   - jpegli-rs (our decoder)
   - zune-jpeg (pure Rust validation)
   - jpeg-decoder (pure Rust validation)

3. Compare pixel outputs:
   ```rust
   let max_diff = original.iter()
       .zip(decoded.iter())
       .map(|(a, b)| (*a as i16 - *b as i16).abs())
       .max()
       .unwrap();

   assert!(max_diff < 5, "Lossy JPEG, small diff expected");
   ```

4. If roundtrip fails:
   - Add logging to encoder scan writing
   - Compare scan structure with C++ output
   - Verify coefficient values match expected

5. Test edge cases:
   - Small images (8x8, 16x16)
   - Non-MCU-aligned dimensions
   - Grayscale progressive
   - Different subsampling modes (4:4:4, 4:2:0)

**Validation:**
```bash
cargo test --test progressive_roundtrip -- --nocapture
```

**Success Criteria:**
- [ ] Rust encodes → Rust decodes successfully
- [ ] zune-jpeg can decode Rust progressive output
- [ ] jpeg-decoder can decode Rust progressive output
- [ ] Max pixel difference < 5 (lossy compression expected)

### Phase 3: Compare YCbCr Progressive File Sizes
**Goal:** Understand efficiency gap vs C++

**Approach:**
1. Compare file sizes:
   ```bash
   # C++ progressive YCbCr
   cjpegli flower.png cpp.jpg -q 90
   # Rust progressive YCbCr
   cargo run --example encode_progressive -- flower.png rust.jpg -q 90

   ls -lh cpp.jpg rust.jpg
   ```

2. If gap exists (expected: Rust larger), investigate:
   - **Huffman tables:** Are we using optimized or standard?
   - **Scan structure:** Same number of scans?
   - **Coefficient values:** Decode both, compare DCT coefficients

3. Dump JPEG structure:
   ```bash
   cargo run --example compare_structure -- cpp.jpg rust.jpg
   ```

   Compare:
   - DHT (Huffman table) markers - count and sizes
   - Scan count and spectral selection
   - Compressed scan data sizes

4. Add C++ instrumentation:
   ```cpp
   // In lib/jpegli/huffman.cc
   void OptimizeHuffmanCodes(...) {
     fprintf(stderr, "Building Huffman for scan ss=%d se=%d\n", ...);
     fprintf(stderr, "Symbol frequencies: [...]\n");
   }
   ```

5. Compare Huffman frequencies:
   - If frequencies match but codes differ → Huffman algorithm issue
   - If frequencies differ → Coefficient values different

**Expected Findings:**
- Rust likely using standard Huffman tables (large)
- C++ building optimized tables per scan (small)
- Need to implement Phase 4 to fix

**Success Criteria:**
- [ ] Understand root cause of size difference
- [ ] Have plan to fix (likely: add optimized Huffman)
- [ ] Document findings

### Phase 4: Add Optimized Huffman for Progressive
**Goal:** Build per-scan Huffman tables like C++

**Approach:**
1. Study C++ Huffman optimization:
   ```cpp
   // lib/jpegli/huffman.cc
   void OptimizeHuffmanCodes(j_compress_ptr cinfo) {
     // Analyze how C++ builds tables per scan
   }
   ```

2. Design Rust implementation:
   ```rust
   struct ProgressiveHuffmanOptimizer {
       // Collect stats for each scan
       scan_stats: Vec<HuffmanFrequencies>,
   }

   impl ProgressiveHuffmanOptimizer {
       fn optimize_scan(&mut self, scan_index: usize) -> OptimizedTables {
           // Build tables from collected frequencies
       }
   }
   ```

3. Modify encoder pipeline:
   ```rust
   // First pass: collect coefficient statistics
   for scan in scans {
       collect_huffman_stats(scan, &mut optimizer);
   }

   // Second pass: build tables and encode
   for (scan_idx, scan) in scans.iter().enumerate() {
       let tables = optimizer.optimize_scan(scan_idx);
       write_dht_if_changed(&mut output, &tables);
       write_scan_data(&mut output, scan, &tables);
   }
   ```

4. Handle table changes between scans:
   - Write DHT marker only when tables change
   - C++ inserts DHT markers mid-stream for refinement scans
   - Track which tables are "active" in decoder state

5. Test file size reduction:
   ```bash
   # Before optimization
   ls -lh rust_standard_huffman.jpg  # ~70KB

   # After optimization
   ls -lh rust_optimized_huffman.jpg  # ~52KB (target: match C++ ~50KB)
   ```

**Reference Implementation:**
- mozjpeg Huffman optimization can be studied
- Our existing `huffman_opt.rs` can be extended
- C++ `CreateHuffmanTree` already ported

**Success Criteria:**
- [ ] File size within 5% of C++ progressive
- [ ] Can decode optimized output with all decoders
- [ ] Performance acceptable (2-pass is slower but worth it)

### Phase 5: Fix XYB Progressive Encoder
**Goal:** Correct XYB progressive encoding

**Current Issues:**
- 140KB vs C++ 49KB (3× larger)
- Decoder fails (UnexpectedEof)

**Approach:**
1. Verify block count calculation:
   ```rust
   // XYB MCU: 16×16 pixels
   // R/X: 4 blocks (2×2 in MCU)
   // G/Y: 4 blocks (2×2 in MCU)
   // B: 1 block (1×1 in MCU)

   let mcu_cols = (width + 15) / 16;
   let mcu_rows = (height + 15) / 16;

   let r_blocks = mcu_cols * mcu_rows * 4; // 2×2
   let g_blocks = mcu_cols * mcu_rows * 4; // 2×2
   let b_blocks = mcu_cols * mcu_rows * 1; // 1×1
   ```

2. Check MCU block ordering:
   ```rust
   // For each MCU at (mcu_x, mcu_y):
   // R: blocks [0,1,2,3] in raster order within MCU
   // G: blocks [0,1,2,3] in raster order within MCU
   // B: block [0]
   ```

3. Compare with C++ block ordering:
   - Add logging to `encode_progressive_scan` showing block indices
   - Add C++ instrumentation showing block indices
   - Verify ordering matches

4. Check if C++ uses different scan script for XYB:
   ```bash
   # Compare structures
   cargo run --example compare_xyb_structure
   ```

   Check:
   - Same number of scans?
   - Same spectral selection (Ss/Se)?
   - Same successive approximation (Ah/Al)?

5. Fix `quantize_all_blocks_xyb`:
   - Currently returns flat Vec of blocks
   - May need to track component boundaries
   - Progressive encoder needs to know which blocks belong to which component

6. Apply optimized Huffman (Phase 4) to XYB:
   - Should reduce size significantly
   - May close most of the 3× gap

**Validation:**
```rust
// Test XYB progressive roundtrip
let xyb_jpeg = Encoder::new()
    .use_xyb(true)
    .mode(JpegMode::Progressive)
    .encode(rgb)?;

let decoded = Decoder::new().apply_icc(true).decode(&xyb_jpeg)?;
assert!(decoded.data.len() > 0);
```

**Success Criteria:**
- [ ] XYB progressive file size within 10% of C++
- [ ] Rust can decode its own XYB progressive output
- [ ] Block count and ordering correct

### Phase 6: Fix XYB Progressive Decoder
**Goal:** Decode C++ XYB progressive JPEGs

**Approach:**
1. Use working YCbCr progressive decoder as template

2. Handle component ID mapping:
   ```rust
   match comp_id {
       82 => 0, // 'R' → X
       71 => 1, // 'G' → Y
       66 => 2, // 'B' → B
       _ => return Err(...),
   }
   ```

3. Handle 2x2,2x2,1x1 subsampling in MCU reconstruction:
   ```rust
   // When decoding XYB progressive:
   // - R/X has 4 blocks per MCU
   // - G/Y has 4 blocks per MCU
   // - B has 1 block per MCU

   // Upscale B from 1 block to 4 for MCU reconstruction
   ```

4. Test with C++ XYB progressive:
   ```bash
   cjpegli flower.png cpp_xyb.jpg -q 90 --xyb
   cargo run --example test_decode -- cpp_xyb.jpg
   ```

5. If fails, compare with baseline XYB decoder:
   - Baseline XYB decoding works
   - Progressive should use same block reconstruction
   - Main difference: multi-scan vs single-scan

**Success Criteria:**
- [ ] Decodes C++ XYB progressive without errors
- [ ] Can apply ICC transform correctly
- [ ] Visual output looks correct (no artifacts)

### Phase 7: Change Default to Progressive
**Goal:** Match C++ cjpegli defaults

**Approach:**
1. Change default in `types.rs`:
   ```rust
   #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
   pub enum JpegMode {
       Baseline,
       Extended,
       #[default]  // Move this line
       Progressive,
       Lossless,
   }
   ```

2. Update tests that assume baseline:
   ```bash
   # Find tests that may break
   git grep "Baseline" tests/

   # Update or add explicit .mode(JpegMode::Baseline) where needed
   ```

3. Update examples:
   - Most examples should use Progressive by default
   - Some may need explicit Baseline for comparison

4. Document the change:
   ```rust
   /// JPEG encoding mode.
   ///
   /// **Default:** Progressive (matches C++ jpegli default)
   ///
   /// Progressive encoding is more efficient at high qualities (Q70+)
   /// and produces better visual progressive rendering.
   ```

5. Run full test suite:
   ```bash
   cargo test
   cargo test --release
   ```

**Success Criteria:**
- [ ] All tests pass with Progressive default
- [ ] File sizes match C++ by default
- [ ] Documentation updated

## Testing Strategy

### Unit Tests
- `tests/progressive_decode.rs` - Decode C++ progressive JPEGs
- `tests/progressive_roundtrip.rs` - Encode/decode roundtrip
- `tests/progressive_structure.rs` - Verify scan structure
- `tests/xyb_progressive.rs` - XYB progressive tests

### Integration Tests
- Compare with zune-jpeg decoder (pure Rust)
- Compare with jpeg-decoder (pure Rust)
- Compare file sizes with C++ cjpegli
- Visual comparison (human review)

### Regression Tests
- Keep all existing examples
- Lock tests for known-good cases
- File size should not regress

## Risk Mitigation

### 1. Don't Change Multiple Things at Once
- Fix decoder before encoder
- Fix YCbCr before XYB
- Add one feature at a time
- Test after each change

### 2. Use Multiple Reference Implementations
- **C++ jpegli:** Ground truth
- **zune-jpeg:** Pure Rust decoder reference
- **jpeg-decoder:** Pure Rust decoder (alternative reference)
- **mozjpeg:** C encoder based on libjpeg-turbo (Huffman, trellis reference)

### 3. Incremental Validation
- Phase 1 validates Phase 2
- Phase 2 validates Phase 3
- Can't skip phases without breaking later ones

### 4. Keep Debugging Examples
- Don't delete comparison examples
- They're invaluable for debugging regressions
- Archive in `examples/debug/` if needed

### 5. C++ Instrumentation
- Add logging to C++ encoder/decoder
- Generate `.testdata` files for intermediate values
- Compare Rust vs C++ step-by-step

## Success Metrics

### Phase 1 Complete
- [ ] Decodes all C++ progressive JPEGs (YCbCr)
- [ ] Matches zune-jpeg output exactly

### Phase 2 Complete
- [ ] Rust progressive roundtrip works
- [ ] Other decoders can decode Rust output

### Phase 3 Complete
- [ ] Understand size gap root cause
- [ ] Have actionable plan to fix

### Phase 4 Complete
- [ ] File size within 5% of C++
- [ ] Optimized Huffman working

### Phase 5 Complete
- [ ] XYB progressive file size reasonable
- [ ] XYB progressive roundtrip works

### Phase 6 Complete
- [ ] Decode C++ XYB progressive

### Phase 7 Complete
- [ ] Default matches C++
- [ ] All tests pass

## Timeline Estimate

**Phase 1:** 4-8 hours (decoder debugging is tedious)
**Phase 2:** 2-4 hours (should be straightforward if Phase 1 works)
**Phase 3:** 2-3 hours (analysis and instrumentation)
**Phase 4:** 6-10 hours (Huffman optimization is complex)
**Phase 5:** 4-6 hours (XYB MCU handling)
**Phase 6:** 2-4 hours (should be similar to Phase 5)
**Phase 7:** 1-2 hours (mostly testing)

**Total:** 21-37 hours of focused work

## Next Steps

Start with **Phase 1: Fix Progressive Decoder (YCbCr)**

Create example:
```bash
cargo run --example compare_progressive_decoders -- \
    --input cpp_progressive.jpg \
    --compare-with zune-jpeg
```

Expected output:
```
Testing: cpp_progressive.jpg

jpegli-rs decoder:
  ✗ Failed: UnexpectedEof at scan 7

zune-jpeg decoder:
  ✓ Success: 510x532, 816480 bytes

Conclusion: jpegli-rs decoder has bugs
```

Then debug until both decoders produce identical output.
