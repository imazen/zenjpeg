# Decoder Refactoring Handoff

## Goal
Split `decode/parser.rs` (2600 lines) into granular modules without introducing performance regressions (no extra copies, allocations, or indirection).

## Completed

### New Files Created
1. **`decode/parser/mod.rs`** (~320 lines)
   - `JpegParser` struct definition
   - `CompInfo` helper struct
   - Core I/O: `read_u8`, `read_u16`, `read_marker`
   - `build_comp_infos()` helper
   - `decode()` orchestration
   - `find_scan_info()` for scanline reader
   - `info()` and `extract_coefficients()`

2. **`decode/parser/markers.rs`** (~250 lines)
   - `read_header()` - marker loop
   - `parse_frame_header()` - SOF
   - `parse_quant_table()` - DQT
   - `parse_huffman_table()` - DHT
   - `parse_restart_interval()` - DRI
   - `skip_segment()`

3. **`decode/parser/scan.rs`** (~450 lines)
   - `parse_scan()` - SOS parsing + dispatch
   - `decode_scan()` - baseline MCU decode
   - `can_use_streaming()`
   - `decode_baseline_streaming_rgb()` - fused decode path

## Remaining

### Files to Create

4. **`decode/parser/progressive.rs`** (~300 lines)
   - Extract `decode_progressive_scan()` from parser.rs lines 993-1291
   - DC-first, DC-refine, AC-first, AC-refine scan handling

5. **`decode/parser/output.rs`** (~1200 lines)
   - Extract from parser.rs lines 1384-2600:
     - `can_use_fast_i16_path()`
     - `can_use_fast_i16_subsampled()`
     - `to_pixels_fast_i16()` - fast path for 4:4:4
     - `to_pixels_fast_i16_subsampled()` - fast path for 4:2:0/4:2:2
     - `to_pixels()` - generic f32 path
     - `to_pixels_f32()` - f32 output
     - `to_ycbcr_planes_f32()` - YCbCr plane output

### Final Steps
6. Update `decode/mod.rs` to use `mod parser;` instead of `mod parser;` (single file)
7. Delete old `decode/parser.rs`
8. Run tests: `cargo test --release -p jpegli-rs --features decoder`
9. Run benchmarks: `cargo bench --bench decode -p jpegli-rs`

## Performance Constraints

**DO NOT:**
- Add `Clone` where `&self` suffices
- Box or Arc any hot-path data
- Add trait objects where concrete types work
- Change `&mut [i16]` to `Vec<i16>` in IDCT paths
- Add any heap allocation in per-block or per-MCU loops
- Change `pub(super)` to `pub` (keep internal)

**VERIFY:**
- `decode_baseline_streaming_rgb` still fuses decode+IDCT+color in one pass
- `to_pixels_fast_i16` still uses `idct_int_tiered` with coefficient counts
- No new `Vec::new()` in hot paths
- All `try_alloc_maybeuninit` calls remain (DoS protection)

## Key Dependencies Between Modules

```
parser/mod.rs
├── imports from crate::* (color, entropy, foundation, huffman, quant, types)
├── markers.rs (impl JpegParser methods)
├── scan.rs (impl JpegParser methods)
├── progressive.rs (impl JpegParser methods)
└── output.rs (impl JpegParser methods)
```

All submodules add `impl` blocks to `JpegParser` - no new structs needed.

## Test Commands
```bash
# Unit tests
cargo test --release -p jpegli-rs --features decoder

# Decode benchmark (compare before/after)
cargo bench --bench decode -p jpegli-rs --features decoder -- --save-baseline before
# ... make changes ...
cargo bench --bench decode -p jpegli-rs --features decoder -- --baseline before

# Full parity tests
cargo test --release -p jpegli-rs --features decoder -- --ignored
```

## Files Reference

| Original Line Range | New Location | Content |
|---------------------|--------------|---------|
| 1-150 | parser/mod.rs | Struct, new(), CompInfo |
| 152-191 | parser/mod.rs | read_u8, read_u16, read_marker |
| 193-492 | parser/markers.rs | read_header, parse_* |
| 494-518 | parser/mod.rs | decode() orchestration |
| 520-991 | parser/scan.rs | parse_scan, decode_scan, streaming |
| 993-1290 | parser/progressive.rs | decode_progressive_scan |
| 1291-1383 | parser/mod.rs | info(), extract_coefficients() |
| 1384-2600 | parser/output.rs | to_pixels* methods |
