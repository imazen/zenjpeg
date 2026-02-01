# XYB Size Difference Investigation - RESOLVED

## Root Cause: Configuration Mismatch

The reported **5-11% size gap** was caused by comparing:
- **Rust XYB**: baseline (sequential) mode (default)
- **C++ XYB**: progressive mode (default)

Progressive encoding with optimized Huffman produces ~6% smaller files than baseline.

## Apples-to-Apples Comparison (Both Progressive)

| Image | Q70 | Q80 | Q90 |
|-------|-----|-----|-----|
| kodak/1.png | **-0.65%** | **-0.60%** | **-0.30%** |
| kodak/5.png | **-0.57%** | **-0.54%** | **-0.34%** |
| kodak/13.png | +1.75% | +0.81% | +0.11% |
| kodak/19.png | **-2.66%** | **-2.35%** | **-1.74%** |

**Result:** Rust is often slightly smaller than C++. All differences are within ±2%.

## What's Verified Identical

1. **Quantization tables**: Byte-for-byte identical (8-bit, 3 tables)
2. **XYB color conversion constants**: Verified in `compare_xyb_constants` example
3. **AQ maps**: 100% match when using distance-based encoding
4. **Subsampling structure**: R:2×2 G:2×2 B:1×1

## Configuration Details

| Property | C++ Default | Rust Default |
|----------|-------------|--------------|
| Progressive | **true** | false |
| Scans | 15 | 1 |
| Huffman tables | 8 | 2 |
| Optimize Huffman | true | true |

## Recommendation

Consider changing Rust's XYB default to progressive mode to match C++ behavior:

```rust
// In EncoderConfig::xyb()
Self {
    progressive: true,  // Match C++ default
    // ...
}
```

Or document the difference clearly in the API.

## Test Files

- `/mnt/v/output/zenjpeg-xyb-size-explore/cpp_xyb_q90.jpg` - C++ progressive (141,450 bytes)
- `/mnt/v/output/zenjpeg-xyb-size-explore/rust_xyb_q90.jpg` - Rust baseline (149,490 bytes)
- `/mnt/v/output/zenjpeg-xyb-size-explore/rust_xyb_prog_q90.jpg` - Rust progressive (141,027 bytes)

## Diagnostic Tools Used

```bash
# Inspect JPEG structure
cargo run --release --example jpeg_inspect -- --quant <file.jpg>

# Compare progressive XYB
cargo run --release --example xyb_prog_comparison

# Generate C++ XYB
./internal/jpegli-cpp/build/tools/cjpegli --xyb -q 90 input.png output.jpg
```
