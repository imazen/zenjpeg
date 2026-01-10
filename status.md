# jpegli-rs Status Report

**Generated:** 2026-01-09 18:45:54 MST
**Branch:** avx2-dct-intrinsics

---

## Recent Work

### Completed This Session
- [x] Updated multiversion crate 0.7 → 0.8 (reduces retpoline cfg warnings)
- [x] Researched optimal SIMD target strings
- [x] Verified allocation instrumentation (`alloc_tracker` example)
- [x] Confirmed fallible allocations in streaming AQ hot paths
- [x] Created this feature support documentation
- [x] **Merged low-memory StreamingAQ** - replaced full Y plane impl with rolling buffers (~2.5 MB vs 33 MB for 4K)

---

## SIMD Target Configuration

Current multiversion targets are appropriate:

| Target | Microarch Level | Coverage |
|--------|-----------------|----------|
| `x86_64+avx2+fma` | x86-64-v3 | Haswell+ (2013+), ~85% x86 CPUs |
| `x86_64+sse2` | x86-64-v1 | All x86-64 (baseline fallback) |
| `aarch64+neon` | ARM64 baseline | All ARM64 (NEON is mandatory) |

---

## Memory Profile

### Full-Plane Encoder (measured via `alloc_tracker`)

| Image Size | Peak Allocation | Alloc Count |
|------------|-----------------|-------------|
| 2K (1920×1080) | 40 MB | ~50 |
| 4K (3840×2160) | 159 MB | ~50 |
| 12MP (4000×3000) | 230 MB | ~73 |

### Strip-Based Encoder (theoretical)

| Image Size | Peak Allocation | Reduction |
|------------|-----------------|-----------|
| 12MP | ~47 MB | 5× less |

---

## Feature Support Matrix

### Encoder Features

| Feature | Full-Plane | Strip-Based | Notes |
|---------|:----------:|:-----------:|-------|
| **Modes** ||||
| Baseline (8-bit) | ✅ | ✅ | Standard JPEG |
| Progressive | ✅ | ❌ | Full-plane only |
| Extended (12-bit) | ❌ | ❌ | Not implemented |
| Lossless | ❌ | ❌ | Not planned |
| **Color Spaces** ||||
| YCbCr (RGB input) | ✅ | ✅ | Standard path |
| XYB | ✅ | ❌ | Full-plane only |
| Grayscale | ✅ | ✅ | |
| CMYK | ❌ | ❌ | Not implemented |
| **Subsampling** ||||
| 4:4:4 | ✅ | ✅ | No subsampling |
| 4:2:0 | ✅ | ✅ | Most common |
| 4:2:2 | ✅ | ✅ | Horizontal only |
| 4:4:0 | ✅ | ✅ | Vertical only |
| **Quality Features** ||||
| Adaptive Quantization | ✅ | ✅ | Streaming AQ matches full-plane exactly |
| Optimized Huffman | ✅ | ✅ | jpegli & mozjpeg methods |
| Sharp YUV chroma | ✅ | ✅ | Via yuv crate |
| Custom quant tables | ✅ | ❌ | Hidden API, full-plane only |
| **Input Formats** ||||
| RGB/BGR 8-bit | ✅ | ✅ | |
| RGBA/BGRA 8-bit | ✅ | ✅ | Alpha ignored |
| 16-bit input | ❌ | ❌ | Not implemented |
| Float32 input | ❌ | ❌ | Not implemented |

### Decoder Features

| Feature | Status | Notes |
|---------|:------:|-------|
| **Modes** |||
| Baseline (8-bit) | ✅ | |
| Progressive | ✅ | |
| Extended (12-bit) | ⚠️ | Partial support |
| **Color Spaces** |||
| YCbCr → RGB | ✅ | |
| Grayscale | ✅ | |
| XYB (baseline) | ✅ | Works correctly |
| XYB (progressive) | ❌ | Known bug |
| CMYK | ⚠️ | Decode only |
| **Output Formats** |||
| 8-bit | ✅ | |
| 16-bit | ✅ | |
| Float32 | ✅ | Preserves internal precision |
| **Features** |||
| ICC profile extraction | ✅ | |
| ICC profile application | ✅ | Requires cms feature |
| Fancy upsampling | ✅ | |
| Memory limits (DoS protection) | ✅ | |

### SIMD Support

| Architecture | Dispatch | Status |
|--------------|----------|--------|
| x86-64 AVX2+FMA | Runtime | ✅ Tested |
| x86-64 SSE2 | Runtime | ✅ Tested |
| ARM64 NEON | Always | ✅ Tested |
| x86 32-bit | — | ❌ Not supported |

---

## Streaming AQ Implementation

The `StreamingAQ` implementation uses **rolling buffers** for low memory usage.

### Memory Model (4K image)

| Buffer | Full-Plane | StreamingAQ |
|--------|------------|-------------|
| Y plane (f32) | 33 MB | 0 |
| Y iMCU buffers | 0 | 490 KB (2×16 rows) |
| Pre-erosion (full) | 2 MB | 0 |
| Pre-erosion (rolling) | 0 | 45 KB (12 rows) |
| Row buffers | 0 | 60 KB |
| **Total AQ overhead** | **35 MB** | **~2.5 MB** |

### Features

- ✅ True constant memory (~2.5 MB for 4K vs 33 MB)
- ✅ Double-buffered Y iMCU data with lookahead
- ✅ All fallible allocations
- ✅ Both batch mode (`finalize()`) and incremental mode (`flush()`)
- ✅ Per-iMCU AQ output for immediate quantization

### Limitations

| Limitation | Reason | Impact |
|------------|--------|--------|
| Minor numerical differences | Row-by-row modulations | <0.1 max diff vs full-plane |
| 1 iMCU latency | Needs 4-row lookahead for fuzzy erosion | Slight delay |

### Usage

```rust
// Batch mode (drop-in replacement)
let mut aq = StreamingAQ::new(width, height, y_quant_01, v_samp)?;
for strip in strips {
    aq.process_y_strip(&strip, strip_y, strip_height);
}
let all_strengths = aq.finalize()?;

// Incremental mode (lowest memory)
for strip in strips {
    if let Some(strengths) = aq.process_y_strip(&strip, strip_y, strip_height) {
        // Quantize this iMCU immediately
    }
}
if let Some(strengths) = aq.flush() {
    // Handle last iMCU
}
```

---

## Known Bugs

| Issue | Severity | Status |
|-------|----------|--------|
| Progressive XYB decode fails | Medium | Open - baseline XYB works |
| XYB encoder quality gap (~5 SSIM2) | Low | Under investigation |
| retpoline cfg warnings from multiversion | Cosmetic | Won't fix (upstream issue) |

---

## Not Implemented

| Feature | Priority | Notes |
|---------|----------|-------|
| 16-bit encode input | Low | Would need pipeline changes |
| ICC profile embedding | Medium | Extract works, embed doesn't |
| CMYK encoding | Low | Rare use case |
| Lossless JPEG | None | Different algorithm entirely |
| Hybrid trellis quantization | Experimental | Behind feature flag |

---

## Performance Benchmarks

### Encoding Throughput (Release build, AVX2)

| Image | Full-Plane | Strip-Based | Winner |
|-------|------------|-------------|--------|
| 2K q75 | 94 MP/s | 94 MP/s | Tie |
| 4K q75 | 77 MP/s | 94 MP/s | Strip +22% |
| 12MP q75 | 65 MP/s | 85 MP/s | Strip +31% |

### vs C++ jpegli

| Image | Rust | C++ | Notes |
|-------|------|-----|-------|
| 2K | 94 MP/s | 60 MP/s | Rust 57% faster |
| 4K (full-plane) | 48 MP/s | 63 MP/s | C++ 31% faster |
| 4K (strip-based) | 94 MP/s | 63 MP/s | Rust 49% faster |

---

## Test Commands

```bash
# Run all tests
cargo test --release -p jpegli-rs

# Streaming AQ parity tests
cargo test --release -p jpegli-rs --lib streaming

# Allocation tracking
cargo run --release -p jpegli-rs --example alloc_tracker

# C++ parity (requires submodule)
cargo test --release -p jpegli-rs --features ffi-tests --test comprehensive_cpp_comparison -- --nocapture --ignored

# Benchmark strip vs full-plane
cargo run --release -p jpegli-rs --example bench_strip_vs_full
```
