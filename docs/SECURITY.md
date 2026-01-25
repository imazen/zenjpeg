# zenjpeg Security Analysis & Priorities

## Executive Summary

This document outlines security vulnerabilities in zenjpeg and remediation priorities based on libjpeg-turbo's battle-tested approach.

## libjpeg-turbo Reference

libjpeg-turbo uses several mechanisms to prevent DoS attacks:

| Mechanism | Value | Purpose |
|-----------|-------|---------|
| `JPEG_MAX_DIMENSION` | 65500 | Max width/height (just under 64K to prevent overflow) |
| `max_memory_to_use` | Application-set | Limit virtual array allocation |
| `TJPARAM_MAXMEMORY` | In megabytes | TurboJPEG API memory limit |
| `TJPARAM_MAXPIXELS` | Total pixels | Limits allocation tied to dimensions |

**Key insight from libjpeg-turbo maintainer**: "Applications wishing to guard against [DoS] should set `cinfo->mem->max_memory_to_use`" - responsibility is shared between library and application.

Sources:
- [libjpeg-turbo jmorecfg.h](https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/jmorecfg.h)
- [Issue #249: Memory DoS](https://github.com/libjpeg-turbo/libjpeg-turbo/issues/249)
- [Issue #735: TurboJPEG memory limits](https://github.com/libjpeg-turbo/libjpeg-turbo/issues/735)
- [Mozilla Bug 1252200](https://bugzilla.mozilla.org/show_bug.cgi?id=1252200)

---

## Priority 0: Critical (Must Fix Before Production Use)

### P0-1: Use Fallible Allocation (`try_reserve`)

**Problem**: All `vec![]` allocations panic on OOM, causing ungraceful crashes.

**Solution**: Use Rust's stable fallible allocation APIs.

```rust
// BEFORE (panics on OOM)
let mut plane = vec![0u8; width * height];

// AFTER (returns error on OOM)
let mut plane = Vec::new();
plane.try_reserve_exact(width * height)
    .map_err(|_| Error::AllocationFailed { bytes: width * height })?;
plane.resize(width * height, 0u8);

// Or use a helper function
fn try_alloc_zeroed<T: Default + Clone>(count: usize) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(count)
        .map_err(|_| Error::AllocationFailed { bytes: count * std::mem::size_of::<T>() })?;
    v.resize(count, T::default());
    Ok(v)
}
```

**Files to update**:
- `decode.rs`: Lines 674, 757, 940, 977, 988, 1004
- `encode.rs`: Lines 227, 464, 499, 558, 1125-1126, 1792-1794
- `adaptive_quant.rs`: Lines 135, 310, 427, 525
- `color.rs`: Lines 335, 404
- `xyb.rs`: Line 281

**Rust API Reference**: [Vec::try_reserve](https://doc.rust-lang.org/std/vec/struct.Vec.html#method.try_reserve)

---

### P0-2: Add Maximum Pixel Limit

**Problem**: 65535×65535 image = 4.3 billion pixels = 12.8GB RGB buffer.

**Solution**: Add configurable pixel limit (like `TJPARAM_MAXPIXELS`).

```rust
// In consts.rs
/// Default maximum pixels (100 megapixels)
/// Matches common server-side limits
pub const DEFAULT_MAX_PIXELS: u64 = 100_000_000;

/// Absolute maximum per JPEG spec (65500 × 65500)
pub const JPEG_MAX_DIMENSION: u32 = 65500;

// In decode.rs validation
fn validate_dimensions(&self, width: u32, height: u32, max_pixels: u64) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(Error::InvalidDimensions { width, height, reason: "zero dimension" });
    }
    if width > JPEG_MAX_DIMENSION || height > JPEG_MAX_DIMENSION {
        return Err(Error::InvalidDimensions {
            width, height,
            reason: "exceeds JPEG_MAX_DIMENSION (65500)"
        });
    }
    let total_pixels = (width as u64).checked_mul(height as u64)
        .ok_or(Error::InvalidDimensions { width, height, reason: "pixel count overflow" })?;
    if total_pixels > max_pixels {
        return Err(Error::ImageTooLarge {
            pixels: total_pixels,
            max: max_pixels
        });
    }
    Ok(())
}
```

---

### P0-3: Checked Arithmetic for All Size Calculations

**Problem**: `width * height * 3` can overflow on 32-bit systems.

**Solution**: Use `checked_mul` chains.

```rust
// BEFORE
let size = width * height * bytes_per_pixel;

// AFTER
let size = width
    .checked_mul(height)
    .and_then(|p| p.checked_mul(bytes_per_pixel))
    .ok_or(Error::DimensionOverflow)?;
```

**Create helper**:
```rust
/// Calculate allocation size with overflow checking
pub fn checked_alloc_size(width: usize, height: usize, bpp: usize) -> Result<usize> {
    width
        .checked_mul(height)
        .and_then(|p| p.checked_mul(bpp))
        .ok_or(Error::AllocationOverflow {
            width, height, bpp,
            reason: "size calculation overflow"
        })
}
```

---

## Priority 1: High (Should Fix Soon)

### P1-1: Add Memory Budget Configuration

**Problem**: No way to limit total memory usage like libjpeg-turbo's `max_memory_to_use`.

**Solution**: Add `DecoderConfig::max_memory_bytes` option.

```rust
pub struct DecoderConfig {
    // ... existing fields ...

    /// Maximum memory to use for decoding (default: 1GB)
    /// Set to 0 for unlimited
    pub max_memory_bytes: usize,

    /// Maximum pixels to decode (default: 100MP)
    pub max_pixels: u64,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            // ...
            max_memory_bytes: 1024 * 1024 * 1024, // 1GB
            max_pixels: 100_000_000, // 100MP
        }
    }
}
```

---

### P1-2: Limit ICC Profile Size

**Problem**: ICC profiles concatenated without limit.

**Solution**: Add size cap (16MB is generous; typical profiles are <1MB).

```rust
// In icc.rs
const MAX_ICC_PROFILE_SIZE: usize = 16 * 1024 * 1024; // 16MB

pub fn extract_icc_profile(jpeg_data: &[u8]) -> Option<Vec<u8>> {
    let mut chunks: Vec<(u8, Vec<u8>)> = Vec::new();
    let mut total_size: usize = 0;

    // ... existing parsing ...

    // Check accumulated size
    total_size = total_size.saturating_add(icc_data.len());
    if total_size > MAX_ICC_PROFILE_SIZE {
        return None; // Or return Result with error
    }

    // ...
}
```

---

### P1-3: Limit Progressive Scan Count

**Problem**: Malformed progressive JPEGs can have excessive scans.

**Solution**: Limit number of scans (libjpeg-turbo allows hundreds but malicious files have thousands).

```rust
const MAX_SCANS: usize = 256; // Generous limit

// In decode loop
let mut scan_count = 0;
loop {
    match marker {
        MARKER_SOS => {
            scan_count += 1;
            if scan_count > MAX_SCANS {
                return Err(Error::TooManyScans { count: scan_count });
            }
            self.parse_scan()?;
        }
        // ...
    }
}
```

---

## Priority 2: Medium (Important for Hardening)

### P2-1: Add Fuzz Testing

Create `fuzz/` directory with libFuzzer targets:

```rust
// fuzz/fuzz_targets/decode_jpeg.rs
#![no_main]
use libfuzzer_sys::fuzz_target;
use zenjpeg::Decoder;

fuzz_target!(|data: &[u8]| {
    let decoder = Decoder::new()
        .max_memory(64 * 1024 * 1024)  // 64MB limit for fuzzing
        .max_pixels(10_000_000);        // 10MP limit
    let _ = decoder.decode(data);
});
```

```toml
# fuzz/Cargo.toml
[package]
name = "jpegli-fuzz"
version = "0.0.0"
edition = "2021"

[dependencies]
libfuzzer-sys = "0.4"
jpegli = { path = "../jpegli" }

[[bin]]
name = "decode_jpeg"
path = "fuzz_targets/decode_jpeg.rs"
test = false
doc = false
```

Run with: `cargo +nightly fuzz run decode_jpeg`

---

### P2-2: Add Huffman Tree Construction Limit

**Problem**: Retry loop can iterate many times with adversarial input.

**Solution**: Add iteration cap.

```rust
// In huffman.rs build_code_lengths()
const MAX_TREE_RETRIES: usize = 32;

let mut retry_count = 0;
loop {
    retry_count += 1;
    if retry_count > MAX_TREE_RETRIES {
        // Force-clamp depths and exit
        for d in &mut depth {
            *d = (*d).min(max_length);
        }
        break;
    }
    // ... existing loop body ...
}
```

---

### P2-3: Validate Restart Marker Sequence

**Problem**: Restart markers (RST0-RST7) not validated against expected sequence.

**Solution**: Track and validate sequence.

```rust
struct EntropyDecoder {
    // ...
    expected_restart_num: u8,
}

fn handle_restart_marker(&mut self, marker: u8) -> Result<()> {
    let rst_num = marker - 0xD0;
    if rst_num != self.expected_restart_num {
        // Warning, not error - matches libjpeg behavior
        log::warn!("unexpected restart marker: got RST{}, expected RST{}",
                   rst_num, self.expected_restart_num);
    }
    self.expected_restart_num = (self.expected_restart_num + 1) & 7;
    Ok(())
}
```

---

## Priority 3: Low (Nice to Have)

### P3-1: Add Allocation Tracking

Track total allocations to enforce `max_memory_bytes`:

```rust
struct AllocationTracker {
    total_allocated: usize,
    max_allowed: usize,
}

impl AllocationTracker {
    fn try_allocate(&mut self, bytes: usize) -> Result<()> {
        let new_total = self.total_allocated.checked_add(bytes)
            .ok_or(Error::AllocationOverflow)?;
        if new_total > self.max_allowed {
            return Err(Error::MemoryLimitExceeded {
                requested: bytes,
                total: new_total,
                limit: self.max_allowed
            });
        }
        self.total_allocated = new_total;
        Ok(())
    }
}
```

---

### P3-2: Add `-strict` Mode

Like libjpeg-turbo's `djpeg -strict`, make warnings fatal:

```rust
pub struct DecoderConfig {
    // ...
    /// Treat warnings as errors (matches libjpeg -strict)
    pub strict: bool,
}
```

---

## New Error Variants Needed

```rust
pub enum Error {
    // ... existing variants ...

    /// Allocation failed (OOM or limit exceeded)
    AllocationFailed { bytes: usize },

    /// Size calculation overflowed
    AllocationOverflow { width: usize, height: usize, bpp: usize },

    /// Image exceeds pixel limit
    ImageTooLarge { pixels: u64, max: u64 },

    /// Memory limit exceeded
    MemoryLimitExceeded { requested: usize, total: usize, limit: usize },

    /// Too many progressive scans
    TooManyScans { count: usize },

    /// ICC profile too large
    IccProfileTooLarge { size: usize, max: usize },
}
```

---

## Implementation Checklist

- [ ] **P0-1**: Replace `vec![]` with `try_reserve` pattern in all files
- [ ] **P0-2**: Add `JPEG_MAX_DIMENSION` and `max_pixels` validation
- [ ] **P0-3**: Use `checked_mul` for all size calculations
- [ ] **P1-1**: Add `DecoderConfig::max_memory_bytes`
- [ ] **P1-2**: Limit ICC profile to 16MB
- [ ] **P1-3**: Limit scans to 256
- [ ] **P2-1**: Set up cargo-fuzz infrastructure
- [ ] **P2-2**: Cap Huffman tree retries at 32
- [ ] **P2-3**: Validate restart marker sequence
- [ ] **P3-1**: Implement allocation tracking
- [ ] **P3-2**: Add strict mode

---

## References

- [libjpeg-turbo jmorecfg.h - JPEG_MAX_DIMENSION](https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/jmorecfg.h)
- [libjpeg-turbo Issue #249 - Memory DoS](https://github.com/libjpeg-turbo/libjpeg-turbo/issues/249)
- [libjpeg-turbo Issue #735 - TJPARAM_MAXMEMORY](https://github.com/libjpeg-turbo/libjpeg-turbo/issues/735)
- [Mozilla Bug 1252200 - DoS via small image with large dimensions](https://bugzilla.mozilla.org/show_bug.cgi?id=1252200)
- [RFC 2116 - Fallible Collection Allocation](https://rust-lang.github.io/rfcs/2116-alloc-me-maybe.html)
- [Vec::try_reserve documentation](https://doc.rust-lang.org/std/vec/struct.Vec.html#method.try_reserve)
- [Microsoft fallible_vec crate](https://github.com/microsoft/rust_fallible_vec)
