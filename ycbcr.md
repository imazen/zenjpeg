# YCbCr8 and YCbCrF32 Support Fix Plan

## Problem Summary

The v2 API's `PixelLayout::YCbCr8` and `PixelLayout::YCbCrF32` variants don't work because:

1. `PixelLayout.to_legacy()` maps both to `PixelFormat::Rgb` (3 bytes/pixel)
2. `StreamingEncoder` calculates `bytes_per_row` using the legacy format's `bytes_per_pixel()`
3. For YCbCrF32 (12 bytes/pixel), this causes `InvalidBufferSize { expected: 3072, actual: 12288 }`

**Location of buffer validation failure:**
- `zenjpeg/src/encode/streaming.rs:1009-1014` - validates against wrong `bytes_per_row`

## Root Cause

```rust
// zenjpeg/src/encode/v2/types.rs:507-508
Self::YCbCr8 | Self::YCbCrF32 => crate::types::PixelFormat::Rgb,  // WRONG

// zenjpeg/src/encode/streaming.rs:865
let bytes_per_row = width * builder.pixel_format.bytes_per_pixel();  // Uses 3 instead of 12
```

## Fix Plan

### Step 1: Add YCbCr variants to legacy PixelFormat

**File:** `zenjpeg/src/types.rs`

```rust
pub enum PixelFormat {
    // ... existing variants ...

    // === Pre-converted YCbCr (skip RGB->YCbCr conversion) ===
    /// YCbCr interleaved, 3 bytes per pixel, u8
    YCbCr8,
    /// YCbCr interleaved, 12 bytes per pixel, f32
    YCbCrF32,
}
```

### Step 2: Update PixelFormat methods

**File:** `zenjpeg/src/types.rs`

```rust
impl PixelFormat {
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            // ... existing ...
            Self::YCbCr8 => 3,      // ADD
            Self::YCbCrF32 => 12,   // ADD
        }
    }

    pub const fn num_channels(self) -> usize {
        match self {
            // ... existing ...
            Self::YCbCr8 | Self::YCbCrF32 => 3,  // ADD
        }
    }

    pub const fn color_space(self) -> ColorSpace {
        match self {
            // ... existing ...
            Self::YCbCr8 | Self::YCbCrF32 => ColorSpace::YCbCr,  // ADD (may need new variant)
        }
    }

    pub const fn is_grayscale(self) -> bool {
        // No change needed - YCbCr is not grayscale
    }

    pub const fn is_fast_path(self) -> bool {
        // YCbCr is fast path (no conversion needed)
        matches!(self, /* existing */ | Self::YCbCr8 | Self::YCbCrF32)
    }
}
```

### Step 3: Add ColorSpace::YCbCr variant (if needed)

**File:** `zenjpeg/src/types.rs`

Check if `ColorSpace` enum exists and add `YCbCr` variant if not present.

### Step 4: Update PixelLayout.to_legacy()

**File:** `zenjpeg/src/encode/v2/types.rs`

```rust
pub fn to_legacy(&self) -> crate::types::PixelFormat {
    match self {
        // ... existing ...
        Self::YCbCr8 => crate::types::PixelFormat::YCbCr8,      // FIX
        Self::YCbCrF32 => crate::types::PixelFormat::YCbCrF32,  // FIX
    }
}
```

### Step 5: Handle YCbCr in strip processor

**File:** `zenjpeg/src/encode/strip/convert.rs`

Add handlers in `convert_strip_to_ycbcr()` match statement:

```rust
match self.pixel_format {
    // ... existing handlers ...

    PixelFormat::YCbCr8 => {
        // Direct copy - input is already YCbCr, just needs level shift
        // Y: copy to y_strip with strided layout
        // Cb/Cr: copy to cb_strip/cr_strip
        for row in 0..strip_height {
            let src_row_start = row * width * 3;
            let y_row_start = row * padded_width;
            let cbcr_row_start = row * width;
            for x in 0..width {
                let idx = src_row_start + x * 3;
                self.y_strip[y_row_start + x] = rgb_strip[idx] as f32;
                self.cb_strip[cbcr_row_start + x] = rgb_strip[idx + 1] as f32;
                self.cr_strip[cbcr_row_start + x] = rgb_strip[idx + 2] as f32;
            }
            // Edge-pad Y row
            if width < padded_width {
                let edge_val = self.y_strip[y_row_start + width - 1];
                for x in width..padded_width {
                    self.y_strip[y_row_start + x] = edge_val;
                }
            }
        }
    }

    PixelFormat::YCbCrF32 => {
        // Direct copy from f32 YCbCr - may need level shift depending on input range
        // Assumes input is [0, 255] range (JPEG convention)
        for row in 0..strip_height {
            let src_row_start = row * width * 12;  // 12 bytes per pixel
            let y_row_start = row * padded_width;
            let cbcr_row_start = row * width;
            for x in 0..width {
                let base = src_row_start + x * 12;
                let y = f32::from_ne_bytes([rgb_strip[base], rgb_strip[base+1], rgb_strip[base+2], rgb_strip[base+3]]);
                let cb = f32::from_ne_bytes([rgb_strip[base+4], rgb_strip[base+5], rgb_strip[base+6], rgb_strip[base+7]]);
                let cr = f32::from_ne_bytes([rgb_strip[base+8], rgb_strip[base+9], rgb_strip[base+10], rgb_strip[base+11]]);

                self.y_strip[y_row_start + x] = y;
                self.cb_strip[cbcr_row_start + x] = cb;
                self.cr_strip[cbcr_row_start + x] = cr;
            }
            // Edge-pad Y row
            if width < padded_width {
                let edge_val = self.y_strip[y_row_start + width - 1];
                for x in width..padded_width {
                    self.y_strip[y_row_start + x] = edge_val;
                }
            }
        }
    }
}
```

### Step 6: Update other match statements

Search for all `match.*pixel_format` and `match.*PixelFormat` to ensure YCbCr8/YCbCrF32 are handled:

- `zenjpeg/src/encode/strip/convert.rs` - gamma-aware downsampling (skip for YCbCr)
- `zenjpeg/src/encode/strip/mod.rs` - any format-specific logic
- Any XYB-related code (YCbCr input incompatible with XYB mode)

### Step 7: Enable test

**File:** `zenjpeg/tests/encode_api.rs`

Remove `#[ignore]` from `test_encode_ycbcr_f32_input`.

### Step 8: Verify

```bash
cargo test --release -p zenjpeg test_encode_ycbcr
cargo clippy -- -D warnings
```

## Testing Notes

- YCbCr8 test already passes (uses 3 bytes/pixel same as Rgb)
- YCbCrF32 test currently fails with buffer size mismatch
- After fix, both should produce valid JPEGs
- Decoded output won't match RGB input exactly (different quantization paths)

## Open Questions

1. **Input range for YCbCrF32:** Should it expect [0, 255] or [-128, 127] centered?
   - JPEG internal uses [0, 255] with level shift
   - Some video formats use centered
   - Recommend: document as [0, 255] to match JPEG convention

2. **XYB mode:** Should we error or silently fall back when YCbCr input + XYB mode?
   - Recommend: return error - XYB requires RGB input for color transform
