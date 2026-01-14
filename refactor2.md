# V2 as Internal Truth Refactoring

## Goal

Make the v2 encoder types (`EncoderConfig`, `BytesEncoder`, etc.) the internal truth,
with `StreamingEncoder` becoming a thin delegation layer.

## Current Architecture

```
encoder/mod.rs                    # Public API - re-exports from encode::v2
  └─ pub use encode::v2::*

encode/
  ├─ v2/                          # V2 API types (thin wrappers)
  │   ├─ config.rs                # EncoderConfig
  │   ├─ types.rs                 # Quality, PixelLayout, ChromaSubsampling
  │   └─ encoders.rs              # BytesEncoder wraps StreamingEncoder
  │
  ├─ streaming.rs                 # StreamingEncoder (ACTUAL LOGIC)
  │   └─ uses StripProcessor
  │   └─ uses Encoder (legacy)
  │
  ├─ strip/                       # Strip-based processing
  │   └─ StripProcessor           # Core encoding engine
  │
  ├─ config.rs                    # Legacy EncoderConfig
  └─ mod.rs                       # Encoder (legacy, deprecated)
```

**The Problem:** BytesEncoder → StreamingEncoder → StripProcessor
- V2 is just a facade over legacy implementation
- All actual encoding logic lives in StreamingEncoder

## Target Architecture

```
encoder/mod.rs                    # Public API - re-exports from encode
  └─ pub use encode::*

encode/
  ├─ config.rs                    # EncoderConfig (from v2, is the truth)
  ├─ types.rs                     # V2 types (Quality, PixelLayout, etc.)
  ├─ encoders.rs                  # BytesEncoder (CONTAINS ACTUAL LOGIC)
  │   └─ uses StripProcessor directly
  │
  ├─ streaming.rs                 # StreamingEncoder (DELEGATES to BytesEncoder)
  │   └─ deprecated compat wrapper
  │
  └─ strip/                       # Strip-based processing (unchanged)
```

**The Goal:** BytesEncoder → StripProcessor (direct)
             StreamingEncoder → BytesEncoder (delegation)

## Implementation Plan

### Step 1: Move V2 types to encode root

```bash
# Move files
mv encode/v2/types.rs    encode/encoder_types.rs
mv encode/v2/config.rs   encode/encoder_config.rs
mv encode/v2/encoders.rs encode/byte_encoders.rs
```

Update imports in moved files to use `crate::encode::*` instead of `super::*`.

### Step 2: BytesEncoder uses StripProcessor directly

Currently BytesEncoder does:
```rust
let inner = StreamingEncoder::new(...).start()?;
self.inner.push_rows(...);
self.inner.finish();
```

Change to:
```rust
let processor = StripProcessor::new(...);
// Direct row buffer management
// Direct strip processing
processor.finish();
```

Copy the row buffering and strip coordination logic from StreamingEncoder.

### Step 3: StreamingEncoder delegates to BytesEncoder

```rust
#[deprecated]
pub struct StreamingEncoder {
    inner: BytesEncoder,  // Now delegates to v2
}

impl StreamingEncoder {
    pub fn push_row(&mut self, row: &[u8]) -> Result<()> {
        self.inner.push(row, 1, row.len(), Unstoppable)
    }

    pub fn finish(self) -> Result<Vec<u8>> {
        self.inner.finish()
    }
}
```

### Step 4: Update encode/mod.rs re-exports

```rust
// Re-export v2 types directly
pub use encoder_config::EncoderConfig;
pub use encoder_types::{Quality, PixelLayout, ChromaSubsampling, ...};
pub use byte_encoders::{BytesEncoder, RgbEncoder, YCbCrPlanarEncoder};

// Keep legacy for compat
#[deprecated]
pub use streaming::StreamingEncoder;
```

### Step 5: Update encoder/mod.rs

```rust
// Just re-export from encode
pub use crate::encode::{
    EncoderConfig,
    BytesEncoder, RgbEncoder, YCbCrPlanarEncoder,
    Quality, PixelLayout, ChromaSubsampling,
    // etc.
};
```

## Key Logic to Move

From `StreamingEncoder::from_builder()`:
1. StripProcessor creation
2. Quantization table generation
3. Row buffer allocation
4. Zero bias computation

From `StreamingEncoder::push_row_with_stop()`:
1. Row buffering into strip
2. Strip flush when full

From `StreamingEncoder::finish()`:
1. Final strip flush
2. Encoder::finish() call

## Files to Modify

| File | Action |
|------|--------|
| `encode/v2/types.rs` | Move to `encode/encoder_types.rs` |
| `encode/v2/config.rs` | Move to `encode/encoder_config.rs` |
| `encode/v2/encoders.rs` | Move to `encode/byte_encoders.rs`, add logic |
| `encode/streaming.rs` | Gut and delegate to BytesEncoder |
| `encode/mod.rs` | Update re-exports |
| `encoder/mod.rs` | Update re-exports |

## Constraints

- Public API (`jpegli::encoder::*`) unchanged
- `StreamingEncoder` deprecated but functional
- All tests pass
- No duplicate code

## Testing Strategy

1. After each step, run `cargo test --release`
2. Specifically test:
   - `cargo test --release --lib` (unit tests)
   - `cargo test --release --test codec_coverage` (integration)
   - `cargo test --release -- streaming` (StreamingEncoder compat)
