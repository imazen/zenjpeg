# Context Handoff - zenjpeg UltraHDR Integration

## Session Summary

Added `ultrahdr` feature to zenjpeg integrating `ultrahdr-core` for HDR gain map support.

## Recent Commits (newest first)

```
97c80ba docs: clean up CLAUDE.md
15dbdf1 refactor(ultrahdr): update to renamed streaming API
5957920 Revert "perf: use linear-srgb mage module for FMA-accelerated sRGB conversion"
35bfdd4 test(ultrahdr): add thorough grayscale and gainmap verification tests
98d2d71 feat(ultrahdr): add streaming interfaces for low-memory processing
284b3a4 style: rustfmt formatting
94f5cb7 fix(ultrahdr): remove potential panics with bounds-checked access
718568e feat(ultrahdr): add reencode_ultrahdr() and roundtrip tests
8c471a2 feat(ultrahdr): add UltraHDR encoding/decoding support
```

## New Files

```
zenjpeg/src/ultrahdr/
├── mod.rs      # Re-exports from ultrahdr-core
├── encode.rs   # encode_ultrahdr(), encode_with_gainmap(), create_gainmap_computer()
└── decode.rs   # reconstruct_hdr(), create_hdr_reconstructor(), UltraHdrExtras trait

zenjpeg/tests/ultrahdr_roundtrip.rs  # 9 integration tests
```

## Key API (ultrahdr feature)

### Encoding
```rust
use zenjpeg::ultrahdr::{encode_ultrahdr, GainMapConfig, ToneMapConfig, Unstoppable};

let jpeg = encode_ultrahdr(&hdr, &GainMapConfig::default(), &ToneMapConfig::default(),
    &EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter), 75.0, Unstoppable)?;
```

### Decoding
```rust
use zenjpeg::ultrahdr::{reconstruct_hdr, UltraHdrExtras, HdrOutputFormat};

let decoded = Decoder::new().decode(&jpeg)?;
if decoded.extras().map(|e| e.is_ultrahdr()).unwrap_or(false) {
    let hdr = reconstruct_hdr(decoded.pixels(), w, h, extras, 4.0, HdrOutputFormat::LinearFloat, Unstoppable)?;
}
```

### Streaming (low-memory)
```rust
// For decode: RowDecoder - full gainmap in memory, process SDR rows -> HDR rows
let decoder = create_hdr_reconstructor(w, h, extras, 4.0, HdrOutputFormat::LinearFloat)?;

// For encode: RowEncoder - process HDR/SDR row pairs -> gainmap
let encoder = create_gainmap_computer(w, h, &config, format, transfer, gamut)?;

// Dual streaming: StreamDecoder/StreamEncoder - parallel decode of base+gainmap
```

## ultrahdr-core API Renames (just completed)

The user renamed streaming APIs in ultrahdr-core:
- `StreamingHdrReconstructor` → `RowDecoder`
- `StreamingGainMapComputer` → `RowEncoder`
- `InputConfig` → `DecodeInput`
- `EncoderInputConfig` → `EncodeInput`
- Added: `StreamDecoder`, `StreamEncoder` for dual streaming

zenjpeg updated to match in commit `15dbdf1`.

## Known Issues

1. **XYB quality gap** - ~5 SSIMULACRA2 behind C++ in XYB mode. Root cause TBD.

## SIMD Status

All SIMD intact - no changes to:
- `zenjpeg/src/quant/aq/simd.rs` (3362 lines) - AQ SIMD
- `zenjpeg/src/encode/mage_simd.rs` (1938 lines) - DCT SIMD
- Only change: whitespace fix in `streaming.rs` comment

## Tests

```bash
cargo test -p zenjpeg --features ultrahdr --test ultrahdr_roundtrip  # 9 tests pass
cargo build -p zenjpeg --features ultrahdr  # builds clean
```

## Next Steps

- User may want to add more streaming examples
- Memory analysis for large images (40MP = ~800MB peak for full-image path)
- Consider adding `encode_ultrahdr_streaming()` for truly low-memory large image encode

## Dependencies

- `ultrahdr-core` at `~/work/ultrahdr/ultrahdr-core` (path dependency)
- Requires `decoder` feature (gain map reconstruction needs decode)
