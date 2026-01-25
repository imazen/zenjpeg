# Context Handoff - zenjpeg UltraHDR Streaming

## Session Summary (2026-01-25)

Fixed critical MPF offset bug in both zenjpeg and ultrahdr-core that prevented gain map extraction from third-party UltraHDR files (e.g., gen-dress.jpg from Android).

## Bug Fixed: MPF Secondary Image Offsets

**Root cause:** MPF offsets for secondary images are relative to the TIFF header within the MPF segment (after "MPF\0"), NOT absolute file offsets or relative to APP2 marker.

**Calculation:** `absolute_offset = tiff_header_pos + mpf_entry_offset`

For gen-dress.jpg:
- MPF marker at offset 69431
- TIFF header at 69439 (marker + 4 for length + 4 for "MPF\0")
- MPF entry offset: 2860045
- Correct gain map SOI: 69439 + 2860045 = 2929484

### zenjpeg Fix

**Files modified:**
- `zenjpeg/src/decode/parser/mod.rs` - Added `mpf_header_pos: usize` field
- `zenjpeg/src/decode/parser/markers.rs` - Record position when processing APP2 MPF

```rust
// In extract_mpf_secondary_images:
let mpf_base = self.mpf_header_pos;
let offset = if entry.offset == 0 {
    primary_eoi_pos
} else if mpf_base > 0 {
    mpf_base + entry.offset as usize
} else {
    entry.offset as usize
};
```

### ultrahdr-core Fix

**Files modified:**
- `ultrahdr-core/src/metadata/mpf.rs` - Multiple fixes:
  1. Parser: Track `tiff_header_pos` and use for offset calculation
  2. Encoder: Fixed VERSION tag (was appended after IFD entry, corrupting structure)
  3. Encoder: Fixed MP_ENTRY offset calculation (+16 not +8)
  4. Changed `parse_mpf()` return from `(offset, size)` to `(start, end)`
  5. Added `mpf_insert_offset` parameter to `create_mpf_header()`

- `ultrahdr/src/container.rs` - Use TIFF header position for offset calculations
- `ultrahdr/src/encode.rs` - Updated to use new API

## Tests Added

- `zenjpeg/tests/grayscale_decode_test.rs` - Tests for:
  - `test_ultrahdr_gainmap_extraction` - Verify gain map bytes extracted
  - `test_gainmap_grayscale_decode_streaming` - Full decode of grayscale gain map

## Next Steps: Streaming Encoder

The plan at `/home/lilith/.claude/plans/wise-forging-sonnet.md` defines the streaming encoder API.

### UltraHdrEncoder API (from plan)

```rust
pub struct UltraHdrEncoder {
    // Internally uses RowEncoder or StreamEncoder from ultrahdr-core
}

impl UltraHdrEncoder {
    pub fn new(
        width: u32,
        height: u32,
        mode: UltraHdrEncodeMode,
        config: UltraHdrEncodeConfig,
    ) -> Result<Self>;

    pub fn push_hdr_rows(&mut self, pixels: &[f32], rows: usize) -> Result<()>;
    pub fn push_sdr_rows(&mut self, pixels: &[u8], rows: usize) -> Result<()>;
    pub fn finish(self) -> Result<Vec<u8>>;
}
```

### Files to Create/Modify

1. **New:** `zenjpeg/src/ultrahdr/streaming_encode.rs` - Core streaming encoder
2. **Modify:** `zenjpeg/src/ultrahdr/mod.rs` - Re-export streaming types

### Implementation Notes

- Use `RowEncoder` from ultrahdr-core for HDR→SDR+gainmap computation
- Integrate with zenjpeg's `ScanlineEncoder` for streaming JPEG output
- MPF header creation now requires insert offset (fixed in this session)
- Consider `assemble_ultrahdr()` utility for post-encoding MPF insertion

## Commits This Session

Check git log in both repos:
- `/home/lilith/work/zenjpeg/` - MPF parser fix + grayscale tests
- `/home/lilith/work/ultrahdr/` - MPF parser/encoder fixes

## Test Commands

```bash
# zenjpeg tests
cargo test -p zenjpeg --features ultrahdr --test grayscale_decode_test

# ultrahdr-core tests
cd ~/work/ultrahdr && cargo test -p ultrahdr-core
cd ~/work/ultrahdr && cargo test -p ultrahdr

# Real-world test file
# /mnt/v/gen-dress.jpg - Android UltraHDR photo
```
