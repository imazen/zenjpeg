# Lossless JPEG Transforms — Design Exploration

## Motivation

Two use cases:

1. **End-to-end lossless JPEG rotation/flip**: JPEG in → transform DCT coefficients → JPEG out.
   No decode to pixels, no re-encode. Zero generation loss. 3-5x faster than decode+rotate+encode.

2. **Fast EXIF orientation pre-streaming**: When a JPEG has EXIF orientation != 1, apply the
   rotation/flip in DCT domain before streaming decode begins. This avoids the current problem
   where Rotate90/Rotate270 require FatStrip (full image buffering) in zenimage's streaming pipeline.

## How It Works

JPEG stores image data as 8×8 blocks of DCT (Discrete Cosine Transform) coefficients. The 2D DCT
basis functions have a symmetry property:

- **Even-indexed** basis functions (0, 2, 4, 6) are symmetric about the block center
- **Odd-indexed** basis functions (1, 3, 5, 7) are antisymmetric (flip sign when mirrored)

This means we can flip/rotate the spatial content of a block by selectively negating coefficients
and/or transposing their positions — without ever computing the inverse DCT.

### The Seven Transforms

Each transform = block rearrangement on the image grid + coefficient manipulation within each block.

Coefficients are in an 8×8 matrix. "Row i, column j" refers to the DCT frequency indices.

| Transform   | Block grid change             | Per-block coefficient change             |
|-------------|-------------------------------|------------------------------------------|
| H-flip      | Mirror columns                | Negate odd columns (j=1,3,5,7)           |
| V-flip      | Mirror rows                   | Negate odd rows (i=1,3,5,7)              |
| Transpose   | Swap (bx,by) ↔ (by,bx)       | Transpose matrix (swap i↔j, no negation) |
| Rotate 90°  | Transpose + mirror cols       | Transpose + negate odd original rows     |
| Rotate 180° | Mirror rows + cols            | Negate where (i+j) is odd                |
| Rotate 270° | Transpose + mirror rows       | Transpose + negate odd columns           |
| Transverse  | Transpose + mirror both       | Transpose + negate where (i+j) is odd    |

### EXIF Orientation Mapping

| EXIF | Meaning          | DCT Transform |
|------|------------------|---------------|
| 1    | Normal           | None          |
| 2    | Flip horizontal  | H-flip        |
| 3    | Rotate 180°      | Rotate 180°   |
| 4    | Flip vertical    | V-flip        |
| 5    | Transpose        | Transpose     |
| 6    | Rotate 90° CW    | Rotate 90°    |
| 7    | Transverse       | Transverse    |
| 8    | Rotate 270° CW   | Rotate 270°   |

### Edge Handling (Non-MCU-Aligned Dimensions)

When image dimensions aren't multiples of the MCU size (8 for 4:4:4, 16 for 4:2:0), partial
blocks at edges cause issues during transforms that move edges:

- **Trim mode**: Discard partial MCU strips. Output is slightly smaller.
- **Perfect mode**: Refuse if not aligned.
- **Default**: Leave partial edge blocks untransformed (preserves reversibility).

For EXIF orientation, most cameras produce MCU-aligned images, so this is rarely an issue.

### Zigzag Order Complication

zenjpeg stores coefficients in **zigzag scan order**, not row-major 8×8. The transform operations
(transpose, negate odd rows/cols) are defined in terms of the 8×8 matrix position (row i, col j).

We need lookup tables that map zigzag positions to their 8×8 (row, col) coordinates.

From `JPEG_NATURAL_ORDER`: zigzag index z → linear index `JPEG_NATURAL_ORDER[z]`,
where linear index = `row * 8 + col`.

So for zigzag index z: `row = JPEG_NATURAL_ORDER[z] / 8`, `col = JPEG_NATURAL_ORDER[z] % 8`.

**Optimization**: Pre-compute permutation tables for each transform. A transform maps
zigzag index z to a new zigzag index z', with optional sign flip. This can be a `[(u8, bool); 64]`
lookup table — one load + conditional negate per coefficient.

## Implementation in zenjpeg

### What Already Exists

1. **`decode_coefficients()`** — Decodes JPEG to `DecodedCoefficients` with:
   - `components: Vec<ComponentCoefficients>` — per-component `Vec<i16>` in zigzag order
   - `quant_tables: Vec<Option<[u16; 64]>>` — quantization tables
   - Block grid: `blocks_wide`, `blocks_high`, `h_samp`, `v_samp`

2. **Entropy encoder** — `encode_block()` takes `&[i16; 64]` in zigzag order

3. **JPEG serializer** — Writes SOI, DQT, DHT, SOF, SOS, markers

4. **Extras preservation** — `PreserveConfig` + `DecodedExtras` + `EncoderSegments` for
   round-tripping EXIF, ICC, XMP, IPTC, MPF

### What's Needed

A new module `zenjpeg::lossless` (or `zenjpeg::transform`) that:

1. Takes JPEG bytes + transform type
2. Parses headers and Huffman-decodes to `[i16; 64]` blocks (no IDCT)
3. Rearranges blocks on the grid and manipulates coefficients
4. Re-encodes with optimized Huffman tables
5. Copies all metadata markers (updating EXIF orientation to 1)
6. Returns new JPEG bytes

### API Sketch

```rust
/// Lossless JPEG transform operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LosslessTransform {
    /// No transform (useful for Huffman re-optimization)
    None,
    /// Horizontal flip
    FlipHorizontal,
    /// Vertical flip
    FlipVertical,
    /// Transpose (swap rows/columns)
    Transpose,
    /// Rotate 90° clockwise
    Rotate90,
    /// Rotate 180°
    Rotate180,
    /// Rotate 270° clockwise (= 90° counter-clockwise)
    Rotate270,
    /// Transverse (transpose + rotate 180°)
    Transverse,
    /// Apply EXIF orientation, then reset it to 1
    ApplyExifOrientation,
}

/// How to handle images with non-MCU-aligned dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeHandling {
    /// Trim partial MCU blocks (output may be slightly smaller)
    Trim,
    /// Error if dimensions aren't MCU-aligned
    Perfect,
    /// Best-effort: transform what's possible
    BestEffort,
}

/// Configuration for lossless JPEG transforms.
pub struct TransformConfig {
    pub transform: LosslessTransform,
    pub edge_handling: EdgeHandling,
    /// Whether to optimize Huffman tables (slightly slower, smaller output)
    pub optimize_huffman: bool,
    /// Whether to preserve metadata (EXIF, ICC, XMP, etc.)
    pub preserve_metadata: bool,
}

/// Perform a lossless JPEG transform.
pub fn transform(
    jpeg_data: &[u8],
    config: &TransformConfig,
    stop: impl Stop,
) -> Result<Vec<u8>>;

/// Apply EXIF orientation losslessly (convenience function).
pub fn apply_exif_orientation(
    jpeg_data: &[u8],
    stop: impl Stop,
) -> Result<Vec<u8>>;
```

### For zenimage Integration (EXIF Pre-Streaming)

The second use case — applying EXIF orientation before streaming decode — is trickier because
we want to avoid buffering the entire decoded image. Options:

**Option A: Full lossless transform (rewrite JPEG, then stream-decode the result)**
- Pro: Cleanest. The streaming decoder sees a normal, correctly-oriented JPEG.
- Con: Requires writing the transformed JPEG to a buffer first. Adds latency.
- Con: Allocates memory for the full compressed JPEG (but much less than a decoded frame).

**Option B: Coefficient-level transform during streaming decode**
- Intercept coefficients after Huffman decode, transform them, then IDCT.
- Pro: No extra JPEG write step.
- Con: Requires modifying the decoder pipeline. Transpose/rotation changes image dimensions
  and block ordering, so the decoder's strip output would need to account for this.
- Con: Very invasive change to the decoder.

**Option C: Keep current approach (pixel-level transform in zenimage)**
- For flips (H, V, 180°): already streaming-efficient (ThinStrip).
- For rotations (90°, 270°, transpose): requires FatStrip (full image buffer).
- Pro: No changes to zenjpeg.
- Con: 90°/270° rotation can't stream.

**Recommendation**: Start with **Option A** for the `transform()` API. For zenimage integration,
use Option A when the overhead is acceptable (compressed JPEG is much smaller than decoded pixels),
and fall back to Option C for non-JPEG formats.

## Performance Expectations

Lossless transform skips:
- IDCT (~40% of decode time)
- Forward DCT + quantization (~40% of encode time)
- Color space conversion (~15% of decode/encode time)

What it still does:
- Huffman decode (serial, ~30% of decode time)
- Block rearrangement + coefficient negation (trivial, memory-bandwidth limited)
- Huffman re-encode (~30% of encode time)

**Expected: 3-5x faster than decode+rotate+encode.**

Memory: Must buffer all coefficients. For a 4000×3000 4:2:0 JPEG: ~35 MB of i16 coefficients.
This is comparable to a decoded pixel buffer but we avoid needing both source and destination buffers.

## Implementation Plan

### Phase 1: Core Transform Logic (this exploration)
- [ ] Implement coefficient manipulation for all 7 transforms
- [ ] Handle zigzag ↔ 8×8 mapping with precomputed permutation tables
- [ ] Unit tests with known coefficient patterns

### Phase 2: End-to-End Pipeline
- [ ] Parse JPEG → extract coefficients + quant tables + metadata
- [ ] Transform coefficients
- [ ] Re-encode with copied quant tables, optimized Huffman tables, preserved metadata
- [ ] Round-trip tests: decode(transform(jpeg)) == decode_and_rotate(jpeg)

### Phase 3: Edge Handling
- [ ] Detect MCU alignment
- [ ] Implement trim mode
- [ ] Handle chroma subsampling (4:2:0, 4:2:2) block relationships

### Phase 4: zenimage Integration
- [ ] Wire into EXIF orientation handling
- [ ] Benchmark against pixel-level transform
