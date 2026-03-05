# Lossless JPEG Transforms

## Motivation

Two use cases:

1. **End-to-end lossless JPEG rotation/flip**: JPEG in → transform DCT coefficients → JPEG out.
   No decode to pixels, no re-encode. Zero generation loss.

2. **Decode-time orientation**: Apply EXIF orientation or an explicit transform during decode,
   in DCT-coefficient space before IDCT. One entropy decode pass, no re-encoding.

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

These form the D4 dihedral group. `LosslessTransform::then()` composes transforms via
a precomputed Cayley table, and `inverse()` returns the group inverse.

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
blocks at edges cause issues for transforms that move those edges to a different position:

- **`TrimPartialBlocks`** (default): Discard partial MCU strips. Output may be up to 15 pixels
  smaller per affected edge. This matches jpegtran's `trim` behavior.
- **`RejectPartialBlocks`**: Return `TransformError::NotMcuAligned` if the image isn't aligned.

The decode-time path (`DecodeConfig::transform()`) handles non-aligned images differently:
it renders the full padded image and crops to the visible region, preserving all pixels.
This goes through IDCT so it's not lossless on re-encode, but produces correct pixel output.

Most cameras produce MCU-aligned images, so this is rarely an issue in practice.

### Zigzag Order

zenjpeg stores coefficients in zigzag scan order, not row-major 8×8. The transform operations
(transpose, negate odd rows/cols) are defined in terms of the 8×8 matrix position (row i, col j).

`BlockTransform::for_transform()` precomputes a `[(u8, bool); 64]` permutation table mapping
each zigzag source position to its destination position and sign. This costs one table lookup
and conditional negate per coefficient.

## API

### End-to-End Pipeline (`zenjpeg::lossless`)

Requires the `decoder` feature. Takes JPEG bytes → Huffman decode → transform coefficients →
Huffman re-encode → JPEG bytes.

```rust
use zenjpeg::lossless::{transform, apply_exif_orientation, LosslessTransform, TransformConfig};

// Rotate 90° losslessly
let rotated = transform(&jpeg_data, &TransformConfig {
    transform: LosslessTransform::Rotate90,
    ..Default::default()
}, enough::Unstoppable)?;

// Auto-correct EXIF orientation (resets tag to 1)
let oriented = apply_exif_orientation(&jpeg_data, enough::Unstoppable)?;
```

All metadata (EXIF, ICC, XMP, IPTC, comments) is preserved. `apply_exif_orientation()`
also resets the EXIF orientation tag to 1 in the output. The `transform()` function
does not modify EXIF — the caller handles that.

Huffman tables are always re-optimized from coefficient frequencies.

### Decode-Time Transform (`DecodeConfig`)

Apply a transform during decode without a separate lossless re-encoding step. The transform
happens in DCT-coefficient space before IDCT, so there's only one entropy decode pass.

```rust
use zenjpeg::decoder::Decoder;
use zenjpeg::lossless::LosslessTransform;

// Auto-orient from EXIF
let result = Decoder::new()
    .auto_orient(true)
    .decode(&jpeg_data, enough::Unstoppable)?;

// Explicit transform
let result = Decoder::new()
    .transform(LosslessTransform::Rotate90)
    .decode(&jpeg_data, enough::Unstoppable)?;

// Both (EXIF first, then explicit transform)
let result = Decoder::new()
    .auto_orient(true)
    .transform(LosslessTransform::FlipHorizontal)
    .decode(&jpeg_data, enough::Unstoppable)?;
```

Works with both buffered `.decode()` and streaming `.scanline_reader()`.

### Low-Level Coefficient Transform

For custom pipelines that need direct coefficient access:

```rust
use zenjpeg::decoder::Decoder;
use zenjpeg::lossless::{transform_coefficients, TransformConfig, LosslessTransform};

let coeffs = Decoder::new().decode_coefficients(&jpeg_data, enough::Unstoppable)?;
let transformed = transform_coefficients(&coeffs, &TransformConfig {
    transform: LosslessTransform::Rotate180,
    ..Default::default()
})?;
// transformed.width, transformed.height, transformed.components, transformed.quant_tables
```

Use `decode_coefficients_with_extras()` when you need both coefficients and metadata
in a single parse pass.

## Known Limitations

1. **4:2:0 scanline path with transforms**: The coefficient-based scanline reader produces
   pixel differences up to ~57 at chroma block boundaries for 4:2:0 subsampled images.
   4:4:4 is exact. The buffered decode path is correct. See CLAUDE.md known bug #5.

2. **No progressive re-encoding**: The lossless pipeline always writes baseline sequential
   output, even if the source was progressive. Coefficients are preserved exactly, but
   the scan structure changes.

3. **Memory**: The end-to-end pipeline buffers all coefficients in memory. For a 4000×3000
   4:2:0 JPEG, that's ~35 MB of i16 data. The decode-time path has the same requirement
   (disables streaming internally to store coefficients for the transform).
