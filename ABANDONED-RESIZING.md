# Abandoned: Shrink-on-Load / Resizing

## What this was

JPEG shrink-on-load: decode at reduced resolution by using smaller IDCT kernels
(4x4, 2x2, 1x1 instead of 8x8). Supports 1/2, 1/4, 1/8, and 1/16 scale factors.
The 1/16 path runs 1/8 internally (DC-only IDCT) then applies a 2x2 box post-filter.

Also includes ShrinkQuality::Best (full 8x8 IDCT + area-average downscale) and
an f32 precision path for high-quality shrink decoding.

## Why abandoned

Shrink-on-load is the wrong abstraction for a codec library. The performance wins
are modest (Huffman entropy decoding dominates — 1/16 is only 1.13x faster than
full decode on 100MP images) while the quality loss is significant (SSIM2 76.7 vs
91.2 for proper resize at 1/16). The memory savings (315 MB → 1.2 MB output buffer)
are real but better addressed by a streaming resize pipeline that doesn't require
the codec to know about target dimensions.

Image processing pipelines that need thumbnails should decode at full resolution
into a streaming resize filter, or use a purpose-built thumbnail library. Baking
resize logic into the JPEG decoder couples concerns that should be separate.

## Preserved work

- Branch `abandoned/resizing` contains all commits from `8751457` through `9201a48`
- Includes reduced IDCT kernels, StripProcessor integration, scanline reader wiring,
  DctScale/ShrinkHint/ShrinkQuality types, comprehensive tests, resamplescope filter
  characterization, and the Sixteenth (1/16) post-filter implementation
- ~20 integration tests, quality benchmarks, and a 100MP performance benchmark

## Commits (oldest first)

- `8751457` feat: add DctScale, ShrinkHint, ShrinkQuality types
- `6807d7e` feat: add reduced IDCT kernels (1x1, 2x2, 4x4)
- `f2e2939` feat: integrate reduced IDCT into StripProcessor and scanline reader
- `58dd380` feat: wire shrink-on-load through decode()
- `c318693` test: add integration tests for shrink-on-load
- `1912c68` feat: zero-copy decode_into via scanline reader
- `69c8899` test: resamplescope filter analysis + SSIM2 quality
- `4f95cfa` feat: ShrinkQuality::Best path (full IDCT + area average)
- `85de6cd` feat: f32 Precise shrink decode path
- `0cf3f21` feat: DctScale::Sixteenth (1/16 shrink-on-load)
- `9201a48` test: add Sixteenth to all test files
