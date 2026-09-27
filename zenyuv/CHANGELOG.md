# Changelog

## [Unreleased]

### Docs

- `YuvContext::new` doc: removed the stale first block that described a
  `max_width × max_strip_height` sizing the constructor never had; the
  accurate "no buffers until first use, grown on demand" contract stands
  alone (zenjpeg #156, item 3).
- README now states the encode input channel order (packed **R, G, B**,
  3 bytes/pixel, tightly packed), documents the actual public method names
  for 4:4:4 / 4:2:0 / f32 / Sharp encode, the panic-on-undersized-buffer
  return contract and exact plane sizes (incl. odd-dimension `ceil` chroma),
  the lack of a stride parameter, per-`Range` output ranges and neutral
  chroma, and the not-yet-public status of the decode path. Found via an
  insulated external-developer usability test (the prior README showed only
  `encode_420_u8` on a grey-128 sample, which can't reveal channel order).

## [0.1.3] - 2026-04-26

### Breaking

- `Matrix` is now `#[non_exhaustive]` — pattern matches outside the
  crate must include a wildcard `_ =>` arm. Future variant additions
  will be non-breaking minor releases (746487a).
- New `Matrix::WebpEncoder` variant breaks exhaustive matches that
  predate this release. Released as a patch with prior 0.1.x versions
  yanked rather than a 0.2.0 minor — early-adopter compensation.

### Added

- `Matrix::WebpEncoder` exposes libwebp's empirical `kWebpMatrix`
  coefficients (16839, 33059, 6420 at PREC=16, halved with ties-away
  for our PREC=15 i16 lanes). Use this when producing WebP files that
  need byte-identical Y to libwebp's WebP encoder. Limited-range only;
  `Range::Full` silently maps to canonical `Bt601 Full` because libwebp
  itself uses `kRec601FullMatrix` for Full range. (746487a)
- `webp_encoder_matches_libwebp_kwebp_matrix` test pins the contract:
  zenyuv's PREC=15 Y matches libwebp's PREC=16 Y bytewise across all
  256 R=G=B inputs. Drift on any input means the variant has lost its
  reason for existing — investigate before changing.
- `webp_encoder_full_range_falls_back_to_canonical_bt601_{forward,inverse}`
  tests pin the silent-fallback semantics so future refactors can't
  change the mapping without updating both crates.

### Fixed

- Zero panics in `src/`. Previously the crate had no panics in public
  paths, but the WebpEncoder PR initially added some; they were removed
  before merge. Documented as the policy going forward.

## [0.1.2] - 2026-04-16

### Added
- `refine_chroma_420_u8` (and `_with_workspace` variant): runs the sharp-YUV
  Newton iteration on pre-seeded Cb/Cr without recomputing Y. Intended for
  callers that produce the initial chroma with a different averaging model
  (e.g., gamma-corrected downsampling) and want the L2 refinement on top
  (c96c7eb).
- Y refinement pass in the sharp-YUV pipeline, controlled by the new
  `SharpYuvConfig::refine_y` field (default: `true`). After chroma
  refinement, adjusts Y to compensate for the luma error introduced by 4:2:0
  chroma subsampling, matching libwebp's `SharpYuvUpdateY` (bb04b84).

### Changed
- `SharpYuvConfig` now derives `Clone`, `Copy`, and `PartialEq` so it can be
  embedded in downstream config structs (e.g., inside `Option<SharpYuvConfig>`
  in zenwebp's `LossyConfig`) without manual plumbing (8c45031).

### Fixed
- Correct `y_offset` sign in the sharp YUV iteration. The offset was applied
  in the wrong direction, leaving a systematic luma bias in refined output
  (1c9631a).

## 0.1.1 - 2026-04-15

### Added
- `YuvContext::encode_420_y_only_u8` — Y-only 4:2:0 encode that skips chroma
  compute entirely. For callers that will replace chroma with a custom
  downsampling (e.g., gamma-corrected averaging) and want to avoid the cost
  of zenyuv's chroma compute+write being overwritten.
- AVX2 kernel `rgb_to_yuv420_y_only_avx2` reuses the existing Y `matrix_row_avx2`
  helper; no chroma compute in the hot loop.

Measured ~22% speedup on 3840×2160 gamma-corrected RGB→YUV420 when paired
with a custom chroma overwrite (downstream in zenwebp).

## 0.1.0 - 2026-04-14

Initial release.

### Encode (RGB → YCbCr)
- BT.601, BT.709, BT.2020 matrices
- Full and Limited (studio) range
- 4:4:4 and 4:2:0 subsampling
- AVX2 `#[arcane]` kernel (32 pixels/iter, pmaddwd + pshufb deinterleave)
- NEON and WASM SIMD128 kernels
- magetypes generic fallback (all platforms)
- Sharp YUV iterative chroma optimization (L2-optimal Newton step, 2 iterations)

### Decode (YCbCr → RGB)
- 4:4:4, 4:2:0, 4:2:2, 4:0:0 (grayscale)
- Bilinear chroma upsampling for 4:2:0
- magetypes generic (AVX2 decode skeleton exists but not dispatched)

### API
- `YuvContext` — reusable encoder/decoder context, 28 bytes on stack
- Lazy workspace allocation (sharp: 432 bytes heap, f32 temps: variable)
- `#![no_std]` + `#![forbid(unsafe_code)]`

### Performance (vs `yuv` crate 0.8, 7950X AVX2)
- 4:4:4 encode: 6-11% faster at 256-512px, equal at 1024+
- 4:2:0 encode: matched (±4%)
- Sharp YUV: 25× faster than the original scalar implementation
