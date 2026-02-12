# JPEG Decoder Crash Regression Test Import

Systematic import of crash/panic/malformed-input test cases from upstream JPEG decoder projects.

## Status

| Source | Issues Scraped | Reproducers Found | In-Repo Tests | Panics Found |
|--------|---------------|-------------------|---------------|--------------|
| zune-jpeg (etemesi254/zune-image) | 356 (all) | 85 (36 phase 1 + 49 extra) | 56 (36 include_bytes + 20 sweep) | **0** |
| jpeg-decoder (image-rs/jpeg-decoder) | 285 (all) | 59 (32 + 27 jd_257) | 17 sweep (small) | **0** |
| libjpeg-turbo (libjpeg-turbo/libjpeg-turbo) | 866 (all) | 41 (38 JPEG + BMP/PPM/GIF) | 33 sweep (small) | **0** |
| codec-corpus jpeg-conformance | — | — | 176 (40 valid + 116 invalid + 20 non-conformant) | **0** |
| codec-corpus zune fuzz | — | — | 1,836 fuzz corpus | **0** |

**Total: 2,062 JPEG files tested, 0 panics. zenjpeg handles all malformed input gracefully.**

## Three-tier classification

Each test file is classified by expected decoder behavior:

| Classification | Count | Assertion |
|---|---|---|
| **should_decode** | 13 in-repo + 40 corpus valid | MUST decode successfully, MUST NOT panic |
| **may_decode** | 3 in-repo + 20 corpus non-conformant | MAY accept or reject, MUST NOT panic |
| **must_reject** | 54 in-repo + 116 corpus invalid + 1,836 fuzz | MUST NOT panic (error expected) |

### should_decode files (valid JPEGs)

These triggered C-specific bugs (segfaults, UB, heap corruption) or output-correctness
bugs in upstream decoders — the JPEG data itself is valid:

- `jd_219_no_app14.jpg` — valid JPEG without optional APP14 marker
- `jd_228_ff00_marker.jpeg` — valid JPEG with FF00 byte stuffing
- `zj_040_decode_diff_1.jpg` — decode produced wrong pixels in zune-jpeg
- `zj_134_luma_decode_bad.jpg` — luma channel decoded incorrectly
- `zj_249_discolored_ycbcr.jpg` — YCbCr→RGB color shift
- `zj_303_app14_adobe.jpg` — Adobe APP14 marker handling
- `ljt_441_segv_jcopy_sample_rows.jpg` — C segfault in row copy
- `ljt_470_ub_null_ptr_skip.jpg` — C null pointer UB
- `ljt_574_heap_corruption.jpg` — C heap buffer overwrite
- `ljt_758_segv_adjust_quant_src.jpg` — C segfault in quant adjustment

### may_decode files (non-conformant edge cases)

- `jd_040_16bit_quant.jpg` — 16-bit quantization tables (non-standard but common)
- `jd_132_subtract_overflow.bin` — arithmetic edge case in coefficient decode
- `zj_172_upsample_assert.jpg` — unusual sampling factors

## Phase 1: zune-jpeg core (complete)

36 reproducers from issues 218, 219, 236, 257, 262, 297, 300, 301, 302, 309, 314, 315, 316, 324, 331.
All pass. See `zenjpeg/tests/zune_crash_repro.rs`.

## Phase 2: Full historical scrape (complete)

### zune-jpeg expanded (49 new files)

Scraped all 356 issues from etemesi254/zune-image. Found 49 additional reproducer files
covering issues: 4, 5, 7, 8, 40, 64, 67, 77, 86, 87, 89, 90, 91, 104, 134, 148, 151,
162, 167, 172, 188, 202, 207, 217, 224, 243, 246, 249, 251, 261, 266, 269, 270, 275,
276, 277, 278, 288, 291, 292, 293, 294, 303, 323, 340, 341, 348.

Bug types: panics (assertion, unwrap, OOB), incorrect output, decode failures,
infinite loops, integer overflows, divide-by-zero.

20 small files (<30KB) in repo at `tests/crash_repro/zune_jpeg_extra/`.
29 large files staged for codec-corpus import.

### jpeg-decoder (59 files)

Scraped all 285 issues from image-rs/jpeg-decoder. Found 32 individual reproducer files
plus 27 "broken images" from issue #257.

Bug types: arithmetic overflow, OOB panics, DoS (OOM, slow parse), decode failures
on real-world cameras (Samsung, Sony Ericsson), incorrect output, lossless JPEG panics.

17 small files (<30KB) in repo at `tests/crash_repro/jpeg_decoder/`.
42 large files staged for codec-corpus import.

### libjpeg-turbo (41 files, 38 JPEG)

Scraped all 866 issues from libjpeg-turbo/libjpeg-turbo. Cataloged 19 CVEs.
Downloaded 38 JPEG reproducer files covering decode-side bugs.

Bug types: heap buffer overflow, use-after-free, integer overflow, divide-by-zero,
algorithmic complexity DoS, infinite loops, logic errors.

33 small JPEG files (<30KB) in repo at `tests/crash_repro/libjpeg_turbo/`.
5 large files staged for codec-corpus import.

## Phase 3: Codec-corpus integration (complete)

### jpeg-conformance (176 files)

Tests run with `cargo test -- --ignored`:

- `valid/` — 40 files, all decode successfully (MUST decode)
- `invalid/` — 116 files, 97 rejected + 17 accepted (lenient), 0 panics (MUST NOT panic)
- `non-conformant/` — 20 files, 6 accepted + 14 rejected, 0 panics (MUST NOT panic)

### zune fuzz corpus (1,836 files)

All 1,836 files from `zune/fuzz-corpus/jpeg/` tested, 0 panics.
16 decoded ok, 1,818 returned errors, 2 not JPEG.

## Large files for codec-corpus

The following files are >30KB and should be added to `imazen/codec-corpus`
under `jpeg-conformance/crash-repro/`. They are currently stored at
`/home/lilith/research/image-tiff-fork/crash-repro/` and verified to cause
no panics in zenjpeg.

- jpeg-decoder: 42 files (36MB) including jd_257 broken images collection
- zune-jpeg-extra: 29 files (29MB) including CMYK, panoramas, progressive
- libjpeg-turbo: 5 files (6.3MB) including signed overflow DC test cases

## Manifests

Detailed per-issue manifests with bug descriptions, root causes, and reproducer status:

- `/home/lilith/research/image-tiff-fork/crash-repro/jpeg-decoder/MANIFEST.md`
- `/home/lilith/research/image-tiff-fork/crash-repro/zune-jpeg-extra/MANIFEST.md`
- `/home/lilith/research/image-tiff-fork/crash-repro/libjpeg-turbo/MANIFEST.md`
- `/home/lilith/research/image-tiff-fork/jpeg-test-corpora.md` (corpora research)
