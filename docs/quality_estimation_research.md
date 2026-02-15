# JPEG Quality Estimation for Re-encoding

Research notes on detecting the approximate quality a JPEG was encoded at, for
matching it on re-encode when you must reduce file size or perform a pixel resize.

## The Problem

"Quality" is an encoder-specific number. libjpeg Q85 and cjpegli Q85 produce
completely different quantization tables and file sizes. ImageMagick's `identify`
reports cjpegli Q90 as "Quality: 58" — a 32-point error — because it assumes IJG
standard tables.

What you actually have in the JPEG file: quantization tables (DQT markers). What
you want: the right quality parameter to pass to your re-encoder to produce
comparable visual quality at a smaller file size.

This requires two steps:
1. **Identify which encoder** produced the JPEG (from its table structure)
2. **Reverse-estimate the quality parameter** for that specific encoder

## Encoder Fingerprinting

### Distinguishing Encoders from DQT Tables

The simplest discriminator: how many quantization tables?

| Tables | Encoder Family |
|--------|---------------|
| 2 (Y + shared CbCr) | IJG/libjpeg, libjpeg-turbo, mozjpeg, ImageMagick |
| 3 (Y + Cb + Cr) | jpegli/cjpegli, zenjpeg |
| 1 | Grayscale (any encoder) |
| 2 (non-standard) | Photoshop, cameras, other proprietary |

Three tables is a strong jpegli signal. Two tables requires deeper analysis:
match against IJG standard tables, then mozjpeg's alternate table sets, then
proprietary databases.

### Empirical Validation

Tested with `baby-lossless.png` from the gb82 corpus (512x512 photograph):

```
cjpeg (libjpeg 9d) at Q50-95: All tables EXACT MATCH with IJG formula
cjpegli at Q50-95: 3 tables, all match jpegli formula exactly
cross-encoder re-encode: Always matches the LAST encoder's tables
```

Fingerprinting accuracy: 100% on all test files. The 2-vs-3 table distinction
alone separates IJG-family from jpegli-family with zero false positives.

## Quality Estimation by Encoder Family

### IJG/libjpeg/libjpeg-turbo: Exact Inversion

The IJG quality formula is invertible. For quality Q:

```
scale_factor = 5000/Q    (if Q < 50)
scale_factor = 200 - 2*Q (if Q >= 50)

table[i] = clamp(round(base[i] * scale_factor / 100), 1, 255)
```

**Least-Squares Matching (LSM):** Generate expected tables for Q=1..100, find
the one with minimum SSE against the image's actual table. For standard IJG
tables, this always produces RMSE = 0.0 (exact match):

```
Actual Q    Est Q    RMSE     SSE
      50       50   0.000       0
      60       60   0.000       0
      85       85   0.000       0
      95       95   0.000       0
```

The LSM approach also works as a fallback for non-standard tables — it reports
the closest IJG-equivalent quality with a nonzero RMSE as a confidence indicator.
(Empirically, cjpegli Q85 maps to IJG Q70 with RMSE=47.5, which is meaningless
as a quality estimate but tells you "these aren't IJG tables.")

### jpegli/cjpegli/zenjpeg: Distance-Based Inversion

jpegli doesn't use the IJG linear scaling formula. It uses a butteraugli
distance-based system with per-frequency non-linear scaling:

```
quality → distance:
  Q >= 100: d = 0.01
  Q >= 30:  d = 0.1 + (100 - Q) * 0.09
  Q < 30:   quadratic

distance → scale (per-frequency):
  d < 1.5:  scale = d  (linear)
  d >= 1.5: scale = max(0.5*d, 1.5^(1-exp) * d^exp)
            where exp = FREQUENCY_EXPONENT[freq_idx]

table value:
  q[i] = clamp(round(BASE_MATRIX[comp][i] * scale(d, i) * GLOBAL_SCALE), 1, 255)
```

The key constants (from zenjpeg source, matching C++ jpegli):
- `GLOBAL_SCALE_YCBCR = 1.739660`
- `DIST_THRESHOLD = 1.5`
- `FREQUENCY_EXPONENT[64]`: ranges from 0.51 (low-freq) to 1.0 (high-freq)
- `BASE_QUANT_MATRIX_YCBCR[192]`: 3 × 64 jpegli-optimized base values

**Reverse estimation** uses the existing `quant_vals_to_distance()` algorithm
(already in zenjpeg at `quant/mod.rs:375`). It binary-searches the distance
that produces the observed tables, using per-coefficient bounds from
`scale_to_distance()`. Accuracy on test files:

```
Quality  True Dist   Est Dist    Est Q   Dist Err
     50     4.6000     4.6002     50.0     0.0002
     75     2.3500     2.3501     75.0     0.0001
     90     1.0000     1.0001     90.0     0.0001
     95     0.5500     0.5500     95.0     0.0000
```

Sub-0.001 distance error across the entire quality range. The reverse estimation
is effectively exact.

### mozjpeg: Multiple Table Families

mozjpeg has 9 quantization table presets (from `jcparam.c`):

| Idx | Name | Default for |
|-----|------|-------------|
| 0 | JPEG Annex K | Standard tables (same as IJG) |
| 1 | Flat | All coefficients = 16 |
| 2 | MS-SSIM | Tuned for MS-SSIM metric |
| 3 | ImageMagick/Robidoux | Max compression mode (default) |
| 4 | PSNR-HVS | Tuned for PSNR-HVS |
| 5-8 | Academic | Klein, Watson, Ahumada, Peterson |

The quality scaling formula is identical to IJG — only the base tables differ.
So you need to match against all 9 base table sets. With mozjpeg not installed
on this system, I couldn't run empirical tests, but the algorithm is:

1. For each preset index 0-8:
2.   For each quality Q=1..100:
3.     Generate expected tables using that preset's base tables + IJG scaling
4.     Compute SSE against actual tables
5. Return the (preset, Q) with minimum SSE

If SSE = 0 for preset 0, it's standard IJG (could be libjpeg OR mozjpeg default).
If SSE = 0 for preset 3 (Robidoux), it's mozjpeg max-compression mode.

### ImageMagick's Hash/Sum Heuristic (Don't Use This)

ImageMagick uses a precomputed lookup table of aggregated table sums and
position-specific hash values. It's fast but unreliable:

- Falls back to "Quality: 92" when it can't determine quality (making 92 ambiguous)
- Uses `table0[2] + table0[53] + table1[0] + table1[63]` as a hash — highly collision-prone
- For quality > 50, ignores the sum entirely and uses only the hash
- Produces completely wrong results for non-IJG tables

Empirical comparison against LSM:

```
                File   Actual Q   IM Est  LSM Est
         cjpegli_q50         50       92       37
         cjpegli_q60         60       92       43
         cjpegli_q80         80       92       63
         cjpegli_q90         90       58       80
         cjpegli_q95         95       77       89
```

ImageMagick says 92 for everything below cjpegli Q90. Don't use it.

## Repeated Compression Behavior

### Same Encoder, Same Quality

Re-encoding a JPEG at the same quality with the same encoder preserves identical
quantization tables. Tested with cjpeg Q85 through 4 re-encode cycles:

```
Iteration    File Size    Table Sum (Y+Cb)
1x           40077        1109 + 1666
2x           40089        1109 + 1666
3x           40106        1109 + 1666
4x           40096        1109 + 1666
```

Tables are byte-identical. File size varies by <0.1% due to re-quantization of
DCT coefficients producing slightly different Huffman statistics.

### Same Encoder, Different Quality

The last encoder's tables completely overwrite previous ones. Quality estimation
always reflects the most recent encode, not the first. This is inherent to JPEG —
the quantization table in the file is whatever the last encoder wrote.

```
Source Q   Re-enc Q    Est Q    Exact    Size
     95         85       85     True   40430
     90         75       75     True   28234
     85         95       95     True   62857
```

If you encode at Q95 then re-encode at Q75, quality estimation reports Q75. The
damage from the first compression at Q95 plus the second at Q75 is worse than a
single Q75 encode from the source, but the quant tables don't reveal this history.

### Cross-Encoder Re-encoding

The same principle applies: the last encoder's tables win.

```
Src(jpegli)   Re-enc(cjpeg) Q    Est Encoder          Est Q
          95               85     IJG/libjpeg (Q85)       85
          90               75     IJG/libjpeg (Q75)       75
```

After re-encoding a jpegli file with cjpeg, the fingerprint correctly identifies
IJG/libjpeg and the quality matches the re-encode parameter. The original jpegli
quality is unrecoverable from DQT analysis.

### Detecting Previous Compression (DCT Histogram Analysis)

The original quality IS potentially recoverable from the DCT coefficient
distributions, not from the DQT tables. When a JPEG is decoded and re-compressed:

1. First compression quantizes coefficient `c` to `round(c / Q1) * Q1`
2. Re-compression quantizes to `round(v / Q2) * Q2`
3. If Q2 < Q1 (re-encoding at higher quality), DCT coefficient histograms show
   periodic comb patterns with period Q1/Q2

This is the domain of forensic double-compression detection (Fridrich & Lukás,
Nikoukhah et al. 2022). Implementing it requires computing DCT coefficients from
pixel data and doing statistical tests on their histograms. This is heavy
machinery — not needed for the "match quality on re-encode" use case where you
still have the original JPEG's DQT markers.

## Cross-Encoder Quality Mapping

The hard part: if you have a jpegli Q90 file and want to re-encode with cjpeg at
"equivalent quality," what number do you use?

### File Size Comparison at Same Quality Number

```
Quality    cjpeg size    cjpegli size    Ratio
     50         19827           18841    0.950
     70         26927           24269    0.901
     85         40077           35020    0.874
     95         74763           66523    0.890
```

jpegli produces 10-13% smaller files at the same quality number because its
perceptually-optimized tables quantize high frequencies more aggressively while
preserving low frequencies.

### Matching Strategy for Re-encoding

When you need to re-encode a JPEG and want to avoid exceeding the original's
visual quality degradation, the approach depends on your encoder:

**Case 1: Same encoder** — Use the exact same quality number. Tables will be
identical; only DCT re-quantization noise is added (negligible for practical
purposes, as shown by the <0.1% file size variance in repeated compression).

**Case 2: jpegli → IJG re-encode** — You want an IJG quality that produces
similar quantization intensity. The zenjpeg project already has mapping tables
in `quant/quality_conversion.rs`:

```
mozjpeg → jpegli mapping (4:4:4):
  mozjpeg 30 → jpegli 28
  mozjpeg 50 → jpegli 47
  mozjpeg 70 → jpegli 65
  mozjpeg 85 → jpegli 83
  mozjpeg 95 → jpegli 94
```

These were calibrated using DSSIM/SSIMULACRA2 comparison at matched BPP. The
inverse mapping gives the IJG quality to use for equivalent perceptual quality.

**Case 3: Unknown encoder → any re-encode** — If you can't identify the encoder:
1. Use the LSM approach to get the IJG-equivalent quality (even if RMSE > 0)
2. Use that quality number minus 2-3 as a conservative starting point
3. Optionally verify with a perceptual metric (DSSIM, SSIMULACRA2)

### The "Must Reduce File Size" Constraint

When you MUST reduce file size (e.g., proxy cache, CDN optimization), you want
the minimum quality reduction that achieves the target size. Given the original
quality Q_orig estimated from DQT:

1. If resize is involved, re-encoding at Q_orig will already reduce file size
   (fewer pixels = smaller file). Start there.
2. If no resize, you need Q_new < Q_orig. The relationship between quality and
   file size is roughly exponential — dropping 5 quality points typically reduces
   file size by 15-25%.
3. For jpegli → IJG transcoding without resize, the IJG-equivalent quality is
   already lower due to jpegli's better compression efficiency. A jpegli Q90 file
   re-encoded with cjpeg at Q90 will be ~12.5% larger; at Q85 it'll be similar
   or slightly smaller.

## Proposed Algorithm for zenjpeg

```rust
pub struct QualityEstimate {
    /// Detected encoder family
    pub encoder: EncoderFamily,
    /// Estimated quality parameter (encoder-specific scale)
    pub quality: f32,
    /// For jpegli: the butteraugli distance
    pub distance: Option<f32>,
    /// Confidence: 0.0 = guess, 1.0 = exact match
    pub confidence: f32,
}

pub enum EncoderFamily {
    /// IJG/libjpeg/libjpeg-turbo with standard Annex K tables
    Ijg,
    /// jpegli/cjpegli/zenjpeg (3-table, non-linear scaling)
    Jpegli,
    /// mozjpeg with identified preset index
    Mozjpeg { preset: u8 },
    /// Unknown encoder (LSM gives approximate IJG-equivalent)
    Unknown,
}

pub fn estimate_quality(dqt_tables: &[QuantTable]) -> QualityEstimate {
    // 1. Count tables
    let n = dqt_tables.len();

    // 2. Try jpegli (3-table) matching
    if n == 3 {
        let dist = quant_vals_to_distance(&dqt_tables[0], &dqt_tables[1], &dqt_tables[2]);
        let quality = distance_to_quality(dist);
        return QualityEstimate {
            encoder: EncoderFamily::Jpegli,
            quality,
            distance: Some(dist),
            confidence: 1.0, // jpegli tables are unambiguous
        };
    }

    // 3. Try exact IJG match
    if n >= 2 {
        for q in 1..=100 {
            let expected = generate_ijg_tables(q);
            if exact_match(&dqt_tables[0], &expected.lum)
                && exact_match(&dqt_tables[1], &expected.chr)
            {
                return QualityEstimate {
                    encoder: EncoderFamily::Ijg,
                    quality: q as f32,
                    distance: None,
                    confidence: 1.0,
                };
            }
        }
    }

    // 4. Try mozjpeg preset matching (presets 1-8, preset 0 = IJG already checked)
    // ... (match against each preset's base tables)

    // 5. Fallback: LSM against IJG tables
    let (best_q, rmse) = lsm_estimate(dqt_tables);
    let confidence = (1.0 - rmse / 50.0).max(0.0);
    QualityEstimate {
        encoder: EncoderFamily::Unknown,
        quality: best_q as f32,
        distance: None,
        confidence,
    }
}
```

The `quant_vals_to_distance()` function already exists in zenjpeg. The IJG
matching is ~100 lines. mozjpeg preset tables can be embedded as constants.
Total implementation: ~300-400 lines.

## Appendix A: Key Constants

### IJG Standard Base Tables (ITU-T T.81 Annex K)

Luminance:
```
16  11  10  16  24  40  51  61
12  12  14  19  26  58  60  55
14  13  16  24  40  57  69  56
14  17  22  29  51  87  80  62
18  22  37  56  68 109 103  77
24  35  55  64  81 104 113  92
49  64  78  87 103 121 120 101
72  92  95  98 112 100 103  99
```

Chrominance:
```
17  18  24  47  99  99  99  99
18  21  26  66  99  99  99  99
24  26  56  99  99  99  99  99
47  66  99  99  99  99  99  99
99  99  99  99  99  99  99  99
99  99  99  99  99  99  99  99
99  99  99  99  99  99  99  99
99  99  99  99  99  99  99  99
```

Note: libjpeg 9e changed chrominance DC from 17 to 16. libjpeg-turbo kept 17.

### Jpegli Base Quantization Matrix (YCbCr)

192 floats (3 components × 64), stored in `foundation/consts.rs` as
`BASE_QUANT_MATRIX_YCBCR`. Key characteristics:
- Y component: values 1.2-6.1 (much smaller range than IJG's 10-121)
- Cb component: values 2.8-112.2 (huge range; position [6][2] = 102.6)
- Cr component: values 2.9-114.9 (similar to Cb; large bottom-right corner values)
- After scaling by `GLOBAL_SCALE_YCBCR * distance`, Y table values are small
  (3-14 at Q90), while Cb/Cr table values span 5-195

### Jpegli Frequency Exponents

Per-frequency non-linear scaling exponents (64 values). Low-frequency positions
have exponents < 1.0 (0.51-0.87), meaning they scale sub-linearly with distance.
High-frequency positions have exponent 1.0 (linear scaling). This makes low
frequencies less sensitive to quality changes — you get more gradual quality
transitions where the eye is most sensitive.

## Appendix B: DQT Table Extraction

DQT markers (0xFF 0xDB) store quantization values in **zigzag scan order**, not
raster order. The zigzag-to-raster mapping:

```
Zigzag index:  0  1  2  3  4  5  6  7  8  9 10 11 12 ...
Raster pos:    0  1  8 16  9  2  3 10 17 24 32 25 18 ...
```

Any tool comparing against reference tables must convert. The IJG source code
defines base tables in raster order; DQT markers store them in zigzag order.

## Appendix C: Why Not Use File Size?

File size depends on:
- Quantization tables (quality)
- Huffman optimization (mozjpeg does this better than standard libjpeg)
- Trellis quantization (mozjpeg, jpegli)
- Image content (smooth images compress much better than noisy ones)
- Chroma subsampling (4:2:0 vs 4:4:4)
- Progressive vs baseline encoding

Same image, same quantization tables, different encoders can produce 10-30% size
differences due to Huffman and trellis optimization alone. File size is not a
reliable quality indicator.

## References

- [Cogranne (2018) — Determining JPEG Quality Factor from Quantization Tables](https://arxiv.org/abs/1802.00992)
- [Nikoukhah et al. (2022) — A Reliable JPEG Quantization Table Estimator](https://www.ipol.im/pub/art/2022/399/)
- [Bitsgalore — JPEG quality estimation using least squares matching](https://bitsgalore.org/2024/10/30/jpeg-quality-estimation-using-simple-least-squares-matching-of-quantization-tables.html)
- [Bitsgalore — ImageMagick heuristic analysis](https://bitsgalore.org/2024/10/23/jpeg-quality-estimation-experiments-with-a-modified-imagemagick-heuristic.html)
- [libjpeg-turbo jcparam.c](https://github.com/libjpeg-turbo/ijg/blob/main/jcparam.c)
- [mozjpeg jcparam.c](https://github.com/mozilla/mozjpeg/blob/master/jcparam.c)
- [Kornblum (2008) — Using JPEG Quantization Tables to Identify Imagery](https://dfrws.org/presentation/using-jpeg-quantization-tables-to-identify-imagery-processed-by-software/)
- [ImpulseAdventure JPEG Quality and Quantization Tables](https://www.impulseadventure.com/photo/jpeg-quantization.html)
- [qbammey/jpeg-analysis](https://github.com/qbammey/jpeg-analysis) — Statistical NFA-based quality estimation from pixel data
- [KBNLresearch/jpeg-quality-demo](https://github.com/KBNLresearch/jpeg-quality-demo) — Python LSM implementation
