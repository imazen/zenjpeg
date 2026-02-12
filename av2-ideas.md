# AV2 Ideas for zenjpeg

Research date: 2026-02-12.

## High-Value Encoder-Only Backports

### 1. Custom Color Transform Matrix (CCTX concept) ← START HERE

JPEG does not mandate YCbCr. The spec allows arbitrary color transform matrices stored in APP headers. AV2's CCTX applies content-adaptive 2D rotations to Cb/Cr to compact energy — the same principle works as a custom RGB→YCC matrix optimized per-image (or per-image-class).

**Why this matters for JPEG:** Standard BT.601 YCbCr is a compromise for broadcast video. Photographic content, especially with saturated colors or skin tones, can have significantly different optimal decorrelation axes. A content-adaptive matrix reduces chroma energy → smaller residuals → better compression at the same quality.

**Implementation:** Analyze image statistics (covariance of RGB channels), compute optimal KLT-like decorrelation matrix, encode using that matrix, store it in the JPEG header. Decoder uses the stored matrix to invert. Fully spec-compliant.

**Compatibility:** Any JPEG decoder that reads the color transform from the header (most do) will decode correctly. Decoders that assume YCbCr will show wrong colors — but this is already true for any non-YCbCr JPEG.

### 2. Trellis Coefficient Optimization

Viterbi search for optimal DCT coefficient levels on the existing quantizer grid. mozjpeg already does this. Improvements from AV2 research:
- Joint optimization of coefficient levels + EOB position
- 52% SIMD speedup techniques for the trellis search
- Better lambda estimation for the R-D tradeoff

### 3. Content-Adaptive Quantization Tables

JPEG's 64-value quantization tables are fully encoder-defined. AV2's perceptual research on coefficient importance can inform per-image (or per-content-class) table generation. Most encoders use fixed tables (IJG defaults, mozjpeg's tuned tables). Per-image optimized tables are legal and universally decodable.

### 4. Progressive Scan Optimization

AV2 coefficient significance research (which frequencies carry the most perceptual weight) can refine progressive scan ordering to front-load the most important coefficients. mozjpeg does some of this already.

### 5. DC-Only Block Optimization (DCTX insight)

In smooth regions (sky, gradients), aggressively quantize near-zero AC coefficients to zero. The R-D cost of signaling tiny AC values often exceeds their quality contribution. AV2's DCTX tool exploits this same observation at the decoder level; for JPEG it's an encoder-side quantization threshold decision.

## Not Applicable

- IST, MRLS, CCSO, SDP, ATC, FSC, PARA — all require decoder changes to formats that support them. JPEG has no mechanism for these.
- WebP-style improvements — zenjpeg targets JPEG only.
- 4:4:4 — JPEG already supports it.

## Priority Order

1. **CCTX-style custom color matrix** — Novel for JPEG, high potential, low implementation risk
2. **Trellis quant improvements** — Incremental over existing, but compounds well
3. **Adaptive quant tables** — Moderate effort, good payoff for diverse content
4. **Progressive scan tuning** — Polish, not transformative
5. **DC-only threshold tuning** — Easy, small but consistent win
