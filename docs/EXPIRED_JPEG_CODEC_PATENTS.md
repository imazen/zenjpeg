# Expired Patents on Improving JPEG-Compliant Codecs (2018-2026)

Research date: 2026-04-13

This document catalogs patents related to techniques that make JPEG-compliant
encoders/decoders better (the output is still standard JPEG, but with improved
quality, compression, speed, or perceptual optimization). Focus is on patents
that expired in the last 8 years, plus still-active patents that represent
techniques used in modern JPEG encoders like jpegli, mozjpeg, and zenjpeg.

**Disclaimer**: This is research notes, not legal advice. Patent expiration
dates are estimated from filing dates + 20 years; actual dates may differ due
to Patent Term Adjustment (PTA), Patent Term Extension (PTE), terminal
disclaimers, or maintenance fee lapses. Always verify with USPTO records.

---

## 1. Expired Patents (now public domain)

### 1.1 Adaptive Quantization

| Patent | Title | Assignee | Filed | Expired | Technique |
|--------|-------|----------|-------|---------|-----------|
| **US5157488** | Adaptive quantization within the JPEG sequential mode | IBM | 1991-05 | ~2011 (Lifetime) | Clears low-magnitude bits in regions where coarser quantization is desired. Spatially varying quantization within standard JPEG syntax. |
| **US6175650** | Adaptive quantization compatible with JPEG baseline sequential mode | Xerox | 1998 | ~2018 | Type-classifies pixel blocks, applies image-type-optimized Q-tables per block to minimize perceptual error. Decoder-compatible (no sideband). |
| **US6252994** | Adaptive quantization compatible with JPEG baseline sequential mode | Xerox | 1998 | ~2018 | Companion to US6175650. Pixel blocks classified by DCT coefficient analysis, quantization modified per type. Standard JPEG output. |
| **US6882753** | Adaptive quantization using code length in image compression | Silicon Integrated Systems | 2001-06 | ~2021 (Lifetime) | Uses Huffman code length as feedback signal for adaptive quantization decisions. |
| **US7092578** | Signaling adaptive-quantization matrices in JPEG using end-of-block codes | (not specified) | 2001-10 | ~2021 (Fee expired) | Repurposes unused EOB Huffman code slots to signal which quantization matrix was used per block. Clever decoder-compatible AQ signaling. |

**Relevance to zenjpeg**: jpegli's adaptive dead-zone quantization modulates
the zero-threshold per block based on psychovisual modeling. The expired Xerox
patents (US6175650/US6252994) cover the general concept of per-block adaptive
quantization within baseline JPEG. The IBM patent (US5157488) covers the
specific bit-clearing approach. All expired; the general technique is free.

### 1.2 Quantization Table Optimization

| Patent | Title | Assignee | Filed | Expired | Technique |
|--------|-------|----------|-------|---------|-----------|
| **US5724453** | Image compression system with optimized quantization tables | Wisconsin Alumni Research Foundation | 1995-07 | ~2015 (Lifetime) | DCT-based preprocessor that generates image-specific optimized quantization tables. The foundational "custom Q-tables" patent. |
| **US5883979** | Method for selecting JPEG quantization tables for low bandwidth | Hewlett-Packard | 1995 | ~2015 (Lifetime) | Perceptually-driven Q-table design: weights table elements by perceptual importance ("supra-threshold" terms). |
| **US6314208** | System for variable quantization in JPEG for compound documents | Hewlett-Packard | 1999 | Expired (Fee) | Different quantization factors for text vs. picture blocks in mixed documents. Auto-classifies content type. |

**Relevance to zenjpeg**: The concept of generating image-specific or
content-adaptive quantization tables (as in jpegli's `jpegli_set_distance()`)
is now unencumbered. The Watson (1993) perceptual Q-table work that influenced
the JPEG standard tables was never patented.

### 1.3 Huffman Coding Optimization

| Patent | Title | Assignee | Filed | Expired | Technique |
|--------|-------|----------|-------|---------|-----------|
| **US6081211** | Minimal buffering for optimized encoding tables in JPEG compression | Xerox | 1998-04 | ~2018 (Lifetime) | Two-pass Huffman optimization with minimal buffering: collects frequency stats and generates optimal codes without buffering the entire image. |
| **US6373412** | Fast JPEG Huffman encoding and decoding | IBM | 2000-12 | ~2020 (Lifetime) | Simplified Huffman using two table formats by code length. Reduces table size and decode time. |
| **USRE39925** | Fast JPEG Huffman encoding and decoding (reissue) | IBM | (reissue of US6373412) | ~2020 | Reissued version of the IBM fast Huffman patent. Same technique. |

**Relevance to zenjpeg**: Optimized Huffman tables (as used in mozjpeg and
jpegli) are standard practice. The two-pass approach where you collect
statistics then generate optimal tables is now fully public domain.

### 1.4 Dead-Zone Quantization

| Patent | Title | Assignee | Filed | Expired | Technique |
|--------|-------|----------|-------|---------|-----------|
| **US6408026** | Deadzone quantization method and apparatus for image compression | Sony | 1999-08 | Expired (Fee) | Optimizes dead-zone (zero bin) width independently from outer bin width. Standard JPEG uses 2:1 ratio; this finds optimal ratio per distribution. |

**Relevance to zenjpeg**: jpegli's "adaptive dead-zone quantization" modulates
the zero-threshold spatially. The Sony patent on dead-zone width optimization
is expired. The general concept of non-uniform dead zones in DCT quantization
is now free.

### 1.5 Artifact Removal / Improved Decoding

| Patent | Title | Assignee | Filed | Expired | Technique |
|--------|-------|----------|-------|---------|-----------|
| **US7079703** | JPEG artifact removal | Sharp Labs | 2002-10 | ~2022 (Lifetime) | Post-decode deringing + deblocking filters via convolution. Smoothes block boundaries and ring artifacts. |

**Relevance to zenjpeg**: Decoder-side deblocking and deringing (as in
cjpegli/Knusperli) is now unencumbered. The Price & Rabbani (2000)
dequantization bias technique (Laplacian coefficient model) was published as
academic research, not patented.

### 1.6 Arithmetic Coding (for JPEG)

| Patent | Title | Assignee | Filed | Expired | Technique |
|--------|-------|----------|-------|---------|-----------|
| **US4652856** | Arithmetic coding (Q-coder foundation) | IBM | 1986 | ~2006 | Core arithmetic coding patent that blocked JPEG arithmetic mode. |
| **US4905297** | Arithmetic coding encoder/decoder system | IBM | 1988 | ~2008 | QM-coder implementation used in JPEG/JBIG. |
| **US4935882** | Probability adaptation for arithmetic coders | IBM | 1988 | ~2008 | Adaptive probability estimation for QM-coder. |

**Relevance to zenjpeg**: JPEG arithmetic coding is now completely free. These
patents were the reason JPEG implementations historically only supported
Huffman coding. Arithmetic coding gives ~5-10% better compression than Huffman
for JPEG. zenjpeg currently uses Huffman (as does jpegli); arithmetic is a
potential future improvement path with zero patent risk.

---

## 2. Still-Active Patents (use with caution)

### 2.1 Parallel JPEG Decoding

| Patent | Title | Assignee | Filed | Expires | Technique |
|--------|-------|----------|-------|---------|-----------|
| **US9542760** | Parallel decoding JPEG images | Google | 2014-12 | **2035-03** | Preprocessing JPEGs to insert restart markers + APP markers for parallel multi-core decoding. |
| **US8538180** | Generating JPEG files suitable for parallel decoding | (not specified) | ~2011 | ~2031 | Analyzing image to determine optimal restart marker placement for parallel decode. |
| **US9936213** | Parallel decode of progressive JPEG bitstream | (not specified) | ~2015 | ~2035 | Per-scan parallel threading for progressive JPEG decode. |

**Note**: Google's US9542760 is relevant to zenjpeg's parallel decode feature
(restart-marker-based parallelism). However, zenjpeg's approach uses DRI
(restart interval) that is already part of the JPEG standard spec (ITU-T T.81,
Annex B.2.4.4, 1992). The patent covers a *specific preprocessing method* to
add restart markers to JPEGs that lack them, not the general concept of
parallel decode using restart markers. Google also provides royalty-free patent
grants for JPEG XL/jpegli implementations.

### 2.2 Adaptive Image Recompression (JPEGmini/Beamr)

| Patent | Title | Assignee | Filed | Expires | Technique |
|--------|-------|----------|-------|---------|-----------|
| **US8452110** | Classifying an image's compression level | Beamr Imaging | 2011-03 | ~2031 | Perceptual quality classifier for JPEG recompression decisions. |
| **US8908984** | Apparatus and methods for recompression of digital images | Beamr Imaging | ~2012 | ~2032 | Iterative JPEG requantization guided by quality metric. |
| + ~51 more | Various image/video compression | Beamr Imaging | 2011-2020 | 2031-2040 | JPEGmini's 53-patent portfolio covers perceptual quality metrics, iterative compression, adaptive Q-table selection. |

**Note**: JPEGmini/Beamr patents cover specific *recompression* workflows
(detect existing quality, requantize to threshold). They do NOT cover the
general concept of perceptual encoding or quality-driven quantization, which
predates their work by decades (Watson 1993, JPEG Annex K).

### 2.3 Perceptual Chroma Subsampling (SharpYUV)

| Patent | Title | Assignee | Filed | Expires | Technique |
|--------|-------|----------|-------|---------|-----------|
| **US9153017** | System and method for optimized chroma subsampling | Google | 2014-08 | **~2034** | Quality-metric-driven chroma subsampling that minimizes perceptual error. Iteratively generates representations and compares quality metrics. |

**Note**: zenjpeg uses SharpYUV-style chroma subsampling (from libwebp's
`sharpyuv` library). Google provides this under Apache 2.0 with patent grant.
The patent covers a specific iterative optimization method, not the general
concept of perceptual chroma downsampling.

### 2.4 Content-Adaptive Quantization Tables (Adobe)

| Patent | Title | Assignee | Filed | Expires | Technique |
|--------|-------|----------|-------|---------|-----------|
| **US20170324958** | Generating custom quantization tables for JPEG based on image content | Adobe | 2017 | ~2037 (if granted) | Content-aware Q-table generation: different thresholds for different image regions, claiming ~2x compression improvement. Status unclear (application, may not have been granted). |

### 2.5 Microsoft ANS Patent

| Patent | Title | Assignee | Filed | Expires | Technique |
|--------|-------|----------|-------|---------|-----------|
| **US11234023** | Features of range asymmetric number system encoding and decoding | Microsoft | 2019-06 | ~2039 | rANS hardware implementations, adaptive symbol width, fragment-by-fragment adjustments. |

**Note**: Not directly relevant to JPEG (which uses Huffman or arithmetic
coding, not ANS). Relevant to JPEG XL which uses ANS. Microsoft has not
declared this essential to JPEG XL at ISO. The JPEG XL spec editor says it
doesn't apply. zenjpeg produces standard JPEG output using Huffman coding;
this patent is not applicable.

---

## 3. Techniques That Were Never Patented

Several key techniques used in modern JPEG encoders were published as academic
research or open-source without patent protection:

| Technique | Origin | Status |
|-----------|--------|--------|
| **Trellis quantization** for JPEG | Academic (R-D optimization theory, 1990s). Implemented in mozjpeg (2014). | Never patented as applied to JPEG. Core R-D theory is textbook material. |
| **Butteraugli** perceptual metric | Google (2016). Apache 2.0 + patent grant. | Open source with explicit patent grant. Used in Guetzli and jpegli. |
| **SSIMULACRA2** quality metric | Cloudinary/Jon Sneyers (2020s). Part of JPEG XL. | Open source (BSD license). No known patents. |
| **Dequantization bias** (Laplacian model) | Price & Rabbani (2000). Academic paper. | Published research, never patented. Implemented in cjpegli decoder. |
| **Perceptual Q-table design** (Watson 1993) | Andrew Watson, NASA. Published in academic literature. | Public domain (US government work). Foundation for JPEG Annex K tables. |
| **Progressive scan optimization** | Academic (R-D ordered spectral selection). | General technique from JPEG spec (1992). No separate patents on the optimization strategy. |
| **XYB color space** | Google (JPEG XL). Apache 2.0 + patent grant. | Open source with explicit royalty-free patent grant from Google. |
| **Adaptive dead-zone modulation** (as in jpegli) | Google (jpegli, 2024). Apache 2.0. | Released under Apache 2.0 with Google's standard patent grant. The expired Sony patent (US6408026) covered a different specific method. |
| **CMA-ES frequency scaling** | Evolutionary optimization (Hansen 2001). Applied to JPEG Q-tables. | CMA-ES algorithm is public domain. Application to Q-table optimization is novel but not patented by anyone. |
| **Deringing during encoding** (pre-DCT smoothing) | Various implementations. Used in jpegli. | The general technique of pre-DCT block smoothing is well-established in literature. |

---

## 4. Summary: What's Free for zenjpeg

### Fully free (expired or never patented)

- Per-block adaptive quantization (expired: US5157488, US6175650, US6252994)
- Dead-zone width optimization (expired: US6408026)
- Image-specific quantization tables (expired: US5724453, US5883979)
- Optimized Huffman table generation (expired: US6081211, US6373412)
- Trellis quantization for JPEG (never patented)
- Dequantization bias / Laplacian model (never patented, academic)
- Butteraugli perceptual metric (Google patent grant, Apache 2.0)
- XYB color space (Google patent grant, Apache 2.0)
- JPEG arithmetic coding (expired: US4652856, US4905297, US4935882)
- Post-decode deblocking/deringing (expired: US7079703)
- Adaptive quantization feedback via code length (expired: US6882753)
- Variable quantization for compound documents (expired: US6314208)
- Content classification for quantization (expired: multiple)

### Covered by Google's patent grant (via jpegli/JPEG XL Apache 2.0 license)

- Adaptive dead-zone quantization (jpegli technique)
- Butteraugli-guided encoding
- XYB perceptual color space
- SharpYUV chroma subsampling (US9153017, but Apache 2.0 licensed)
- Parallel JPEG decode via restart markers (US9542760, but patent grant)

### Still active (not used by zenjpeg, or covered by license)

- JPEGmini/Beamr recompression patents (~2031-2040) - irrelevant, different technique
- Adobe content-adaptive Q-tables (US20170324958, ~2037) - may not be granted
- Microsoft ANS patent (US11234023, ~2039) - not applicable to JPEG Huffman

### Conclusion

**All techniques used in zenjpeg are either (a) covered by expired patents,
(b) never patented, or (c) covered by Google's royalty-free patent grants
through the Apache 2.0 / JPEG XL PATENTS file.** There are no known active
patent barriers to any technique currently implemented in zenjpeg.

---

## Sources

- [US5157488 - Adaptive quantization JPEG (IBM)](https://patents.google.com/patent/US5157488)
- [US6175650 - Adaptive quantization JPEG baseline (Xerox)](https://patents.google.com/patent/US6175650)
- [US6252994 - Adaptive quantization JPEG baseline (Xerox)](https://patents.google.com/patent/US6252994)
- [US6882753 - Adaptive quantization code length](https://patents.google.com/patent/US6882753)
- [US7092578 - Signaling AQ in JPEG via EOB codes](https://patents.google.com/patent/US7092578)
- [US5724453 - Optimized quantization tables](https://patents.google.com/patent/US5724453A/en)
- [US5883979 - JPEG Q-tables for low bandwidth (HP)](https://patents.google.com/patent/US5883979)
- [US6314208 - Variable quantization JPEG compound docs](https://patents.google.com/patent/US6314208)
- [US6081211 - Minimal buffering optimized Huffman (Xerox)](https://patents.google.com/patent/US6081211)
- [US6373412 - Fast JPEG Huffman (IBM)](https://patents.google.com/patent/US6373412B1/en)
- [US6408026 - Deadzone quantization (Sony)](https://patents.google.com/patent/US6408026)
- [US7079703 - JPEG artifact removal (Sharp)](https://patents.google.com/patent/US7079703B2/en)
- [US9542760 - Parallel decoding JPEG (Google)](https://patents.google.com/patent/US9542760)
- [US9153017 - Optimized chroma subsampling (Google)](https://patents.google.com/patent/US9153017)
- [US11234023 - rANS encoding (Microsoft)](https://patents.google.com/patent/US11234023B2/en)
- [US8948530 - Adaptive image compression](https://patents.google.com/patent/US8948530B2/en)
- [JPEG XL PATENTS file](https://github.com/libjxl/libjxl/blob/main/PATENTS)
- [Beamr Patents](https://beamr.com/virtual-patent)
- [jpegli introduction (Google)](https://opensource.googleblog.com/2024/04/introducing-jpegli-new-jpeg-coding-library.html)
- [Dequantization bias - Price & Rabbani 2000](https://ieeexplore.ieee.org/document/844179/)
- [Trellis quantization Wikipedia](https://en.wikipedia.org/wiki/Trellis_quantization)
- [mozjpeg trellis implementation](https://github.com/mozilla/mozjpeg/issues/3)
- [JPEG Optimization Algorithms Review](https://fastcompression.medium.com/jpeg-optimization-algorithms-review-eb2dc1a2e154)
- [ANS patent controversy (The Register)](https://www.theregister.com/2022/02/17/microsoft_ans_patent/)
- [ANS patent - ESP Wiki](https://wiki.endsoftwarepatents.org/wiki/Asymmetric_numeral_systems)
