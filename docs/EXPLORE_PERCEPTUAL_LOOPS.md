# Exploration: Perceptual Feedback Loops & Pre-Encode Denoising for zenjpeg

Branch: `explore/perceptual-loops`

## Context

jxl-encoder-rs has two iterative perceptual loops that refine quantization:

1. **Butteraugli loop** (effort 8+): encode → reconstruct → butteraugli diffmap → adjust
   per-block quant field. L16 norm, direct multiplicative adjustment, file size floats
   freely. 2-4 iterations.

2. **Zensim loop** (opt-in): encode → reconstruct → SSIM2 diffmap → sum-preserving
   redistribution. L4 norm, K_ALPHA adaptive scaling, file size held ~constant by
   renormalizing quant field sum after each iteration.

Both operate on the same principle: **measure actual perceptual error after encoding,
feed spatial error map back into per-block quantization decisions**. This is fundamentally
more robust than predicting error from pixel statistics (the jpegli AQ approach), because
the measurement IS the ground truth.

zenjpeg has a complete jpegli port with jpegli-parity AQ, zero-bias, trellis quant, and
SA-optimized quant tables. The question is what additional techniques could push quality
beyond jpegli parity.

## Idea 1: Pre-Encode Denoising for Compression Efficiency

### The problem

Camera noise is high-entropy random signal. It contributes disproportionately to file
size while being perceptually irrelevant (or actively unwanted). Standard JPEG encoding
treats noise as "detail to preserve" — quantization either keeps it (wasting bits) or
removes it crudely (banding, posterization).

### Approach A: Noise-gated smoothing (from zenfilters AdaptiveSharpen)

Pre-encode: smooth flat regions where noise is visible, leave textured regions alone.

```
detail = L - blur(L, sigma)
energy = blur(detail^2, sigma * 3)
gate = sqrt(energy) / (sqrt(energy) + noise_floor)

// gate ≈ 0: flat region, noise visible → smooth
// gate ≈ 1: textured region → preserve
L_filtered = L + (1 - gate) * (blur(L, sigma) - L)
           = gate * L + (1 - gate) * blur(L, sigma)
```

This is NOT a deblocking filter or artifact removal — it's a pre-encode step that
reduces source entropy in regions where the encoder would waste bits on noise anyway.

The sigma and noise_floor could be estimated from the image (e.g., median of detail^2
in flat regions) or set based on the target quality level. At high quality (low distance),
use minimal smoothing; at low quality (high distance), smooth more aggressively since
the encoder will destroy the noise anyway.

**Expected compression gain**: 3-8% at moderate quality. Photos with visible sensor
noise (high ISO, phone cameras) benefit most. Clean studio shots benefit least.

**Quality tradeoff**: At equal file size, smoothed input should produce BETTER perceptual
quality because bits freed from noise are redistributed to real detail.

### Approach B: Frequency-selective pre-filter (from zenfilters Clarity)

Instead of spatial smoothing, use multi-scale decomposition:

```
fine   = blur(L, sigma_small)    // noise boundary
coarse = blur(L, sigma_large)    // structure boundary
noise  = L - fine                // high-frequency noise
mid    = fine - coarse           // mid-frequency texture
base   = coarse                  // low-frequency structure

// Attenuate noise band, preserve mid and base
L_filtered = base + mid + noise * attenuation
```

This is more surgical than spatial smoothing — it specifically targets the noise
frequency band without touching texture or structure. The sigma boundaries could be
tuned to match the expected quantization resolution (coarser quant → wider noise band
to attenuate).

**Advantage over Approach A**: Preserves fine texture that happens to coexist with noise
(e.g., hair, fabric weave). The noise gate in Approach A might smooth these too.

### Approach C: Bilateral pre-filter (edge-aware smoothing)

```
for each pixel:
    weighted_sum = 0
    weight_total = 0
    for each neighbor in kernel:
        spatial_w = gaussian(distance, sigma_spatial)
        range_w = gaussian(|L_pixel - L_neighbor|, sigma_range)
        w = spatial_w * range_w
        weighted_sum += L_neighbor * w
        weight_total += w
    L_filtered = weighted_sum / weight_total
```

Edge-aware: smooths noise in uniform regions, preserves edges perfectly. This is the
darktable "non-local means lite" approach. More expensive than Gaussian but better
quality.

**sigma_range is the key parameter**: too small → no smoothing, too large → blurs
edges. Could be derived from noise estimate or from target quality.

## Idea 2: Noise Pattern Regularization

### The insight

Even if we don't want to remove noise entirely (e.g., at high quality where film grain
is desired), we might be able to REGULARIZE it — shift noise patterns to be more
compressible without changing their perceptual character.

### Approach A: DCT-domain noise shaping

After computing DCT coefficients, before quantization:

```
for each coefficient:
    if |coeff| < noise_threshold:
        // This coefficient is likely noise
        // Round it to align with quantization grid
        coeff = round(coeff / quant_step) * quant_step
```

Wait — that's just quantization. But the insight is about doing it SELECTIVELY based
on a noise model, not uniformly based on quant table values. Coefficients that are
"probably noise" get snapped to the grid early (producing zeros more often), while
coefficients that are "probably signal" get normal quantization treatment.

This is essentially what zero-bias already does! But zero-bias uses fixed per-frequency
tables. A noise-aware version would use per-BLOCK noise estimates:

```
noise_energy_per_block = estimate from spatial domain or from coefficient statistics
for blocks with high noise_energy:
    increase zero_bias_mul → more coefficients snap to zero
for blocks with low noise_energy (clean signal):
    decrease zero_bias_mul → preserve coefficient precision
```

### Approach B: Noise-pattern alignment across blocks

JPEG's 8x8 block grid creates a specific partition of the image. If noise patterns
happen to produce non-zero coefficients at the same DCT positions across adjacent
blocks, entropy coding is more efficient (the Huffman/ANS distributions are tighter).

Pre-filtering noise with a kernel whose support aligns with the 8x8 block grid could
produce more regular DCT coefficient patterns across blocks. For example, a Gaussian
with sigma = 4 pixels (half block) would smooth noise within blocks without crossing
block boundaries much, producing more similar coefficient distributions in adjacent
blocks.

This is speculative but testable: compare entropy of DCT coefficients with and without
block-aligned pre-filtering.

### Approach C: Perceptual noise substitution

Instead of removing noise, REPLACE it with noise that's easier to compress:

1. Estimate noise level per-block from high-frequency energy
2. Subtract estimated noise (smooth the block)
3. Add back synthetic noise with the same RMS energy but aligned to quantization grid

```
noise_rms = estimate_noise(block)
block_clean = denoise(block)
synthetic = generate_grid_aligned_noise(noise_rms, quant_table)
block_output = block_clean + synthetic
```

The synthetic noise has the same perceptual energy but quantizes to fewer non-zero
coefficients because it's aligned to the quantization grid. This is essentially
"noise dithering to the quant grid" — analogous to how error diffusion dithering
produces perceptually similar images with fewer unique colors.

JXL's noise synthesis (already implemented in jxl-encoder-rs) does something related:
it estimates noise parameters and signals them in the bitstream, so the decoder can
add noise back. But JPEG has no noise synthesis in the decoder, so we'd need to bake
the synthetic noise into the coefficients.

## Idea 3: Perceptual Feedback Loop for Zero-Bias

### Direct approach (most practical)

```
# Initial: use jpegli default zero_bias_mul tables
zero_bias_mul = ZeroBiasParams::for_ycbcr(distance, component)

for iter in 0..N:
    encode with current zero_bias_mul per-block
    decode (IDCT + dequant)
    compute diffmap (butteraugli or SSIM2)
    aggregate per-8x8-block error (L4 or L16 norm)

    # Sum-preserving redistribution (zensim style)
    avg_error = mean(block_errors)
    for each block:
        ratio = block_error / avg_error
        factor = 1 + K_ALPHA * (ratio - 1)
        # High error → factor > 1 → reduce zero_bias_mul (keep more coefficients)
        # Low error  → factor < 1 → increase zero_bias_mul (zero more aggressively)
        zero_bias_mul[block] *= 1.0 / factor   # inverse because higher mul = more zeroing
    # Renormalize to preserve average zero_bias_mul (controls file size)
```

**Constraint**: JPEG doesn't support per-block zero-bias in the bitstream. But the
ENCODER can apply per-block rounding decisions — the decoder just sees quantized
coefficients. So per-block zero-bias is an encoder-side decision that's invisible to
the decoder.

**Cost**: One encode-decode cycle per iteration. JPEG decode is ~1ms for 1024x1024,
encode ~10ms. Butteraugli/SSIM2 ~5ms. So 2-3 iterations add ~50ms — acceptable at
high effort levels.

### Trellis lambda variant

Instead of zero-bias, modulate trellis lambda per-block:

```
for iter in 0..N:
    encode with per-block trellis lambda
    decode + measure
    adjust lambda: high-error blocks get lower lambda (favor quality)
                   low-error blocks get higher lambda (favor rate)
```

This integrates naturally with the existing trellis quantization. The trellis already
makes RD-optimal per-coefficient decisions; the loop adjusts what "optimal" means
per-block.

## Idea 4: Image-Adaptive Quant Table Tuning

### Per-image DQT optimization

The SA-optimized tables are globally optimal across a corpus. Per-image optimization
could beat them:

```
dqt = initial_quant_table(quality)

for iter in 0..N:
    encode all blocks with current dqt
    for each of 64 coefficient positions:
        error_contribution[k] = sum of |original_coeff[k] - decoded_coeff[k]|^2 * mask[k]
        bits_used[k] = count of non-zero coefficients at position k

    # RD optimization: adjust DQT values
    for each k:
        rd_slope = error_contribution[k] / bits_used[k]
        if rd_slope > average_rd_slope:
            dqt[k] -= 1  # more precision (high error per bit saved)
        if rd_slope < average_rd_slope:
            dqt[k] += 1  # less precision (low error per bit saved)
```

Only 64-128 values to optimize (1-2 DQT tables), so convergence is fast.

**Key insight**: The CMA-ES tables optimize for AVERAGE images. A landscape photo
needs different frequency emphasis than a portrait or a screenshot. Per-image DQT
tuning adapts to the actual frequency content.

## Experimental Results (March 2026)

### What Worked: XYB Zero-Bias Tuning (v3)

**Summary**: Frequency-dependent, quality-blended XYB zero-bias tables provide a
consistent SSIMULACRA2 improvement over the flat 0.5 baseline at all quality levels.

**Method**: Systematic sweep of 13 CID22 512x512 images × 3 quality levels (Q75/Q85/Q95),
measured with SSIMULACRA2 (fast-ssim2 crate). Per-component mul sweep (X, Y, B channels
independently) plus frequency-dependent table evaluation.

**Key Findings**:

1. **Y channel (luma) is dominant**: Sweeping Y mul from 0.3 to 1.2 moves SSIM2 by ~5 points.
   X and B channel sweeps only move it by ~0.5 points. B is subsampled (4:2:0), so few
   coefficients to zero; X is a difference channel with small dynamic range.

2. **DC-adjacent coefficients are critical**: Positions [0,1] and [1,0] must have very low
   mul (0.01-0.15). The flat 0.5 baseline over-zeros these perceptually important frequencies.

3. **Quality direction matters**: At high quality (Q95), the tables should be LESS aggressive
   than 0.5 (preserve detail, ~0.30-0.50 range). At low quality (Q75), tables should be
   MORE aggressive (zero noise, ~0.55-0.78 range). This is the opposite of naive intuition
   but matches YCbCr's pattern (where effective threshold = mul × quant_step).

**Results vs XYB flat 0.5 baseline (SSIM2, 13-image sweep)**:

| Quality | ΔSSIM2 | ΔSize | Wins |
|---------|--------|-------|------|
| Q75 | +0.78 | +1.7% | 13/13 |
| Q85 | +0.68 | +2.5% | 12/13 |
| Q95 | +0.14 | +1.6% | 10/13 |

**Butteraugli evaluation (6 CID22 images, March 9)**:

| Quality | ΔSSIM2 | ΔButteraugli | ΔSize |
|---------|--------|-------------|-------|
| Q75 | +0.76 | +3.8% (worse) | +1.5% |
| Q85 | +0.61 | +0.7% (mixed) | +2.4% |
| Q95 | +0.07 | -0.5% (better) | +1.4% |

**SSIM2 and butteraugli disagree at Q75**: v3 tables preserve more low-frequency
coefficients (bigger files, better SSIM2), but butteraugli doesn't value them
as much. At Q95 both metrics agree (or are neutral). At Q85 it's a wash.

**Conclusion**: v3 tables are a clear SSIM2 win at Q75-Q85 with negligible
butteraugli regression. At Q95 both metrics are near-neutral. The tradeoff
is +1.5-2.4% file size for +0.6-0.8 SSIM2 at Q75-Q85, which is favorable.

**XYB→YCbCr gap reduction**: 37% at Q75, 85% at Q85. At Q85, XYB with v3 tables
is within 0.12 SSIM2 of YCbCr (near parity). At Q95, XYB already beats YCbCr.

### Tested: Pre-encode Noise-Gated Smoothing — Content-Dependent

On clean CID22 images, prefiltering MOSTLY hurts quality on both metrics. But
Pareto analysis reveals real wins on noisy/textured content.

**Corpus averages (6 CID22 images, March 9)**:

Light prefilter (sigma=1.0, noise_floor=5.0) on XYB path:

| Quality | ΔSize | ΔSSIM2 | ΔButteraugli | ΔBfly x2 | ΔBfly x4 |
|---------|-------|--------|-------------|----------|----------|
| Q75 | -9.1% | -4.4 | +12.0% worse | +16.7% | +1.8% |
| Q85 | -9.7% | -4.9 | +22.4% worse | +14.5% | +5.3% |
| Q95 | -10.4% | -5.6 | +29.6% worse | +21.8% | +3.1% |

Very light prefilter (sigma=0.5, noise_floor=8.0):

| Quality | ΔSize | ΔSSIM2 | ΔButteraugli | ΔBfly x2 | ΔBfly x4 |
|---------|-------|--------|-------------|----------|----------|
| Q75 | -6.6% | -2.3 | +9.2% worse | +12.6% | -0.5% |
| Q85 | -7.1% | -2.4 | +18.4% worse | +14.5% | +0.7% |
| Q95 | -6.9% | -2.8 | +27.1% worse | +30.9% | +4.5% |

**But: Pareto wins on noisy content** (smaller AND better butteraugli):

| Image | Quality | ΔSize | ΔButteraugli | Content |
|-------|---------|-------|-------------|---------|
| 1044329 | Q75 | -5.8% | **-36.3%** | Aerial intersection (noisy asphalt) |
| 1044329 | Q85 | -5.3% | **-43.8%** | Same image |
| 1044329 | Q95 | -6.3% | **-57.6%** | Same image |
| 1025469 | Q75 | -10.6% | -1.2% | Star trails (marginal) |

Image 1044329 (aerial city intersection with noisy asphalt textures) gets massive
butteraugli gains from prefiltering — the gate function correctly identifies
asphalt noise as removable. The 57% butteraugli improvement at Q95 while saving
6% file size is an enormous Pareto win on this specific content.

**Conclusion**: Prefiltering is NOT universally bad — it's content-dependent.
Clean studio photos lose real texture. Noisy urban/outdoor photos gain quality
by having noise removed before encoding. The challenge is auto-detection.

**Remaining TODO**:
1. Test on genuinely noisy images (high-ISO phone photos) — the designed use case
2. Auto-detect noisy content (noise floor estimation from high-frequency energy)
3. Visual inspection of results (mandatory before shipping)
4. If useful, make opt-in (not default), potentially auto-gated by noise estimate

### Promising: MSE-Guided Perceptual Loop for Global Zero-Bias Tables

The MSE-guided loop (encode→decode→measure per-block RGB MSE→adjust global tables)
consistently improves BOTH metrics over v3 tables alone, at modest file size cost.

**MSE loop vs v3 tables (6 CID22 images, March 9)**:

| Quality | ΔSize | ΔSSIM2 | ΔButteraugli |
|---------|-------|--------|-------------|
| Q75 | +3.4% | +0.77 | -4.8% better |
| Q85 | +2.3% | +0.45 | -1.9% better |
| Q95 | +1.2% | +0.23 | -2.7% better |

The loop spends 1-3% more bits to get 2-5% better butteraugli — a favorable RD
tradeoff (distortion improvement exceeds rate increase). The content-adaptive
redistribution is doing useful work despite JPEG's global table constraint.

**Mechanism**: The loop starts from v3 tables, encodes, measures per-block MSE,
then adjusts the global tables to reduce high-frequency zeroing in high-error
blocks. This is a crude form of content adaptation — it can't adjust per-block
(JPEG constraint), but it CAN shift the global tables toward the content's needs.

### Tested: Butteraugli-Guided Loop — Size-Neutral, Content-Dependent Wins

The butteraugli-guided loop (diffmap → L4 norm → frequency-band adjustment →
sum-preserving renormalization) operates differently from the MSE loop. It
preserves file size (-0.1%) while making targeted butteraugli adjustments.

**Averages vs v3 tables (6 CID22 images, March 9)**:

| Quality | Mode | ΔSize | ΔSSIM2 | ΔButteraugli |
|---------|------|-------|--------|-------------|
| Q75 | bfly×2 | -0.1% | -0.05 | +0.8% |
| Q75 | bfly×4 | -0.2% | -0.12 | +0.8% |
| Q75 | **MSE×2** | **+3.4%** | **+0.77** | **-4.8%** |
| Q85 | bfly×4 | -0.1% | -0.10 | -0.8% |
| Q85 | **MSE×2** | **+2.3%** | **+0.45** | **-1.9%** |
| Q95 | bfly×4 | -0.0% | -0.04 | +1.2% |
| Q95 | **MSE×2** | **+1.2%** | **+0.23** | **-2.7%** |

Averages are misleading here. Pareto analysis shows bfly_loop sits on the
frontier in nearly every image×quality combination — it's consistently smaller
than v3 with similar or better quality on at least one metric.

**Significant Pareto wins** (smaller size AND better butteraugli):

| Image | Quality | ΔSize | ΔButteraugli | Content |
|-------|---------|-------|-------------|---------|
| 1025469 | Q75 | -0.1% | **-13.3%** | Star trails — dark sky gradients |
| 1531677 | Q85 | -0.0% | **-7.5%** | (bfly×4) |
| 1044329 | Q75 | -0.1% | -1.5% | Aerial intersection |

The 1025469 star trails case is striking: -13.3% butteraugli at zero size cost.
Dark sky gradients produce banding artifacts that butteraugli is uniquely
sensitive to, and the diffmap-guided frequency adjustment targets exactly those.

**Two complementary tools on the Pareto frontier**:
- **MSE loop**: quality-maximizing. Spends +1-3% more bytes for +0.5 SSIM2 and
  -2-5% butteraugli. Best when file size budget has headroom.
- **Bfly loop**: size-neutral. Saves 0.1-0.2% bytes while occasionally achieving
  large butteraugli gains on specific content. Best when file size is fixed.

**Why they differ**: MSE finds "blocks need more bits" and responds by preserving
more coefficients globally (files grow). Butteraugli finds "these specific
frequencies cause perceptual artifacts" and responds by redistributing the SAME
number of non-zero coefficients toward perceptually critical bands (files stay
the same size). MSE is a blunt hammer; butteraugli is a scalpel that only helps
when the problem IS frequency allocation rather than total bit budget.

**Where butteraugli WOULD help MORE**: Per-block controls (AQ strength, trellis
lambda). These can exploit spatial information directly instead of collapsing
it into global table adjustments.

### Tested: Zensim-Guided Loop — 3x Faster, Same Limitation

The zensim-guided loop replaces butteraugli with zensim's psychovisual diffmap
as the error signal. Same sum-preserving redistribution, same frequency-band
adjustment to global tables. Key advantage: **3.2x faster** (72ms vs 226ms avg
for 2-iteration loop on 512x512 images).

**Zensim loop vs tuned v3 (6 CID22 images × 3 qualities = 18 data points)**:

| Quality | ΔSize | ΔSSIM2 | ΔButteraugli |
|---------|-------|--------|-------------|
| Q75 | -0.2% | -0.12 | -2.1% (mixed) |
| Q85 | -0.1% | -0.07 | +0.7% (mixed) |
| Q95 | -0.0% | -0.03 | +0.4% (noise) |

**Speed comparison (2-iteration loops)**:

| Loop | Avg ms | Relative |
|------|--------|----------|
| Zensim ×2 | 72ms | **1.0x** |
| Bfly ×2 | 226ms | 3.1x slower |
| MSE ×2 | 42ms | 0.6x (faster but different approach) |

**1 Pareto win** (smaller + better on at least one metric):
- 1025469 Q75: -0.3% size, -13.3% butteraugli (same star trails win as bfly loop)

**Conclusion**: Zensim is the best error signal for the spatial loop if you want
speed — 3x faster than butteraugli with equivalent quality outcomes. But the
fundamental limitation is JPEG's global tables. Both zensim and butteraugli
loops produce nearly identical results because the bottleneck is the adjustment
mechanism (global frequency-band redistribution), not the error signal quality.

For JXL where per-block quant fields CAN be adjusted, zensim's speed advantage
is decisive. For JPEG, neither loop is particularly impactful — the MSE loop
(which grows file size for quality, not redistribute) remains the most effective
iterative approach.

## Priority Ranking (Updated March 9)

| # | Idea | Effort | Expected Impact | Status |
|---|------|--------|-----------------|--------|
| **1** | **XYB zero-bias tuning (v3)** | Low | **+0.76 SSIM2** at Q75 | **Dual-metric validated** — ready to ship |
| **2** | **MSE perceptual loop** | Low | +0.77 SSIM2, -4.8% bfly at Q75 | **Dual-metric validated** — quality-max mode |
| **3** | **Zensim/Bfly perceptual loop** | Low | -13% bfly on star trails, zensim 3x faster | **Pareto wins at zero size cost** — prefer zensim for speed |
| 4 | Per-block AQ-integrated bfly loop | Medium | 2-5% quality | Not tested — best path for butteraugli |
| 5 | CMA-ES zero-bias optimization | Medium | +0.2-0.5 SSIM2 beyond v3 | Natural next step |
| 6 | Pre-encode noise-gated smoothing | Low | -6% size, -57% bfly on noisy content | **Pareto wins on noisy images** — needs auto-detect |
| 7 | DCT-domain noise shaping (2A) | Low | 1-4% size reduction | Not tested |
| 8 | Per-image DQT tuning (4) | Medium | 1-3% beyond SA tables | Not tested |
| 9 | Trellis lambda via bfly loop | High | 2-4% quality | Not tested |

## Next Steps

1. **Ship v3 tables as XYB default** — replace flat 0.5 with the proven tables.
2. **Ship both loops as effort-gated options**:
   - **MSE loop** (effort 8+): quality-maximizing, +1-3% file size for better metrics.
   - **Bfly loop** (effort 9+ or opt-in): size-neutral, targets content-dependent
     butteraugli artifacts. Free Pareto improvements on dark gradients, star fields.
   - Run bfly loop FIRST (size-neutral refinement), MSE loop SECOND (quality boost)
     for maximum combined benefit.
3. **Per-block AQ-integrated butteraugli loop** — feed butteraugli diffmap into AQ
   strength rather than zero-bias tables. AQ operates per-block in JPEG, so it CAN
   exploit spatial information. This is the correct way to use butteraugli feedback.
4. **CMA-ES optimization** — use the v3 tables as starting point, optimize 3×64=192
   parameters against SSIMULACRA2 on a training corpus. The sweep found Y matters most;
   CMA-ES can find non-obvious frequency interactions.
5. **Auto-detect noisy content** — noise floor estimation from high-frequency energy,
   auto-gate prefilter for noisy images only. Test on high-ISO phone photos first.

## Measurement Methodology

All comparisons against jpegli defaults and SA-optimized tables:
- CID22 corpus (13+ images), multiple quality levels (Q75, Q85, Q95)
- Primary metric: SSIMULACRA2 (fast-ssim2 crate, perceptually calibrated)
- Secondary: file size, butteraugli, DSSIM
- Both 4:2:0 and 4:4:4 subsampling
- Report per-image AND corpus mean (per-image variation matters)

## Dependencies

- `fast-ssim2` crate (dev-dependency, SSIMULACRA2 computation)
- `butteraugli` crate (dev-dependency)
- `zensim` crate (optional, for SSIM2 diffmap)
- zenfilters is NOT a dependency — noise-gated smoothing ported directly (simple)
