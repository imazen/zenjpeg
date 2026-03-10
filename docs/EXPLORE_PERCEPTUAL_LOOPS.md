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

## Priority Ranking

| # | Idea | Effort | Expected Impact | Risk |
|---|------|--------|-----------------|------|
| 1 | Pre-encode noise-gated smoothing (1A) | Low | 3-8% size at equal quality | Low — preprocessing only |
| 2 | DCT-domain noise shaping (2A) | Low | 1-4% size reduction | Low — extends zero-bias |
| 3 | Per-block zero-bias via loop (3) | Medium | 2-5% quality improvement | Medium — iteration cost |
| 4 | Frequency-selective pre-filter (1B) | Medium | 3-6% size | Low |
| 5 | Noise pattern regularization (2C) | High | Unknown | High — speculative |
| 6 | Per-image DQT tuning (4) | Medium | 1-3% beyond SA tables | Medium — convergence |
| 7 | Trellis lambda via loop (3 variant) | High | 2-4% quality | Medium — complex integration |

## Implementation Plan

### Phase 1: Pre-encode denoising (highest ROI)

1. Add optional pre-encode Gaussian smoothing with noise gate
2. Estimate noise level from image statistics (median of high-freq energy in flat blocks)
3. Scale smoothing strength by target quality (more smoothing at lower quality)
4. Benchmark: encode with/without, compare file size at equal SSIM2/butteraugli
5. Test on CID22 corpus + high-ISO phone photos + clean studio shots

### Phase 2: Noise pattern regularization

1. DCT-domain noise shaping: per-block noise estimates modulate zero-bias
2. Block-aligned pre-filtering: test sigma=4 Gaussian (half-block support)
3. Perceptual noise substitution: denoise + add grid-aligned synthetic noise
4. Measure entropy reduction in DCT domain with/without each technique

### Phase 3: Perceptual feedback loop for zero-bias

1. Add encode-decode-measure iteration
2. Start with butteraugli diffmap (already have the crate)
3. Zensim-style sum-preserving redistribution on zero_bias_mul
4. 2-3 iterations at high effort, gated behind effort level
5. Benchmark: is the loop worth the iteration cost?

## Measurement Methodology

All comparisons against jpegli defaults and SA-optimized tables:
- CID22 corpus (10-20 images), multiple quality levels (Q50, Q75, Q85, Q95)
- Metrics: file size, SSIM2, butteraugli, DSSIM
- Both 4:2:0 and 4:4:4 subsampling
- Report per-image AND corpus mean (per-image variation matters)
- Use `just quality-compare` or equivalent standardized benchmark

## Dependencies

- `butteraugli` crate (already a dev-dependency)
- `zensim` crate (optional, for SSIM2 diffmap)
- zenfilters is NOT a dependency — we'd port the relevant algorithms (Gaussian blur
  + noise gate) directly into zenjpeg, they're simple enough
