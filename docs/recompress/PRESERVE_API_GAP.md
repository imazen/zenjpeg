# Preserve strategy API gap

The `Preserve` strategy described in [DESIGN.md](../DESIGN.md) — coefficient-
domain quantizer scaling plus per-block AQ zero-bias — is the headline
differentiator vs the Tuned/Deblock strategies because it eliminates the
pixel-domain IDCT/FDCT round-trip and the corresponding generation-loss.

In v0.1 we cannot ship it: `zenjpeg::lossless::pipeline::encode_from_coefficients`
is module-private. The only public coefficient-domain re-emit path is
`zenjpeg::lossless::restructure`, which preserves the original quantization
tables and is therefore useless for tightening to a lower target quality.

## Workplan

**Phase 1 — Upstream surface (recommended).**
Add a public `zenjpeg::lossless::encode_from_coefficients(&DecodedCoefficients,
preserved_segments, restart_mcus, stop)` function to zenjpeg, plus an entry
point that takes a `DecodedCoefficients` *with overridden quant tables* and
re-quantizes the coefficients before emit. Signature sketch:

```rust
pub fn requantize_and_encode(
    coeffs: &DecodedCoefficients,
    new_quant_tables: &QuantTableSet,
    aq_mask: Option<&AqMask>,        // optional per-block zero-bias mask
    config: &RestructureConfig,
    stop: impl Stop,
) -> Result<Vec<u8>>;
```

`requantize_and_encode` does:

1. `new_coeff[k] = round(old_coeff[k] * old_q[k] / new_q[k])`
2. Apply `aq_mask` per block, zeroing `AC[i]` where `mask.zero_ac[block_idx][i]`
3. Emit DQT segments with `new_quant_tables`
4. Run Huffman optimization over the new coefficient distribution
5. Optionally restructure into progressive scans

**Phase 2 — Own emitter (fallback).**
If upstreaming isn't feasible, port the relevant portion of zenjpeg's
`encode_from_coefficients` into `zenjpeg-recompress/src/preserve_emit.rs`.
Approximate scope: ~500 LOC for marker writing, DQT serialization, DHT
serialization, scan data Huffman-encoding. Higher maintenance burden;
deviates from zenjpeg's encoder state-machine and risks output mismatch
on edge cases (XYB, 16-bit DQT, restart markers).

## Decision deadline

v0.2 cycle. By that point either:

- the upstream PR has merged and zenjpeg-recompress depends on the new
  symbol, OR
- the fallback emitter is implemented and tested for byte-exactness
  against `zenjpeg::lossless::restructure` when `new_q == old_q` and no
  AQ mask is provided.

Until then, the router never selects Preserve and `run_preserve()` is
documented as a stub that delegates to Tuned.

## AQ zero-bias mask design

For when Preserve does ship:

- One bit per AC coefficient position per 8×8 block per component.
- Encoded as `Box<[u64]>` — 64 bits × 1 block fits in one u64. For a 4K
  image: ~768K blocks × 8 bytes = 6 MB worst case. Acceptable.
- Decided pre-emit by classifying each block (activity, AC energy
  distribution, low-band coefficient amplitude) and consulting a
  thresholded lookup table or a small MLP.
- Default thresholds derived from `zenanalyze` tier-2 features. The MLP
  ships in v0.3 once corpus + holdout validation justifies the model
  cost.

## Calibration impact

Until Preserve actually exists, the `projected_size_ratio` and
`projected_zensim_a` for the Preserve column in the calibration table
should be set to **strictly worse than Tuned** so the router never
picks it. The current seed table sets `preferred = false` on Preserve;
the router additionally filters out Preserve from candidate selection
when `cfg!(feature = "expert")` is off and a real-coefficient-domain
implementation is absent — implemented via the `preserve_available`
constant in `src/router.rs`.
