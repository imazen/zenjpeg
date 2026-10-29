# The mathematics of JPEG generation loss, and why Preserve avoids it

This is the theoretical foundation for the `Preserve` strategy. It is
**metric-independent** — it concerns the algebra of JPEG coefficients, not
perception. Perceptual metrics (zensim, butteraugli, cvvdp) only enter when
deciding the rate-distortion *tradeoff* layered on top (see the closing
section and `docs/AQ_DIRECTION.md`).

## 1. The operators

For one 8×8 block, decode and encode compose these maps:

- **F** — forward DCT with level shift: `F(B) = DCT(B − 128)`. The
  normalized DCT-II is **orthonormal**, so `F⁻¹ = Fᵀ` and **Parseval's
  identity** holds: `‖F(x)‖₂ = ‖x‖₂`.
- **Qₐ** — quantize to integer levels at table `a`:
  `levels = round(C ⊘ Qₐ)`; dequantize `Ĉ = levels ⊙ Qₐ`.
  (`⊘`, `⊙` are elementwise divide/multiply over the 64 positions.)
- **R** — the pixel nonlinearity: `round` to integers **and clamp** to
  `[0, 255]`.

A source JPEG stores integer levels `qₛ`. Its dequantized coefficients are
`Ĉₛ = qₛ ⊙ Qₛ`; its decoded pixels are `Bₛ = R(F⁻¹(Ĉₛ) + 128)`.

Two re-encode operators, given a new table `Q`:

```
pixel-domain  (Tuned/Deblock):  T_Q(Ĉ)  = Q_Q( F( R(F⁻¹(Ĉ)) ) )
coeff-domain  (Preserve):       T'_Q(Ĉ) = Q_Q(Ĉ)
```

Generation loss is `‖T(Ĉₛ) − Ĉₛ‖` per cycle and what accumulates under
iteration.

## 2. Result — coefficient-domain re-encode at the same table is exactly idempotent

At `Q = Qₛ`:

```
T'_Qₛ(Ĉₛ) = round(qₛ ⊙ Qₛ ⊘ Qₛ) ⊙ Qₛ = round(qₛ) ⊙ Qₛ = qₛ ⊙ Qₛ = Ĉₛ.
```

Zero loss, by algebra. The lossless re-pack and identity-Preserve are
**fixed points by construction** (verified empirically pixel-for-pixel in
`tests/api.rs::preserve_identity_emit_is_pixel_identical`).

The pixel path is **not** idempotent: `T_Qₛ(Ĉₛ) ≠ Ĉₛ` in general because
`R(F⁻¹(Ĉₛ)) ≠ F⁻¹(Ĉₛ)`. A pixel re-encode at *identical quality* still
drifts; a coefficient re-encode does not. This is the whole reason Preserve
exists.

## 3. Result — the pixel round-trip is broadband; it resurrects killed frequencies

`R` rounds (and clamps) pixels. Rounding is **not band-limited** — its
error `ε`, bounded `|ε| ≤ ½`, has spectral energy at *every* frequency. By
Parseval the perturbation lands in the coefficient domain with total energy

```
E‖F(ε)‖² = E‖ε‖² ≈ 64 · (1/12) ≈ 5.3 per block   ⟹   per-coefficient RMS ≈ 0.29.
```

The source had killed its high frequencies — those coefficients are
**exactly 0** in `Ĉₛ`. After a pixel round-trip, `F(R(F⁻¹(Ĉₛ)))` has
*nonzero* energy at those positions. Requantizing keeps whatever exceeds
`Q/2`. Iterate and spurious high-frequency content accumulates — the
textbook mosquito-noise / ringing of repeated JPEG saves.

**Support monotonicity** states the advantage cleanly:

- Coefficient-domain requantize/AQ with `Q ≥ Qₛ` can only **zero**
  coefficients or shrink magnitudes: `supp(output) ⊆ supp(Ĉₛ)`. It cannot
  create a nonzero where there was a zero.
- The pixel path *can and does* grow the support (broadband `R` energy
  surviving requantization at fine-Q positions).

The "frequency artifact" of generation loss is exactly this support growth,
and it is **strictly a pixel-path phenomenon**.

AQ zeroing is the same projection applied deliberately: idempotent
(`zero ∘ zero = zero`), support-shrinking, generation-loss-safe by
construction. AQ's only cost is rate-distortion — never generation loss.

## 4. Result — the clamp is a *biased* error, reapplied every generation

`round` is approximately zero-mean. The **clamp** to `[0, 255]` is not.
Near highlights/shadows, and at Gibbs overshoot around sharp edges,
`F⁻¹(Ĉₛ)` exceeds the valid range and the clamp truncates it — always
inward, a systematic non-zero-mean bias. The pixel path reapplies this
ratchet every cycle; the coefficient path never applies it.

## 5. Result — exact-multiple requantize is loss-minimal

If `Q = k·Qₛ` with integer `k`, then

```
q_t = round(qₛ ⊙ Qₛ ⊘ (k·Qₛ)) = round(qₛ / k)
```

— pure integer level-division, **no DCT, no cross-frequency coupling, no
R**. The only loss is the inherent k→1 rebinning, which is
information-theoretically unavoidable for that rate reduction. This is why
`UniformScale` with near-integer factors is the gentlest tightening, and
why `TargetQuality` is clamped never to go *finer* than the source: refining
a position the source already coarsened cannot recover information; it can
only invite pixel-path-style resurrection if a later stage round-trips.

## 6. The dynamical-systems view

Iterating the pixel re-encode `T_Q` converges toward a nearby fixed point —
a coefficient block that survives decode-encode unchanged. This is why the
first 1–3 saves lose the most and then the file stabilizes: you are relaxing
onto the fixed-point set of `T_Q`. Preserve sidesteps the dynamics entirely:
`T'_Q` *is* a projection (idempotent), so you land on the fixed point in one
step with no transient loss.

## 7. Generation counts per strategy

| strategy | pixel round-trips (`R∘F⁻¹` then `F`) | clamp applied | can grow support |
|---|---|---|---|
| `Lossless` (restructure) | 0 | no | no |
| `Preserve` (coeff-domain) | 0 | no | no |
| `Tuned` | 1 | yes | yes |
| `Deblock` | 1 (+ deblock filter) | yes | yes |

`Preserve` and `Lossless` are the zero-generation-loss strategies. `Tuned`
and `Deblock` each pay exactly one pixel round-trip — acceptable when the
source has block artifacts worth removing (Deblock) or when its tables can't
express the target (Tuned), but never "free".

## 8. Where the perceptual metrics enter

Everything above is coefficient algebra. zensim, butteraugli, and cvvdp all
agree that the coefficient path injects less error than the pixel path,
because that is a statement about the bytes, not about perception.

The metrics diverge only on the **rate-distortion decisions** built on top:

- which target quality hits a requested perceptual level,
- whether AQ's removed high-frequency content is perceptually free,
- whether resurrected ringing is visible.

butteraugli weights high-frequency error through XYB opponent channels plus
a masking model; cvvdp adds an explicit contrast-sensitivity function and
luminance adaptation. Both model contrast masking more explicitly than
SSIMULACRA2 / zensim. The cross-check that re-runs the AQ-direction and
Preserve-vs-Tuned experiments under all three metrics lives in
`benchmarks/tri_metric_*` and is summarized in `docs/TRI_METRIC_CROSSCHECK.md`.
