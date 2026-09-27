# Complete candidate binding in the existing Zq loop — September 8, 2026

Registered before candidate encodes. This continues the recovered July 18
`46a6ff3083d4` experiment in the current checkout, without importing its stale
manifest, process-global profile/gradient cache or obsolete global-q correction.
The current `encode/zq.rs` already owns global-q correction and AQ redistribution.

## Concrete implementation delta and callers

Add the research feature `__zensim-research = ["target-zq",
"zensim/custom-profiles", "zensim/feature-regime-v2", "dep:zenpredict-serving"]`.
Its concrete callers are the restored `zq_rd_probe` example and the existing
private Zq iteration loop. No public Rust type or function signature changes.
The exact zensim and zensim-regress dependency revision moves together; the
candidate Model alias matches that zensim revision's predictor source.

Under this feature only, `ZENJPEG_ZQ_BAKE` names an exact model file,
`ZENJPEG_ZQ_SEED_Q` supplies a finite starting jpegli q in [1,100], and
`ZENJPEG_ZQ_SPATIAL=scalar|neutral|active` selects the control. Explicit candidate
seeds bypass the historical B/profile bucket tables and picker. No mutable
profile alias or implicit candidate seed is accepted. Formula revision 1 must
be explicit for this initial study. Unset bake preserves the named-profile path.

Each encode call owns its complete BakeScorer, reference cache and reusable
Fused944Session. Every current decoded reconstruction, including pass zero,
feeds the complete candidate surface. Scalar uses no map; neutral measures the
same map but applies unit scales; active feeds absolute integrated score
contributions per 8x8 block to the existing next_scales policy. This is a named
research interpretation, not the old trained-diffmap peak unit: candidate
BlockArtifactBound is refused. Unsupported spatial terms and corruption gates
are refused, never silently dropped. Partial edge blocks remain represented.

Candidate sources initially support packed opaque sRGB8 only. The existing
named-profile RGB/f32 and color/alpha behavior is preserved. An optional fresh
`ZENJPEG_ZQ_TRACE_DIR` records each complete JPEG and each measured map/consumed
scale field, allowing independent accounting and actuator checks. The example
independently decodes and scores the returned bytes through BakeScorer.

## Bounded first screen and decision

Start with canonical imazen-26 training origin 2010, original registered
256-long-side variant bytes, exact D by-ID artifact, 444, jpegli q=80. First
measure its scalar score; then request five points below that score with zero
overshoot tolerance and two correction passes, forcing the existing claw-back
branch. Compare scalar, neutral, active and active repeat. No claim about
realistic seed accuracy comes from this deliberately fixed-seed engagement
screen. The existing max_passes means correction budget, so two corrections
can cost three full encodes; report observed counts, not its stale prose.

Require exact neutral/scalar bytes and scores, exact active repeat, independent
terminal scalar parity within 1e-5, current-map changes after changed pixels,
and recorded non-neutral scale application. Also run missing/invalid seed,
invalid mode, profile alias, unsupported layout/peak bound and unsupported-map
rejection checks. Run existing Zq behavior tests before and after, exact example
Clippy/build, and scoped formatting. Do not wait on CI.

If active fields are consumed but emitted bytes remain unchanged, record the
AQ actuator as ineffective on this control and inspect its coefficient effect
before any larger matrix. Engagement alone does not prove rate-distortion
benefit or a shippable model. Full train-only calibration, per-image bounds,
actual 1/2/3 encode comparisons, independent judging and release gates remain
subsequent requirements owned by the shared zensim targeting protocol.

## First screen findings and repair registration

The frozen initial binary is preserved in the artifact root. It exposed two
pre-existing controller errors: a unit scale clamps 756/806 visited AQ strengths
to 0.20, changing a neutral correction from 16,864 to 18,737 bytes; and the last
26 blocks bypass the callback at final flush (806 visited out of 832). Active
scales do reach the callback and change correction bytes, but the selected
result remains the initial JPEG. This is not evidence of an inert actuator.

The canonical `quant_field_to_aq_strength` definition has no upper 0.20 bound:
`max(0, 0.6 / quant_field - 1)`. Remove the erroneous absolute-strength clamp
while retaining existing bounded multiplicative scales; unit factors must
preserve all normal strengths exactly. Route final-flush and fallback AQ through
the same controller callback as earlier strips. Test neutral full-range identity
and require every iMCU callback, including finalize. No-controller encoding is
unchanged. These intentional fixes apply to the existing Zq controller too.

The newer named scorer also exposes the existing secant clamp panic when
99 < current q < 100. Bound the minimum proposed q by 100 before clamping the
next probe. Existing convergence/strict-bound tests must pass after the repair.
Re-run the exact registered first screen with a separately pinned binary and
fresh outputs; preserve both measurements, no gain/seed/target tuning.

The old peak-bound test's fixed 0.0002 total limit became attainable (measured
0.0001261128), so accepting that output was correct. Replace the stale
"basically impossible" numeric assumption with a permissive zero-ceiling
probe of the same search, then set strict finalization slack to half its
measured positive peak. This preserves a real strict-failure test while keeping
its search trajectory identical; no product bound or gate is weakened.
