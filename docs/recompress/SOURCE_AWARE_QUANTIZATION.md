# Source-aware quantization for recompression and pixel encoding

Investigation: 2026-09-27, against `fd36fd47` after syncing `main`.
Status: implementation design based on code inspection; no new quality or
performance measurements. “Jpeglish” below means the jpegli-style YCbCr encoder.

## Recommendation

Build one internal quantization-table adapter that consumes the actual source
DQT tables and the destination encoder's resolved target tables. Use it from
both the coefficient recompressor and, subsequently, per-image pixel encode
requests. Keep the destination table family as the starting point.

First evaluate it through the existing coefficient-domain Preserve emitter.
That gives exact source coefficients, isolates table selection from pixel
round-trip error, and avoids new whole-image floating-point buffers. Then
evaluate the same candidate tables through jpeglish, moz, and classic pixel
encoding, including their actual quantization rules.

There are two contracts:

- **JPEG input with retained coefficients:** exact added coefficient error is
  computable, and unchanged dequantized coefficients can be guaranteed.
- **Pixels plus previous JPEG tables:** the tables are a useful hint when the
  component and block grids survive, but cannot guarantee preservation. The
  encoder may be seeing edited pixels, clipping, resampling, or decode noise.

Do not make table snapping a default until matched-size experiments establish
where it wins. Retain the current encoder/recompressor as a candidate.

## Existing implementation to reuse

| Location | Existing capability | Needed extension |
|---|---|---|
| `zenjpeg/src/encode/plan.rs::resolve_quant_tables` | Shared resolution for streaming encode and `resolve_plan`; produces final integer tables and zero-bias parameters | Common point for destination-table adaptation; introspection must describe the adapted result |
| `zenjpeg/src/encode/streaming.rs::from_builder` table setup | Builds `QuantContext` from resolved tables before processing strips | Adapt before constructing SIMD reciprocals/context or writing any header |
| `zenjpeg/src/encode/tuning.rs::EncodingTables` | `ScalingParams::Exact`, per-component tables and zero-bias parameters | Existing vehicle for prototype tables; avoid accidental second quality scaling |
| `zenjpeg/src/encode/tables/presets.rs::MozjpegTables::generate_ex` | Annex K, Robidoux, and other bases with IJG scaling | Classic target generator already exists |
| `zenjpeg/src/recompress/strategies/preserve_emit.rs` | Source table scaling, Annex K/Robidoux retargeting, coefficient requantization, entropy optimization | Private explicit-table emission entry point and candidate evaluation |
| `zenjpeg/src/decode` coefficient decode | Source DQTs, component-to-table mapping, sampling, integer coefficients | Source facts and coefficient statistics without RGB decoding |
| `zenjpeg/src/quant/identify.rs` | Table-law identification | Optional provenance/prior; actual DQT values remain authoritative |
| `zenjpeg/src/recompress/measure.rs` | Cached source reference and generation-loss measurement | Reuse for final candidate comparison |

Preserve currently builds tables in `build_new_quant_tables`/`build_new_table`
and requantizes with floating-point old/new ratios in `edit_coefficients`.
`TargetQuality` and `RobidouxTargetQuality` clamp entries to at least the old
entry. None of these strategies searches divisors or neighboring alignment
candidates. `refine.rs` changes per-block zero masks, not DQT entries.

The Tuned path currently decodes to RGB8 and re-encodes with
`HybridMaxCompression`; it does not pass source DQTs to the pixel encoder.
The router returns NoOp when its estimated source quality is already at or
below the requested target. Preserve's existing public target is cumulative
zensim-A, not an instruction to manufacture a finer DQT on quality increases.

Classic encoding is expressible using `MozjpegTables::generate_ex(q,
QuantTablePreset::JpegAnnexK, true)` as exact custom tables, with AQ,
deringing, and trellis disabled. There is no dedicated classic optimization
preset in the inspected enum. `JpegliBaseline` still uses jpegli tables and
AQ; “baseline” only selects a sequential scan here. This classic configuration
does not imply byte parity with C libjpeg's DCT/color/downsampling pipeline.

## What can actually be guaranteed

For source integer coefficient `k`, old step `a`, and candidate step `b`, use
JPEG dequantized units:

```
x = k * a
k_new = nearest_integer(x / b)
error = k_new * b - x
```

If `b` divides `a`, every source coefficient is exactly representable. Keeping
`b = a` is already sufficient for an unchanged image. A smaller divisor is
useful when finer tables are required or edited blocks need new detail, but
cannot recover lost detail and may cost more bytes. Prime old entries can
make divisor-only choices much too coarse or much too expensive.

At lower quality, odd multiples align nearest-rounding bin boundaries in the
ideal scalar model. This does **not** imply that snapping to `3*a` beats a
generic `1.2*a` at a comparable rate: `3*a` is much coarser. Even multiples put
old reconstruction levels on new decision boundaries; ties toward zero are a
candidate worth evaluating, not a universally optimal rule.

The bin-boundary equivalence also assumes the original coefficient was formed
by nearest rounding. Source AQ/zero bias/trellis can violate that assumption.
The stored reconstruction `k*a`, and its representability under a divisor,
remain exact regardless of how the source encoder chose `k`.

Changing a table, zeroing coefficients, or applying trellis can all introduce
generation loss without a pixel round trip. Some older statements in
`GENERATION_LOSS_THEORY.md` and `RECOMPRESSION_COMPENDIUM.md` use “zero
generation loss” to mean “no pixel round-trip loss” and claim universal
optimality of integer multiples. Those stronger claims must not be used as
implementation guarantees. The historical tri-metric comparison also used
different output sizes, so it is not proof of a matched-rate advantage.

Reference for scalar requantization and rounding:
[Bauschke et al., A Requantization-Based Method for Recompressing JPEG Images](https://cmps-people.ok.ubc.ca/bauschke/Research/c06.pdf).

## Candidate selection

Start from final, rounded and precision-clamped target tables generated by the
chosen destination mode. Handle each frequency independently when generating
candidates: changing families may make some entries finer and others coarser
even when a single quality number moves in one direction.

For each source step `a` and destination step `t`, consider a bounded set:

1. Unmodified `t` and nearby legal integers.
2. `a`, if retaining that frequency fits the search budget.
3. Divisors of `a` near `t` when finer steps are appropriate.
4. Nearby multiples, odd multiples, and integers adjacent to even multiples.

Enforce precision limits during candidate generation. Clamping a selected
divisor or multiple afterward can destroy its alignment. Include DC in the
correctness model, but initially hold its table entry at the baseline choice
to limit brightness/block-boundary regressions during AC optimization.

With coefficient histograms, compute the actual added squared error:

```
D[c, f, b] = sum_k histogram[c, f, k]
                  * (b * round(k * a[c, f] / b) - k * a[c, f])^2
```

Use fixed frequency/component weights for comparing table candidates. If the
weights themselves change with the candidate step, the scoring objective can
reward a table just for changing the units of distortion. This cost is relative
to the source reconstruction; it does not estimate error against the unknown
original without an additional model.

Histograms support distortion screening, but not exact JPEG byte estimates:
AC zero runs, DC prediction, Huffman adaptation, scan structure, and table
overhead couple coefficients. Use a cheap rate approximation to propose a few
whole-table candidates, then encode those candidates and compare actual bytes
and decoded quality. A bounded coordinate search is a practical starting
point; “round every entry to the nearest multiple” is only a control variant.

When only the old table is available, use a conservative candidate heuristic
and call it a hint. There is no content-independent guarantee that it beats
the original target at equal file size. Leave pixel streaming intact: richer
image statistics require an explicit replay/prepass budget, not hidden storage
of all floating-point DCT blocks.

If Cb and Cr share an output table, aggregate their costs before choosing its
entries. If their source tables differ, exact preservation by a shared new
step requires it to divide both source entries. A third table can avoid this
constraint, at a measurable header cost. Resolve every distinct table once;
the repo already fixed a shared-chroma double-scaling bug in this area.

## Integration by destination mode

| Mode | Initial integration | Preservation limit |
|---|---|---|
| Classic Annex K | Adapt resolved exact tables; neutral zero bias; AQ/trellis/deringing off for the control | Pixel rounding, clipping, color conversion and chroma resampling can still cause drift |
| Jpeglish | Adapt integer tables while retaining destination zero-bias parameters for the first ablation; compare normal AQ with reduced/disabled AQ separately | AQ/zero bias and deringing can change representable coefficients; divisibility alone is insufficient |
| Moz | Adapt final Robidoux tables before `TrellisContext` consumes them; evaluate with the actual AC/DC trellis enabled | Trellis may select a cheaper coefficient even when an exact one exists |

For the jpeglish prototype, export both quant tables **and resolved zero-bias
parameters** into `EncodingTables::Exact`. Constructing fresh default custom
tables would silently change the bias. Likewise `Custom` currently selects a
separate-chroma layout, so record/control that difference in comparisons.
Normal jpegli resolution infers effective distance from the written tables to
set zero bias. Whether to recompute that estimate after adaptation should be
a measured second ablation: adapted tables need not fit the original family.

For a later source-aware moz trellis experiment, use `k*a` as the distortion
reference when the actual source coefficients are available. In the current
trellis API `src` is scaled by 8, so the corresponding input is `8*k*a`, with
checked intermediate range handling. Do not pass quantized `k` directly.
For strict preservation, bypass coefficient optimization altogether.

Use per-image request context for the eventual pixel hint, rather than leaving
source-image tables attached to a reusable encoder configuration. The hint
needs actual tables, component mapping, sampling, color interpretation, and
the caller's grid/transform validity information. A source quality number or
encoder fingerprint alone is insufficient.

Apply grid alignment only to unchanged component grids: a resize, arbitrary
crop, chroma resampling, YCbCr-to-XYB conversion, or color transform invalidates
the direct correspondence. Block-aligned lossless transforms can preserve it
if coefficient geometry and tables are transformed together. Table hints must
never silently snap intentional pixel edits back to the old source values.

## Implementation sequence

1. **Internal prototype:** add `quant/requant.rs` with bounded candidate
   generation, integer requantization and distortion accumulation. Use wide
   products/accumulators, explicit signed tie handling, and checked output
   representability. The existing emitter clamps all coefficients to
   `[-1024, 1023]`; audit that assumption before claiming exactness for newly
   supported finer-table/extended-input cases.
2. **Reuse emission:** factor Preserve's emission after table resolution into
   a private function accepting explicit tables. Keep its metadata, geometry,
   DQT ordering and entropy paths. Do not add a second JPEG serializer or
   change the existing expert `QuantStrategy` enum just for the experiment.
3. **Compare the three encoders:** extend the recompression experiment harness
   with generated tables from each destination mode and the pixel paths above.
   First keep AQ masking off in the coefficient control. Existing exact custom
   tables make this possible without committing to a new stable API.
4. **Choose policy from measurements:** compare source-only table hints against
   coefficient-informed selection; identify useful effort/quality ranges.
   Keep the generic candidate available. Validate actual output size and
   quality before accepting a candidate.
5. **Production integration:** add per-image request plumbing into the shared
   resolver/streaming builder and `resolve_plan` reporting. Preserve a single
   authoritative final-table resolution. Expose a stable API once the policy
   and applicability rules have evidence.
6. **Recompression routing:** introduce the winning behavior with fresh
   calibration. The existing achieved-quality and GEXP tables describe current
   Preserve/Tuned behavior; silently changing that behavior would invalidate
   their predictions. Keep source-relative measured quality separate from
   estimated cumulative quality against an unavailable original.

## Required evidence before enabling by default

Use real source files from pinned libjpeg-turbo, mozjpeg and jpegli encoders,
plus zenjpeg sources. Encode every source through each destination mode.
Cover equal quality, small increases/decreases, larger reductions, shared and
separate chroma tables, grayscale, 4:4:4/4:2:0/4:2:2/4:4:0, and partial MCUs.
Include photos, text/screenshots, gradients and clipping-heavy edges.

Report byte-matched generation loss versus the decoded source, cumulative
quality versus known originals, speed, memory, and the distribution of
per-image regressions. Use zensim plus a second perceptual metric for finalists.
Do not compare equal quality-slider settings as if they implied equal rate.
For quality increases on unchanged JPEG input, include lossless repacking as
the strongest control.

Correctness coverage should include identity, exact divisors, positive and
negative midpoint ties, mixed finer/coarser entries, zero and DC-only blocks,
natural/zigzag conversion, nonstandard component IDs, shared table slots,
16-bit source DQTs, and unchanged defaults with no source hint. Check the
integer scalar properties independently of AQ/trellis and separately verify
actual decoded outputs. No tests or benchmarks were run for this design-only
investigation.
