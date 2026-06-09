# zenjpeg Encoder Knob Space

Charted 2026-06-09 against `main` @ `7df309e3`. Every constant below was read
from source, not from docs; file:line references are to `zenjpeg/src/`.

This document answers three questions:

1. **What is the full encode-knob space?** (§1–§7)
2. **What does `auto_optimize` actually do?** (§5.3)
3. **Why is the optimization search space clumsy, and how could it be
   simplified?** (§8–§9)

---

## 0. TL;DR — the five structural problems

1. **"Hybrid" means two different things.** The `Hybrid*` presets
   (`OptimizationPreset::HybridProgressive` etc.) set jpegli AQ + **standalone**
   trellis — they never touch `HybridConfig` (encoder_config.rs:855–867). The
   `HybridConfig` type is the AQ→λ *coupling* engine. And `auto_optimize()`
   sets a `HybridConfig` whose coupling is **0.0** — so its "hybrid" is
   standalone-trellis math wearing the hybrid code path.

2. **One engine, three config types, six activation surfaces, two opposing
   priority rules.** Standalone trellis and hybrid both run
   `HybridQuantContext`; hybrid literally converts itself to a per-block
   `TrellisConfig` (`to_trellis_config`, trellis/hybrid.rs:512–535). The
   public surface splits this single engine across `TrellisConfig`,
   `HybridConfig`, and two *different public types both named* `ExpertConfig`.
   Setter-side docs say "hybrid takes priority" (encoder_config.rs:1237);
   encode-time dispatch checks **trellis first** (streaming.rs:335–339). The
   contradiction is only papered over by setters that clear each other.

3. **Chroma quality is five disconnected mechanisms** on different paths that
   don't compose (§6), two of which are silently inert in every shipped
   configuration, and chroma zero-bias doesn't follow the chroma distance at
   all. In XYB mode, `chroma_distance_scale` scales the **Y and B** channels
   (component indices 1, 2), not the "chroma" channels — X keeps the luma
   distance (streaming.rs:169 + XYB component order X,Y,B in
   foundation/consts.rs:170).

4. **Six quant-table families; the best-validated one is orphaned.**
   `sa_piecewise_v4` (+6.602 mean pareto vs jpegli on CID22 training, +6.09
   holdout) is exported from `encode/tables/mod.rs:71` and consumed by
   nothing. The CMA-ES global-scale/frequency-exponent constants are
   `#[cfg(test)]`/dead (quant/mod.rs:43–77).

5. **The flat optimizer surface (`search::ExpertConfig`) is ~784 scalars**
   with four documented-dead axes, a redundant double representation of
   zero-bias, and a λ₁/λ₂ ridge — while the axes that matter most
   (subsampling, color mode, chroma scaling) live *outside* it.

---

## 1. Pipeline stages and where each knob bites

```mermaid
flowchart LR
    A[Input rows<br/>PixelLayout ×16] --> B[S0 pre-filter<br/>pre_blur σ]
    B --> C[S1 color transform<br/>color_mode:<br/>YCbCr / XYB / Gray]
    C --> D[S2 chroma downsample<br/>ChromaSubsampling ×4<br/>XybSubsampling ×2<br/>DownsamplingMethod ×3]
    D --> E[S3 AQ field<br/>aq_enabled<br/>custom_aq_map - internal<br/>AqController - scaffold]
    E --> F[S4 deringing<br/>deringing bool]
    F --> G[S5 DCT f32<br/>no knobs]
    G --> H[S6 quantize<br/>Quality ×6 units<br/>QuantTableConfig ×5<br/>chroma_distance_scale<br/>chroma_quality<br/>allow_16bit<br/>zero-bias auto]
    H --> I[S7 coefficient opt<br/>trellis engine:<br/>standalone / hybrid<br/>boundary-rd feature]
    I --> J[S8 entropy<br/>scan_mode ×4<br/>huffman ×4<br/>restart_mcu_rows<br/>tiny_file_mode]
    J --> K[S9 container<br/>metadata segments<br/>force_sof1 XYB<br/>DQT 8/16-bit]
    L[Closed loop: Quality::Zq<br/>feature target-zq<br/>max_passes ≤ 4] -.re-runs S3..S9.-> E
```

| Stage | Knobs | Defaults | Where |
|---|---|---|---|
| S0 | `pre_blur: f32` (σ, 0.0 = off) | 0.0 | encoder_config.rs:488 |
| S1 | `color_mode: ColorMode{YCbCr{sub}, Xyb{sub}, Grayscale}` | per constructor | encoder_types.rs:280 |
| S2 | `ChromaSubsampling{None,HalfHorizontal,Quarter,HalfVertical}`; `XybSubsampling{Full,BQuarter}`; `DownsamplingMethod{Box,GammaAware,GammaAwareIterative}` (`sharp_yuv()` = Box↔GammaAwareIterative) | ctor arg; BQuarter; Box | encoder_types.rs:304,373,385 |
| S3 | `aq_enabled: bool`; `custom_aq_map` (pub(crate), streaming_builder.rs:343); `AqController` hook (unwired scaffold, strip/mod.rs:992) | true; None; None | encoder_config.rs:474 |
| S4 | `deringing: bool` (uses DC quant value; only fires on saturated pixels) | true | encoder_config.rs:470 |
| S6 | see §2–§4 | — | — |
| S7 | see §5; plus `boundary_rd_mode` (feature `boundary-rd`: SeamPenalty{alpha 2.0, threshold 0.02, drift_gain 0, retry_beta 1.0}, RetryPolicy{shrink 0.5, max_retries 2}, NeighborScope) | Off | encoder_config.rs:501 |
| S8 | `scan_mode: ProgressiveScanMode{Baseline,Progressive,ProgressiveMozjpeg,ProgressiveSearch}`; `huffman: HuffmanStrategy{Optimize,Fixed,FixedAnnexK,Custom}`; `restart_mcu_rows: u16`; `force_restart_markers`; `tiny_file_mode{Auto,Off,Force}` | Progressive; Optimize; 4; false; Auto | encoder_config.rs:438–494 |
| S9 | metadata segments (ICC/EXIF/XMP/MPF — non-RD); `allow_16bit_quant_tables`; `force_sof1` (auto for XYB) | —; false; auto | encoder_config.rs:477,483 |

Non-RD knobs (parallelism, metadata) excluded from the rest of this doc.

---

## 2. Quality resolution: six unit systems → one scalar → three distances

```
Quality::ApproxJpegli(f32)  ──────────────┐  identity
Quality::ApproxMozjpeg(u8)  ── 10-pt LUT ─┤  encoder_types.rs:149
Quality::ApproxSsim2(f32)   ── 8-pt LUT ──┼─→ internal q (jpegli 0–100)
Quality::ApproxButteraugli  ── 7-pt LUT ──┤        │
Quality::Zq / ZqExplicit ── starting q ───┘        │ to_distance()  (exact C++ jpegli formula,
                                                   ▼                encoder_types.rs:129–144)
                       butteraugli distance d:  q≥100→0.01;  q≥30→0.1+(100−q)·0.09;
                                                else 53/3000·q² − 23/20·q + 25
                                                   │
                                                   │ × chroma_distance_scale  (components 1,2)
                                                   ▼
                                  [d, d·cs, d·cs]  per-component distances   streaming.rs:167–169
                                                   │
            ┌──────────────────────────────────────┼────────────────────────────────┐
            ▼                                      ▼                                ▼
   Jpegli family (default)              MozjpegRobidoux family            Custom / Glassa
   BASE_QUANT_MATRIX_{YCBCR|XYB}[c]     quality = for_mozjpeg_tables()    EncodingTables
   scaled by distance_to_scale(d_c,k)   ── BYPASSES the distance          {quant 3×64,
   per-coeff FREQUENCY_EXPONENT[k]         pipeline entirely               zero_bias_mul 3×64,
   × GLOBAL_SCALE_*                     (+ chroma_quality: Option<u8>      offsets 3+3,
   (quant/mod.rs:31, streaming.rs:225)     overrides chroma only,          scaling: Exact|Scaled}
                                           robidoux.rs:90–94)             (tuning.rs:253)
            └──────────────────────────────────────┼────────────────────────────────┘
                                                   ▼
                              quant tables (1–3) → clamp to 255 | 32767 (allow_16bit)
                                                   │
                          quant_vals_to_distance(Y,Cb,Cr tables)  ← INVERSION, streaming.rs:255
                                                   ▼
                          ONE global effective_distance → zero-bias blend
                          HQ table @ d≤1.0 … LQ table @ d≥3.0 (quant/mod.rs:125)
                          per-component mul[64] + offset_dc + offset_ac
```

Things to notice:

- **`ApproxMozjpeg` forks both ways.** `to_internal()` remaps it to jpegli-q
  for everything distance-flavored, but `for_mozjpeg_tables()` passes the raw
  q to the Robidoux generator (encoder_types.rs:118–123). One enum variant,
  two simultaneous meanings depending on which table family is active.
- **Zero-bias does not see the per-component distances.** It sees a single
  effective distance inverted *back out of the generated tables*. Scaling
  chroma distance therefore moves chroma zero-bias only via a diluted global
  average — and moves *luma* zero-bias too.
- **`Quality::Zq` / `ZqExplicit`** (`ZqTarget{target, max_overshoot:
  Some(1.5), max_undershoot: None, block_artifact, max_passes: 2}`, zq.rs:38)
  trigger the closed loop **only when built with `--features target-zq`**
  (byte_encoders.rs:97). Without the feature they silently degrade to a
  one-shot encode at the starting q.

---

## 3. Quant-table families (six, two of them dead/orphaned)

| # | Family | Entry point | Tables | Status |
|---|---|---|---|---|
| 1 | Jpegli perceptual | `QuantTableConfig::Jpegli` (default) | 3 (Y, Cb, Cr; Cb uses base matrix 1) | live, default |
| 2 | Jpegli shared chroma | `QuantTableConfig::JpegliSharedChroma` | 2 (Cb+Cr both use Cr base matrix 2; streaming.rs:218) | live |
| 3 | Mozjpeg Robidoux | `QuantTableConfig::MozjpegRobidoux` | 2, quality-scaled, optional independent `chroma_quality` | live |
| 4 | Custom | `QuantTableConfig::Custom(Box<EncodingTables>)` | user-defined 3×64 + zero-bias + scaling mode | live (expert) |
| 5 | Glassa low-BPP | `QuantTableConfig::GlassaLowBpp(u8 3–25)` | SA-optimized for 0.15–0.50 BPP, interpolated (`tables/glassa.rs`) | live, niche |
| 6 | SA piecewise v4 | `encode/tables/sa_piecewise_v4.rs` — 20 quality-anchored sets, linear interpolation | **orphaned**: exported (tables/mod.rs:71), consumed by nothing | +6.602 pareto vs jpegli (CID22 train), +6.09 holdout; its own doc says it should be `adaptive()`'s default |
| † | CMA-ES scale/exponents | `OPTIMIZED_GLOBAL_SCALE` 5.608994, `OPTIMIZED_FREQUENCY_EXPONENT[64]` (+444 variants) | **dead**: `#[cfg(test)]` / `#[allow(dead_code)]` (quant/mod.rs:43–77) | never reachable in production |

---

## 4. AQ (adaptive quantization)

- Computed from the Y plane per 8×8 block (jpegli algorithm: pre-erosion →
  fuzzy erosion → per-block modulation), `quant/aq/`.
- **No strength knob.** On/off only (`aq_enabled`, default true). The only
  modulation is the internal quality-driven `dampen` ramp
  (`K_DAMPEN_RAMP_START` keyed on the Y quant value, quant/aq/streaming.rs:854).
- Consumed twice:
  1. At quantization: per-block strength scales the effective quant step.
  2. In hybrid trellis mode: adjusts per-block λ (§5).
- Per-block override hooks exist but are internal: `custom_aq_map`
  (streaming_builder.rs:343, pub(crate)) and the `AqController` trait
  scaffold (strip/mod.rs:992, `#[allow(dead_code)]`, "wired up by external
  callers in PR-D"). The conflicted bookmark `feat/aq-controller-scaffold`
  (918f581f) is this work.

---

## 5. Coefficient optimization: one engine wearing three coats

### 5.1 The engine

There is exactly one trellis engine: `HybridQuantContext` with
`TrellisMode::{Standalone(TrellisConfig), Hybrid(HybridConfig)}`
(trellis/hybrid.rs:853–858). In Hybrid mode each block converts the config
*into a `TrellisConfig`* before quantizing (hybrid.rs:512–535):

```
adjustment = 0                                  if !enabled or aq < aq_threshold
adjustment = aq^aq_exponent × aq_lambda_scale   otherwise
adjustment ×= dampen                            if quality_adaptive
adjustment ×= chroma_scale                      if chroma component
adjustment  = clamp(±max_adjustment)            if max_adjustment > 0
scale1 = base_lambda_scale1 + adjustment        (additive, default)
       | base_lambda_scale1 × (1 + adjustment)  (multiplicative)
λ_eff  = 2^scale1 / (2^scale2 + block_norm)     (search.rs:204–222 docs)
```

**Coupling = 0 ⇒ scale1 ≡ base ⇒ identical math to Standalone mode.** The
standalone/hybrid distinction is purely which struct carried the numbers in —
except the two structs have *different defaults and different knob subsets*:

| | `TrellisConfig::default()` (compat.rs:155) | `HybridConfig::default()` (hybrid.rs:283) | `auto_optimize(true)` | `adaptive()` oracle |
|---|---|---|---|---|
| λ₁ (`lambda_log_scale1`) | 14.75 | 14.75 | **14.5** | 12.0–16.0 by bucket/q-bin |
| λ₂ | 16.5 | 16.5 | 16.5 | 16.5 |
| DC trellis | **true** | **false** | false | Standard pick: true; Hybrid pick: false |
| AQ→λ coupling | n/a (≡0) | 0.0 | 0.0 | 0.0 |
| speed_mode | Adaptive | n/a | n/a | Adaptive / Thorough |
| delta_dc_weight | 0.0 | n/a | n/a | n/a |
| `enabled` default | true | **true** (!) | — | — |

`HybridConfig::default().enabled == true` means `config.hybrid_config(HybridConfig::default())`
*activates* the engine — and its own doc admits the defaults "emerged from
limited testing (~5 images)" (hybrid.rs:278–282).

### 5.2 Six activation surfaces, two priority systems

Encode-time truth (the only dispatch that matters, streaming.rs:333–340):

```rust
if let Some(ref trellis) = builder.trellis { processor.set_trellis(*trellis); }
else if builder.hybrid_config.enabled     { processor.set_hybrid(builder.hybrid_config); }
```

**Standalone trellis wins at encode time.** Build-time setters enforce the
*opposite* invariant by clearing:

| Call | Sets | Clears | Net winner at encode |
|---|---|---|---|
| `.trellis(t)` (doc-hidden) | `trellis = Some(t)` | nothing | **t** (even if hybrid enabled earlier) |
| `.hybrid_config(h)` (doc-hidden) | `hybrid_config = h` | `trellis = None` if `h.enabled` (encoder_config.rs:1244) | h |
| `.auto_optimize(true)` | hybrid{λ₁ 14.5} **iff YCbCr ∧ d<5.0**; `scan_mode = Progressive` **always** | `trellis = None` (only if gate passed) | hybrid — unless `.trellis()` later |
| `.optimization(Hybrid*/Mozjpeg*)` | `trellis = Some(...)` + AQ per lineage | **hybrid untouched** (encoder_config.rs:855–867) | **standalone trellis**, even after auto_optimize |
| `.expert(e)` (`encoder_types::ExpertConfig`) | hybrid first if `Some+enabled` (clears trellis), else trellis (encoder_config.rs:1290–1308) | per rule | per rule |
| `EncoderConfig::adaptive(...)` | oracle: `.trellis(TrellisConfig)` or `.hybrid_config(HybridConfig{λ})` (adaptive.rs:626–640) | via setters | per oracle |

Order-dependence demonstrations (all verified semantics):

- `.trellis(t).auto_optimize(true)` → t silently destroyed, hybrid λ14.5 runs.
- `.auto_optimize(true).trellis(t)` → t runs, the hybrid config sits enabled
  but ignored.
- `.auto_optimize(true).optimization(HybridProgressive)` → the preset's
  *standalone* trellis (λ14.75, DC on) beats auto_optimize's *hybrid*
  (λ14.5, DC off). Both are called "hybrid" in their own docs.

### 5.3 What `auto_optimize(true)` actually does (encoder_config.rs:1396–1426)

1. `d = quality.to_distance()`.
2. If `color_mode` is YCbCr **and** `d < 5.0` (≈ q ≥ 50):
   `hybrid_config = {enabled, λ₁ = 14.5, λ₂ = 16.5, coupling = 0.0, dc = false}`;
   `trellis = None`.
3. **Always** (even when the gate fails, even for XYB/Grayscale):
   `scan_mode = Progressive`.
4. Nothing else. No CMA-ES (those constants are test-only), no table change,
   AQ stays at its default (on), Huffman untouched.

Net effect: **"fixed-λ₁=14.5 AC-only trellis + progressive scan."** The AQ→λ
coupling that defines "hybrid" is zero. For XYB input it is effectively a
no-op (re-asserts Progressive, which is already the default). Below q50 it
is also a no-op apart from re-asserting Progressive.

The CLAUDE.md note "AutoOptimize = HybMax-L14.5 (confirmed identical)" is
consistent with this: HybMax-L14.5 *was* coupling-zero hybrid at λ14.5.

### 5.4 boundary-rd (feature-gated, off by default)

Independent post-quantization refinement, orthogonal to the trellis knot:
`BoundaryRd::On(BoundaryRdConfig{seam: {alpha 2.0, threshold 0.02,
drift_gain 0.0, retry_beta 1.0}, retry: {shrink 0.5, max_retries 2},
neighbors: LeftAndAbove})` (encoder_config.rs:28–318). Feature-off and
default builds are byte-identical (enforced by test).

---

## 6. Chroma quality: five mechanisms that don't compose

| # | Mechanism | Path it works on | Granularity | Gotchas |
|---|---|---|---|---|
| 1 | `ChromaSubsampling` / `XybSubsampling` | all | resolution (the biggest lever) | XYB B-channel only via `XybSubsampling` |
| 2 | `chroma_distance_scale: f32 [0.1–5.0]` | **jpegli table-gen only** | Cb+Cr locked together | **XYB mislabel**: scales components 1,2 = **Y and B**; X keeps luma distance. Doc says "(Cb, Cr) only" — false in XYB mode |
| 3 | `chroma_quality: Option<u8>` (doc-hidden) | **Robidoux path only** | single chroma table | inert on the default Jpegli path |
| 4 | `QuantTableConfig` layout (3T vs 2T) | jpegli | which base matrix Cb gets (matrix 1 vs Cr's matrix 2) | quality side-effect disguised as a compat switch |
| 5 | `HybridConfig::chroma_scale` | hybrid trellis | scales only the **coupling adjustment** | **no-op when coupling = 0** — i.e. in every shipped configuration including `auto_optimize` |
| + | zero-bias | jpegli | — | chroma zero-bias follows the single *global* effective distance inverted from all three tables (streaming.rs:255), not the chroma distance |
| + | `sharp_yuv` | YCbCr 4:2:x | downsample quality | auto-set by `adaptive()` for natural/detailed photos |

An optimizer that wants "spend 10% fewer bits on chroma" needs a different
lever depending on which table family and which coefficient-opt mode are
active — and two of the five levers are silently inert in common configs.
There is no way to move Cb independently of Cr on any path, and no way to
move chroma zero-bias with chroma distance on the default path.

---

## 7. `adaptive()`, presets, effort — the meta-knobs

### 7.1 `EncoderConfig::adaptive(image, quality)` (adaptive.rs:375, 544–647)

- Consumes 13 zenanalyze features (incl. `XybBquarterChromaLoss`, id 139).
- `infer_bucket` → {PhotoNatural, PhotoDetailed, PhotoFlat, ScreenContent,
  Illustration} × `q_bin` (7 bins) × metric (ssim2|butter) — distilled from
  the 70-cell `selector_tree_rules.json` oracle (2026-04-25 run).
- Emits: subsampling, XYB on/off (gated: `allow_xyb` ∧ ≥0.25 MP), XYB
  B-subsampling (Full iff `XybBquarterChromaLoss > 4.0`), sharp_yuv (natural/
  detailed photos), scan mode (Progressive; ProgressiveSearch at Effort::Max),
  and a `TrellisChoice`:
  - low q (<40): always **Hybrid(λ)** on 4:2:0, λ ∈ {12.0 flat/illustration,
    16.0 screen, 13.5 natural, 14.7 detailed, 14.0 else} (adaptive.rs:682–697)
  - high q: XYB 4:4:4 with **Standard** or **Off** per bucket/metric;
    PhotoNatural below q90 stays hybrid λ16.0 on 4:2:0.
- `Effort::Fast` forces TrellisChoice::Off; Balanced/Max honor the oracle.

So `adaptive()` is a *fourth* λ-policy author (after TrellisConfig defaults,
HybridConfig defaults, and auto_optimize), each with different numbers.

### 7.2 `OptimizationPreset` ×8 / `Effort` ×3 (encoder_types.rs:883–968)

| Preset | scan | tables | trellis (standalone!) | AQ | dering |
|---|---|---|---|---|---|
| JpegliBaseline | Baseline | Jpegli 3T | None | ✓ | ✓ |
| JpegliProgressive (≡ default config) | Progressive | Jpegli 3T | None | ✓ | ✓ |
| MozjpegBaseline | Baseline | Robidoux 2T | Thorough | ✗ | ✗ |
| MozjpegProgressive | ProgressiveMozjpeg | Robidoux 2T | Thorough | ✗ | ✗ |
| MozjpegMaxCompression | ProgressiveSearch | Robidoux 2T | Thorough | ✗ | ✓ |
| HybridBaseline | Baseline | Jpegli 3T | default (Adaptive) | ✓ | ✓ |
| HybridProgressive | Progressive | Jpegli 3T | default (Adaptive) | ✓ | ✓ |
| HybridMaxCompression | ProgressiveSearch | Jpegli 3T | Thorough | ✓ | ✓ |

`Effort::{Fast→JpegliBaseline, Balanced→HybridProgressive,
Max→HybridMaxCompression}`. Note again: **no preset sets `HybridConfig`**.

---

## 8. The optimizer search space, quantified

### 8.1 `search::ExpertConfig` (the flat surface, search.rs:138–357)

| Axis group | Dims | Notes |
|---|---|---|
| `tables.quant` | 192 | the big lever: halve → +65% size, double → −54% |
| `tables.zero_bias_mul` | 192 | jpegli presets only; all-zeros = +31%, all-ones = −14% |
| `zero_bias_hq` + `zero_bias_lq` endpoints | 384 | **redundant with the above** — blended *into* `zero_bias_mul` by quality |
| `tables.zero_bias_offset_dc/ac` | 6 | |
| `zero_bias_hq_distance`, `lq_distance` | 2 | ~0.1–2% |
| λ₁, λ₂, delta_dc_weight | 3 | λ₁: −46%..+12%; λ₂: −19%..+11% |
| coupling, exponent, threshold, chroma_scale, max_adjustment | 5 | hybrid family |
| `quality` | 1 | **dead under `ScalingParams::Exact`** (mozjpeg presets) |
| bools: trellis_enabled, dc, quality_adaptive, multiplicative, deringing, allow_16bit, aq_enabled | 7 | trellis_enabled ≈ −15% alone |
| categoricals: scaling mode, scan_mode(4), speed_mode(4+, **output-dead**), downsampling(3) | 4 axes | |

≈ **784 continuous dims + 11 bool/categorical axes**, of which the source
itself documents: `speed_mode` dead (identical bytes), `quality` dead under
Exact scaling, `allow_16bit` ~dead (clamped coefficients quantize to zero
anyway), `deringing` content-gated (zero effect without saturated pixels).

### 8.2 Structural pathologies

- **Multiplicative chains = non-identifiability.** The same operating point
  is expressible as quality scalar × `GLOBAL_SCALE` × per-frequency exponent
  × raw table value × AQ multiplier × zero-bias mul. Five aliases for "make
  coefficient k coarser" ⇒ ridge-shaped fitness landscapes.
- **λ₁/λ₂ ridge.** `λ = 2^s1/(2^s2 + norm)`: for low-energy blocks only
  `s1 − s2` matters; for high-energy blocks only `s1`. Two parameters
  encoding ~1.5 effective degrees of freedom.
- **Zero-bias double representation.** Endpoints (384) *and* blended result
  (192) are both writable; writing both makes one a lie depending on
  whether `blend_zero_bias()` runs.
- **Mode discontinuities.** trellis-vs-hybrid-vs-neither and the table-family
  switch change which axes are live; a black-box optimizer wastes samples
  learning dead zones per mode (this is why the jump from "pure-jpegli AQ" to
  "hybrid/λ + table choice" feels like a cliff).
- **The strongest axes live outside the struct.** subsampling, color mode
  (XYB), chroma scaling, pre_blur are EncoderConfig-level, so a "full" sweep
  needs two nested config systems with different override semantics.
- **Quality gates add cliffs**: auto_optimize d<5.0; Glassa Q3–25; adaptive's
  7 q-bins; XYB ≥0.25 MP; zq feature gate.

---

## 9. Simplification proposal

The goal: one resolved plan, one trellis engine config, one quality currency,
and a search surface whose axes are live, orthogonal-ish, and identifiable.

### 9.1 One quality currency: per-component distance, resolved early

```rust
/// THE internal quality representation. Everything else is a front-end.
pub(crate) struct ResolvedQuality {
    /// Per-component butteraugli distance. [Y, Cb, Cr] or [X, Y, B].
    d: [f32; 3],
}
```

- All six `Quality` variants resolve to `d` (Zq resolves its *starting* `d`).
- `chroma_distance_scale` and `chroma_quality` collapse into how the
  front-end fills `d[1]`, `d[2]` — and gain per-channel control for free.
- Zero-bias takes `d[c]` directly per component, killing the
  invert-the-tables-then-blend roundtrip (streaming.rs:255). One mapping per
  family: `zero_bias(d_c, c)`.
- XYB gets the *correct* semantics by construction (the front-end knows which
  index is the luma-like channel).
- The Robidoux path keeps its own table math but consumes the same vector
  through a documented `d→q` inverse, ending `ApproxMozjpeg`'s double life.

### 9.2 One coefficient-opt config; delete the trellis/hybrid split

```rust
pub enum CoeffOpt {
    None,
    Trellis(TrellisOpt),
}
pub struct TrellisOpt {
    lambda1: f32,           // 14.75
    lambda2: f32,           // 16.5
    dc: bool,               // ONE default, not two
    delta_dc_weight: f32,   // 0.0
    speed: TrellisSpeed,    // output-neutral; speed only
    coupling: AqCoupling,   // scale=0.0 ⇒ exactly today's standalone mode
}
pub struct AqCoupling { scale: f32, exponent: f32, threshold: f32,
                        max_adjustment: f32, chroma_mul: f32, multiplicative: bool }
```

- `EncoderConfig.coeff_opt: CoeffOpt` is **one field; last set wins.** No
  cross-clearing, no encode-time priority that contradicts setter docs.
- `TrellisConfig`, `HybridConfig`, `TrellisMode`, both `enabled` flags, and
  the `create_hybrid_ctx` priority dance all delete. `coupling.scale == 0`
  *is* standalone — which is what the engine already does internally
  (`to_trellis_config`).
- Presets and `adaptive()` author `TrellisOpt` values instead of choosing
  *types*; "Hybrid" stops meaning two things because the word disappears
  from the type system.

### 9.3 `auto_optimize` → a preset, not a method

Its honest content is `coeff_opt = Trellis{λ₁ 14.5, dc off, coupling 0} +
Progressive`. Either delete it (Effort::Balanced is ~the same intent) or
re-express it as data: `OptimizationPreset::Tuned { lambda1: 14.5 }`. A
config *method* that conditionally rewrites two other fields based on a
quality gate, with an unconditional scan-mode side effect, is how it got
un-trackable in the first place.

### 9.4 `EncodePlan`: make resolution inspectable (the "never lose track again" fix)

```rust
impl EncoderConfig {
    /// Resolve every knob to its encode-time value. Pure; no encoding.
    pub fn resolve_plan(&self, width: u32, height: u32) -> EncodePlan;
}
/// Debug + serde: per-component distances, final quant tables (or family+params),
/// zero-bias params, CoeffOpt with effective λ-policy, AQ on/off + dampen,
/// scan script, huffman strategy, restart interval, tiny-file resolution,
/// SOF type, DQT precision.
```

One call answers "what did this builder chain actually configure" — for
humans, for tests (golden-plan snapshots), and for sweep logs (provenance
column = serialized plan instead of a codec-name string).

### 9.5 Canonical search vector for optimizers (coefficient, sweeps)

Stop exposing 784 raw dims. The identifiable core RD vector is ~8-D:

| dim | meaning | replaces |
|---|---|---|
| `d_y` | luma distance | quality scalar + global scale + table scaling |
| `d_chroma_ratio` (or `d_cb`,`d_cr`) | chroma/luma distance ratio | chroma_distance_scale, chroma_quality |
| `s_diff = λ₁−λ₂` | λ at low block energy | half the λ ridge |
| `s1 = λ₁` | λ rolloff at high energy | other half |
| `coupling.scale` | AQ→λ | hybrid family |
| `zb_mul_global` | zero-bias aggressiveness scalar | 192-dim zero_bias_mul (table *shape* trained offline) |
| `dc_trellis` (bool), `delta_dc_weight` | DC behavior | |

Categorical: table family (incl. piecewise-v4), scan mode, subsampling,
color mode. Table *shapes* (192-dim) get trained offline per family
(sa_piecewise_v4 anchors are exactly this) and enter the loop as the family
categorical — not as free parameters. Dead axes (`speed_mode`, `quality`
under Exact, `allow_16bit`) leave the surface entirely.

### 9.6 Kill / fix list (independent, mostly small)

1. **Wire or delete `sa_piecewise_v4`** — the best-validated tables in the
   tree currently do nothing. Natural home: a `QuantTableConfig::PiecewiseV4`
   variant + adaptive() photo cells.
2. **Fix `chroma_distance_scale` in XYB mode** (scales Y+B today) — decide
   per-channel semantics via §9.1, or reject the combination loudly.
3. **Rename one of the two `ExpertConfig`s** (`encoder_types::ExpertConfig`
   overlay vs `search::ExpertConfig` flat) and fix the `.expert()` doc link
   that points at the wrong one (encoder_config.rs:1235 links
   `super::search::ExpertConfig`; the signature takes
   `super::encoder_types::ExpertConfig`).
4. **`HybridConfig::default().enabled == true`** is a trap; defaults that
   self-arm get assigned by `..Default::default()` accidentally. (Moot if
   §9.2 lands.)
5. **Per-component zero-bias from per-component distance** (kill the
   `quant_vals_to_distance` inversion), or document why the global average
   is intentional.
6. **Make `chroma_quality` public or fold it** into the unified chroma
   policy; today it's doc-hidden and path-conditional.
7. **Delete the test-only CMA-ES constants** or move them to
   `benchmarks/` provenance docs; as `pub(crate) const` under `#[cfg(test)]`
   they read like a live feature.
8. **CLAUDE.md corrections** (pending user sign-off; see audit notes in the
   session log): the "CMA-ES auto_optimize()" section conflates the two; the
   trellis-ordering note describes the symptom but not the
   opposite-priority mechanism.

### 9.7 Suggested landing order

| Step | Change | Risk | Unlocks |
|---|---|---|---|
| 1 | `EncodePlan` introspection (additive) | none | auditability, golden tests for every later step |
| 2 | `ResolvedQuality` internal refactor (bit-identical at `cs=1.0`; lock with hashes) | low | per-channel chroma, XYB fix, zero-bias sanity |
| 3 | `CoeffOpt`/`TrellisOpt` merge (0.x breaking; no external users per project policy) | medium | deletes the priority knot, single λ author |
| 4 | auto_optimize → preset data; presets express `TrellisOpt` | low | one place to read tuning |
| 5 | Wire piecewise-v4 + canonical 8-D search vector for coefficient | medium | sane optimizer landscapes |

Steps 2–3 are where the "clumsy search space" actually dies: after them, an
optimizer sees `(d_y, d_chroma, λ-policy, coupling, family, scan,
subsampling)` — all live, all documented, one override rule.

---

## Appendix A. Constructor / entry-point map

| Entry | Sets |
|---|---|
| `EncoderConfig::ycbcr(q, sub)` (encoder_config.rs:548) | quality, YCbCr{sub}, all defaults |
| `::xyb(q, b_sub)` (:581) | XYB{b_sub}, `allow_16bit=false`, force_sof1 auto |
| `::grayscale(q)` (:610) | Grayscale |
| `::ycbcr_effort/xyb_effort/grayscale_effort(q, sub, effort)` (:626–646) | ctor + `optimization(effort.to_preset())` |
| `::adaptive(image, q)` / `adaptive_with` (adaptive.rs:375/381) | §7.1 |
| `.optimization(preset)` | §7.2 |
| `.auto_optimize(true)` | §5.3 |
| `.expert(encoder_types::ExpertConfig)` | tables/trellis/hybrid overlay |
| `search::ExpertConfig::{default_ycbcr, from_preset}` + `.to_encoder_config(color_mode)` | flat optimizer surface → config |

## Appendix B. Per-stage knob count (RD-relevant only)

| Stage | Public knobs | Hidden/internal | Dead/orphaned |
|---|---|---|---|
| quality targeting | 6 unit systems + ZqTarget(5 fields) | — | zq without `target-zq` feature |
| tables | 5 families × params + allow_16bit | chroma_quality (doc-hidden) | piecewise-v4, CMA-ES consts |
| chroma | 4 subsampling + 2 xyb + downsampling(3) + chroma_distance_scale | — | HybridConfig.chroma_scale (inert at coupling 0) |
| AQ | aq_enabled | custom_aq_map, AqController scaffold | — |
| coeff-opt | TrellisConfig(6) + HybridConfig(11) + boundary-rd(8) | both setters doc-hidden | speed_mode (output-dead) |
| scan/entropy | scan_mode(4), huffman(4), restart(2), tiny(3) | force_restart_markers | — |
| pre/post | pre_blur, deringing | — | — |
