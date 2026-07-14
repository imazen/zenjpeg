# Review-sweep A/B benchmark: ebcb9882 (base) vs 8764ed65 (final)

**Question:** did the maintenance review sweep (R1–R8, commits ce31248d..8764ed65)
change encode or decode performance?

**Verdict: performance-neutral.** Three independent measurements agree; the
strongest (callgrind instruction counts, deterministic and load-immune) shows
the decode and encode paths execute instruction-identical work:

| path | base Ir | final Ir | delta |
|---|---|---|---|
| decode baseline 4:2:0 Q85, 2048² | 899,624,600 | 899,629,422 | **+0.0005%** |
| decode progressive Q85, 2048² | 1,553,627,700 | 1,553,791,776 | **+0.011%** |
| encode YCbCr Q85 4:2:0, 2048², 50 iters | 61,430,868,316 | 61,430,853,484 | **−0.00002%** |

## Setup

- **base** = ebcb9882 — after R1/R2/R3 (docs, warning cleanup, silent-skip
  tests), before R4–R8 (strictness predicates + padding-helper dedup, buffer
  invariant asserts, scanline/byte-encoder dedup, codec.rs split, sys-crate
  graceful build).
- **final** = 8764ed65 — full sweep applied.
- The #186 XYB padding fix (0064e34a) predates base, so it sits on both sides
  and is not part of this delta.
- Benches: `cargo bench -p zenjpeg --bench decode_zenbench --features zencodec`
  and `--bench encode_zenbench` (zenbench interleaved harness; CID22-512
  training corpus via `CODEC_CORPUS_CACHE`, 209 images; larger sizes
  generated deterministically).
- Callgrind: `valgrind --tool=callgrind` on `--example valgrind_decode`
  (jpegli 2048 [progressive]) and `--example profile_encode`, identical
  committed harness files in both trees, release builds, local Ryzen 9 7950X
  (instruction counts are load-immune).

## Primary wall-clock: zen-bench-x86 (Hetzner CCX33, 8 dedicated x86_64 cores)

Dedicated idle box (recreated after the original box was torn down mid-run by
fleet GC). Both trees built offline from the same vendored dep set; the base
tree carried a build-only backport of the R8 graceful sys-crate build
(dev-dependency FFI machinery only — no library code).

### decode_zenbench — zen/mozjpeg and zen/zune throughput-ratio deltas

| group | lane | Δ(zen/moz) | Δ(zen/zune) | raw ms base→final |
|---|---|---|---|---|
| baseline_4:2:0_Q85 | default (Jpegli IDCT) | −1.0% | −5.8% | 9.10→9.30 |
| baseline_4:2:0_Q85 | LibjpegCompat | +2.4% | −2.6% | 9.30→9.10 |
| baseline_4:2:0_Q85 | LibjpegCompat + Jpegli ups. | +3.4% | −1.6% | 9.00→8.80 |
| baseline_4:2:0_Q85 | Triangle + Libjpeg IDCT | +5.4% | +0.2% | 9.20→8.80 |
| baseline_4:2:0_Q85 | NearestNeighbor | +0.0% | −4.9% | 7.50→7.60 |
| progressive_4:2:0_Q85 | default | −4.8% | −4.6% | **15.50→15.50** |
| deblock_4:2:0_Q20 | Off / 4Tap / Knusperli / Auto | (no ref lane) | | ±3% mixed |
| scanline_4:2:0_Q85 | Off / Boundary | (no ref lane) | | −2% / +5.6% |
| dequant_bias_Q85 | default / bias | (no ref lane) | | +12% / +1.9% |
| sink_4:2:0_Q85 | native (Vec out) | −11.2% | (no zune) | 8.60→9.10 |

**Noise floor, demonstrated on this data:** the group-local reference lanes
are themselves unstable at these round counts — mozjpeg (an *unchanged binary
from identical vendored source on an idle box*) moved 8.60→8.10 ms (−6%)
between the two runs in the sink group, and moz vs zune references disagree by
~5 pp systematically in the baseline group. Every flagged lane fails at least
one corroboration test: progressive's raw time is identical (the −4.8% is
entirely the moz lane's own move); the dequant_bias "default" lane (+12% raw,
4-round group) runs the *same code path* as baseline "default", which measures
+2% at higher round counts; sink combines a +6% zen swing with a −6% moz swing
in a 4-round group. No lane exceeds the harness's demonstrated reference noise
at these round counts — and callgrind independently pins the flagged paths at
≤0.011% instruction delta.

### encode_zenbench — raw mean-time deltas (no reference lanes in this bench)

| group | lane | base ms | final ms | delta | max ±MAD |
|---|---|---|---|---|---|
| encode_q85_4k | 4:2:0 progressive | 301.30 | 303.00 | +0.6% | 1.1% |
| encode_q85_4k | 4:2:0 sharp progressive | 311.30 | 314.00 | +0.9% | 1.5% |
| encode_q85_4k | xyb Full progressive | 468.10 | 468.60 | +0.1% | 1.1% |
| encode_q85_4k | xyb Full baseline | 245.20 | 242.70 | −1.0% | 0.4% |
| encode_q85_1k_xyb | ycbcr 4:2:0 progressive | 33.20 | 34.00 | +2.4% | 2.1% |
| encode_q85_1k_xyb | ycbcr 4:2:0 baseline +Fix | 17.40 | 16.90 | −2.9% | 3.4% |
| encode_q85_1k_xyb | ycbcr 4:2:0 baseline +Opt | 15.90 | 15.90 | +0.0% | 4.4% |
| encode_q85_1k_xyb | xyb BQuarter progressive | 43.70 | 43.70 | +0.0% | 4.3% |
| encode_q85_1k_xyb | xyb Full progressive | 52.00 | 53.20 | +2.3% | 4.1% |
| encode_q85_1k_xyb | xyb BQuarter baseline | 22.80 | 22.80 | +0.0% | 5.3% |
| encode_q85_1k_xyb | xyb Full baseline | 25.90 | 26.10 | +0.8% | 2.3% |

Zero lanes flagged at |Δ| ≥ max(3%, 2·MAD). The µs-scale
`sharp_yuv_isolated/1024` lanes swing ±5–18% with 12–49% MADs (tiny
workloads, 4 rounds); sharp-YUV code is untouched by the sweep.

## Corroborating wall-clock: local WSL2 (Ryzen 9 7950X, contended)

Local runs were contaminated by co-tenant load, and the mozjpeg reference lane
itself moved +28% raw between runs (72.2→92.3 MiB/s) while zune stayed flat
(75.5→76.5) — moz-referenced ratios from the local pair are unusable. Against
the stable zune reference, every decode lane with a zune lane in its group is
within ±3% (baseline lanes +3.0/+2.4/−0.9/+0.2/−0.7%, progressive −0.1%).

## Notes

- Runtime-plausible sweep changes, all confirmed neutral: R4 strictness
  predicate helpers (decoder, `#[inline]`), R3/R4 `edge_replicate_h_padding`
  dedup (decoder), R5 `debug_assert` buffer invariants (compiled out in
  release), R6 scanline entry + byte-encoder bridge dedup (encoder setup),
  R6 `PlaneStripMut` padding abstraction (encoder), R7 codec.rs file split
  (zero runtime), R8 sys-crate graceful build (build-time only).
- Methodology lesson recorded: for sub-5% questions, group-local reference
  ratios at 4-round budgets have a ±6% empirical noise floor even on a
  dedicated idle box — settle them with callgrind instruction counts (or
  raise the zenbench group time budget), not with more wall-clock reruns.
