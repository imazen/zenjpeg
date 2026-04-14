# Triage quality-gated vs ungated — full gb82+cid22 corpus

Source CSV: `/mnt/tower/output/zenjpeg/deblock/results/deblock_triage_gated_2026-04-13_54ed3d6b.csv` (1.9 MB, 10,098 measurements).
Commit: 54ed3d6b. Command: `cargo run --release -p zenjpeg --example deblock_harness --features decoder -- --measure --corpus gb82+cid22 --strategy triage,triage_gated`

## Gate design

From the cid22 threshold sweep (`triage_t16/t32/t48/t64/t128/t192/t256/t384/t512`):

| Condition (luma DC quant) | Action |
|---|---|
| `< 50` (≈ Q > 15) | skip, return baseline decode |
| `50–119` (≈ Q 10–15) | triage with `uniform_threshold = 64` |
| `>= 120` (≈ Q ≤ 5) | triage with `uniform_threshold = 128` |

## Results

Mean dSS2 across `{turbo-420, mozjpeg-420, cjpegli}`:

| Strategy | Q5 | Q10 | Q15 | Q20 | Q30 | Q50 | Q75 | Q85 | Q95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| triage | +2.21 | +2.39 | +0.76 | −0.25 | −2.26 | −4.78 | −7.85 | −10.03 | −14.10 |
| triage_gated | +2.34 | +2.74 | +1.53 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |

**No regression anywhere**: at Q20+ the gate falls back to baseline and output is bit-identical (max `|dSS2|` = 0.000).
**Small boost at the low-Q end**: +0.13 to +0.77 over raw triage at Q5–Q15, from the t128 threshold at Q≤5 and image-level selectivity.

## Competitive position (full-corpus mean dSS2)

| Strategy | Q5 | Q10 | Q20 | Q50 |
|---|---:|---:|---:|---:|
| knusperli | **+9.63** | **+9.50** | +4.26 | −0.01 |
| sgr | +3.75 | +5.31 | +4.16 | **+1.77** |
| boundary_4tap | +6.44 | +7.14 | +3.44 | +0.88 |
| triage_gated | +2.34 | +2.74 | — | +0.00 |
| triage | +2.21 | +2.39 | −0.25 | −4.78 |

Triage-gated is a safe "never regress" pixel-domain strategy; knusperli
and sgr stay well ahead at every quality level where triage is positive.

## Takeaway

- The gate eliminates the Q50+ cliff seen in raw triage (−14 SS2 → 0).
- Threshold tuning alone is marginal (≤ 0.08 SS2 uplift); the gate is
  what makes the strategy safe.
- Triage's value proposition is as a pixel-domain-only option (no DCT
  coefficient access required), not as a frontier quality strategy.
