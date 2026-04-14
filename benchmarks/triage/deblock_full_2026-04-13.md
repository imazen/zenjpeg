# Deblock strategy comparison — full gb82+cid22 corpus

Source CSV: `/mnt/tower/output/zenjpeg/deblock/results/deblock_full_2026-04-13_triage_716435d5.csv` (8.1 MB, 43,758 measurements).
Commit: 716435d5. Command: `cargo run --release -p zenjpeg --example deblock_harness --features decoder -- --measure --corpus gb82+cid22`

## Mean dSS2 across {turbo-420, mozjpeg-420, cjpegli}

| Strategy | Q5 | Q10 | Q20 | Q50 | Q75 | Q85 | Q95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive | +6.45 | +7.14 | +3.43 | +0.21 | +0.40 | +0.49 | +0.63 |
| bilateral_bnd | +6.46 | +7.17 | +3.49 | +0.92 | +0.71 | +0.77 | +0.68 |
| boundary_4tap | +6.44 | +7.14 | +3.44 | +0.88 | +0.68 | +0.75 | +0.67 |
| cdef_direction | +0.82 | +2.79 | +2.54 | +1.11 | +0.92 | +0.95 | +0.77 |
| coeff_refine_tv | -1.90 | -2.82 | -3.05 | -2.44 | -1.30 | -0.52 | +0.22 |
| coeff_smooth | -0.52 | -0.68 | -0.81 | -1.10 | -0.72 | -0.36 | -0.06 |
| dequant_bias | -0.84 | -0.93 | -0.60 | -0.34 | +0.21 | +0.52 | +0.61 |
| knusperli | +9.63 | +9.50 | +4.26 | -0.01 | -0.42 | -0.18 | +0.21 |
| pocs | +3.84 | +4.51 | +2.58 | +0.69 | +0.53 | +0.63 | +0.61 |
| quantsmooth_bilateral | -0.95 | -1.44 | -1.49 | -1.39 | -0.78 | -0.37 | +0.03 |
| sgr | +3.75 | +5.31 | +4.16 | +1.77 | +1.17 | +1.04 | +0.75 |
| triage | +2.21 | +2.39 | -0.25 | -4.78 | -7.85 | -10.03 | -14.10 |

## Ranking at Q5, per encoder (dSS2, higher=better)

### turbo-420

| Rank | Strategy | dSS2 |
|---:|---|---:|
| 1 | knusperli | +14.52 |
| 2 | bilateral_bnd | +9.34 |
| 3 | boundary_4tap | +9.34 |
| 4 | adaptive | +9.30 |
| 5 | pocs | +5.26 |
| 6 | sgr | +4.34 |
| 7 | triage | +3.84 |
| 8 | cdef_direction | +0.07 |
| 9 | coeff_smooth | -0.05 |
| 10 | dequant_bias | -0.39 |
| 11 | quantsmooth_bilateral | -0.60 |
| 12 | coeff_refine_tv | -1.05 |

### mozjpeg-420

| Rank | Strategy | dSS2 |
|---:|---|---:|
| 1 | knusperli | +8.29 |
| 2 | bilateral_bnd | +6.00 |
| 3 | adaptive | +6.00 |
| 4 | boundary_4tap | +6.00 |
| 5 | pocs | +3.39 |
| 6 | sgr | +2.79 |
| 7 | triage | +2.56 |
| 8 | cdef_direction | +0.02 |
| 9 | coeff_smooth | -0.20 |
| 10 | quantsmooth_bilateral | -0.39 |
| 11 | dequant_bias | -0.81 |
| 12 | coeff_refine_tv | -1.09 |

### cjpegli

| Rank | Strategy | dSS2 |
|---:|---|---:|
| 1 | knusperli | +6.08 |
| 2 | sgr | +4.12 |
| 3 | adaptive | +4.06 |
| 4 | bilateral_bnd | +4.03 |
| 5 | boundary_4tap | +3.99 |
| 6 | pocs | +2.87 |
| 7 | cdef_direction | +2.38 |
| 8 | triage | +0.24 |
| 9 | coeff_smooth | -1.31 |
| 10 | dequant_bias | -1.31 |
| 11 | quantsmooth_bilateral | -1.87 |
| 12 | coeff_refine_tv | -3.56 |
