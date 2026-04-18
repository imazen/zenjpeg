# Post-audit re-run — container scan + probe (2026-04-20)

Re-measurement after the deep audit + fixes on 2026-04-19/20. Confirms
no regression from the audit fixes:

- `Wants` inner field privatized (struct-literal construction disabled)
- `#[non_exhaustive]` added to `Wants`, `OverflowFlags`, `ProbeSof`,
  `ContainerItem`, `MpfEntry`
- `Wants::ALL` derived from named constants rather than hardcoded
- Multi-XMP-APP1 fingerprint correction
- Missing regression tests added: `mp_entry_offset == 0`,
  `is_oversized_for_u32`, oversize-`create_mpf_header`
- Roundtrip tolerances tightened (0.01 → 1e-5) on ISO / XMP tests

CSV: [`container_scan_2026-04-20.csv`](container_scan_2026-04-20.csv).

## `find_jpeg_boundaries`

| Input | Baseline (ultrahdr-core) | New (zenjpeg::container) | Speedup |
|---|---:|---:|---:|
| synth_256 (8 KB) | 2.69 GiB/s | **35.81 GiB/s** | **13.3×** |
| synth_1024 (120 KB) | 2.90 GiB/s | **38.14 GiB/s** | **13.1×** |
| pixel_ultrahdr (2.85 MB) | 2.90 GiB/s | **14.87 GiB/s** | **5.1×** |

## `primary_bounds`

| Input | Baseline (naive segment walk) | New (zenjpeg::container) | Speedup |
|---|---:|---:|---:|
| synth_256 | 3.92 GiB/s | **36.41 GiB/s** | **9.3×** |
| synth_1024 | 4.27 GiB/s | **41.13 GiB/s** | **9.6×** |
| pixel_ultrahdr | 4.39 GiB/s | **17.25 GiB/s** | **3.9×** |

## `probe_workflow` (sequential 4-walks vs single_probe_all vs is_ultrahdr)

| Input | sequential | single_probe | Speedup | is_ultrahdr |
|---|---:|---:|---:|---:|
| synth_256 | 9.58 GiB/s | **19.39 GiB/s** | **2.02×** | 1,079 GiB/s |
| synth_1024 | 10.64 GiB/s | **21.67 GiB/s** | **2.04×** | 2,427 GiB/s |
| pixel_ultrahdr | 7.16 GiB/s | **11.30 GiB/s** | **1.58×** | **20,441 GiB/s** |

## Significance

| Row | Wilcoxon p |
|---|---:|
| all find_jpeg_boundaries new-impl | 0.000000 |
| primary_bounds/synth_256 | 0.000000 |
| primary_bounds/synth_1024 | 0.000003 |
| primary_bounds/pixel_ultrahdr | 0.000004 |
| probe_workflow/synth_256 (single_probe_all) | 0.000000 |
| probe_workflow/synth_1024 (single_probe_all) | 0.000009 |
| probe_workflow/pixel_ultrahdr (single_probe_all) | 0.000000 |
| is_ultrahdr rows | 0.000000 – 0.000006 |

All new-impl rows are statistically significant at 95% CI.

## Zero-allocation

Not empirically verified in this run — the `__alloc-instrument`
feature was not enabled. By code inspection, `ContainerProbe` has
only fixed-size inline arrays (no `Vec`) and `probe()` calls no
allocating APIs. Verify with `cargo bench --features
__alloc-instrument` if/when allocator instrumentation is wanted
empirically.
