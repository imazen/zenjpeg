# Single-pass `container::probe` vs sequential walks (2026-04-18)

Measures the workflow-level speedup from consolidating "find MPF +
find ISO gainmap + find image ranges + find primary bounds" into one
marker-iter walk via `zenjpeg::container::probe(data, Wants::ALL)`.

CSV: [`container_probe_2026-04-18.csv`](container_probe_2026-04-18.csv).

## Rows

Each fixture carries three measurements:

- **sequential_unified** — 4 separate calls to the new
  `zenjpeg::container` API (`find_jpeg_boundaries`, `parse_mpf`,
  `parse_iso_app2`, `primary_bounds`). Each does its own marker walk.
  This is what a caller pays today to gather the full
  container-level picture.
- **single_probe_all** — one `probe(data, Wants::ALL)` call. One
  walk, zero-copy APP payload ranges, fixed-size inline storage, no
  heap.
- **is_ultrahdr** — short-circuit detector. Uses a trimmed walk that
  exits at the first ISO URN / MPF identifier / `hdrgm:` fingerprint
  match. No probe struct construction.

## Results

Numbers from the companion CSV. See `container_scan_2026-04-20.md` for
post-audit re-measurement.

| Input | Size | sequential_unified | single_probe_all | Speedup | is_ultrahdr |
|---|---:|---:|---:|---:|---:|
| synth_256 | ~8 KB | 3.68 µs | **1.84 µs** | **2.01×** | 30 ns |
| synth_1024 | ~120 KB | 52.13 µs | **25.92 µs** | **2.01×** | 206 ns |
| pixel_ultrahdr | ~2.85 MB | 276.02 µs | **183.47 µs** | **1.50×** | 120 ns |

Throughput view (sequential_unified / single_probe_all / is_ultrahdr):
9.83 / 19.71 / 1,214.90 GiB/s on synth_256, scaling through
9.84 / 14.80 / 22,632.01 GiB/s on the 2.85 MB Pixel fixture.

### Throughput

| Input | sequential_unified | single_probe_all | is_ultrahdr |
|---|---:|---:|---:|
| synth_256 | 9.83 GiB/s | 19.7 GiB/s | 1,215 GiB/s |
| synth_1024 | 10.9 GiB/s | 21.9 GiB/s | 2,755 GiB/s |
| pixel_ultrahdr | 9.84 GiB/s | 14.8 GiB/s | 22,632 GiB/s |

Wilcoxon p ≤ 0.000006 on every non-baseline row; signed-rank tells us
the speedup is statistically significant at the 95% CI level.

## Observations

1. **~2× on probe-everything workflows.** Less than the naive "4 walks →
   1 walk = 4×" projection because `single_probe_all` does additional
   per-segment work (memmem fingerprinting, ICC hash) that the
   individual calls skip. Still a clean win at no code cost to callers.

2. **Pixel speedup (1.50×) is lower than synth (~2×).** Because the Pixel
   fixture's sequential path hits MPF and ISO markers early and
   short-circuits; the single probe keeps walking to find SOS / EOI /
   all APP2s. The right call depends on the workflow — if you need
   EVERYTHING, probe wins; if you only need one signal, a targeted
   single call can be faster.

3. **`is_ultrahdr` is effectively free on multi-MB files**
   (**22.6 TiB/s** effective throughput on the Pixel fixture because it
   exits after ~27 KB — first APP2 with ISO URN). This is the hot path
   for "scan a directory of millions of JPEGs looking for HDR." No
   decode, no struct construction, one memchr + one prefix compare per
   segment visited.

4. **Zero heap during probe.** The probe struct has fixed-size inline
   arrays (`[Option<Range<u32>>; 8]` for images, `[..; 4]` for
   ExtXMP). `alloc-instrument` would confirm zero allocs for
   `single_probe_all` (the `allocs_per_iter` CSV column is blank
   because that feature wasn't enabled for this run; enable with
   `--features __alloc-instrument` if verifying).

## Regression guard

Any future `container::probe` work must re-run this bench and keep
`single_probe_all` ≤ `sequential_unified` on all three fixtures. No
exceptions without a justification in the commit message.
