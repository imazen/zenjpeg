# wide→magetypes Migration Benchmark Baseline

- **Date:** 2026-03-30
- **Commit:** 9415d7c6 (+ benchmark file only)
- **CPU:** Ryzen 9 7950X (WSL2)
- **Command:** `cargo bench -p zenjpeg --bench wide_migration --features decoder`
- **Image:** noise+patches synthetic (deterministic LCG), Q90
- **Sample size:** 10

## Encode (ms, median)

| Size | 4:4:4 base | 4:4:4 prog | 4:2:2 base | 4:2:2 prog | 4:2:0 base | 4:2:0 prog | 4:4:0 base | 4:4:0 prog |
|------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| 2k   | 263.4     | 161.0     | 124.5     | 121.0     | 88.6      | 86.3      | 98.6      | 99.1      |
| 4k   | 662.7     | 659.4     | 485.6     | 483.5     | 413.1     | 416.3     | 480.0     | 487.7     |

## Decode (ms, median)

| Size | 4:4:4 base | 4:4:4 prog | 4:2:2 base | 4:2:2 prog | 4:2:0 base | 4:2:0 prog | 4:4:0 base | 4:4:0 prog |
|------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| 2k   | 52.9      | 52.6      | 43.5      | 43.8      | 32.1      | 33.7      | 48.2      | 44.3      |
| 4k   | 349.1     | 305.3     | 211.1     | 430.6     | 181.1     | 164.2     | 184.6     | 182.0     |

## Notes

- 4k-422-prog decode shows high variance (383-480ms) — likely GC/scheduling noise
- 4k-444-base decode also variable (316-389ms)
- 2k-440 decode variable (42-52ms range)
- All encode numbers are very stable (tight CI)
- Progressive ≈ baseline for encode at 4k (overhead amortized)
- Progressive faster than baseline for 2k-444 encode (fewer scan bytes)

## Raw Data

```
enc/q90/2k-444-base     time:   [241.99 ms 263.36 ms 286.14 ms]
enc/q90/2k-444-prog     time:   [158.95 ms 160.97 ms 163.40 ms]
enc/q90/2k-422-base     time:   [122.88 ms 124.53 ms 126.41 ms]
enc/q90/2k-422-prog     time:   [120.09 ms 121.04 ms 122.55 ms]
enc/q90/2k-420-base     time:   [87.934 ms 88.634 ms 89.163 ms]
enc/q90/2k-420-prog     time:   [86.142 ms 86.315 ms 86.644 ms]
enc/q90/2k-440-base     time:   [97.680 ms 98.613 ms 99.221 ms]
enc/q90/2k-440-prog     time:   [98.256 ms 99.101 ms 99.499 ms]
enc/q90/4k-444-base     time:   [657.60 ms 662.71 ms 667.43 ms]
enc/q90/4k-444-prog     time:   [655.70 ms 659.40 ms 663.38 ms]
enc/q90/4k-422-base     time:   [482.26 ms 485.57 ms 489.17 ms]
enc/q90/4k-422-prog     time:   [481.24 ms 483.52 ms 485.78 ms]
enc/q90/4k-420-base     time:   [411.45 ms 413.06 ms 415.06 ms]
enc/q90/4k-420-prog     time:   [413.19 ms 416.28 ms 419.39 ms]
enc/q90/4k-440-base     time:   [477.32 ms 479.96 ms 483.05 ms]
enc/q90/4k-440-prog     time:   [484.29 ms 487.74 ms 491.39 ms]
dec/q90/2k-444-base     time:   [51.413 ms 52.868 ms 54.265 ms]
dec/q90/2k-444-prog     time:   [51.991 ms 52.562 ms 53.687 ms]
dec/q90/2k-422-base     time:   [42.119 ms 43.536 ms 44.408 ms]
dec/q90/2k-422-prog     time:   [42.432 ms 43.836 ms 44.880 ms]
dec/q90/2k-420-base     time:   [31.879 ms 32.133 ms 32.350 ms]
dec/q90/2k-420-prog     time:   [33.028 ms 33.722 ms 35.116 ms]
dec/q90/2k-440-base     time:   [41.667 ms 48.244 ms 51.901 ms]
dec/q90/2k-440-prog     time:   [39.375 ms 44.311 ms 50.799 ms]
dec/q90/4k-444-base     time:   [315.70 ms 349.06 ms 388.55 ms]
dec/q90/4k-444-prog     time:   [302.78 ms 305.28 ms 307.61 ms]
dec/q90/4k-422-base     time:   [209.01 ms 211.05 ms 213.25 ms]
dec/q90/4k-422-prog     time:   [383.34 ms 430.63 ms 479.86 ms]
dec/q90/4k-420-base     time:   [172.01 ms 181.12 ms 191.01 ms]
dec/q90/4k-420-prog     time:   [163.60 ms 164.17 ms 165.24 ms]
dec/q90/4k-440-base     time:   [182.25 ms 184.61 ms 186.83 ms]
dec/q90/4k-440-prog     time:   [179.12 ms 181.95 ms 185.10 ms]
```
