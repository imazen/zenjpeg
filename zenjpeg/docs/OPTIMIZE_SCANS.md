# Progressive Scan Optimization (`optimize_scans`)

Lossless optimization that searches for the progressive scan script producing the
smallest file. Decoded pixels are bit-identical; only the scan structure changes.

## Usage

```rust
let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
    .optimize_scans(true);  // auto-enables progressive + optimize_huffman
```

## Background: Progressive JPEG Scan Scripts

A progressive JPEG encodes DCT coefficients across multiple scans. Each scan
specifies:
- **Ss, Se**: spectral selection (which AC frequencies, 0=DC, 1-63=AC)
- **Ah, Al**: successive approximation (bit planes — Al=2 means send bits 7..2
  first, then refine with bits 1 and 0 in later scans)
- **Components**: which color channels (Y, Cb, Cr)

The scan script is the ordered list of these scans. Different scripts produce
different file sizes because:
1. Huffman coding efficiency depends on symbol distributions within each scan
2. Successive approximation can reduce entropy by separating significant/refinement bits
3. More scans = more overhead (DHT markers, SOS headers)
4. Fewer scans = less opportunity for targeted Huffman tables

The optimal script depends on the image content — no single fixed script is best
for all images.

## How Other Encoders Handle This

### jpegli (Google)

**No automatic optimization.** Uses fixed hard-coded progressive scripts at 3
compression levels in `SetDefaultScanScript()` (`encode.cc:114-159`):

- Level 0: Single scan (0-63, full spectrum) — effectively baseline
- Level 1: DC separate, then AC with 1 refinement pass
- Level 2: DC separate, low frequencies (1-2), mid frequencies (3-63) with 2-3
  refinement passes

Users can provide custom scan scripts via `cinfo->scan_info`, but there's no
search or trial encoding.

### mozjpeg

**Extensive trial-encode optimization.** When `optimize_scans` is enabled:

1. **Generate 64 candidate scans** (YCbCr; 23 for grayscale):
   - 1 DC scan
   - 23 luma scans: varying Al levels (0-3) and frequency splits
   - 40 chroma scans: varying Al levels (0-2), interleaved vs separate DC
   - Frequency split points: `{2, 8, 5, 12, 18}`

2. **Trial-encode each scan** — full entropy coding to temporary buffers,
   storing `scan_size[i]` for each

3. **Iteratively select best configuration** (`select_scans()` in `jcmaster.c:774-962`):
   - Test Al levels — every 3rd scan (at Al boundaries), compare cumulative size
   - Select highest Al that doesn't increase file size
   - Test frequency split points
   - Early termination: if first 3 candidates show no improvement, stop

4. **Concatenate best scans** using `copy_buffer()` from pre-encoded data

Cost: ~64 trial encodes per image. Savings: 1-3%.

## zenjpeg Algorithm

### Overview

zenjpeg uses a hybrid approach: Huffman frequency estimation for pre-filtering,
then trial-to-buffer encoding for final selection.

### Pipeline

```
Image DCT blocks
    |
    v
Phase 1: Mixed-SA search (15 variants)
    - 5 split points x 3 Al levels
    - Rank by frequency estimate (within-category, accurate)
    - Keep best 1
    |
Phase 2: mozjpeg-style 64-candidate search
    - Generate 64 trial scans (same as mozjpeg)
    - Estimate all scan sizes via FrequencyCounter
    - ScanSelector picks best Al + frequency split
    - Build optimized script
    |
Phase 3: Assemble 3 candidates
    - Default jpegli script (safety baseline)
    - Optimizer's pick from Phase 2
    - Best mixed-SA variant from Phase 1
    - Deduplicate identical scripts
    |
Phase 4: Trial-to-buffer
    - Full encode each candidate (tokenize + Huffman optimize + serialize)
    - Keep the smallest
    - Guarantees zero regressions vs default
```

### Candidate Types

**Default jpegli script** — the fixed script from jpegli level 2:
- Separate DC scans per component
- AC 1-2 at al=0 (full precision)
- AC 3-63 at al=2 (successive approximation)
- AC 3-63 refinement ah=2->al=1
- AC 3-63 refinement ah=1->al=0
- 15 scans for 3-component color

**Optimizer script** — uniform Al across all AC frequencies:
- Tests Al levels 0-3 for luma, 0-2 for chroma
- Tests frequency splits {2, 8, 5, 12, 18}
- Picks the combination minimizing estimated total size
- Typically 6-15 scans depending on chosen Al

**Mixed-SA script** — different Al for low vs high frequencies:
- AC 1-split at al=0 (low frequencies, full precision)
- AC (split+1)-63 at chosen al with refinement passes
- 15 variants: 5 split points x 3 Al levels
- Best picked by frequency estimate

### Files

| File | Purpose |
|------|---------|
| `encode/scan_optimize/mod.rs` | `generate_candidate_scripts()` — main entry point |
| `encode/scan_optimize/config.rs` | `ScanSearchConfig` — search parameters |
| `encode/scan_optimize/generate.rs` | 64 trial scan generation (mozjpeg-compatible) |
| `encode/scan_optimize/select.rs` | `ScanSelector` — Al/frequency-split selection |
| `encode/scan_optimize/estimate.rs` | `estimate_script_cost()` — Huffman frequency estimation |
| `encode/progressive.rs` | Trial-to-buffer encoding loop |

## Frequency Estimator

### How It Works

`estimate_script_cost()` in `estimate.rs` evaluates a scan script without
encoding:

1. For each scan in the script, collect per-scan Huffman histograms from the
   block data
2. Cluster DC and AC histograms separately (matching encoder behavior)
3. Build optimal Huffman code lengths from clustered histograms
4. Sum `frequency[symbol] * code_length[symbol]` for all symbols
5. Add non-Huffman bits (value extra bits, refbits, EOB run extra bits) per scan
6. Add SOS header overhead (84 bits) per scan
7. Return total estimated bits

Uses `FrequencyCounter::estimate_encoding_cost()` which generates optimal code
lengths via the package-merge algorithm, then computes the weighted sum.
`cluster_histograms()` merges scans with similar symbol distributions into
shared tables, matching how the actual encoder shares DHT tables.

### Accuracy

The estimator is accurate across all scan counts after the signed-shift fix
(2026-02-01). Per-scan comparison on first CID22 image at Q75:

| Scan type | Est/actual ratio | Notes |
|-----------|-----------------|-------|
| DC scans | 1.007 | Nearly exact |
| AC first al=0 | 1.006 | Nearly exact |
| AC first al=2 (Y) | 1.005 | Fixed from 2.9x overestimate |
| AC refinement | 0.998-1.002 | Nearly exact |
| AC first al=2 (chroma) | 1.2-2.5x | Tiny scans (<100 bits), negligible |

Full 15-scan script total: est/actual = **1.007** (was 1.22 before fix).

### Historical: Signed-Shift Bug (fixed 2026-02-01)

The original estimator had a critical bug in AC first-pass scans at `al > 0`:

```rust
// BUG: signed right shift fills with 1-bits for negative numbers
let coeff = block[k] >> al;        // (-1i16) >> 2 = -1 (non-zero!)
let abs_coeff = coeff.unsigned_abs(); // abs(-1) = 1 → counted as non-zero

// FIX: unsigned_abs() BEFORE shift, matching the actual encoder
let abs_coeff = block[k].unsigned_abs() >> al; // abs(-1) >> 2 = 0 (correct!)
```

For negative coefficients with `|value| < 2^al`, signed shift produced -1
(non-zero) while the actual encoder correctly got 0. This caused massive
overcounting of non-zero coefficients — especially for chroma where nearly all
coefficients are small. The fix matched the actual encoder's approach at
`entropy/encoder.rs:420` which uses `coeffs[k].unsigned_abs() >> al`.

Impact: 15-scan est/actual ratio dropped from **1.22** to **1.007**. CID22
savings improved from 0.62% to 0.78% at Q75 because the estimator can now
correctly rank scans at different al levels.

### Clustering Mechanics (from the actual encoder)

The clustering algorithm in `cluster_histograms()`:

- Each AC scan gets a context (one per component per scan)
- For each context's histogram, calculate the cost delta of merging with each
  existing cluster: `delta = combined_cost - cluster_cost`
- If `delta < cost_of_new_table`, merge; otherwise create new cluster
- JPEG allows max 4 AC table slots (0-3), but the encoder can define >4 clusters
  by re-emitting DHT markers between scans (slot cycling)
- DC tables are emitted upfront; AC tables are emitted on-demand before each scan
  that needs a new one

### Remaining Improvement Path

1. **Reduce trial-to-buffer** — now that the estimator is accurate cross-structure,
   we may only need 1-2 trial encodes instead of 3, cutting encode overhead.

2. **Model chroma scan overhead better** — tiny chroma scans at high al still
   overestimate by 1.2-2.5x, but these contribute <0.1% of total size so the
   practical impact is negligible.

## Benchmark Results

### CID22 Corpus (207 images, 512x512, 3 candidates)

| Quality | Total Normal | Total Optimized | Savings | Benefited |
|---------|-------------|-----------------|---------|-----------|
| Q75 | 6,881,436 | 6,827,680 | **+0.78%** | 193/207 (93%) |
| Q85 | 9,065,094 | 9,022,702 | **+0.47%** | 162/207 (78%) |
| Q90 | 11,454,820 | 11,414,433 | **+0.35%** | 155/207 (75%) |

Per-image range: 0.00% to 4.75%. Zero regressions (trial-to-buffer always picks
the default when the optimizer finds nothing better).

### Known Limitations

- **Interleaved DC scans not supported** — requires MCU-aware block iteration for
  subsampled images. Separate DC scans work fine.
- **DC successive approximation not implemented** — negligible savings for DC
  coefficients.
- **XYB mode not supported** — returns default fixed script (XYB has different
  coefficient distributions that need separate tuning).
- **Trial-to-buffer cost** — currently 3x progressive encode time (3 candidates).
  Estimator is now accurate enough to potentially reduce to 2x or 1x.
- **Synthetic/chart images** — can see up to 4-5% savings. Real photos typically
  0-1%.
