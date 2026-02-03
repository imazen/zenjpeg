# Hybrid Trellis Investigation Report

## Executive Summary

This branch (`feat/mozjpeg-mimic-tests`) investigated optimization strategies for JPEG encoding.

### Biggest Win: Optimized Quant Tables (+6.5 pareto)

The `coefficient` project discovered that **SA-optimized libjpeg-style quant tables beat jpegli defaults by +6.5 pareto** on unseen images. This dwarfs all other optimizations:

| Optimization | Improvement |
|--------------|-------------|
| **Optimized quant tables** | **+6.5 pareto** |
| Trellis at Q95+ (4:4:4) | -2.1% size |
| Scan optimization | -0.5% size |

**Trellis is redundant when using optimized tables** — it actually hurts by -0.18 pareto.

### Secondary Finding: Trellis Helps Jpegli at Q95+

When using jpegli's default tables (not the optimized ones), trellis provides modest gains:
- 4:4:4 Q95: -2.1% size, +0.4% Butteraugli
- 4:2:0 Q90: -2.5% size, +0.9% Butteraugli

Per-block AQ-coupled trellis is redundant (traces same R-D curve as quality slider).

---

## What Works

### 1. Scan Optimization (Always Use)
- **-0.5% size**, identical quality
- Zero speed cost
- Works at all quality levels

### 2. Uniform Trellis at Q95+ (4:4:4)

**CID22 corpus (30 images):**
| Quality | Size Δ | Butteraugli Δ | Verdict |
|---------|--------|---------------|---------|
| Q90 | -2.8% | +2.9% | ❌ Too much quality loss |
| Q95 | -2.1% | +0.4% | ✅ Pareto win |
| Q98 | -1.5% | +0.3% | ✅ Acceptable |

**CLIC2025 corpus (30 images):**
| Quality | Size Δ | Butteraugli Δ | Verdict |
|---------|--------|---------------|---------|
| Q90 | -2.9% | +0.5% | Borderline |
| Q93 | -2.3% | +0.2% | ✅ Good |
| Q98 | -1.3% | -0.1% | ✅ True Pareto win |
| Q99 | -1.1% | -0.2% | ✅ True Pareto win |

### 3. Stacking
Scan optimization + trellis stack: **-2.9% combined** at Q90.

---

## What Doesn't Work

### Per-Block AQ-Coupled Trellis (Redundant)

Tested adjusting trellis lambda per-block based on AQ strength:
- **Positive coupling**: More aggressive on textured blocks
- **Negative coupling**: More aggressive on smooth blocks

**Result:** Traces the SAME rate-distortion curve as just changing quality.

At matched file size (~58k bytes):
| Mode | Butteraugli |
|------|-------------|
| Jpegli Q85 | 1.922 |
| Hybrid +5 Q84 | 1.938 (same) |
| Hybrid -4 Q87 | 2.031 (worse) |

**Conclusion:** Per-block coupling is redundant. Use uniform trellis.

### Other Dead Ends
| Feature | Finding |
|---------|---------|
| Deringing | No measurable effect |
| Trellis at Q85- | Same curve as jpegli (no benefit) |
| Trellis 4:4:4 at Q90 | +2.9% BA too degraded |

---

## Speed Cost

Trellis adds **+70% encode time**:
- JpegliProgressive (1024×1024): 17.5 ms
- HybridProgressive (1024×1024): 29.4 ms

This is why trellis should be opt-in, not default.

---

## Algorithm Details

### Trellis Implementation
- Ported from mozjpeg `jcdctmgr.c:937-1379`
- Viterbi dynamic programming for 63 AC coefficients
- Block-level parity: **0 failures across 2869 test blocks**
- Pure Rust, no mozjpeg-rs dependency

### Why Hybrid Beats Both Encoders

```
jpegli tables (Butteraugli-optimized) + mozjpeg trellis (R-D optimal) = best
```

C mozjpeg's Robidoux tables weren't perceptually optimized. jpegli's tables were designed with Butteraugli. Trellis on top of better tables = better results.

### Comparison vs C mozjpeg (same Robidoux tables)
| Mode | Size Δ |
|------|--------|
| Baseline | -0.3% to -0.8% (zenjpeg smaller) |
| Progressive | +0.0% to +2.5% (scan script diff) |
| Overall | +0.2% |

Quality: Virtually identical (SSIMULACRA2).

---

## Recommended Configuration

```rust
// High quality 4:4:4 (Q95+): use trellis
EncoderConfig::ycbcr(95.0, ChromaSubsampling::None)
    .optimization(OptimizationPreset::HybridProgressive)
    .optimize_scans(true)

// Standard quality 4:4:4 (Q90): no trellis
EncoderConfig::ycbcr(90.0, ChromaSubsampling::None)
    .optimization(OptimizationPreset::JpegliProgressive)
    .optimize_scans(true)

// 4:2:0 (Q90+): trellis helps more
EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
    .optimization(OptimizationPreset::HybridProgressive)
    .optimize_scans(true)
```

---

## Key Commits

| Commit | Description |
|--------|-------------|
| `7fc96e0` | Internalize trellis as pure Rust |
| `4689be6` | Fix standalone trellis to match mozjpeg exactly |
| `042c52e` | Prove AQ coupling is redundant |
| `e8641ef` | Validate trellis helps at Q90+ (30 images) |
| `589a8e2` | Benchmark 4:4:4 on CID22 and CLIC2025 |
| `d8d1093` | Add scan optimization module |

---

## Coefficient Project: Optimized Quant Tables

The `coefficient` project discovered **dramatically better quant tables** for MozJPEG-style encoding using simulated annealing optimization on the CID22 corpus.

### Key Discovery

**Libjpeg-style quant tables beat jpegli's custom tables by +6.2 mean pareto** across q1-100.
SA optimization adds another +0.6, for a total of **+6.8 pareto on training, +6.5 on holdout**.

### Results (v4 Hybrid Tables)

Validated on 41 unseen images (CID22 holdout):

| Metric | Value |
|--------|-------|
| Quality levels that beat baseline | **100/100** |
| Mean pareto distance | **+6.471** |
| Min pareto (worst case, q100) | **+2.629** |
| BPP range | 0.36–3.27 |

### Sample Quality Points (Holdout)

| Q | BPP | Butteraugli | Pareto |
|---|-----|-------------|--------|
| 50 | 0.75 | 4.34 | +6.815 |
| 75 | 1.27 | 2.90 | +5.944 |
| 85 | 1.59 | 2.76 | +5.314 |
| 90 | 1.93 | 3.44 | +4.532 |
| 95 | 2.27 | 3.49 | +3.844 |

### Trellis is Redundant with Optimized Tables

| Configuration | Mean Pareto | Notes |
|---------------|-------------|-------|
| v4 tables, no trellis | +6.471 | Best |
| v4 tables + trellis | +6.291 | Worse (-0.18) |
| Trellis baseline | 0.0 | Reference |

**Custom tables without trellis beat jpegli+trellis at all 100 quality levels.**

### Where to Find the Tables

Production files at `/mnt/v/output/coefficient/piecewise/`:

| File | Description |
|------|-------------|
| `combined_hybrid_v4.json` | **Best tables** (JSON, 20 anchors) |
| `best_tables_v4.rs` | **Rust source** (26KB, embeddable) |

The Rust source includes interpolation between 20 anchor points (q5, q10, ..., q100).

### Integration Path

```rust
// From best_tables_v4.rs
use optimized_tables::{PiecewiseQuantTables, ANCHOR_LUMA, ANCHOR_CHROMA};

let tables = PiecewiseQuantTables::new();
let (luma, cb, cr) = tables.interpolate(quality);
```

These tables should be integrated into zenjpeg as an alternative to jpegli's tables,
especially for users who don't need XYB mode and want maximum compression efficiency.

---

## Open Questions

1. **Integration priority?** The coefficient project tables provide +6.5 pareto improvement
   over jpegli defaults. This is much larger than trellis (+2-3%). Should be high priority.

2. **4:2:0 thresholds?** Current data shows trellis helps 4:2:0 at Q90, but 4:4:4 needs Q95. More corpus validation needed.

3. **Speed optimization?** Trellis is 70% slower. SIMD optimization of the Viterbi DP could reduce this.

4. **Combine optimized tables + trellis?** The coefficient project found trellis hurts with
   optimized tables, but this was tested against jpegli baseline. Worth retesting trellis
   on top of optimized tables vs optimized tables alone.
