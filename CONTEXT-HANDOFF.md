# Context Handoff: XYB Block-Boundary Coefficient Patterns

## What Was Done This Session

Fixed critical XYB B-channel encoding bug:
- B channel had ~51 mean error vs ~0.1 for R/G (catastrophic corruption)
- Root cause: `StripProcessor` dimensions not recalculated for XYB mode
- Fix: Added `b_blocks_h`, `b_blocks_v`, `padded_b_width` fields; use `with_xyb()` constructor
- Commits: `6d34ed5`, `c82c243`, `2287d94`, `6bb9e82`

## Remaining Investigation: Block-Boundary Coefficient Patterns

**Problem**: XYB visual diff shows U/M patterns at 8×8 block boundaries in ΔR and ΔB channels.

**Repro commands**:
```bash
just xyb-diff                     # 5-panel: C++ | Rust | ΔR×10 | ΔG×10 | ΔB×10
just xyb-diff ~/path/to/image.png # Custom image
```

**Observed patterns (kodak/1.png q90)**:
- ΔR (3rd panel): Block patterns with U/M shapes, sometimes strong
- ΔG (4th panel): Uniform even noise (good)
- ΔB (5th panel): Similar to ΔR but less intense

**Numeric results**:
```
XYB:   Rust 156720 bytes, C++ 141450 bytes (+10.8%)
       Mean |diff|: R=0.310, G=1.300, B=0.200

YCbCr: Rust 130270 bytes, C++ 143695 bytes (-9.3%)
       Mean |diff|: R=2.056, G=1.877, B=2.102
```

**Hypothesis**: Block patterns suggest systematic differences in:
1. Zero-bias calculation at block boundaries
2. AQ strength interpolation between blocks
3. Coefficient quantization thresholds

**Key files to investigate**:
- `zenjpeg/src/encode/strip/mod.rs:quantize_pending_imcu` (lines 1245-1350)
- `zenjpeg/src/quant/aq/simd.rs` - AQ strength calculation
- `zenjpeg/src/quant/zero_bias.rs` - Zero-bias params

**Investigation approach**:
1. Run `just xyb-diff` on solid-color or gradient test images to isolate pattern
2. Compare zero-bias values between Rust and C++ FFI
3. Check if pattern correlates with AQ strength boundaries
4. Look for off-by-one errors in block indexing for edge blocks

**Related notes in CLAUDE.md**:
- "Visual Diff Interpretation (2026-01-31)" section
- "Known Bugs" section #1

## Git Status

```
main branch, ahead of origin by 11 commits
Last commit: 6bb9e82 docs: add visual diff interpretation and XYB B-channel fix
```

## Delete This File

After loading context into new session, delete this file:
```bash
rm CONTEXT-HANDOFF.md
```
