# XYB Progressive Table Assignments

This document clarifies the two different types of tables used in JPEG and how they're assigned for XYB progressive mode.

## Two Types of Tables

### Huffman Tables (DHT markers)
Used for entropy coding - compress the quantized DCT coefficients into the bitstream.

**XYB Progressive assignment** (after fix in commit 50d2cf4):
- R (X channel): DC Huffman table 0, AC Huffman table 0
- G (Y channel): DC Huffman table 0, AC Huffman table 0
- B channel: DC Huffman table 0, AC Huffman table 0

**All components use the same Huffman tables** (table 0 for both DC and AC).
This matches baseline XYB behavior in `write_scan_header_xyb()`.

### Quantization Tables (DQT markers)
Define how to quantize DCT coefficients - control quality vs compression tradeoff.

**XYB assignment** (verified correct, matches C++ jpegli):
- R (X channel): Quantization table 0
- G (Y channel): Quantization table 1
- B channel: Quantization table 2

**Each component uses a separate quant table** as defined in C++ `encode.cc`:
```cpp
if (cinfo->master->xyb_mode) {
  // Use separate quantization tables for each component
  cinfo->comp_info[1].quant_tbl_no = 1;
  cinfo->comp_info[2].quant_tbl_no = 2;
}
```

## Summary

| Component | Huffman Table | Quantization Table |
|-----------|---------------|-------------------|
| R (X) | 0 (DC), 0 (AC) | 0 |
| G (Y) | 0 (DC), 0 (AC) | 1 |
| B | 0 (DC), 0 (AC) | 2 |

**Key takeaway**:
- Huffman tables are SHARED (all use table 0)
- Quant tables are SEPARATE (each component has its own: 0, 1, 2)

## What Was Fixed

**Commit 50d2cf4** fixed the **Huffman table assignment** for XYB progressive:
- Before: Scan headers tried to use Huffman tables 0, 1, and 2
- Problem: Only tables 0 and 1 existed in the DHT markers
- After: All components correctly use Huffman table 0
- Result: mozjpeg now decodes XYB progressive (was crashing before)

**Quant tables were already correct** - they were never the issue.
