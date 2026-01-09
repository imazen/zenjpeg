# Analysis: C++ jpegli Streaming AQ Implementation

## Summary

C++ jpegli achieves ~90% memory reduction by:
1. **Computing AQ incrementally** per iMCU row (not buffering Y plane)
2. **Quantizing immediately** during DCT, storing i32/i16 (not f32)
3. **Reusing rolling buffers** of fixed height

Our Rust `StreamingAQParity` buffers the **entire Y plane** for byte-identical output, defeating the memory savings.

---

## C++ Memory Architecture

### Buffer Sizes (from `encode.cc:AllocateBuffers`)

| Buffer | Size | Purpose |
|--------|------|---------|
| `input_buffer` | 3 × iMCU_height rows | Rolling input (Y, Cb, Cr) |
| `diff_buffer` | xsize_blocks × DCTSIZE + 8 floats | Per-row difference computation |
| `fuzzy_erosion_tmp` | 2 rows | Temporary for erosion |
| `pre_erosion` | 6 × max_v_samp_factor rows | Rolling pre-erosion buffer |
| `quant_field` | max_v_samp_factor rows | Current iMCU's AQ values |

**For 4K (3840×2160):**
- Input buffers: 3 × 16 × 3840 × 4 × 3 = **2.2 MB** (rolling, reused)
- AQ buffers: ~**50 KB** total
- DCT scratch: **256 bytes**
- **Total working set: ~2.5 MB**

### Our Rust Memory (StreamingAQParity)
- Y plane: 3840 × 2160 × 4 = **32 MB**
- f32 DCT storage: **~65 MB** (before quantization)
- **Total: ~100 MB**

---

## Key C++ Functions

### 1. `ComputeAdaptiveQuantField` (adaptive_quantization.cc:384)

```cpp
void ComputeAdaptiveQuantField(j_compress_ptr cinfo) {
  // Called once per iMCU row
  size_t y0 = m->next_iMCU_row * iMCU_height;

  // Process rows in this iMCU
  for (size_t y = y0; y < y_end; ++y) {
    ComputePreErosion(row, diff_buffer, pre_erosion.Row(y), width);
  }

  FuzzyErosion(pre_erosion, fuzzy_erosion_tmp, ...);
  PerBlockModulations(quant_field.Row(0), ...);
}
```

Key insight: **Only computes AQ for current iMCU row**, stores in `quant_field.Row(0)`.

### 2. `ProcessiMCURow` (encode_streaming.cc:112)

Three modes controlled by template parameter:
- `kStreamingModeCoefficients` - Store to coefficient buffers (non-streaming)
- `kStreamingModeTokens` - Compute Huffman tokens (optimize_coding=true)
- `kStreamingModeBits` - Write bits directly to output (true streaming!)

```cpp
template <int kMode>
void ProcessiMCURow(j_compress_ptr cinfo) {
  const float* qf = m->quant_field.Row(0);  // Current iMCU's AQ

  for (int mcu_x = 0; mcu_x < xsize_mcus; ++mcu_x) {
    for each block in MCU:
      aq_strength = qf[bx];  // Per-block AQ strength

      // DCT + quantize in ONE call (outputs int32_t)
      ComputeCoefficientBlock(pixels, stride, qmc, last_dc_coeff[c],
                              aq_strength, zero_bias_offset, zero_bias_mul,
                              m->dct_buffer, block);

      if (kMode == kStreamingModeBits) {
        WriteBlock(...);  // Write to output immediately!
      }
  }
}
```

### 3. `IsStreamingSupported` (encode.cc:387)

```cpp
bool IsStreamingSupported(j_compress_ptr cinfo) {
  if (cinfo->restart_interval > 0) return false;  // No restart markers
  if (cinfo->num_scans > 1) return false;         // Single scan only (no progressive)
  if (cinfo->master->psnr_target > 0) return false;
  return true;
}
```

**Progressive mode requires buffering** - can't stream when num_scans > 1.

---

## Why We Can't Match C++ Output With True Streaming

The C++ AQ algorithm uses a **causal filter** - it only looks at previous rows for the pre-erosion step. However, `FuzzyErosion` needs 3×3 neighborhood context, which requires buffering 3 rows.

The `pre_erosion` buffer stores **6 rows** to provide context for the fuzzy erosion step. This is the "lookahead" - it processes rows slightly delayed to have both above and below context.

To match C++ output exactly, we would need to:
1. Port the exact rolling buffer sizes
2. Match the border handling in `RowBuffer::PadRow()`
3. Match the row delay (process iMCU N when receiving iMCU N+1)

---

## Implementation Options

### Option A: True Streaming (Different Output)
Port C++ algorithm exactly:
- Rolling buffers for input and AQ
- Quantize during strip processing
- **~95% memory reduction**
- Output differs slightly from full-plane encoder

**Pros:** Massive memory savings, matches C++ behavior
**Cons:** Output differs from current Rust full-plane encoder

### Option B: Keep Current Parity Mode
Keep `StreamingAQParity` for byte-identical output when needed:
- Add `low_memory: bool` flag to strip encoder
- Default to true streaming, opt-in to parity mode

**Pros:** User choice
**Cons:** Two code paths to maintain

### Option C: Immediate Quantization Only
Keep current AQ, but quantize immediately:
- Still buffer Y plane for AQ computation
- But quantize to i16 after DCT (not store f32)
- **~50% memory reduction**

**Pros:** Simple change, still byte-identical
**Cons:** Only moderate savings

---

## Recommended Approach: Option A

Port the C++ streaming AQ algorithm:

```rust
pub struct StreamingAQ {
    // Rolling buffers (fixed height, reused)
    pre_erosion: RowBuffer<f32>,      // 6 × v_samp rows
    fuzzy_erosion_tmp: RowBuffer<f32>, // 2 rows
    diff_buffer: Vec<f32>,             // 1 row

    // Output: just current iMCU
    quant_field: Vec<f32>,  // 1 × xsize_blocks

    // State for rolling
    current_row: usize,
}

impl StreamingAQ {
    /// Process one row of Y plane, returns None until iMCU complete
    pub fn process_row(&mut self, y_row: &[f32]) -> Option<&[f32]> {
        // Compute pre-erosion for this row
        compute_pre_erosion(y_row, &mut self.diff_buffer,
                           self.pre_erosion.row_mut(self.current_row));

        self.current_row += 1;

        // When iMCU complete, compute quant field
        if self.current_row % imcu_height == 0 {
            fuzzy_erosion(&self.pre_erosion, &mut self.fuzzy_erosion_tmp);
            per_block_modulations(&self.fuzzy_erosion_tmp, &mut self.quant_field);
            return Some(&self.quant_field);
        }
        None
    }
}
```

### Step 2: Quantize During Strip Processing

```rust
impl StripProcessor {
    pub fn process_strip(&mut self, rgb_strip: &[u8], strip_y: usize) {
        // Color convert strip
        // ...

        // Compute AQ for this strip
        let aq = self.streaming_aq.process_strip(&y_strip);

        // DCT + quantize immediately (store i16, not f32)
        for block in blocks {
            let coeffs = dct_and_quantize(block, aq[block_idx]);
            self.y_blocks.push(coeffs);  // i16, not f32!
        }
    }
}
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `quant/aq/streaming.rs` | Replace with C++ rolling buffer algorithm |
| `encode/strip.rs` | Quantize during `process_strip`, not `finalize` |
| `encode/blocks.rs` | Add `dct_and_quantize_block()` function |

---

## Verification

1. **Memory benchmark**: Should show ~90% reduction for 4K
2. **Quality comparison**: SSIMULACRA2 between streaming and full-plane
3. **C++ comparison**: DSSIM against C++ jpegli output

---

## Notes

- C++ `RowBuffer` uses `border` parameter for edge replication
- Pre-erosion uses 4× downsampling (processes every 4th row conceptually)
- The 6-row buffer provides lookahead for fuzzy erosion's 3×3 kernel
- `quant_field` only stores **1 iMCU row** worth of AQ values
