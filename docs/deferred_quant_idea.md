# Deferred Quantization for Adaptive Quality Targeting

## Problem Statement

Currently, quantization tables are generated **before** any image content is analyzed:

```
Current Flow:
┌─────────────────────────────────────────────────────────────┐
│  Setup:                                                     │
│    quality → distance → generate_quant_table()              │
│    (no image content used)                                  │
│                                                             │
│  Per-Strip:                                                 │
│    RGB → YCbCr → DCT (f32) → AQ computed → quantize (i16)  │
│                                            ▲                │
│                                            │                │
│                              quant tables baked in here     │
└─────────────────────────────────────────────────────────────┘
```

The quantization table values are committed before we know anything about:
- Image texture/complexity distribution
- Actual coefficient magnitudes
- Predicted file size at this quality

## Opportunity

The JPEG header (containing DQT markers) is written **after** all strips are processed. We could defer quantization until we have full image statistics, then choose optimal tables.

```
Proposed Flow:
┌─────────────────────────────────────────────────────────────┐
│  Per-Strip:                                                 │
│    RGB → YCbCr → DCT (f32) → AQ computed → store f32       │
│                                            + accumulate     │
│                                              stats          │
│  Finalize:                                                  │
│    Analyze stats → choose optimal distance → quantize all  │
│    → write header with final tables → encode               │
└─────────────────────────────────────────────────────────────┘
```

## Use Cases

1. **File size targeting**: "Encode this image to ~100KB"
2. **Quality floor**: "Use lowest quality that stays above SSIM 0.95"
3. **Bandwidth budgets**: Proxy servers with per-image byte limits
4. **A/B quality selection**: Try two distances, pick better size/quality tradeoff

## Implementation Design

### Branch Point

Single branch in `quantize_pending_imcu`:

```rust
fn quantize_pending_imcu(&mut self, buffer_idx: usize, aq_strengths: &[f32]) {
    if self.defer_quantization {
        // Deferred path: store f32 blocks, accumulate stats
        self.deferred_y_dct.extend(self.pending_y_blocks[buffer_idx].drain(..));
        self.deferred_aq.extend_from_slice(aq_strengths);
        self.stats.accumulate(&self.pending_y_blocks[buffer_idx]);
    } else {
        // Normal path: quantize immediately to i16 (existing code)
        for (i, dct) in self.pending_y_blocks[buffer_idx].iter().enumerate() {
            let zigzag = quant.y_quant_simd.quantize_with_zero_bias_zigzag(...);
            self.y_blocks.push(zigzag);
        }
    }
}
```

### Performance Impact (Normal Path)

| Overhead | Cost |
|----------|------|
| Branch check `if self.defer_quantization` | ~0 (100% predicted after 1st iter) |
| Field storage (`Option<Vec<Block8x8f>>`) | 0 bytes when `None` |
| Code size | Minimal (deferred path only in binary if used) |

**The normal path is unchanged** - one predictable branch per iMCU.

### Memory Cost (Deferred Path)

| Resolution | Normal (i16) | Deferred (f32) | Overhead |
|------------|--------------|----------------|----------|
| 1080p | ~9 MB | ~36 MB | +27 MB |
| 4K | ~36 MB | ~144 MB | +108 MB |
| 8K | ~144 MB | ~576 MB | +432 MB |

Acceptable for the niche use cases (proxy servers, quality targeting tools).

## Statistics to Accumulate

### Already Available
- `all_aq_strengths: Vec<f32>` - per-block AQ values
- `AQStrengthMap::stats()` → (min, max, mean, std)

### Cheap to Add (O(1) per block)
```rust
struct DeferredStats {
    // AQ distribution
    aq_sum: f32,
    aq_sum_sq: f32,
    aq_min: f32,
    aq_max: f32,

    // DCT energy (predicts compressibility)
    dc_sum: [f64; 3],           // Sum of DC coefficients per component
    ac_energy: [f64; 3],        // Sum of |AC coefficients|
    ac_energy_sq: [f64; 3],     // For variance

    // Coefficient histogram buckets (for entropy estimation)
    coeff_buckets: [[u32; 16]; 3],  // Log-scale magnitude buckets

    block_count: usize,
}
```

### Derived at Finalize
```rust
impl DeferredStats {
    /// Estimate encoded size at a given distance
    fn estimate_bits(&self, distance: f32) -> u64 {
        // Model: bits ≈ k * ac_energy / distance^α + overhead
        // Calibrated from empirical data
    }

    /// Find distance that produces target file size
    fn distance_for_target_size(&self, target_bytes: usize) -> f32 {
        // Binary search using estimate_bits()
    }

    /// Suggest quality based on image complexity
    fn suggested_distance(&self) -> f32 {
        // High AQ variance → more texture → can use higher distance
        // Low AQ variance → flat image → needs lower distance for quality
    }
}
```

## API Surface

### Builder Pattern
```rust
let encoder = StreamingJpegEncoder::new(width, height)
    .defer_quantization(true)
    .build()?;

// Process strips normally
for strip in strips {
    encoder.push_rows(&strip)?;
}

// Get deferred output with stats
let deferred = encoder.finalize_deferred()?;
```

### Inspection and Decision
```rust
// Inspect statistics
println!("AQ: mean={:.3}, std={:.3}", deferred.stats.aq_mean, deferred.stats.aq_std);
println!("AC energy: {:.0}", deferred.stats.ac_energy[0]);
println!("Est. size at q85: {} KB", deferred.stats.estimate_bits(1.0) / 8 / 1024);

// Option A: Target specific file size
let distance = deferred.stats.distance_for_target_size(100_000);

// Option B: Use suggested quality
let distance = deferred.stats.suggested_distance();

// Option C: Try multiple, pick best
let candidates = [0.5, 1.0, 2.0];
let (best_distance, _) = candidates.iter()
    .map(|&d| (d, deferred.stats.estimate_bits(d)))
    .min_by_key(|(_, bits)| (*bits as i64 - target_bits as i64).abs())
    .unwrap();
```

### Final Quantization
```rust
// Quantize with chosen distance
let jpeg = deferred.quantize_with_distance(distance)?;

// Or with explicit tables
let tables = CustomQuantTables::from_distance(distance);
let jpeg = deferred.quantize_with_tables(tables)?;
```

## File Size Estimation Model

The relationship between DCT coefficients and encoded size:

```
encoded_bits ≈ Σ entropy(coeff / quant_value)
             ≈ k₁ * Σ|AC| / distance + k₂ * num_blocks + k₃
```

Where:
- `Σ|AC|` = total AC coefficient energy (accumulated during encoding)
- `distance` = quality parameter (higher = more compression)
- `k₁, k₂, k₃` = calibration constants (derived empirically)

For better accuracy, use per-frequency models:
```rust
fn estimate_bits_detailed(&self, quant_table: &[u16; 64]) -> u64 {
    let mut total = 0u64;
    for freq in 0..64 {
        let energy = self.freq_energy[freq];
        let quant = quant_table[freq] as f64;
        // Laplacian distribution model for DCT coefficients
        total += (energy / quant * ENTROPY_FACTOR).ceil() as u64;
    }
    total
}
```

## Implementation Phases

### Phase 1: Core Infrastructure
- [ ] Add `defer_quantization: bool` to `StripProcessor`
- [ ] Add `deferred_y_dct: Option<Vec<Block8x8f>>` storage
- [ ] Branch in `quantize_pending_imcu`
- [ ] Add `DeferredStats` accumulator
- [ ] Add `finalize_deferred() -> DeferredOutput`

### Phase 2: Quantization API
- [ ] `DeferredOutput::quantize_with_distance(f32) -> Result<Vec<u8>>`
- [ ] `DeferredOutput::quantize_with_tables(QuantTables) -> Result<Vec<u8>>`
- [ ] Verify output matches normal path at same distance

### Phase 3: Size Estimation
- [ ] Implement `estimate_bits(distance)` with basic model
- [ ] Calibrate against real images (codec-corpus)
- [ ] Add `distance_for_target_size(bytes)`
- [ ] Accuracy target: ±10% for typical images

### Phase 4: Advanced Features
- [ ] Per-frequency energy tracking for better estimation
- [ ] Quality suggestion based on image characteristics
- [ ] Multi-pass refinement (encode, measure, adjust, re-encode)

## Open Questions

1. **Should deferred mode support streaming output?**
   - Currently assumes all blocks in memory
   - Could chunk into segments with independent tables (rare JPEG feature)

2. **What about progressive mode?**
   - Progressive already buffers all coefficients
   - Could share infrastructure

3. **Memory-mapped storage for very large images?**
   - 8K deferred = 576 MB
   - Could mmap temp file instead of heap allocation

4. **Parallel quantization at finalize?**
   - All blocks independent
   - Easy to parallelize with rayon

## References

- Current strip processing: `zenjpeg/src/encode/strip.rs`
- Quant table generation: `zenjpeg/src/quant/mod.rs`
- AQ computation: `zenjpeg/src/quant/aq/mod.rs`
- Streaming AQ: `zenjpeg/src/quant/aq/streaming.rs`
