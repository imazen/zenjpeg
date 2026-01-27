# Bounded-Memory Streaming Encoder Exploration

## Goal

Implement a streaming JPEG encoder with **bounded peak memory** that produces **optimized Huffman tables** without buffering the entire image's DCT coefficients.

## The Problem

Currently, optimized Huffman encoding requires:
1. Process ALL strips → DCT → quantize → store coefficients (~9MB for 3MP image)
2. Count symbol frequencies from ALL coefficients
3. Build Huffman tables
4. Encode ALL coefficients
5. Write JPEG

This means memory usage is O(image_size), which is problematic for:
- Proxy servers encoding many images concurrently
- Memory-constrained environments
- Very large images (100MP+)

## The Constraint

JPEG structure requires Huffman tables (DHT markers) to appear **before** scan data. RST markers between segments do NOT allow new DHT markers - all segments share the same tables. This means we cannot build per-segment tables.

## The Solution: Threshold-Based Streaming

Accumulate coefficients and count frequencies until a memory threshold, then transition to streaming mode:

### Phase 1: Accumulation (until threshold)
```
for each strip:
    DCT → quantize → store coefficients + count frequencies
    if accumulated_bytes > threshold:
        TRANSITION
```

### Phase 2: Transition (one-time)
```
1. Build optimized Huffman tables from accumulated frequencies
2. Write JPEG header (SOI, DQT, SOF, DHT, DRI, SOS)
3. Encode ALL accumulated coefficients → flush to output
4. RELEASE coefficient memory
5. Set streaming_mode = true
```

### Phase 3: Streaming (rest of image)
```
for each strip:
    DCT → quantize → encode immediately → flush
    (no coefficient storage)
```

### Memory Profile
```
         Accumulation    Transition    Streaming
Memory:  grows to 10MB → spike at     → flat ~500KB
                         encode+flush
```

## Key Implementation Details

### DC Prediction Consistency (CRITICAL)

The main bug risk is DC prediction mismatch. Frequency counting and encoding must use **identical** DC differences.

**Solution:** Count frequencies **inline during quantization**, not in a separate pass:

```rust
// In quantize_pending_imcu() - count frequencies as blocks are produced:
fn quantize_pending_imcu(&mut self, ...) {
    for block in pending_blocks {
        let quantized = quantize(block);

        // Count frequencies using actual DC difference
        let dc_diff = quantized[0] - self.dc_state.prev_y_dc;
        self.freq_counters.dc_luma.count(category(dc_diff));
        // ... AC coefficients ...

        self.dc_state.prev_y_dc = quantized[0];
        self.y_blocks.push(quantized);
    }
}
```

### Files to Modify

1. **`zenjpeg/src/encode/strip/mod.rs`** (~200-270)
   - Add fields:
     ```rust
     freq_counters: Option<FrequencyCounters>,  // DC/AC luma/chroma
     accumulated_bytes: usize,
     memory_limit: Option<usize>,
     streaming_mode: bool,
     ```
   - Modify `process_strip()` to count frequencies inline
   - Add `take_accumulated_blocks()` for transition

2. **`zenjpeg/src/encode/streaming.rs`** (~767-795)
   - Add fields:
     ```rust
     output_buffer: Option<Vec<u8>>,
     huffman_tables: Option<OptimizedHuffmanTables>,
     streaming_encoder: Option<EntropyEncoder<'static>>,
     header_written: bool,
     dc_state: DcPredictionState,
     ```
   - Add `set_memory_limit(bytes: usize)` method
   - Add `check_memory_and_maybe_transition()` method
   - Modify `finish()` to handle streaming mode

3. **New: `DcPredictionState` struct**
   ```rust
   struct DcPredictionState {
       prev_y_dc: i16,
       prev_cb_dc: i16,
       prev_cr_dc: i16,
       mcu_count: usize,
       restart_counter: usize,
   }
   ```

### Existing Infrastructure to Reuse

- `FrequencyCounter` in `huffman/optimize/frequency.rs` - symbol counting
- `build_optimized_tables()` in `encode/blocks.rs` - table building (adapt for streaming)
- `EntropyEncoder` in `entropy/encoder.rs` - block encoding
- Restart marker logic in `parallel.rs` - DC prediction reset

## Testing Strategy

1. **Parity test**: Same image, with/without memory limit → identical output (when limit > image size)
2. **Transition test**: Force transition at various points → valid JPEG output
3. **DC prediction test**: Compare frequency counts from streaming vs batch → must match
4. **Restart marker test**: Verify RST markers at correct positions across transition
5. **Memory test**: Verify peak memory stays under threshold

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| DC prediction mismatch | HIGH | Count frequencies inline during quantization |
| Restart boundary alignment | HIGH | Continuous MCU counter across transition |
| Partial strip at transition | MEDIUM | Complete strip before transition |
| Memory accounting drift | LOW | Explicit 128 * block_count tracking |

## Estimated Scope

| Component | Lines Changed | New Lines |
|-----------|---------------|-----------|
| strip/mod.rs | ~50 | ~100 |
| streaming.rs | ~80 | ~150 |
| New FrequencyCounters helper | - | ~50 |
| Tests | - | ~200 |
| **Total** | ~130 | ~500 |

## Open Questions

1. Should threshold be configurable per-encode or global?
2. Should we support "no-limit" mode that behaves exactly like current code?
3. What's the right default threshold? 10MB? 5MB? Percentage of available memory?
4. Should we emit a warning/log when transitioning?

## How to Start

1. Read current `StripProcessor` in `strip/mod.rs` to understand coefficient flow
2. Read `build_optimized_tables()` in `blocks.rs` to understand frequency counting
3. Start with a simple prototype that:
   - Adds frequency counting inline to `quantize_pending_imcu()`
   - Doesn't actually limit memory yet
   - Verifies frequencies match existing `build_optimized_tables()` output
4. Once frequencies match, add the transition logic

## Commands

```bash
# Build and test
cargo build --release -p zenjpeg
cargo test --release -p zenjpeg

# Run allocation profiler to verify memory behavior
cargo run --release --example real_alloc_profile --features alloc-instrument

# Benchmark encode performance
cargo bench --bench encode
```
