# Context Handoff - Bounded-Memory Streaming Heuristics

## Session Summary (2026-01-27)

Implemented custom Huffman tables API for bounded-memory streaming encoding, enabling corpus-based table optimization.

## New API: Custom Huffman Tables

### Types Added

```rust
/// A complete set of frequency counters for Huffman table optimization.
pub struct HuffmanFrequencyCounts {
    pub dc_luma: FrequencyCounter,
    pub ac_luma: FrequencyCounter,
    pub dc_chroma: FrequencyCounter,
    pub ac_chroma: FrequencyCounter,
}

/// Result from encoding that includes both JPEG data and Huffman statistics.
pub struct EncodingResult {
    pub jpeg: Vec<u8>,
    pub frequency_counts: HuffmanFrequencyCounts,
    pub huffman_tables: OptimizedHuffmanTables,
}
```

### Builder Methods Added

```rust
// Use pre-built custom tables (highest priority)
StreamingEncoder::new(w, h)
    .custom_huffman_tables(tables)
    .start()?;

// Generate tables from custom frequency counts
StreamingEncoder::new(w, h)
    .custom_frequency_counts(counts)
    .start()?;
```

### Finish Method Added

```rust
// Get JPEG + frequency counts + tables used
let result = encoder.finish_with_tables()?;
println!("JPEG size: {}", result.jpeg.len());
println!("AC entropy: {:.2}", result.frequency_counts.ac_luma.entropy());
```

### Use Cases

1. **Build "universal" tables from corpus:**
   ```rust
   let mut corpus_counts = HuffmanFrequencyCounts::new();
   for image in corpus {
       let result = encode_image(image)?;
       corpus_counts.add(&result.frequency_counts);
   }
   let corpus_tables = corpus_counts.generate_tables()?;
   ```

2. **Use corpus tables for streaming encoding:**
   ```rust
   let encoder = StreamingEncoder::new(w, h)
       .memory_limit(1024 * 1024)
       .custom_huffman_tables(corpus_tables)
       .start()?;
   // Tables used immediately - no optimization pass needed
   ```

3. **Analyze symbol distributions:**
   ```rust
   let result = encoder.finish_with_tables()?;
   let ac_entropy = result.frequency_counts.ac_luma.entropy();
   let dc_coverage = result.frequency_counts.dc_luma.dc_symbol_coverage();
   ```

## Previous Session Findings

### Transition Reason Tracking

Added `TransitionReason` enum to understand WHY images transition to streaming mode:

```rust
pub enum TransitionReason {
    ForcedByRows,       // Testing API forced transition
    HeuristicsPassed,   // Memory limit + heuristics OK
    MinPercentReached,  // min_transition_percent gate only
    SafetyValve,        // 50% safety valve
    NoTransition,       // Full buffering mode
}
```

### CLIC 2025 Test Results (32 images)

| Min % | Failures | Max Overhead | Mean Trans% |
|-------|----------|--------------|-------------|
| 25% | 2/32 | 14.62% | 47.1% |
| 30% | 1/32 | 13.41% | 47.8% |
| 35% | 1/32 | 7.91% | 48.4% |
| 40% | 1/32 | 6.24% | 49.0% |
| **50%** | **0/32** | **3.62%** | **50.2%** |

### Pathological Image Analysis

Two images pass heuristics at 25% but produce poor tables because their early frequency distributions are NOT representative - the distribution continues to diverge significantly through the image.

## Files Modified This Session

- `zenjpeg/src/encode/streaming.rs` - Added HuffmanFrequencyCounts, EncodingResult, builder methods, finish_with_tables()
- `zenjpeg/src/encode/mod.rs` - Re-exported new types
- `zenjpeg/examples/custom_huffman_tables.rs` - Demo of the new API

## Test Commands

```bash
# Run the custom Huffman tables example
cargo run --release -p zenjpeg --features test-utils --example custom_huffman_tables

# Run library tests
cargo test --release -p zenjpeg --features test-utils --lib
```

## Next Steps

1. Test corpus-based tables on CLIC 2025 validation set
2. Compare overhead: corpus tables vs optimized-from-partial vs standard tables
3. Find optimal corpus size for convergent tables
4. Consider whether tables should be tuned per-quality-level
