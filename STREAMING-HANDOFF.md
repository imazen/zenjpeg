# Bounded-Memory Streaming Encoder Exploration

## Implementation Status

**✅ CORE IMPLEMENTATION COMPLETE** - All tasks implemented and tested.

| Task | Status |
|------|--------|
| Inline frequency counting in StripProcessor | ✅ Complete |
| Frequency parity verification | ✅ Complete (tests pass) |
| Memory limit and streaming transition | ✅ Complete |
| Basic parity tests | ✅ Complete |
| Full parity tests with FFI | ⏸️ Blocked (submodule unavailable) |

## What Was Implemented

### 1. Inline Frequency Counting (`strip/mod.rs`)

Added fields to `StripProcessor`:
```rust
// === Inline frequency counting for bounded-memory streaming ===
dc_luma_freq: FrequencyCounter,
ac_luma_freq: FrequencyCounter,
dc_chroma_freq: FrequencyCounter,
ac_chroma_freq: FrequencyCounter,
freq_prev_dc: [i16; 3],          // Y, Cb, Cr DC prediction state
freq_mcu_count: usize,           // MCU counter for restart intervals
restart_interval: u16,           // Restart interval (0 = disabled)
total_mcus: usize,               // Total MCUs in image
freq_imcu_rows_counted: usize,   // iMCU rows with frequencies counted
```

Added methods:
- `count_frequencies_for_imcu_row()` - counts frequencies in MCU order
- `collect_block_frequencies_inline()` - helper to count DC/AC for a block
- `frequency_counters()` - returns references to all frequency counters
- `estimate_block_storage()` - returns current block storage in bytes
- `take_blocks()` - takes ownership of blocks for streaming transition

**Key insight**: Frequencies must be counted in MCU order (not raster order) to ensure DC prediction matches the actual encoding order.

### 2. Bounded-Memory Streaming (`streaming.rs`)

Added fields to `StreamingEncoder`:
```rust
memory_limit: Option<usize>,
streaming_mode: bool,
streaming_output: Option<Vec<u8>>,
streaming_tables: Option<OptimizedHuffmanTables>,
streaming_prev_dc: [i16; 3],
streaming_mcu_idx: usize,
streaming_restart_count: u8,
```

Added builder method:
- `memory_limit(limit: usize)` - sets memory threshold for transition

Added internal methods:
- `check_and_maybe_transition()` - checks memory and triggers transition
- `transition_to_streaming()` - builds tables, writes header, encodes accumulated blocks
- `encode_blocks_mcu_order_ex()` - encodes blocks with state continuation
- `encode_blocks_mcu_order_static()` - static version for finish_streaming
- `encode_new_blocks_streaming()` - encodes new blocks immediately
- `finish_streaming()` - finalizes streaming mode output

### 3. Supporting Infrastructure

**BitWriter extensions** (`bitstream.rs`):
- `flush_without_eoi()` - flushes bits to external buffer without EOI marker
- `flush_restart_marker()` - writes restart marker and resets state

**Entropy encoding** (`entropy/mod.rs`):
- `encode_block_to_writer()` - encodes a single block to an external BitWriter

### 4. Tests

**Inline frequency parity tests** (in `strip/mod.rs`):
- `test_inline_frequency_parity_420` - verifies 4:2:0 frequencies match batch
- `test_inline_frequency_parity_444` - verifies 4:4:4 frequencies match batch

**Streaming encoder tests** (in `streaming.rs`):
- `test_memory_limit_not_compatible_with_progressive` - error handling
- `test_bounded_streaming_basic` - end-to-end streaming test
- `test_bounded_streaming_no_transition_if_below_limit` - no transition when below limit

## How It Works

### Phase 1: Accumulation (until threshold)
```
for each strip:
    process_strip() → DCT → quantize → store blocks
    count_frequencies_for_imcu_row() → count in MCU order
    if estimate_block_storage() > memory_limit:
        TRANSITION
```

### Phase 2: Transition (one-time)
```
transition_to_streaming():
    1. Build Huffman tables from accumulated frequencies
    2. Write JPEG header (SOI, APP0, DQT, SOF, DHT, DRI, SOS)
    3. take_blocks() → encode_blocks_mcu_order_ex() → flush to output
    4. Release block storage
    5. streaming_mode = true
```

### Phase 3: Streaming (rest of image)
```
for each strip after transition:
    process_strip() → DCT → quantize
    encode_new_blocks_streaming() → encode immediately → flush
    (no coefficient storage)
```

### Memory Profile
```
         Accumulation    Transition    Streaming
Memory:  grows to limit → spike at     → flat ~strip size
                         encode+flush
```

## Validation

All tests pass:
```
test encode::streaming::tests::test_bounded_streaming_basic ... ok
test encode::streaming::tests::test_bounded_streaming_no_transition_if_below_limit ... ok
test encode::streaming::tests::test_memory_limit_not_compatible_with_progressive ... ok
test encode::strip::tests::test_inline_frequency_parity_420 ... ok
test encode::strip::tests::test_inline_frequency_parity_444 ... ok
```

## Usage

```rust
let mut encoder = StreamingEncoder::new(4000, 3000)
    .quality(85)
    .subsampling(Subsampling::S420)
    .progressive(false)  // Required for bounded streaming
    .memory_limit(8 * 1024 * 1024)  // 8 MB limit
    .start()?;

// Push rows - streaming transition happens automatically
for row in image_rows {
    encoder.push_row(row)?;
}

let jpeg = encoder.finish()?;
```

## Limitations

1. **Progressive mode not supported** - Progressive encoding requires multiple passes over all coefficients, which is incompatible with bounded streaming.

2. **Huffman tables are "good enough"** - Tables are built from accumulated data only, not the full image. For images where accumulation covers >25% of blocks, tables should be near-optimal.

3. **Restart interval handling** - Restart markers are supported but must be configured before encoding starts.

## Research Findings

### Corpus-Trained Huffman Tables (2026-01)

Trained Huffman tables on CLIC 2025 validation set (32 images) at 30 quality levels.

**Key finding**: jpegli-style adaptive quantization (erosion + zero biasing) produces
a VERY different coefficient distribution than standard JPEG. The JPEG Annex K tables
have 4-11% overhead vs optimal for jpegli-encoded images.

Pre-trained tables achieve ~2-3% overhead consistently (vs 4-11% for Annex K).

Files:
- `zenjpeg/src/huffman/trained/mod.rs` - Pre-trained tables
- `zenjpeg/data/trained_tables/` - Raw frequency data and validation results

### Frequency Blending for Streaming Transitions (2026-01)

Tested whether blending partial image frequencies with pre-trained corpus frequencies
provides better Huffman tables during streaming transitions.

**Strategies tested:**
- **Partial**: Use only frequencies from rows seen so far
- **Trained**: Use pre-trained corpus frequencies
- **Blended**: Combine partial + trained prior for rare symbols

**Results** (10 images @ Q85):

| Coverage | Partial | Trained | Blended | Best |
|----------|---------|---------|---------|------|
| 25% | +1.9% | +2.3% | +2.8% | Partial |
| 40% | +0.7% | +2.3% | +0.9% | Partial |
| 50% | +0.5% | +2.3% | +0.7% | Partial |
| 75% | +0.1% | +2.3% | +0.3% | Partial |

**Conclusions:**
1. Partial frequencies win at all coverage levels
2. Blending with trained prior adds noise that hurts more than helps
3. At 40% coverage, partial frequencies reach <1% overhead vs optimal
4. Trained tables only useful as fallback at 0% coverage (start of image)

**Recommendation**: For bounded-memory streaming, use partial frequencies directly
without blending. Wait until 40%+ coverage before transition for <1% overhead.

Files:
- `zenjpeg/examples/test_frequency_blending.rs` - Test comparing strategies
- `zenjpeg/src/huffman/optimize/frequency.rs` - Added `blend_with_prior()`, `from_counts()`

## Future Work

1. **FFI parity tests** - Verify output matches C++ jpegli (blocked on submodule)
2. **Memory profiling** - Verify actual peak memory with heaptrack
3. **Benchmarking** - Measure performance impact of streaming mode
4. **Transition heuristics** - Wait until 40%+ coverage for optimal tables

## Files Modified

| File | Changes |
|------|---------|
| `zenjpeg/src/encode/strip/mod.rs` | +~200 lines (frequency counting, take_blocks) |
| `zenjpeg/src/encode/streaming.rs` | +~300 lines (transition, streaming mode) |
| `zenjpeg/src/foundation/bitstream.rs` | +~40 lines (flush_without_eoi, flush_restart_marker) |
| `zenjpeg/src/entropy/mod.rs` | +~60 lines (encode_block_to_writer) |
| `zenjpeg/examples/verify_inline_frequencies.rs` | New example file |

## Commands

```bash
# Build and test
cargo build --release -p zenjpeg
cargo test --release -p zenjpeg --lib --features "test-utils,decoder" -- streaming
cargo test --release -p zenjpeg --lib --features "test-utils,decoder" -- inline_frequency

# Run verification example (requires working build)
cargo run --release -p zenjpeg --example verify_inline_frequencies
```

## Original Problem Statement

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

**Solution**: Accumulate until memory threshold, then transition to streaming mode where blocks are encoded immediately after quantization.
