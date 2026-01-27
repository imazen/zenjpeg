# Context Handoff - Semi-Optimal Huffman Tables for Streaming JPEG

## Problem Statement

Bounded-memory streaming JPEG encoding faces a chicken-and-egg problem:

1. **Optimal Huffman tables** require seeing ALL image data first (two-pass encoding)
2. **Standard JPEG tables** work for any image but have 5-10% overhead
3. **Partial-data optimization** (build tables from first N% of image) fails on "pathological" images where early content differs from later content

We want tables that are:
- Usable immediately (no buffering the whole image)
- Better than standard tables (< 5% overhead)
- Robust across diverse image content

## Solution: Corpus-Based "Universal" Tables

Build Huffman tables from frequency distributions aggregated across a representative corpus. These tables can be used for any new image with predictable overhead.

### API Implemented

```rust
use zenjpeg::encode::{HuffmanFrequencyCounts, StreamingEncoder, EncodingResult};

// Step 1: Collect frequency counts from corpus
let mut corpus_counts = HuffmanFrequencyCounts::new();
for image_path in corpus {
    let result: EncodingResult = encode_image(image_path)?;
    corpus_counts.add(&result.frequency_counts);
}

// Step 2: Generate tables from combined counts
let universal_tables = corpus_counts.generate_tables()?;

// Step 3: Use for streaming encoding
let encoder = StreamingEncoder::new(width, height)
    .quality(Quality::ApproxJpegli(85.0))
    .memory_limit(1024 * 1024)  // Bounded memory
    .custom_huffman_tables(universal_tables)
    .start()?;
```

### Types

```rust
/// Frequency counters for all 4 Huffman tables
pub struct HuffmanFrequencyCounts {
    pub dc_luma: FrequencyCounter,    // 12 symbols (categories 0-11)
    pub ac_luma: FrequencyCounter,    // 162 symbols (EOB, ZRL, run/size)
    pub dc_chroma: FrequencyCounter,
    pub ac_chroma: FrequencyCounter,
}

impl HuffmanFrequencyCounts {
    pub fn new() -> Self;
    pub fn add(&mut self, other: &Self);           // Combine counts
    pub fn generate_tables(&self) -> Result<OptimizedHuffmanTables>;
}

/// Result from finish_with_tables()
pub struct EncodingResult {
    pub jpeg: Vec<u8>,
    pub frequency_counts: HuffmanFrequencyCounts,
    pub huffman_tables: OptimizedHuffmanTables,
}
```

### Builder Methods

```rust
StreamingEncoder::new(w, h)
    // Use pre-built tables directly (highest priority)
    .custom_huffman_tables(tables)

    // OR: Generate tables from provided counts
    .custom_frequency_counts(counts)

    // OR: Use JPEG standard tables (fallback)
    .use_standard_huffman_tables(true)

    .start()?;
```

## Key Findings

### Partial-Data Optimization Fails for Some Images

Testing on CLIC 2025 validation set (32 images):

| Transition % | Failures (>4% overhead) | Max Overhead |
|--------------|------------------------|--------------|
| 15% | 4/32 | 18.2% |
| 25% | 2/32 | 14.6% |
| 50% | 0/32 | 3.6% |

**Root cause**: Some images have frequency distributions that continue changing throughout. Early data isn't representative.

Example pathological image:
- 25%: KL divergence 0.015, passes heuristics
- 100%: KL divergence 0.55, significantly different distribution

### Heuristics Don't Catch All Cases

Current heuristics (entropy + symbol coverage) detect "not enough data" but NOT "unrepresentative data":
- Low entropy → likely smooth region → wait for more data ✓
- Low coverage → few symbol types seen → wait for more data ✓
- High entropy + high coverage but WRONG distribution → not detected ✗

### Standard Tables Baseline

JPEG standard tables overhead varies by content:
- Photographic: 5-8%
- Graphics/text: 8-12%
- Noise/high-frequency: 3-5%

## Research Questions

1. **Optimal corpus size**: How many images needed for convergent tables?
2. **Quality-specific tables**: Should tables differ by quality level?
3. **Content-specific tables**: Separate tables for photos vs graphics?
4. **Adaptive quantization impact**: Does jpegli's AQ + zero-bias change optimal tables?

## Files

### Core Implementation
- `zenjpeg/src/encode/streaming.rs` - HuffmanFrequencyCounts, EncodingResult, builder methods

### Examples
- `zenjpeg/examples/custom_huffman_tables.rs` - Demo of corpus-based tables
- `zenjpeg/examples/compare_table_strategies.rs` - Compare optimized vs standard
- `zenjpeg/examples/analyze_distribution_change.rs` - KL divergence analysis
- `zenjpeg/examples/compare_min_thresholds.rs` - Transition % comparison

### Test Data
- CLIC 2025 validation: `/home/lilith/work/codec-corpus/clic2025/validation/` (32 PNG images)

## Commands

```bash
# Run custom tables demo
cargo run --release -p zenjpeg --features test-utils --example custom_huffman_tables

# Compare table strategies on CLIC corpus
cargo run --release -p zenjpeg --features test-utils --example compare_table_strategies

# Analyze distribution stability
cargo run --release -p zenjpeg --features test-utils --example analyze_distribution_change

# Run streaming threshold tests
cargo test --release -p zenjpeg --features test-utils --test streaming_threshold -- --nocapture --ignored
```

## Next Steps

### Immediate
1. Build tables from CLIC 2025 training set (larger corpus)
2. Measure overhead on validation set with corpus tables
3. Compare: corpus tables vs partial-25% vs partial-50% vs standard

### Research
1. Test if quality level affects optimal table structure
2. Measure whether AQ/zero-bias changes symbol distribution significantly vs libjpeg
3. Consider clustering images by content type for specialized tables

### Production
1. Ship default "universal" tables derived from large corpus
2. Allow users to provide custom tables for domain-specific optimization
3. Document expected overhead ranges for different scenarios
