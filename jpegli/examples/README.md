# jpegli Examples

Debugging and comparison tools for jpegli development.

## C++ Parity Comparison

These tools compare Rust output against C++ jpegli for parity validation:

| Example | Description |
|---------|-------------|
| `compare_cpp_quant` | Compare quantization tables against C++ |
| `compare_dct` | Compare DCT implementations |
| `compare_coefficients` | Compare DCT coefficients between encoded JPEGs |
| `compare_decoded` | Compare decoded pixels |
| `quality_comparison` | Quality comparison between Rust and C++ |

```bash
cargo run --release --example compare_cpp_quant
cargo run --release --example compare_coefficients
```

## Encoder Comparison

Compare jpegli against other encoders (mozjpeg, etc.):

| Example | Description |
|---------|-------------|
| `compare_encoders` | Compare against mozjpeg |
| `compare_quality` | Quality/size comparison at various Q levels |
| `corpus_comparison` | Corpus-wide comparison with DSSIM/SSIMULACRA2 |
| `multi_codec_comparison` | Compare against CID22 dataset encoders |

```bash
cargo run --release --example corpus_comparison -- /path/to/images output.html
MAX_FILES=50 cargo run --release --example corpus_comparison -- /path/to/images output.html
```

## Decoder Validation

| Example | Description |
|---------|-------------|
| `decoder_compare` | Compare jpegli decoder vs jpeg-decoder |
| `validate_jpeg` | Validate JPEGs with third-party decoders |
| `compare_to_original` | Compare decoded output to original |

## XYB Mode

| Example | Description |
|---------|-------------|
| `xyb_comparison` | Compare XYB vs YCbCr with ICC handling |
| `compare_cms` | Compare lcms2 vs moxcms ICC transforms |

## Corpus Testing

| Example | Description |
|---------|-------------|
| `roundtrip_corpus` | Roundtrip test for corpus images |
| `huffman_corpus_validation` | Validate Huffman optimization on corpus |

```bash
cargo run --release --example roundtrip_corpus -- /path/to/images
```

## Reports

| Example | Description |
|---------|-------------|
| `low_q_report` | Generate low-quality comparison report |

## Debug Tools

| Example | Description |
|---------|-------------|
| `jpegli_debug` | **Unified debug CLI** - replaces 27 individual scripts |
| `debug_aq_values` | Debug adaptive quantization strength values |
| `debug_quant_field` | Debug quant_field computation |

### jpegli_debug Commands

```bash
# Show help
cargo run --example jpegli_debug -- help

# Trace encoding pipeline for an image
cargo run --release --example jpegli_debug -- trace image.png

# Compare Rust vs C++ encoding at Q100
cargo run --release --example jpegli_debug -- compare image.png 100

# Dump JPEG structure (markers, tables)
cargo run --release --example jpegli_debug -- dump image.jpg

# Analyze quality metrics
cargo run --release --example jpegli_debug -- analyze original.png encoded.jpg

# Show quant tables at quality level
cargo run --release --example jpegli_debug -- quant 90

# Analyze single block patterns
cargo run --release --example jpegli_debug -- block gradient
```

## Utilities

| Example | Description |
|---------|-------------|
| `create_test_jpeg` | Create test JPEG with mozjpeg |

## Archived

Old/obsolete debugging tools (27 files) were removed in commit that added this
README. They can be recovered from git history if needed. These were one-off
debugging sessions for investigating specific issues.
