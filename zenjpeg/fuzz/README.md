# jpegli Fuzz Testing

This directory contains fuzz targets for testing jpegli's decoder and encoder.

## Requirements

- Nightly Rust toolchain
- cargo-fuzz: `cargo install cargo-fuzz`

## Fuzz Targets

| Target | Description | Priority |
|--------|-------------|----------|
| `fuzz_decode` | Decode arbitrary JPEG data | **Critical** |
| `fuzz_read_info` | Parse JPEG headers only | High |
| `fuzz_truncation` | Every prefix of a stream: no panic, header dims, monotone accept, Strict==Balanced pixels (#92) | High |
| `fuzz_roundtrip` | Encode then decode with structured input | Medium |
| `fuzz_differential` | Compare against zune-jpeg | Medium |

## Running

```bash
# List available targets
cargo +nightly fuzz list

# Run primary decoder fuzzer
cargo +nightly fuzz run fuzz_decode

# Run with seed corpus
cargo +nightly fuzz run fuzz_decode corpus/seed/

# Run for a limited time (e.g., 60 seconds)
cargo +nightly fuzz run fuzz_decode -- -max_total_time=60

# Run with multiple jobs
cargo +nightly fuzz run fuzz_decode -- -jobs=4 -workers=4

# Minimize a crash
cargo +nightly fuzz tmin fuzz_decode <crash_file>
```

## Seed Corpus

The `corpus/seed/` directory contains initial test cases:
- `minimal_1x1.jpg` - Minimal valid 1x1 grayscale JPEG
- `flower_420.jpg` - 4:2:0 subsampling
- `flower_444.jpg` - 4:4:4 subsampling
- `flower_gray.jpg` - Grayscale
- `flower_progressive.jpg` - Progressive JPEG
- `flower_cmyk.jpg` - CMYK color space
- `1x1_exif.jpg` - 1x1 with EXIF/XMP metadata

## Adding More Seeds

Good seed files include:
- JPEGs from [codec-corpus](https://github.com/AcrossTheCloud/codec-corpus)
- Edge case images (1x1, max dimensions, odd sizes)
- Progressive vs baseline
- Various subsampling modes
- ICC profiles
- Restart markers

## Coverage

To see code coverage:

```bash
cargo +nightly fuzz coverage fuzz_decode
```

## Security Considerations

The primary goal is ensuring the decoder:
1. Never panics on malformed input
2. Never causes undefined behavior
3. Gracefully rejects invalid data
4. Matches behavior of reference decoders
