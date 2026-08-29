# zenjpeg Fuzz Testing

This directory contains fuzz targets for testing zenjpeg's decoder, encoder,
and JPEG container parsers.

**This is its own Cargo workspace** (`[workspace] members = ["."]`), excluded
from the repo root. Nothing at the root — not `cargo test`, not clippy —
compiles a line of it, which is exactly how it silently stopped resolving on
2026-08-29. See the `[patch.crates-io]` comment in `Cargo.toml` for why that
table has to be carried here, and note it needs the `../../../zenanalyze`
sibling checkout on disk.

## Requirements

- **Fuzzing**: nightly Rust + cargo-fuzz (`cargo install cargo-fuzz`)
- **Compiling** the targets: stable, nothing else. `just fuzz-check` (=
  `cargo check --all-targets` here) is what `.github/workflows/fuzz.yml` runs
  on every push and PR so API drift fails CI the day it happens.

## Fuzz Targets

All 13 are compiled by the CI gate on every push.

| Target | Description | Priority |
|--------|-------------|----------|
| `fuzz_decode` | Decode arbitrary JPEG data across every pixel format, output target, strictness and decode mode | **Critical** |
| `fuzz_decode_limits` | Decode under strict `max_pixels` + `max_memory` | **Critical** |
| `fuzz_read_info` | Parse JPEG headers only | High |
| `fuzz_truncation` | Every prefix of a stream: no panic, header dims, monotone accept, Strict==Balanced pixels (#92) | High |
| `fuzz_decode_paths` | Structured: varies decoder config *and* data via `arbitrary` | High |
| `fuzz_push_decode` | Streaming `decode_rows` / `decode_rows_f32` row callbacks | High |
| `fuzz_container_probe` | `container::probe` / `is_ultrahdr`, range + gain-map-presence invariants | High |
| `fuzz_container_marker` | `container::{iter, primary_bounds, find_jpeg_boundaries}` zero-copy span invariants | Medium |
| `fuzz_container_mpf` | `container::{parse_mpf, parse_mpf_segment}` + `create_mpf_header` roundtrip | Medium |
| `fuzz_container_xmp` | `container::{parse_xmp, parse_xmp_full}` on arbitrary UTF-8 | Medium |
| `fuzz_roundtrip` | Encode then decode with structured input | Medium |
| `fuzz_encode` | Encoder config matrix (grayscale / XYB / YCbCr, progressive, Huffman opt) | Medium |
| `fuzz_differential` | Compare against zune-jpeg | Medium |

## Regression seeds

`regression/` holds minimized inputs for bugs that have since been fixed.
They are replayed on **stable** by `zenjpeg/tests/fuzz_regression.rs` — every
seed through every entry point above, under the same tight limits — so
`cargo test -p zenjpeg` (and `just fuzz-regression`) gates them on every CI
run rather than only during hand-run fuzzing. Drop a new
`cargo fuzz tmin`-minimized crash file in there and it is picked up
automatically; no registration step.

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
