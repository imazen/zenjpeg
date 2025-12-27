# butteraugli Examples

Debugging and tracing tools for butteraugli parity development.

## Unified Debug Tool

| Example | Description |
|---------|-------------|
| `butteraugli_debug` | **Unified CLI** for perceptual quality analysis |

### butteraugli_debug Commands

```bash
# Compare two images
cargo run --release --example butteraugli_debug -- compare img1.png img2.png

# Analyze JPEG compression at specific quality
cargo run --release --example butteraugli_debug -- jpeg photo.png 90

# Sweep quality levels 50-100
cargo run --release --example butteraugli_debug -- sweep photo.png

# Test uniform color behavior
cargo run --release --example butteraugli_debug -- uniform 128 128 128
```

## Analysis

| Example | Description |
|---------|-------------|
| `analyze_score` | Analyze butteraugli scores in detail |

## Debugging

| Example | Description |
|---------|-------------|
| `debug_values` | Debug intermediate values |
| `test_uniform` | Test with uniform images |

## Tracing (for C++ parity debugging)

| Example | Description |
|---------|-------------|
| `trace_ac` | Trace AC components for uniform images |
| `trace_detailed` | Detailed computation tracing |
| `trace_full_diff` | Trace full diff computation |
| `trace_no_multires` | Test without multiresolution |
| `trace_opsin` | Trace OpsinDynamicsImage |
| `trace_uniform` | Trace values for uniform images |
| `trace_values` | General intermediate value tracing |

## Usage

```bash
cargo run --release --example analyze_score -- image1.png image2.png
cargo run --release --example trace_values
```
