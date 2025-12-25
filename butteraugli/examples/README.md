# butteraugli Examples

Debugging and tracing tools for butteraugli parity development.

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
