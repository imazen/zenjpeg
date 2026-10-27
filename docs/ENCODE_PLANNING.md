# Effective-job planning and colour binding (zencodec adapter)

zencodec's shared hooks (ideal-api `f5a7fd46`, review API): `EncodeJob::plan_encode`
and `Encoder::begin_declared`.

```rust
use zencodec::encode::{EncodeDeclaration, EncodeJob, Encoder, EncoderConfig};
use zenjpeg::JpegEncoderConfig;

let job = JpegEncoderConfig::new().job();
// The colour the rows will carry, declared before any header byte exists.
let mut declaration = EncodeDeclaration::new(width, height, pixels.descriptor());
if let Some(context) = pixels.color_context() {
    declaration = declaration.with_color(context);
}
let plan = job.plan_encode(&declaration)?;
// ... admit the plan; retain this exact job and declaration ...
let mut encoder = job.encoder()?;
let started = encoder.begin_declared(&declaration)?;
plan.validate_started(&started)?; // before any row is pushed or byte sent
```

`plan_encode` and `begin_declared` run one preparation: validation, limits, colour
binding and the native builder's geometry and streaming eligibility. Planning
starts no pixel or coefficient storage. It can allocate configuration and
metadata, for example a synthesized ICC profile. `begin` and `begin_output` are
`begin_declared` without a colour context. JPEG never patches earlier bytes, so
seek-back access changes nothing. The plan is always one ordered row pass that
emits in that pass (`InputRequirement::Rows`, `OutputPhase::FinalPass`); only the
timing varies. Colour carriers are not reported yet (`carriers: None`, which means
unknown, not absent). Planning errors are the job's own `At<CodecError>`, boxed.

The plan reports byte availability only. It is not a memory bound, privacy
evidence or permission to apply backpressure. Input is always one pass of rows.
Progressive and optimized-Huffman configurations retain coefficients until
`finish`. An MPF index (gain map, depth map) is written at `finish`. The generic
job refuses gain maps. A pre-blur filters the whole image, so planning and row
encodes of RGB8 input refuse it; one-shot and accumulated encodes apply it.

## Colour

The written ICC describes the stored samples:

1. The pixels' colour context: an ICC profile (written verbatim), otherwise CICP.
2. Otherwise the descriptor's primaries when they are not BT.709 or unknown.
3. Otherwise the job metadata, as before. Metadata also wins over descriptor-only
   primaries. An explicit context that contradicts it refuses.

Linear input (f32, 16-bit linear) is stored sRGB-encoded, so its carrier is its
primaries with the sRGB transfer. A profile for the linear samples is not carried,
and an unidentified profile refuses. sRGB-encoded 16-bit input is decoded to
linear f32 on the way in and encodes at its stored values: one 16-row strip at a
time in a one-shot encode, each pushed batch in a row encode. XYB refuses non-sRGB
pixels. A policy that leaves no ICC for colour identified as non-sRGB refuses; an
unidentified metadata ICC can still be stripped.

`tests/source_color_binding.rs` (jpeg-decoder reads back samples and ICC) and
`tests/row_planning.rs` (3,332 plan-vs-startup cells, streamed header colour,
MPF, cancellation) hold these rules.

## Measured heap

`examples/encode_planning_probe.rs` under `scripts/profile-encode-planning.py`
(heaptrack event replay, cross-checked against `heaptrack_print`): 2048×2048, all
jobs and encoders retained at once, dev profile (opt-level 2). Results:
[`benchmarks/encode_planning_alloc_2026-10-27.json`](../benchmarks/encode_planning_alloc_2026-10-27.json).
These were measured with the inherent experimental `plan_rows` / `begin_declared`
(the JSON records the probe revision). The probe now calls the shared hooks, which
run the same preparation; the numbers were not re-measured after that port.

| Configuration | Colour | Jobs | Prepared | Planned | Begun |
|---|---|---:|---:|---:|---:|
| Fixed tables, sequential | sRGB | 1 | 75,464 B | 75,464 B | 1,836,476 B |
| Fixed tables, sequential | sRGB | 4 | 76,456 B | 76,453 B | 7,119,326 B |
| Fixed tables, sequential | P3 descriptor | 4 | 76,465 B | 82,558 B | 7,127,847 B |
| Progressive, optimized | sRGB | 4 | 76,462 B | 76,459 B | 58,071,436 B |
| Progressive, optimized | P3 context ICC | 4 | 76,960 B | 77,437 B | 58,073,854 B |

These are whole-process heap peaks. They are not RSS, not native-only totals, and
not bounds for other sizes or configurations. Equal peaks do not prove zero
planning allocations. A descriptor-only P3 plan synthesizes its profile.
