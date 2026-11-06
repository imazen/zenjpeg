# Sans-I/O recovery qualification, 2026-11-06

## Remaining gates

The full JPEG workspace does not resolve the selected pixel cohort: enabling its
UltraHDR SIMD graph selects stable-only `archmage >=0.9.30, <0.10.0`, conflicting
with the cohort's explicit beta.2 requirement. Root-lock regeneration fails
before compilation. Full unit/integration/features, public API snapshot and
platform CI therefore remain unqualified. Do not merge this review candidate
based only on the focused consumer.

Default triangle input remains whole-file backed. Native resumption is qualified
only for the explicit nearest-neighbor eight-bit fixture configuration. Deep,
CMYK, arithmetic, arbitrary transformed outputs and complete attached color/
completion details need their own controls. Progressive input remains buffered.
The root erased factory retains its existing implementation. No performance,
latency or whole-operation memory claim is made.

## Recovered graph and behavior

The forward merge retains published `sansio` ancestry `7c59f19e` and live main
`cae3a9f3`; it does not rewrite the published branch. Old push/pull consumer loops
are migrated to the shared slice, synchronous and Tokio drivers. Current
`Option<&dyn Pulse>` calls preserve the owned job Pulse with both `None` and an
explicit no-op. Native MCU work checks the composed controls.

The split-row path yields one sink loan per step. Failure poisons the decoder
before callbacks; ordinary errors release native input/scratch owners. Sink
errors remain typed and retain priority when the same callback signals timeout.
Concrete native ErrorKind and its whereat envelope survive typed and erased
steps. Successful Done repeats without further effects.

Failed-first controls found and repaired two native defects: incomplete-row
rollback omitted the coefficient scratch state; EOF truncation failed rather
than producing the same balanced-recovery samples as one-shot. A third control
found completion before late XMP discovery. The adapter now captures marker
payloads without entropy, inspects them through the existing metadata code, and
finishes discovery before Done. Captured bytes are never used for source offsets
or published as a JPEG. ExifTool independently finds the late title; the native
comparison retains exact XMP bytes. Libjpeg-turbo independently reproduces the
fixture samples with nearest-neighbor upsampling.

## Frozen qualification graphs

| Graph | Core family | Pixel family | garb |
|---|---|---|---|
| Main | `1fb997bb88e7405e6a315bf1f0b22f71e99be272` | `fde59b96632557592cffa9b7301acd8db96ed078` | `605276d7` |
| Unmerged PR199 compatibility only | `88fc5871f38f34153d1dfe77379c94e1a2abb35c` | `b6b12bbe702e7ca2f93db559858ae1f233a066ec` | `58ce9cad` |

PR199 was not merged or repinned into main. Each graph uses a separate export
and Cargo target. The native lock comparison changes exactly garb, zencodec,
zencodec-io, zencodec-testkit, zenmeta, zenpixels and zenpixels-convert. Each has
one core/pixel/Pulse/whereat identity; shared versions include enough and
almost-enough 0.4.5, how-far 0.1.0 and whereat 0.1.5. The small UltraHDR clone
compatibility repair is `aadf1021`, verified on its remote main with green CI.

Both graphs pass the shared incremental testkit, coroutine and all-feature I/O
library/integration suites, strict I/O/coroutine Clippy, WASM and freestanding
no-default checks. These are shared-framework results, separate from native
codec qualification. The focused native consumer passes its pixel, truncation,
early-row/backpressure, cancellation, source/sink cause, independent decoder,
late-XMP and synchronous/Tokio controls on both graphs. No assertions or ignores
were weakened. The consumer's strict native and harness Clippy pass on main. The selected native
library also compiles for wasm32-unknown-unknown (rc0, 9s, peak RSS 0.47 GiB);
this is a build check, not WASM runtime or native no_std qualification.

Commands are retained in `tools/sansio-validation` and `sansio-native-check`.
The independent tools were libjpeg-turbo 2.1.5 and ExifTool 13.50. ExifTool's
brute-force scan exposes parsed tags rather than the raw XMP block; the control
checks its title independently and checks exact bytes through native metadata.
Earlier raw-block oracle attempts remain failures in the local logs.

Builds ran serially through run-heavy on r7900x at 3 GiB/jobs2 with command
 timeouts. Shared main wrapper: rc0, 14s, peak RSS 0.30 GiB. Shared PR199 wrapper:
rc0, 48s, peak RSS 0.43 GiB. Native main test/lint wrapper: rc0, 5s, peak RSS
0.56 GiB. Native PR199 test wrapper: rc0, 16s, peak RSS 0.65 GiB. These measure
validation processes, not native memory requirements or codec speed.

Earlier failures are retained: nonexistent fixture feature, prerelease
resolution, old fallible-clone call, old pixel helper paths, adapter compile
errors, nearest-neighbor mismatch, truncated EOF mismatch, small pump allowance,
and late metadata. An initial shared all-target test command also ran executable
bench targets; it was terminated (rc143) and replaced by explicit library/tests.
Those partial timings are not qualification measurements.
