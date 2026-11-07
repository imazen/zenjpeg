# Sans-I/O recovery qualification, 2026-11-06

## Remaining gates

The complete reviewed root graph resolves with exact remote source pins.
Full strict Clippy now PASSES for library, tests and benches with the unchanged
CI feature list, PNG owner facade candidate6b740345 and UltraHDR30249993.
The instrumented pinned C++ oracle is enabled. Complete release user/default/FFI/
expert/recompress runtime gates, supported examples/doctests and API snapshots
pass. Broad colorspace oracles pass; platform gates remain open. Keep the draft; no merge on a focused test or lint result.

The former PNG errors were fixed by its owning facade-only candidate atop
[manifest-only draft PR55](https://github.com/imazen/zenpng/pull/55).
That exposed JPEG HDR and test private-helper calls, now migrated to equivalent
public facades. The private HDR helper returns the existing At<CodecError>
envelope and retains typed converter causes and their original location traces,
including orientation fallback failures. An actual destination-refusal control
recovers the original ConvertError before/after erasure and checks every trace
frame. No public signature, unwrap or silent fallback was added. Original ICC,
orientation, dimension and pixel assertions remain unchanged.

All manifest-only cohort candidates preserve their original dependency floors,
source kernels and numeric assertions. Their temporary <=0.9.31-beta.2 caps
exclude stable0.9.31; replace them after the stable cohort publishes and passes
unchanged qualification. Live registry checks found stable0.9.30 only.

The root now pins reviewed remote source commits rather than local paths.
The lock was regenerated from committed 88635489 Cargo.lock outside ambient
workspace overrides, preserving unrelated versions. Exact removed/added package
identities are in [the lock delta](sansio_cohort_lock_2026-11-06.json).
There is one core/pixel/enough/whereat identity. The legacy HDR oracle retains
core0.4.1 separately from core0.6.0 and mozjpeg-rs retains archmage0.4.0;
these distinct major families are deliberate, not silently unified.

[UltraHDR draft PR43](https://github.com/imazen/ultrahdr/pull/43) separates
SIMD admission from BT.2446 feature forwarding. Exact contents first supply
bt2446a at e70065f; older same-version pixels do not. It pins reviewed fde59b96
and forwards the feature explicitly. Unchanged all-feature core numeric tests
pass on main and, in a distinct matching export, PR199. Main strict library/test
Clippy passes. The initial broader all-target benchmark deprecation was fixed
in a separate benchmark-only alias migration: exact beta.2 mul_add directly
delegates to mul_add_relaxed. A dev-only beta.2 edge leaves runtime floors
unchanged. Main all-target strict Clippy and doctests now pass; no public API,
rounding, numeric assertion or suppression changes. Whole
UltraHDR workspace/platform gates remain open; no independent landing.

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

## Standalone fuzz and FFI graph preparation

Remote fuzz CI on3a1627aa reproduced the PNG errors in seed replay, while the
separate fuzz-workspace compiler gate still failed stable-only SIMD resolution.
Its patch table now matches the reviewed main root cohort, and actual
`cargo check --manifest-path zenjpeg/fuzz/Cargo.toml --all-targets` passes
without removing a target or feature. Its regenerated lock is committed;
seed replay remains part of the still-blocked root suite.

The full FFI gate requires the exact C++ gitlink631732a5f73cd547255998478cf7b248e04a5880,
Highway457c891775a7397bdb0376bb1031e6e027af1c48 and reference
headers8ecba3647edb6dd940463fedf38ca33a8e2a73d1. These were independently
materialized for the export, rather than borrowing an unidentified sibling build.
The unchanged CI CMake configuration and both cjpegli/djpegli tool builds
succeed. Actual FFI suite qualification remains separate. `sansio-ffi-tools` retains the existing CI configuration.

PNG owner facade candidate6b740345e9550c72da462311024961a413a15abc is
directly atop the manifest-only repair. Actual full Clippy now compiles PNG and
reaches four JPEG HDR private-facade uses (two ICC synthesis calls and the HDR
orientation in-place/fallback calls). Those failures are retained history; the subsequent public-facade migrations
and complete strict Clippy now pass. No PNG source was edited by this lane.

## Full runtime and diagnostics follow-through

The unchanged complete user-feature release library/test gate passes
(jpeg-full-user-features-test.log rc0,415s). The unchanged complete default
release library/test gate also passes (jpeg-default-release-test.log rc0,187s).
Pre-existing ignores remain unchanged; these are command-scope results, not
claims that every ignored experiment or native configuration is qualified.
Strict default library/test/bench and true-default lib-only Clippy pass after
compiling private incremental helpers with their sole caller's zencodec feature.
No warning suppression or target exclusion was used.

The independent Encoder diagnostics CI workspace still resolved published
zensim0.2.7's stable-only SIMD edge. Its candidate now preserves the exact
published runtime source at2894042dc0a444dd4df7e6c311ddf30996781deb; every
Rust runtime source matches cargo-read0.2.7 byte-for-byte (aggregate SHA256
27611673b4c8a6befe4a7f91b1ce13708e6ac750bd867c10efdcf46d90677a81).
Only the two0.9.23 SIMD requirements change in reviewed
4255242af5e84cde7099fd5bc8b4bee47df93dd8. It keeps0.2.7's profile/model/API;
it does not substitute the research0.3 graph. The temporary beta.2 cap excludes
stable0.9.31 and requires the same post-publication removal/qualification follow-up.

The diagnostics workspace repeats exact reviewed main-cohort patches and retains
its seeded lock delta. Actual CI wasm-pack release/web build passes
(diagnostics-wasm-cohort-build.log rc0,28s), not just metadata resolution.
Broader viewer/browser gates remain
separate; no weights, scientific thresholds, fixture expectations or UI changed.

The typed error review also retained the established color-refusal category:
a real RowConverter CMYK→RGB refusal reproduced Internal(Bug) instead of the
existing CmsRequired category before correction. The private color mapper now
preserves CmsRequired plus the concrete ConvertError and original trace. Both
typed/erased controls pass (jpeg-color-category-before.log rc101;
jpeg-color-category-after.log rc0). No public API or pixel/threshold change.

The initial complete FFI invocation is NOT a passing gate: its library stage
passed, but the bundled stage was terminated withrc143 after139s and no reported
test failure. Its zero-RSS wrapper receipt is invalid, and the full log remains
retained. All children exited. The initial journal-only search found no OOM record,
but privileged kernel messages subsequently verified bundled-process cgroup
OOM kills for both3G attempts. DefaultOOMPolicy=stop terminated remaining scope
children, explainingrc143. Rerun the full gate at6G with two test threads; no
test, feature, assertion or existing ignore is removed. The wrapper receipts
from the killed attempts are invalid memory evidence.

The complete FFI gate subsequently PASSES at6G/jobs2 with
RUST_TEST_THREADS=2 (jpeg-full-ffi-test-3.log rc0,83s, run-heavy peakRSS1.39GiB).
Every original test/feature/assertion and pre-existing ignore remains. This
limits test concurrency only; the prior default-concurrency3G scope failures
remain recorded. These process resources are not codec memory bounds.

The strict expert/boundary-RD gate initially found eight deprecated vector
mul_add calls (expanded into sixteen diagnostics). They now explicitly name
mul_add_relaxed, exactly the operation to which the selected beta.2 method
delegates. No arithmetic order, runtime floor, hash constant, sample assertion
or threshold changed. Strict expert library/test Clippy passes; the existing
complete expert release test command, including frozen boundary hashes, remains
a distinct runtime gate.

The unchanged complete expert release library/test command passes, including
the frozen boundary-hash controls (jpeg-full-expert-test.log rc0,212s). The
remaining static recipe also passes (jpeg-static-gates.log rc0,122s): full
library/test/bench and recompress strict Clippy, examples in default and all
user configurations including layout, and both default/user-feature doctests.
Existing ignored documentation examples remain unchanged; this does not claim
those ignored examples executed. No hash, sample or quality expectation changed.

Recompress's original API and filtered library runtime commands pass unchanged
(jpeg-recompress-runtime.log rc0,49s). Frozen legacy0.2.7 native numeric controls
pass with explicit beta.2 and the transfer-enabled dependency closure: default
metric configuration rc0,44s; matching viewer no-default configuration with
linear default/avx512/std/transfer rc0,48s. Runtime Rust/profile/weight sources
remain identical to the published0.2.7 contents; optional zenpixels interop and
ignored corpus studies were not promoted as qualified.

The separate linear528 std-only export fails six unguarded PQ imports. That
feature-off result remains failed; the actual viewer enables transfer, verified
from its resolved feature graph. No production feature or test assertion was
changed to hide the failure. Whole dependency release qualification stays open.

The original API check detects expected snapshot drift. Regeneration with
zenutils-apidoc0.1.0 then fails on featureless zenyuv's virtual manifest. Its
published0.1.1 uses disambiguated cargo rustdoc per package; update the isolated
API tool rather than excluding a crate or disabling the guard.

## Complete API and frozen feature ownership

The API tool0.1.0 fails on featureless zenyuv's virtual manifest. Its published
0.1.1 invokes cargo rustdoc with the exact package, retaining every package and
feature configuration. After that tooling repair the real default zenyuv graph
fails on linear528's unguarded private transfer imports. Review ref
[2969ee84](https://github.com/imazen/linear-srgb/commit/2969ee846e92a4f00f34ab447edaf1eb6857ebe0)
reuses existing main506004d9 feature ownership and cb3ef992's equivalent
NaN/nonpositive predicates, on frozen528. Arithmetic expressions, floors,
rounding and expectations remain unchanged. Feature-off std105 library tests,
all-feature165 and all3 existing tier tests pass, as do fmt and strict Clippy
for both feature configurations (rc0,22s, peakRSS0.20GiB). Later main arithmetic
changes are outside this graph. Full platform/assembly gates remain separate.

The root lock delta from the previous reviewed graph is only linear-srgb's
source identity. Standalone fuzz/viewer/main consumer locks have the same single
identity delta. Main/PR199 core and pixel identities remain distinct; the older
PR199 qualification retains its recorded linear528.

Full API regeneration passes (jpeg-api-regenerate-3.log rc0,16s); strict check
passes (jpeg-api-check-2.log rc0,9s), both with pinned nightly2026-07-13. Supported
default and internal API surfaces are unchanged. The feature surface adds only
JpegIncrementalDecoder, its explicit constructor and IncrementalDecode impl,
with the recorded auto-trait exceptions. Recovery helpers stay private; unused
new helper methods were removed. No pre-existing public method was removed.

After private-helper scoping and the linear pin, full original strict Clippy,
default strict library/test/bench Clippy and all11 standalone native controls
pass (jpeg-final-cohort-check.log rc0,42s, peakRSS0.88GiB). The latter include
independent djpeg/ExifTool and exact pixels, early rows, both stop reasons,
whereat causes and synchronous/Tokio consumers. Full runtime suites recorded
above precede these visibility/feature-guard-only edits; no active arithmetic
changed.

The complete existing colorspace_oracles target passes both controls with no
filtered/ignored tests (jpeg-colorspace-oracles.log rc0,100s, peakRSS2.04GiB).
All three independent readers run: Pillow/libjpeg-turbo2.1.5, IJG9f and the
exact pinned jpegli binary. Existing exact/one-code thresholds are unchanged.
The test-only IJG source comes from https://www.ijg.org/files/jpegsrc.v9f.tar.gz,
SHA25604705c110cb2469caa79fb71fba3d7bf834914706e9641a4589485c1f832565b.
The build helper pins that archive and verifies a real decode plus Pillow's
libjpeg-turbo feature. An initial unsupported `djpeg -version` probe returned1;
the corrected real-decode probe and the complete oracle suite both pass.
These are color-model fixture controls, not full native incremental color
qualification or a default-triangle promotion.
