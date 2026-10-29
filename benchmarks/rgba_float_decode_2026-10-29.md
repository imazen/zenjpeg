# RGBA float decode negotiation

Before this change, the advertised RGBAF32 preference returned RGBF32 on both
baseline and progressive RGB JPEGs. The paired zencodec required-output path
correctly returned OutputDescriptor; the preference path exposed RGBF32. The
initial diagnostic is retained in jpeg-admission-matrix.log.

The adapter now appends exactly opaque alpha (1.0f32) to decoded RGB triples.
Allocation is fallible. Source color/metadata handling and RGB sample bits stay
unchanged. This does not add a whole-operation memory bound or pre-allocation
admission contract. No public signature changes.

The controlled_decode target passes all 17 tests, including a new baseline and
progressive regression comparing every RGB byte and alpha value, preference
ordering, metadata, dimensions, and checked required-output success. Native
library/test strict Clippy passes on the standalone pinned 0.2 pixel graph.
The paired 0.3 pixel consumer also passes its seven-layout/two-scan-mode matrix
and strict native-library/consumer Clippy.

Final command group: cargo test -p zenjpeg --features controlled-decode
--test controlled_decode --locked; cargo clippy with the same feature and
--lib --test controlled_decode -- -D warnings, then paired-consumer Clippy.
Run-heavy receipt: rc0, 8s, peak RSS 0.59GiB. Logs under
~/tmp/zenpixels-release-reconcile-20261029: jpeg-rgba-native4.log (final),
jpeg-rgba-native2.log (ambiguous test trait call), jpeg-rgba-native3.log
(constant chunks lint), jpeg-rgba-fix2.log (paired matrix pass followed by
unsupported dependency-package test invocation). Setup failures are retained.

## Ranked precision selection

A subsequent shared-consumer regression on e49c76f0 reproduced RGBF32 output
for `[RGB8_SRGB, RGBF32_LINEAR]`: the one-shot adapter enabled float output
when any preference was float. Precision now follows the existing first-supported
layout selector after the header supplies component count. This retains one
header parse and does not change the HDR reconstruction branch.

The regression checks RGB/RGBA, U8-first/F32-first and baseline/progressive
inputs against singleton-preference native decoding, including exact row bytes
and metadata. `just rgba-float-check` passes all 18 controlled-decode tests and
strict library/test Clippy. Receipt: rc0, 20s, peak RSS 0.68 GiB. C++ FFI parity
was not run; this change concerns shared adapter preference negotiation.
