# JPEG reconstruction scan metadata

`DecodeConfig::decode_coefficients_with_jbrd_metadata` retains one scan record
for each baseline or progressive Huffman SOS. Baseline records use Ss=0,
Se=63 and Ah=Al=0. Their reset-point array is empty; a trailing ZRL sequence is
retained by scan-block index so a reconstruction writer can reproduce the
original entropy bytes. A nonzero AC value ends that sequence. Indices count
all decoded data units, including interleaved components and MCU padding,
and continue across restart markers.

The shared baseline entropy kernel has a private tracking specialization.
The ordinary specialization retains its coefficient arithmetic. Tracking
allocations return the established allocation error. No public name or
signature is added.

The original pinned public baseline test reports zero scan records. The
encoder's independent 16x8 controls separately show a legal ZRL-before-EOB
input reconstructing incorrectly in both Rust and pinned native decoders;
the unmodified v0.12 source control reconstructs exactly. Progressive reset
and extra-zero-run controls already reconstruct exactly.

The first restored consumer control passes for one/three components, restart
boundaries, both padding-bit choices and ZRL followed by a nonzero AC value;
coefficient buffers equal the ordinary API. Log `zenjpeg-baseline-consumer-after-v2.log`:
23 seconds, peak RSS 1.06 GiB, minimum available 18,897 MiB. This is the frozen
encoder source graph. It does not qualify newer dependency cohorts, all
arithmetic/truncation modes or the full decoder.

Thirteen native v0.12 source controls also cover 10- and 11-bit Huffman codes,
nonzero AC before and after ZRL, multiple runs, exact block completion, gray
absolute sampling, restart markers and odd interleaved component grids. Their
ordinary and tracked coefficient buffers agree under Strict, Balanced, Lenient
and Permissive settings. Truncated scan prefixes retain the ordinary API's
acceptance/error policy and coefficients.

Frozen consumer coverage observes 23/23 dispatch regions, all three reset sites
and both increment sites (3/3 each), and 7/7 scan-record construction/emission
regions. Trailing-run storage is 13/18: five allocation-error regions remain
unexecuted. Four caller windows observe 11/12, 12/14, 12/12 and 14/15; missing
regions are error propagation and the strict missing-padding return. This is
selected baseline metadata coverage, not complete JPEG or decoder coverage.
The final capture is `zenjpeg-baseline-metadata-restored.log` (25 seconds;
peak RSS 1.05 GiB, minimum available 19,034 MiB).

Eight independent faults fail at runtime and restore the exact source: removing
each reset or run increment, holding the block index fixed, shifting stored
indices, and discarding the scan record. Retained objects, fault sources and
full logs are bound in `zenjpeg-baseline-metadata-faults` (199 seconds; peak RSS
1.06 GiB, minimum available 18,953 MiB).

The forward merge preserves current main's progress reporting and newer pixel
dependency family. An isolated consumer copies that source's complete pinned
patch table, then passes minimal/default/parallel/CMS scan checks and strict
consumer Clippy. Direct current-main minimal library Clippy and scoped format
checks also pass (`zenjpeg-current-main-consumer-v2.log`, 27 seconds; peak RSS
0.52 GiB, minimum available 20,016 MiB). The first consumer run failed two
new-test constant-chunk lints after passing runtime checks; `as_chunks` fixes
those without changing assertions. This does not qualify every optional API
or the complete current-main workspace. The encoder can pin the narrow
frozen-family backport while main keeps its newer dependency cohort.
