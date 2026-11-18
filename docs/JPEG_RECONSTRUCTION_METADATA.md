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
