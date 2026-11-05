# Qualified JPEG sibling graph (2026-11-05)

Base: `762fa7128c2124b42cfa6e23b30024a688f5b7a3`.
Root and standalone fuzz use core `1816b4d082234c04c3805411b437b8396040f522`
and pixels `fa836abf72afe9826e866dc25712b7c800323962`.
Root PNG is `02a7f0df5f72dd79b0afa58d66ced7cdd8395e31`; UltraHDR core is
`85cc309c07526341e6fa8b7e4d6130aaf64c8fea`. Its old/new source diff contains
only manifests, lock and changelog changes. Both research dev-tool pins use
qualified zensimx `02243a4ba84ed71611f2cf445553eb4d41c2fbac`.

Published zenresize0.3.1 requires zenpixels^0.2.10. Root-patch its existing
`e3975fb9d6d6b7baa96038a0eb8e27febb37c012` source, which admits0.3, and
refresh the old pixel lock. Both inverse production queries resolve one
fa836abf pixel source and one1816b4d0 core source. Dependency patches do
not propagate from zensimx. Root and fuzz lockfiles are retained.

Validation on a Ryzen9 7950X under WSL2, isolated source/target and shared lock:

- Default: 2,466 passes, zero failures,112 existing ignores across69 targets.
- Production features parallel,moxcms,ultrahdr,controlled-decode,inventory,layout,
  recompress-expert:2,954 passes, zero failures,118 existing ignores across70
  targets. Quality-floor, size-regression, dispatch and content-checksum gates pass.
- Doctests:16 pass,110 existing ignores.
- Explicit f32 layout rerun:30 library and63 integration tests pass, zero failures.
- Strict library/test/example/bench clippy passes; standalone fuzz all-target
  check passes. No warning is allowed past the Rust lint gate.
- Existing public-API snapshots match under CI's nightly-2026-07-13 toolchain:
  one runner test passes; no snapshot or API expectation is changed.

Default+production+docs+first lint scope:
`rc=101 702s | peak-RSS 1.35GiB | min-avail 6662MiB | peak-load 4.30`;
the nonzero result is two example-format lints after runtime/docs passed.
The focused layout tests pass in a74s scope (peak RSS1.23GiB) that then
stopped on another example iterator lint. Collecting the remaining independent
example diagnostics took13s, peak RSS0.91GiB. Final strict lint+fuzz+API scope:
`rc=0 60s | peak-RSS 0.91GiB | min-avail 7659MiB | peak-load 3.48`.

Maintenance makes the compiler's already-inferred layout quality type explicitly
f32, removes redundant formatting borrows and uses equivalent fixed-size chunk
iteration in comparison tools. Numeric values, assertions, thresholds and
existing ignores are retained. No comparison tool or old encoder sweep was run
to regenerate historical results. The optional C++ FFI parity suite remains
outside these selected features; its submodule is absent.

The first attempted graph failed before tests with two dev-tool PixelBuffer
errors at85577588; the approved prerequisite fixes that hold. A subsequent
production build found nine HDR type errors before the resize patch/lock
refresh. Both failures and all successful stage logs are retained.
Baseline formatting+library clippy:rc0 12s, peak RSS0.54GiB.

Raw logs, per-target results, resolved locks, source patch and hashes, checked
on dev: `/mnt/v/output/zensim-siblings-20261105/zenjpeg/`.
Tower mirror: `/mnt/tower/output/zensim-siblings-20261105/zenjpeg/`.
