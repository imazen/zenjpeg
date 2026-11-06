# Native sans-I/O migration candidate

This standalone consumer qualifies the recovered `sansio` implementation against
zencodec main `1fb997bb`, pixels `fde59b96`, and the exact dependency lock. Run
outside workspace path overrides:

```sh
just sansio-native-check
cargo clippy --locked --manifest-path tools/sansio-validation/Cargo.toml -p zenjpeg -- -D warnings
cargo clippy --locked --manifest-path tools/sansio-validation/Cargo.toml --all-targets -- -D warnings
```

Install `djpeg` (libjpeg-turbo) and `exiftool`; missing oracles fail the tests.
The recipe copies `dependencies.lock` to the ignored `Cargo.lock` before testing.
The full reviewed JPEG graph now resolves and passes the complete user/default/FFI/
expert/recompress runtime gates plus strict linting, examples and doctests. The
standalone consumer remains a focused no-default-features + zencodec check; it
does not qualify every feature or platform. Its main graph uses the reviewed
linear-srgb2969ee84 feature-ownership fix. The separately tested PR199 export
retains its original linear528 graph and remains unmerged.

`JpegDecodeJob::into_incremental_decoder(&[PixelDescriptor])` returns the concrete
`JpegIncrementalDecoder`, which implements the current `step(Input, sink,
Option<&dyn Pulse>)` contract. This is a review API; the ordinary erased factory
continues using `WholeFileIncremental` until promotion is reviewed.

The explicit nearest-neighbor, ordinary eight-bit route can produce rows before
EOF. Default triangle, policy, CMS, crop, deep output and deblock configurations
use the existing whole-file adapter; unsupported native frame modes refuse.
Progressive input remains buffered. This is not a general native incremental
JPEG implementation.

Each step copies at most 4096 offered input bytes into retained header/scan
storage. Native row reconstruction retries an incomplete MCU row; output uses an
aligned scratch batch and one copy into at most one sink loan per step.
Metadata capture retains marker structure/payloads and discards entropy bytes,
then uses the existing metadata inspector. Its internal bytes never serve as an
encoded output, source identity, or source-offset report. Retention depends on
headers, entropy-row size and caller configuration; no operation memory bound or
throughput improvement is established here.

See [qualification](../../benchmarks/sansio_sync_2026-11-06.md) for failures,
separate main/PR199 graphs, exact scope and remaining gates.
