# Optional per-call Pulse

`DecodeConfig::decode_with_pulse` and `scanline_reader_with_pulse` accept
`Option<&dyn how_far::Pulse>`. Use `None` without an additional call control,
or `Some(pulse)` to retain planning, sharing, reporting and typed cancellation.
Shared metadata services forward the original option. An absent per-call
control never disables the cancellation retained by a shared job.

## Qualified root graph

Source `265a0447` passed the following on Linux with core `feb09f9f` and pixels
`283655c5`, using the committed portable lock outside parent workspace patches:

```sh
cargo test --locked -p zenjpeg --features controlled-decode,metadata-rewrite,inventory --test native_hardening --test controlled_decode --test metadata_rewrite --test row_planning
cargo clippy --locked -p zenjpeg --features controlled-decode,metadata-rewrite,inventory --lib --test native_hardening --test controlled_decode --test metadata_rewrite --test row_planning -- -D warnings
```

Results: 50 passed, one inherited ignored case, strict Clippy passed. The
controls exercise optional native calls, retained cancellation, controlled
adoption, metadata rewrite and row-route policy selection. Pixel algorithms,
existing behavioral assertions and the ignored case were not changed.
These commands do not qualify the complete codec, new platform runs, the
standalone fuzz graph or generated public API snapshots.

The three-codec wrapper recorded `rc1 373s; peak-RSS 0.81GiB;
min-avail 13902MiB; peak-load 2.45`. JPEG steps passed; the aggregate failed
on bitmap Clippy. This is neither JPEG throughput nor its isolated memory use.
The combined result remains in the
[zencodec receipt](https://github.com/imazen/zencodec/blob/encoding-route/benchmarks/optional_pulse_2026-10-29.json).

Native PNG tests now use PNG `7b6d1436` and its required zenflate `a0ea0363`.
The old PNG 0.1.4 brought a second core and failed its fixed layout assertion;
that assertion was preserved. The paired metric revision `85577588` has the
same consumed crate source as `280b77d6` and permits the newer PNG dependency.
All root lock changes were explicitly reviewed, including the required rustix
update. The root resolves one shared core, pixel, converter and Pulse package.
The fuzz workspace needs its own paired pixel patches because dependency patch
tables do not propagate; that source update remains unqualified.

## Generated API follow-up

On `31b789f3`, `just api-doc` and `just api-doc-check` pass. The generated
surface changes exactly the two native helper arguments to `Option`; the
snapshot is committed. Existing ambiguous rustdoc links in `error.rs` still
warn and are not represented as a warning-free documentation build.

The standalone fuzz workspace failed dependency resolution before compiling:
its old lock retains pixels 0.2.16 while core requires 0.2.17. A targeted pixel
lock update is required; no tests were skipped or signatures weakened. The
follow-up wrapper recorded `rc1 62s; peak-RSS 0.62GiB; min-avail 14490MiB;
peak-load 0.99`, retaining this resolution failure separately from the passing
API generation/check steps.
