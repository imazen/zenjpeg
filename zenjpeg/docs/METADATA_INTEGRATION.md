# Shared metadata integration and published API audit

The integration combines [PR210](https://github.com/imazen/zenjpeg/pull/210),
[PR221](https://github.com/imazen/zenjpeg/pull/221) and
[PR225](https://github.com/imazen/zenjpeg/pull/225). JPEG framing, entropy
consumption, MPF placement and extended-packet reassembly stay in zenjpeg.
Generic gain-map XMP and GContainer XML delegate privately to zenmeta.
`XmpError`, `ContainerItem` and `ItemSemantic` remain JPEG-owned types;
private conversion helpers prevent zenmeta types leaking into codec signatures.

`extended-metadata` and `inventory` are separate default-off features. The
single-component metadata filter remains usable with neither feature enabled.
Inventory uses the configured job's input limits, policy and cancellation, and
returns a typed unsupported operation when unavailable. Part caps retain their
actual error type. The Ultra HDR scratch preflight propagates cancellation and
part-limit failures instead of treating them as an ordinary decode rejection.

The owner approved strict PR210 XMP behavior on 2026-10-12. Malformed values and
ambiguous XML no longer become defaults; namespace aliases remain valid. This
does not replace the released decoder's permissive extended-XMP concatenation
with the new inspector's strict reassembly. Those acceptance differences remain
covered by [the existing metadata tests and specification notes](METADATA.md).
The inventory's generated GContainer fixture now declares its XML namespaces;
its pinned part/disposition assertions are unchanged.

## Published API comparison

Baseline: [crates.io zenjpeg 0.8.4](https://crates.io/crates/zenjpeg/0.8.4).
`just metadata-api-check` builds rustdoc JSON directly with this workspace's
coordinated pins and compares against the downloaded published source. This
avoids the standalone-workspace resolution problem noted in older changelog
entries. It forces a patch comparison despite the existing 0.9.0 checkout.
The selected features are zencodec and ultrahdr, plus the new inventory and
extended-metadata features on the current side; this is not an exhaustive
comparison of every feature combination.

On 2026-10-12 cargo-semver-checks 0.50.0 passed 218 checks, failed five and
skipped 31. All reported breaks below already exist in parent main `b88f02e3`;
the integration does not modify these definitions:

| Check | Published API difference |
|---|---|
| Auto traits | `BytesEncoder`, `RgbEncoder`, `YCbCrPlanarEncoder` lose `Sync`, `UnwindSafe`, `RefUnwindSafe`; `JpegEncoder` loses `Sync` |
| Discriminants | Nine `DecodeWarning` variants from `PaddingBlockError` through `ExtraneousBytesSkipped` move up one; `EncoderFamily::Unknown` moves 7 to 8; `ProgressiveScanMode::ProgressiveSearch` moves 3 to 5 |
| Tuple variant shape | `QuantTableConfig::GlassaLowBpp` becomes a unit variant |
| Unit variant shape | `QuantTableConfig::{Jpegli,JpegliSharedChroma,MozjpegRobidoux}` gain fields |
| Removed methods | `ScanlineReader::{read_rows_ycbcr_planes,read_rows_planar_i16}` and `QuantTableConfig::custom_tables` |

The scanline removals are from `de79d22e`; quant-table changes are from
`22485f82`. Manual inspection also retains the already queued seven associated
error-type and four convenience-method return-type breaks (`a548bb98`); the
automated tool did not report those across dependency revisions. The published
0.8.4 source has no `container` module: the metadata additions being reconciled
here are unreleased. No extra version bump is introduced by this integration.

## Validation

The original container, metadata corpus, stable fuzz replay and inventory review
suites passed, including disabled support and the combined inventory/ultrahdr
build. The shared parser's original strictness tests run through the JPEG
wrappers. Doctests and strict clippy passed; the standalone inventory/metadata
fuzz targets compile. Public snapshots are regenerated.

The existing ExifTool 12.40 oracle passed on 2,216 files: 1,798 agreed exactly,
3,506 segments were identity-checked and 16,494 field values compared. The
remaining differences use the original pinned, documented settlement rules;
308 files were compared through the prefix before a settled divergence. The
APP12 regex differential checked 414,716 inputs with zero mismatches. The
separate inventory oracle checked 89 files, with one existing explained
difference and zero unexplained mismatches. These are structural/metadata
checks, not a claim of full pixel-decoder conformance.

The oracle version is deliberate: use ExifTool tag `12.40`, revision
`537cba56936c21b729aeed404b117e31c3c6fed0`, on PATH for `just metadata-oracle`.
Changing its pinned expectations requires re-settling the differences. Approved
fixtures are unchanged, with [hashes and source revisions](../fuzz/METADATA_FIXTURES.md).

## CI corpus provisioning

CI run `38253462548` exposed missing-corpus failures on Windows ARM and i686.
`metadata_corpus_prefetch` fetches all seven metadata datasets serially before
parallel tests. `CODEC_CORPUS_CACHE` is an explicit absolute path, mounted into
cross through `Cross.toml`, so cross does not need to fetch image-rs itself.
The previous two image-rs skips are removed. `just metadata-prefetch-check`
passed locally; Windows ARM and container execution still require remote CI.

## Additional metadata source reconciliation

The unmerged metadata stack ending at `3c8f4057` is separate from PRs210/221/225.
Its EXIF fixes `ea224f98` and `a76f7c41` are now integrated: encode prefixes
EXIF exactly once, and the zencodec adapter transports bare TIFF while native
`DecodedExtras::exif` retains the APP1 payload. This changes metadata bytes
relative to 0.8.4 and is recorded in the changelog, even though signatures are
unchanged. `just metadata-exif-check` passed, including existing independent
img-parts/kamadak readers, new round trips and strict clippy.

ISO-first gain-map parsing, ISO-only output and SDR-base metadata/color
preservation are integrated below. The interrupted GContainer/lossless stack
and HDR reconstruction color interpretation still require review.

## Reconciled ISO-first metadata stack

The existing pending source commits `1581bf08`, `da8d2982` and `e008374e`
are integrated with shared XMP delegation. Their original tests cover ISO-only,
XMP-only and combined output, conflicting metadata precedence, malformed-ISO
fallback, streaming readers, SDR caller metadata and ICC synthesis. The new
inventory check requires the selected secondary ISO record to be GainMap
metadata in Components mode and unconsumed in BaseOnly mode. A primary version
record is not interpreted as the secondary's full parameters.

`just metadata-iso-check` passes the native decode, roundtrip, inventory review
and clippy checks. Two existing ignored external HDR tests remain ignored;
this run is not independent HDR reconstruction validation. Existing independent
framing/metadata oracles and final cross-platform checks remain release gates.

Remaining reconciliation: interrupted GContainer/lossless work, HDR
reconstruction's base-gamut assumption, and mismatch reporting.
No claim is made that all JPEG metadata work is complete.

## Codec-owned metadata rewriting

The default-off `metadata-rewrite` feature exposes `zenjpeg::metadata::rewrite`
with zencodec request/plan/error types. The planner and existing framing and
Ultra HDR scrub assertions moved from zencodecs `aa21cba0`; tests now call the
codec directly. Neither zencodecs nor zenpipe is required.

`just metadata-rewrite-check` passes without default features. The gain-map
recipe verifies native full-HDR output, parameters, unchanged entropy scans,
MPF offsets and explicit XMP service behavior, and runs framing tests on i686.
`just metadata-rewrite-oracle` decodes both primary and gain-map JPEGs before
and after rewriting with libjpeg's `djpeg` and compares PPM bytes exactly;
ExifTool reports `Validate: OK` for the rewritten container.

The planner supports one primary plus one MPF gain map, and explicitly refuses
other MPF layouts. It is stricter than a display decoder: malformed authoritative
ISO or conflicting ISO/XMP parameters prevent rewriting, while a decoder can
fall back to XMP. Extended-XMP editing still requires reassembly support.

## Metadata carrier bounds and explicit gain-map policy

The carrier boundary regression reproduced a panic for an oversized prepared
comment and omission of request-level prepared EXIF. `716b0e13` forwards
`EncodeRequest::segments` and validates APP/COM sizes before encoder construction.
Raw EXIF larger than 65,527 TIFF bytes now returns an error instead of being
truncated. The new boundary tests use img-parts to compare retained bytes;
the existing 26 img-parts/kamadak metadata tests and strict clippy pass.

The unreleased `encode_with_gainmap_metadata` entry point from `ab83a512` is
restored with zenmeta filtering and the common format-aware assembler. Its
source metadata is authoritative; private segments from the encoder config
are not copied around that policy. ISO-only output stays ISO-only. Public
parameters use zencodec types and existing codec types, never zenmeta types.
