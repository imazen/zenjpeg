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
`metadata_corpus_prefetch` fetches metadata datasets serially before
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
preservation are integrated below. The recovered GContainer/lossless stack and HDR reconstruction color
interpretation are reconciled in the sections below.

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

The recovered GContainer/lossless work and HDR base-gamut reconciliation are
implemented below. The API findings section records the remaining design limits.

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

### Ultra HDR convenience-reader color

The Display P3 regression reproduced a BT.709 descriptor from
`decode_ultrahdr`, `decode_ultrahdr_hdr`, and the streaming reconstructor's
hardcoded gamut. These paths now use zenpixels' shared ICC resolver, retaining
primaries and transfer without a second profile parser. XYB decoder output
retains its sRGB interpretation; an ICC profile requiring a CMS is refused
instead of silently labeled sRGB. The existing Ultra HDR round-trip suite
and strict clippy pass with the new P3 regression (`just metadata-policy-check`).
No public signatures change; the descriptor and unsupported-profile behavior
changes are observable and must be retained in release notes.

## Recovered lossless container work

Lossless transforms and sequential/progressive restructuring carry secondary
JPEGs and rebuild the MPF index after emission. GContainer lengths are matched
by semantic rather than MPF order. Namespace validation and span editing use
zenmeta's `Directory`; no zenmeta nominal type appears in the JPEG public API.
Malformed/ambiguous directories fail rather than preserving stale lengths.
Decoder extraction accepts bounded, disjoint secondary ranges; default-mode
compatibility for embedded ranges is described below. Rewriting requires
secondary ranges after the primary EOI.
Nested secondary containers and proportional-crop mismatches are rejected.

The additive public entry point is
`lossless::apply_exif_orientation_with_edge_handling(&[u8], EdgeHandling, impl Stop)`.
The owner approved default edge rejection and the corresponding aligned layout
fixture on 2026-10-12. The original partial-MCU input remains an explicit rejection test.
`just metadata-lossless-check` passes the recovered coefficient, edge, MPF,
bounds and lossless unit suites. `just metadata-lossless-oracle` passes exact
jpegtran/djpeg comparison and ExifTool secondary-image reachability. The raw
and decoded gain-map tests also run with HDR rendering disabled.

### Native compatibility and recompression admission

Native and extended ICC reassembly share one chunk validator (`c7d341a9`).
Default native decoding additionally recognizes the complete single-profile
sequence-zero compatibility case described below. Other invalid chunk sets
return None through the published native Option accessor; inspection retains
its typed errors. Native extended-XMP concatenation remains
legacy behavior; strict extended inspection stays explicit.

The recovered lossless path preserves auxiliary JPEGs. Lossy recompression's
single-image strategies now refuse MPF/ISO/gain-map/container signaling and
concatenated images instead of silently dropping auxiliaries. This is an
accepted-input behavior change with no public signature change. It does not
implement lossy gain-map recompression. Existing recompression/ICC tests and
new auxiliary-image admission tests pass.

Validation on 2026-10-12: both complete library configurations pass (parallel/
HDR/codec and inventory/extended/HDR/codec). After owner approval, the layout
suite passes all 59 selected tests and all six edge tests. Lossless native/i686,
independent-reader, stable fuzz-seed replay, API snapshot, and recompression
admission checks have passed locally. Remote CI must validate the combined revision.

### Embedded MPF compatibility

The combined-CI regression used a synthetic layout: the test moves a corpus
Ultra HDR gain map into APP15 and appends an unrelated JPEG. It is not evidence
of an original real-world file with that layout. Default (Balanced), Lenient
and Permissive decoding retain embedded MPF JPEGs. Strict decoding rejects an
MPF secondary inside the primary, including early gain-map extraction. Native
extraction and inventory share the readable/disjoint-range admission predicate;
duplicate ranges still cannot multiply copied secondary bytes. The original
inventory assertion that the embedded image is the gain map remains intact.

Lossless transformation/restructuring independently refuses embedded MPF
images: those emitters cannot relocate the references inside the original APP
payload. Decoder acceptance therefore does not promise rewrite support.
`just metadata-mpf-compat-check` passes the original inventory review suite,
strict-mode parity, repeated/overlapping extraction bounds and lossless refusal.

### Native depth and MPF bounds

The new Dynamic Depth regression reproduced shifted extraction: padding bytes
were returned as the depth prefix and its tail was lost. Native extraction now
adds the primary item's padding before walking secondary resources, matching
[Dynamic Depth 1.0](https://developer.android.com/static/media/camera/camera2/Dynamic-depth-v1.0.pdf)
and the inventory. Oversized padding returns no depth result.
Native MPF's compatibility parser independently needed checked IFD and MP-entry
offsets; strict inspection's checked arithmetic had not covered it.
`just metadata-depth-bounds-check` passes native depth/bounds tests and the
native-parser overflow regression on i686, plus Clippy.

### Standalone markers and absent metadata

TEM is parameterless in both header and body parsing; inventory no longer
swallows the following bytes into its segment. Components decoding preserves
base pixels on ordinary XMP. The adapter retains the shared parser's distinction
between `NotUltraHdr` and malformed gain-map metadata instead of converting both
to decode failure. `just metadata-admission-check` covers both paths and their
inventory agreement. Existing assertions were updated with owner approval.

### Orientation selection

Native orientation and inventory now share first-EXIF selection through the
container marker iterator's private decoder-metadata mode. The public iterator
retains its structural stop-on-stray behavior. Selection handles fill bytes,
stray bytes and native SOS header sizing, includes EXIF after scan data, and
never searches inside another segment's payload. An orientation-free first
EXIF remains authoritative instead of falling through to another packet.
Streaming metadata reports the same packet and transformed dimensions.
`just metadata-orientation-check` passes the corpus inventory, orientation
reporting and byte-identical streaming/buffered pixel regressions.

The full inventory walker and structural inspector remain different consumers;
this removes their duplicated orientation lookup, not all marker parsing.

### Extraction policy

The new eight-combination policy regression failed on the original probe's
EXIF output with extraction disabled. A shared adapter helper now maps policy
to preservation for headers and decoding; the unconditional preserve-all
override is removed. Probe, buffered output, streaming metadata, native extras
and inventory agree. Gain-map resolution can disable XMP fallback in both the
primary and secondary packet while retaining ISO parameters.
`just metadata-extraction-policy-check` passes those combinations, orientation
and inventory regressions, gain-map policy tests and strict Clippy.

These flags govern metadata extraction, not privacy rewriting of the original
file or the opaque bytes of a preserved secondary JPEG. Required internal color
and orientation interpretation remains a separate concern; see the shared API
findings below before treating these flags as parser-isolation controls.

### HDR adapter reconciliation (2026-10-13)

The adapter's literal `hdrgm:` dispatch bypassed valid ISO-only and aliased-XMP
metadata. Reconstruction now decides from decoded extras using the native
ISO-first resolver. Ordinary images still return their base rendition. The
native and adapter outputs match byte-for-byte for XMP, ISO, both, a namespace
alias, and a Display P3 base at the same headroom. Mastering primaries follow
the reconstructed descriptor. Run `just metadata-hdr-dispatch-check`.

The existing `rgba_uhdr.jpg` corpus profile is Display P3 with sRGB transfer,
but was absent from zenpixels' identification table. Its generator admits the
profile with maximum u16 reference error 1 and all rendering intents; the fix
is on zenpixels main at `8f7ee871`. JPEG pins that unpublished revision. This
keeps color interpretation in the shared pixel crate and preserves the original
inventory tests. Profiles requiring a CMS still return an explicit error.

### API design findings from the codec integrations

- **Presence, validity, and support are different results.** Ordinary XMP is
  not an invalid gain map. Preserve zenmeta's absence/malformed distinction at
  the codec boundary; `Option<Result<...>>` accessors must not erase it.
- **Rendering must consume resolved metadata.** The ISO/prefix bug survived a
  correct shared parser because adapter dispatch duplicated its decision. Keep
  parsing and source selection together, and test adapter pixels against native
  output for each metadata representation.
- **Extraction and interpretation need distinct policies.** Denying returned
  ICC or EXIF is different from ignoring color or orientation while rendering.
  Current extraction flags filter probe/output/native extras. HDR refuses a
  denied ICC it needs, rather than silently assigning sRGB. A future shared
  policy should express these intents separately.
- **Compatibility, strict validation, and rewrite capability differ.** An
  embedded MPF image can be readable and bounded while unsafe to relocate.
  Share parsing and bounds checks, but pass validation policy explicitly. Do
  not make a lossless rewrite restriction a default decode rejection.
- **Truncation policy currently changes more than truncation.** JPEG maps
  `allow_truncated = false` to its full strictness mode. This also tightens
  marker/layout admission. That coupling remains and deserves a shared-policy
  review; this integration does not silently change its existing contract.
- **Normalization is observable API behavior.** zencodec EXIF uses bare TIFF;
  native JPEG extras retain the APP1 prefix. Signature-only semver checks do
  not catch this. The changelog and boundary tests must accompany the API audit.
- **Shared primitives do not yet mean one complete walker.** First-EXIF
  selection and MPF admission are shared. Native recovery, structural inspection,
  and inventory still have distinct walks; malformed DRI/DAC handling needs
  continued differential coverage. Native MPF's i686 overflow demonstrated why
  duplicated offset arithmetic is particularly costly.
- **Inventory needs its own limits and availability contract.** The codec set
  exposed mismatched input/part-limit expectations. Typed limits and feature-off
  errors belong to zencodec; the optional scanner must not require pixel decode.
- **Ownership boundaries work without exposing zenmeta types.** JPEG-owned
  wrappers and private conversions preserve independent versions. Reuse parser
  implementations and tests, not public implementation-crate types.

One implementation cost remains: explicit HDR requests for an ordinary image
can decode it once to resolve metadata and again for the requested base output.
A future metadata-only resolved-component view can remove that work without
reintroducing format-signaling heuristics; it must retain the same selection,
bounds, and policy semantics. No claim of a fully unified scanner is made here.

HDR rendering also now applies requested output gamut conversion after gain-map
application, with strided row conversion and fallible output allocation. The
progressive-image admission policy reaches this path too. Tests cover the exact
native-reconstruct-then-convert sequence and inventory/decode rejection parity.
This is another consequence of keeping output conversion separate from source
metadata interpretation; converting the base first changes the gain-map inputs.

### Real-file ICC compatibility (2026-10-13)

`jpeg-conformance/crash-repro/jpeg-decoder-257/amazed2k1424ktv.jpg` carries one
ICC chunk numbered 0 of 1. ExifTool reads its complete 3,144-byte sRGB profile.
Native decoding now retains that exact payload when its ICC header signature
and declared size establish a complete single profile. The shared strict chunk
assembler still supplies size/allocation checks; duplicate chunks do not take
this compatibility path. Strict decoding rejects sequence zero, and inventory
reflects that rejection. Structural inspection and rewriting stay strict.

`just metadata-legacy-icc-check` reuses the original corpus file, compares native
pixels with canonical sequence numbering, and retains every existing malformed
chunk assertion and the inspector's pinned corpus discrepancy. The earlier
question about changing the synthetic sequence-zero expectation is no longer
needed: its three-byte payload remains rejected. No new binary is committed.
