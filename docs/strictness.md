# JPEG Decoder Strictness Comparison

Comparison of marker parsing and validation behavior across libjpeg-turbo (mozjpeg),
zune-jpeg, and zenjpeg's four decode strictness levels.

Source analysis from `jdmarker.c`, `jdinput.c` (libjpeg-turbo/mozjpeg),
`zune-jpeg 0.5.x` (`decoder.rs`, `headers.rs`, `marker.rs`),
and zenjpeg (`decode/parser/markers.rs`, `decode/parser/scan.rs`, `decode/parser/mod.rs`).

## Architecture

| | libjpeg-turbo | zune-jpeg | zenjpeg |
|---|---|---|---|
| **Language** | C | Rust | Rust |
| **Input model** | Streaming with suspension | In-memory, `Result` on EOF | In-memory slice, `Result` on EOF |
| **State machine** | Resumable (`unread_marker` persists) | Single forward pass | Single forward pass |
| **Strictness config** | None (hardcoded behavior) | `strict_mode()` bool | 4-level `Strictness` enum |
| **Error recovery** | `WARNMS` (continue) vs `ERREXIT` (fatal) | Always error | `Strict`/`Balanced`/`Lenient`/`Permissive` with warning collection |
| **Decode paths** | 1 (all modes through same loop) | 1 (baseline + progressive) | 5 (baseline streaming, baseline buffered, parallel fused, progressive, arithmetic) |

## Supported JPEG Modes

| SOF | libjpeg-turbo | zune-jpeg | zenjpeg |
|-----|--------------|-----------|---------|
| SOF0 Baseline | Yes | Yes | Yes |
| SOF1 Extended Sequential | Yes | Yes (treated as baseline) | Yes (treated as baseline) |
| SOF2 Progressive | Yes | Yes | Yes |
| SOF3 Lossless Huffman | Yes | Error | Error |
| SOF5-7 Differential | Error | Error | Error |
| SOF9 Arithmetic Sequential | Yes | Error | Yes |
| SOF10 Arithmetic Progressive | Yes | Error | Yes |
| SOF11 Arithmetic Lossless | Yes | Error | Error |
| SOF13-15 Diff Arithmetic | Error | Error | Error |
| **Precision** | 8, 12, 16-bit | 8-bit only | 8 and 12-bit |

## Marker-Level Structural Validation

| Marker/Check | libjpeg-turbo | zune-jpeg | zenjpeg |
|---|---|---|---|
| **SOI validation** | First 2 bytes must be `FF D8`, no junk | First 2 bytes must be `FF D8` | First 2 bytes must be `FF D8` |
| **Duplicate SOF** | Fatal | Fatal | Fatal |
| **Duplicate SOI** | Fatal | Not checked | Not checked (SOI not in main loop) |
| **SOF length mismatch** | Fatal | Fatal | Fatal |
| **Sampling factor = 0** | Fatal (`JERR_BAD_SAMPLING` in `initial_setup`) | Fatal | Fatal |
| **Sampling factor > 4** | Allowed (max factor check is > `MAX_SAMP_FACTOR`) | Allowed | Fatal (> 4 rejected) |
| **Quant table idx >= 4** | Not validated at SOF parse time | Fatal | Fatal |
| **Zero quant value** | Allowed (division by zero in dequant) | Allowed | Fatal (caught during DQT parse) |

## Table Markers (DQT, DHT, DAC, DRI)

| Check | libjpeg-turbo | zune-jpeg | zenjpeg |
|---|---|---|---|
| **DQT precision > 1** | Not checked (inferred from length) | Fatal | Fatal |
| **DQT length mismatch** | Fatal | Fatal | Fatal |
| **DQT table idx >= 4** | Fatal | Fatal | Fatal |
| **DHT table class > 1** | Implicitly handled (0x10 bit) | Fatal | Fatal |
| **DHT symbol count > 256** | Fatal | Fatal | Validated via length |
| **DHT length mismatch** | Fatal | Fatal | Fatal |
| **DAC marker** | Parsed (if compiled in) or skipped | Fatal ("not supported") | Parsed (L/U and Kx conditioning) |
| **DAC L > U** | Fatal | N/A | Fatal |
| **DRI length != 4** | Fatal | Fatal | Not validated (reads 2 bytes regardless) |
| **DNL marker** | Skipped (`skip_variable`) | Fatal ("not supported") | Parsed (updates height if SOF height=0, warns on conflict) |

## SOS (Start of Scan) Validation

| Check | libjpeg-turbo | zune-jpeg | zenjpeg |
|---|---|---|---|
| **SOS before SOF** | Fatal | Implicit failure | Implicit failure (no components to match) |
| **SOS length mismatch** | Fatal | Fatal | Not validated (trusts num_components) |
| **Component ID not found** | Fatal | Fatal | Fatal |
| **Duplicate component in scan** | Fatal | Fatal | Not checked |
| **Huffman table idx >= 4** | Not validated (table used later) | Not validated per se (index stored) | Fatal |
| **Ss > 63** | Not checked (deferred to entropy) | Fatal | Fatal |
| **Se > 63** | Not checked (deferred to entropy) | Fatal | Fatal |
| **Ah > 13** | Not checked | Fatal | Fatal (validated, commit d12d699) |
| **Al > 13** | Not checked | Fatal | Fatal (validated, commit d12d699) |

## APP Markers

| Check | libjpeg-turbo | zune-jpeg | zenjpeg |
|---|---|---|---|
| **APP0 (JFIF)** | Special: parse JFIF header, extract density/version | Special: check for AVI1 (MJPEG), skip rest | Skip (no JFIF parsing) |
| **APP1 (EXIF/XMP)** | Skip (or save if configured) | Special: parse Exif and XMP | Preserved if configured |
| **APP2 (ICC/MPF/HDR)** | Skip (or save if configured) | Special: parse ICC chunks, HDR gainmap, MPF | ICC extracted upfront; MPF/HDR preserved if configured |
| **APP13 (IPTC)** | Skip | Special: parse IPTC (Photoshop 3.0) | Preserved if configured |
| **APP14 (Adobe)** | Special: parse Adobe color transform | Special: parse Adobe color transform | Special: parse Adobe color transform |
| **APP3-12, APP15** | Skip (or save if configured) | Mapped to `UNKNOWN` in `from_u8()`, then skipped | Skip |
| **Unknown transform in APP14** | Stored as-is (no validation) | Fatal (only 0,1,2 accepted) | Mapped to `Unknown` (not fatal) |

## Unknown/Reserved Markers

| Marker type | libjpeg-turbo | zune-jpeg | zenjpeg |
|---|---|---|---|
| **DHP (0xDE)** | Fatal | Skipped | Skipped |
| **EXP (0xDF)** | Fatal | Skipped | Skipped |
| **JPG0-JPG13 (0xF0-FD)** | Fatal | Skipped | Skipped |
| **Other unknown** | Fatal | Skipped (warn) | Skipped |
| **Stray RST between scans** | Logged (parameterless) | Not handled | Silently skipped |

libjpeg-turbo treats reserved markers as fatal ("likely to be used to signal
incompatible JPEG Part 3 extensions"). Both Rust decoders skip them.

## Inter-Marker Junk Bytes

| Behavior | libjpeg-turbo | zune-jpeg | zenjpeg |
|---|---|---|---|
| **Non-FF bytes before marker** | Counted in `discarded_bytes`, warned | >3 bytes: strict mode = fatal, else logged | Silently skipped (no count/warn) |
| **FF padding bytes** | Silently consumed | Silently consumed | Silently consumed |
| **FF 00 stuffed zeros** | Discarded, loop retries | Consumed in fill-byte loop | Treated as byte-stuffing, outer loop retries |
| **EOI before any SOS** | Graceful: `JPEG_REACHED_EOI` (tables-only stream) | Fatal ("Premature End of image") | Fatal ("unexpected EOI before frame header") |

## Error Recovery (Mid-Scan)

| Situation | libjpeg-turbo | zune-jpeg | zenjpeg |
|---|---|---|---|
| **Truncated scan data** | Fills zeros via `insufficient_data` | `ExhaustedData` error | **Balanced/Lenient/Permissive**: fills zeros, collects `TruncatedScan` warning. **Strict**: error |
| **Missing Huffman table** | Uses standard tables (compiled in) | Error | **Balanced/Lenient/Permissive**: uses ITU-T T.81 K.3 standard tables, warns. **Strict**: error |
| **Wrong restart marker** | 3-action resync (discard/scan/leave) | Not explicitly handled | **Lenient**: warn. **Permissive**: resync forward. **Strict/Balanced**: error |
| **RST sequence wrong** | 3-action resync | Not handled | **Permissive**: accept any RST value. **Others**: error |
| **AC index overflow** | Tolerated silently (`jpeg_natural_order` carries 16 dummy slots; value bits consumed, block ends) | Not recovered | **Balanced/Lenient/Permissive**: consume value bits, treat as EOB, `AcIndexOverflow` warning — on every decode path since #92 (the fast_ac path had tolerated it silently while the bit-by-bit path errored). **Strict**: error |
| **Invalid Huffman code** | `insufficient_data` flag, fills zeros | Error | **Lenient/Permissive**: treat as EOB, warn |
| **Padding block decode error** | Fills zeros | Error | **Balanced/Lenient/Permissive**: fills zeros, warns |
| **Zero quant value** | Allowed (division by zero in dequant) | Allowed | **Permissive only**: clamp to 1. **All others**: fatal |
| **Malformed segment** | Varies | Fatal | **Permissive only**: skip. **All others**: fatal |
| **Bad Huffman table idx** | Implicit | Fatal | **Permissive only**: clamp to 0. **All others**: fatal |
| **Malformed DNL** | Skipped | Fatal | **Permissive only**: skip. **All others**: fatal |

## Summary

**libjpeg-turbo**: Battle-hardened reference. Fatal on reserved/unknown markers (paranoid
about future incompatibility) but forgiving about data corruption (restart marker resync,
extraneous byte tolerance, tables-only streams). "Decode anything that might be JPEG."

**zune-jpeg**: Binary strict/lenient. Rejects the most modes (no arithmetic, no lossless,
no 12-bit, no DNL, no DAC). Validates the most SOS parameters (Ah/Al bounds). Most
forgiving about unknown markers (skips everything). Prioritizes fast failure on unsupported input.

**zenjpeg**: Four strictness levels (`Strict`, `Balanced`, `Lenient`, `Permissive`) with
warning collection. Supports the most modes (arithmetic sequential/progressive, 12-bit, DNL).
Validates the most at the structural level (zero quant values, sampling factor bounds, table
indices in SOS). `Balanced` targets libjpeg-turbo compatibility, `Strict` exceeds it,
`Lenient` goes beyond for damaged files, and `Permissive` clamps/skips errors that would
be fatal in all other modes (zero quant values, bad Huffman indices, malformed segments).
