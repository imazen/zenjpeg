# JPEG metadata model (`container::metadata`, feature `extended-metadata`)

`zenjpeg::container::metadata::JpegMetadata` walks a JPEG buffer once and
returns every structural unit in file order as an `Item`: marker segments,
appended data after the primary image's EOI (trailers), and runs of bytes the
walk could not structure. Items tile the input: offsets are contiguous from 0
and the lengths sum to the input length. Nothing is dropped.

The planned direction is one JPEG walker for the crate: the item list is to
be built from a walker that follows the decoder's own marker loop, so that
metadata reading and decoding see one structure, and the decoder's metadata
readers (`detect::scanner`, `color::icc::extract_icc_profile`,
`decode::mod::find_exif_orientation`, the gain-map byte scanner, the
`decode::extras` reassembly and segment classification, `jpeg_inspect`) are
to be routed through this module's identification and reassembly. Until then
they stay as they are, and the walk below is this module's own. The code is
always compiled for that reason; the `extended-metadata` feature makes the
module public. With the feature off the module is crate-private and nothing
calls it yet, so behavior and output bytes are unchanged.

What is stable, now and once the walker changes: the items tile the input in
file order; each item's `kind` describes the bytes it covers; and the
accessors' results on well-formed files (files whose primary image has one
SOI, one frame, no reserved markers (TEM), length words that match their
content and no stray bytes). Not yet stable: item boundaries on
malformed or unusual layouts, `Unparsed` extents, `primary_image_end()` on
such files, and the item list after the primary EOI. The differences known
today are listed under "Follow-up" below, as information, not as a contract.
Accessor results can also change for reasons other than the walker: the
0xB002 scan planned for `container::mpf` (Follow-up item 3) and a
namespace-aware XMP reader (PR #210's) may change `mpf()` and
`extended_xmp()` on files they read differently. Any change to the published
items or results goes in the CHANGELOG.

The walk's byte search over entropy-coded data and its allocation profile
(the item list, plus a trailer list when vendor trailers end the file) are
properties of this walker, not promises.

## Walk

Built on `container::marker::MarkerIter`, which now records why it stopped
(`IterStop`, crate-private). Around its spans the walk adds:

| bytes | item | rule |
|---|---|---|
| fill `0xFF` before a marker | `Padding` | T.81 B.1.1.2 |
| input not starting `FF D8` | one `Unparsed(NotJpeg)` | libjpeg `first_marker` |
| bytes that are not a marker | `Unparsed(ExtraneousBytes)`, resync | libjpeg `next_marker` |
| `FF 00` where a marker belongs | `Unparsed(ExtraneousBytes)` | stuffed zero, not a marker (libjpeg) |
| `RSTn` while resynchronizing | inside the same item | libjpeg `read_markers` skips parameterless markers |
| length word < 2 | `Unparsed(BadLength)` (4 bytes); the walk resumes right after it, so a marker there (`RSTn` included) is its own item | libjpeg skips nothing and searches on |
| segment past the end | `Unparsed(TruncatedSegment)` with its marker and partial payload | |
| more than `max_items` | last item `Unparsed(LimitExceeded)` covering the rest | |

An SOS item covers its header and the entropy-coded data up to the next
marker that is not `RSTn`, as `MarkerIter` defines it. The header length `Ls`
is trusted as T.81 B.2.3 defines it; a decoder reads `6 + 2*Ns` bytes
whatever `Ls` says, and the shared walker may follow it.

Known gap, inherited from `MarkerIter` (`skip_entropy_scan`): fill bytes
inside entropy-coded data end the scan. T.81 B.1.1.2 allows fill before an
`RSTn`, and libjpeg's `fill_bit_buffer` accepts it, but `FF FF D0` (or
`FF FF 00`) inside a scan stops the SOS item; the fill becomes `Padding`, the
`RSTn` a `Restart` item, and the following entropy bytes
`Unparsed(ExtraneousBytes)`. The items still tile the input. Fixing it
changes `MarkerIter`'s public behavior, so it is left for a separate change.

## Identification and parsed fields

Signatures follow ExifTool's `ProcessJPEG`; layouts follow the ExifTool table
named per row. A known signature whose payload is shorter than the format's
fixed layout is `Unknown`, never half-parsed.

| marker | kind | signature | fields |
|---|---|---|---|
| SOFn | `Sof` | — | process, precision, height, width, component count and records, `ycbcr_subsampling()` (ExifTool's ratio, whole numbers only) |
| SOS | `Sos` | — | spectral selection, successive approximation, component selectors, entropy data |
| DRI / DNL | `Dri` / `Dnl` | — | restart interval / lines |
| DHT, DQT, DAC | identified | — | payload |
| APP0 | `Jfif` | `JFIF\0` | version, units, densities, thumbnail size and pixels (the declared `3*w*h` bytes, or fewer if the segment ends) |
| APP0 | `Jfxx` | `JFXX\0` + 0x10/0x11/0x13 | extension code, data |
| APP0 | `Ciff` | `II`/`MM` + 4 + `HEAPJPGM` | payload |
| APP0 | `Avi1` | `AVI1` | InterleavedField |
| APP1 | `Exif` | `Exif\0` + any byte | TIFF bytes from byte 6 (ExifTool `DirStart` 6); from byte 5 when byte 5 starts a TIFF header (a leniency, see below) |
| APP1 | `Xmp` | `http://ns.adobe.com/xap/1.0/\0` | packet |
| APP1 | `ExtendedXmp` | `http://ns.adobe.com/xmp/extension/\0` | GUID, full length, offset, data |
| APP2 | `IccProfile` | `ICC_PROFILE\0` | sequence, count, data |
| APP2 | `Fpxr` / `Mpf` / `GainMapIso21496` | `FPXR\0` / `MPF\0` / ISO 21496-1 URN | payload / TIFF / metadata bytes |
| APP3 | `KodakMeta` | `Meta\0\0`, `META\0\0`, `Exif\0\0` | TIFF |
| APP5 | `RicohRmeta` | `RMETA\0` | body |
| APP6 | `Eppim` | `EPPIM\0` | TIFF, PrintIM block (IFD0 tag 0xC4A5; `II`/`MM`, any magic, IFD0 at >= 8 as in `DoProcessTIFF`), its version (`ProcessPrintIM` checks) |
| APP8 | `Spiff` | `SPIFF\0` | all T.84 header fields (see below) |
| APP10 | `PhotoStudioComment` | `UNICODE\0` | UTF-16BE text |
| APP11 | `Jumbf` | `JP` (>= 16 bytes) | instance, sequence, box type, box bytes |
| APP12 | `Ducky` | `Ducky` | records (read past a zero tag, as `ProcessDucky`); quality, comment and copyright from the first record of each tag (ExifTool gives all three priority 0, so a repeated tag keeps its first value) |
| APP12 | `PictureInfo` | ExifTool's token regex matches (`[section]` or `tag=value`) | identified only; payload |
| APP13 | `Photoshop` | `Photoshop 3.0\0`, or `Adobe_Photoshop2.5:` with its 27-byte header | resource blocks |
| APP13 | `AdobeCm` | `Adobe_CM` | AdobeCMType |
| APP14 | `Adobe` | `Adobe` | DCTEncodeVersion, flags0, flags1, ColorTransform (raw byte) |
| APP15 | `GraphicConverter` | 4 bytes, `Q\s*\d+` (Perl `\s`, vertical tab included) | quality |
| COM | `Comment` | — | payload |
| anything else | `Unknown` | — | payload |

SPIFF: ITU-T T.84 F.2.1 puts the image height right after the component
count (24-byte body). ExifTool identifies SPIFF only for a 26-byte body and
reads every later field two bytes further on, citing its single Kodak sample.
Both layouts are accepted, each at its own offsets.

APP14 is parsed as metadata only. How the decoder interprets the colour
transform stays in `decode/parser/markers.rs` (the colour-model owner).

## Accessors and reassembly

The accessors (`exif`, `xmp`, `extended_xmp`, `icc_profile`, `photoshop`,
`mpf`) read the header: the primary image's items before its first SOS; with
no SOS, the items before the primary EOI; with neither, every item. A second
SOI does not start a new header (ExifTool reads on, libjpeg-turbo stops with
an error; we follow ExifTool), and streams appended after the primary EOI are
never part of it. This is where libjpeg-turbo reads metadata, and where
ExifTool does by default: with `-ee` or `-validate` ExifTool also reads
segments between scans. The same segments after the first SOS stay visible as
items and are not used, so a stray ICC chunk between progressive scans cannot
turn a valid header profile into an error. Each accessor returns
`Result<Option<_>, MetadataError>`. When `MetadataLimits::max_items` cut the
walk before the header ended, a payload may lie past the cut: the
single-segment accessors (`exif`, `xmp`, `mpf`) still return a segment found
before the cut, `photoshop` a run that ended before it, and everything else
is `ItemLimitReached` rather than a silent miss or a misleading error. A limit
item whose bytes begin (after `0xFF` fill) with SOS or EOI took the place of
the item that ends the header, so that header is complete and no accessor
reports the cut. An MPF directory that does not parse is
`MetadataError::Mpf`.

Reassembly borrows when one segment holds the whole payload and otherwise
allocates the returned buffer, fallibly, only after its checks pass. Either
way the result (ICC profile, extended XMP, Photoshop resources) is at most
`max_reassembled_bytes` long.

- **ICC** (`icc_profile`): ICC.1:2010 Annex B.4 with the validation of
  libjpeg-turbo's `jpeg_read_icc_profile`: every chunk declares the same
  count, sequence numbers lie in `1..=count`, none repeats, none is missing,
  and the chunks hold at least one byte; file order does not matter. Errors
  name the first violation in file order.
- **Extended XMP** (`extended_xmp`, returns `Cow`): XMP Specification Part 3,
  1.1.3.1. The GUID is the last `HasExtendedXMP` value in the header's
  standard packets (as an attribute with either quote, or as an opening
  element tag; the name must follow a `:`, and the namespace prefix is not
  resolved), as ExifTool keeps the last value it reads. With several
  standard packets, `xmp()` returns the first while the declaration may come
  from a later one. It selects the chunks, comparing GUIDs without regard
  to hex digit case; they must agree on the full length and cover it exactly
  once by offset; the MD5 of the result must equal the GUID. The digest is
  computed over the chunks in place (`foundation::md5::Md5`). One chunk
  holding the whole packet is borrowed, with no allocation; several are
  listed (24 bytes per chunk on 64-bit targets, reserved once, fallibly) and
  copied into one buffer after the digest matches. Chunks with other GUIDs
  stay visible as items. The standard and extended packets are separate
  (`xmp()` and `extended_xmp()`), not concatenated. Finding the GUID is
  linear in the packet: candidates are tried from the end, and one whose
  closing quote or `<` would have to lie past the last one fails without a
  scan.
- **Photoshop** (`photoshop`): the first run of APP13 Photoshop segments,
  each segment's header stripped. The run continues while the next segment
  is `Photoshop 3.0`, and an `Adobe_Photoshop2.5:` segment can only start a
  run. ExifTool finds the next marker by reading up to the first `0xFF` and
  skipping `0xFF` fill, so fill and stray bytes without an `0xFF` do not end a
  run, while stray bytes holding an `FF 00` or an `RSTn` do (ExifTool reads
  that as the next marker). `resources()` walks the blocks (signature `8BIM`,
  `PHUT`, `DCSR`, `AgHg`, `MeSa`; Pascal name padded to even; data padded to
  even) and stops at the first block that does not fit; `iptc()` is resource
  0x0404.
- **MPF** (`mpf`): parsed by `container::mpf::parse_mpf_segment`, absolute
  offsets. Its entry list is capped at 1000 entries by that parser; it is not
  counted against `max_reassembled_bytes`.

## Trailers

After the primary EOI, vendor trailers are found from the end of the input
backward, the way ExifTool's `IdentifyTrailer`/`ProcessTrailers` find them:
the footer signature picks the type (checked in ExifTool's order), the footer
gives the length, and the search repeats on the bytes before it.

| kind | footer | length rule (ExifTool reader) |
|---|---|---|
| `Afcp` | `AXS!`/`AXS*` + header offset + 4 | header at that absolute offset starts with the same 4 bytes (`AFCP.pm`) |
| `FotoStation` | per record: tag u16, size u32, `A1 B2 C3 D4` | records chained backward (`FotoStation.pm`); one item per record |
| `PhotoMechanic` | size u32 + `cbipcbbl` | `size + 12`, high bit clear (`PhotoMechanic.pm`) |
| `CanonVrd` | 0x40-byte footer `CANON OPTIONAL DATA\0` | size at 0x14 + 0x5C; header must match (`CanonVRD.pm`) |
| `Mie` | `zmie` signature + group terminator | group lengths chained backward; the first group opens with the `0MIE` element (`MIE.pm`) |
| `Samsung` | `\0\0SEFT` or `\0\0QDIOBS` | `SEFH` directory; farthest entry (`Samsung.pm`) |

ExifTool 12.40 also names Insta360 and NikonApp trailers. Their readers are
not implemented here, so such a trailer, and any vendor trailer before it, is
`UnknownTrailer` (or appended JPEG streams where those walk cleanly).

What precedes them is scanned forward for appended JPEG streams
(`EmbeddedJpeg`): an `FF D8` from which the same walk reaches an EOI with no
unparsed bytes on the way. A failed candidate resumes the search where its
walk stopped. Candidate walks never overlap, and the search re-reads only the
stray-byte run a failed walk stopped on, so the scan is linear (about two
passes at worst). Gaps are `Padding` (all `0x00` or all `0xFF`) or
`UnknownTrailer`.

## ExifTool cross-check

`just metadata-oracle` runs `tests/extended_metadata_exiftool.rs` (feature
`__oracle-exiftool`; a missing `exiftool`, or a version other than 12.40, fails
it) over every codec-corpus JPEG in `tests/support/metadata_corpus.rs` plus the
coverage seeds in `fuzz/seeds/metadata/`. It compares `exiftool -v3 -a -u`
(marker sequence with payload sizes, unknown-APP warnings, the first content
line of each segment, the extended XMP packet ExifTool parses after the walk,
trailers with offset and size) and `exiftool -j -n -G1 -a -u` (field values)
with our items. CI compiles it (Clippy job); running it needs ExifTool.

Measured 2026-10-11 with ExifTool 12.40 over 2216 files (2208 corpus files and
8 seeds): 1798 agree exactly; 3506 segments have the same identified format
(extended XMP packets included, by size); 16494 field values (SOF, JFIF,
Adobe, Adobe_CM, AVI1, Ducky, COM) are equal. Field values are compared over
the whole file when the marker sequences agree, and over the segments before
the divergence when a settled rule explains it (308 files; not for files
ExifTool does not read as JPEG), since ExifTool printed, and so processed,
every segment before it. Every other difference falls
under one of these settled rules. The table has 421 entries over 418 distinct
files: three files fall under two rules (`154.jpg`: JPEG 2000 markers and
trailers after a different EOI; `21ad703b….jpg`: `FF D8 FF` and trailers
after a different EOI; the `app-formats-a` seed: ISO 21496-1 and
HasExtendedXMP). The test pins each rule's file count and a hash of each
file's name and settlement detail, the number of exactly agreeing files, and
the numbers of identity checks (3491) and compared field values (16426), all
outside the moving datasets, so a regression that lands inside a rule,
changes how a settled file is settled, or makes a parser skip its checks,
fails; datasets that track a moving upstream (image-rs) are compared
but not pinned. `tests/support/metadata_corpus.rs` pins each other dataset's
file count, and requires at least one file in a moving one, so an empty or
changed fetch fails before any rule is counted. The counts assume
codec-corpus 1.1.0 content (`Cargo.lock`).

| rule | files | settlement |
|---|---|---|
| ExifTool needs `FF D8 FF` | 79 | ExifTool rejects the file or scans for an embedded image; we follow libjpeg: SOI, then resync |
| exiftool stops at a bad or truncated segment | 170 | ExifTool reads one segment ahead and stops without printing the segment before; we report the bad bytes and continue. ExifTool's list ends one event before our first bad segment, or at it |
| exiftool drops the last marker before EOF | 116 | the same read-ahead: a file without EOI loses its last marker in ExifTool's list |
| JPEG 2000 markers are standalone to ExifTool | 28 | ExifTool's marker table gives 0x30..=0x3F and 0x4F no length; T.81 Table B.1 makes only SOI, EOI, RSTn and TEM standalone |
| SOS header ExifTool reads differently (a marker in it) | 17 | we skip `Ls` bytes (T.81 B.2.3); ExifTool never reads `Ls` (its marker table gives SOS no length) and scans every byte after the marker, skipping `0xFF` fill and stuffed `FF 00`. So an `0xFF` in the header whose fill run ends, within the header or at the first byte after it, in a byte other than `0x00` or a restart marker is a marker to ExifTool and not to us; a run that reaches further is read by our walker too, from the header's end. A wrong `Ls` alone changes nothing for ExifTool, nor does a restart marker in the header, a bad-length SOS (its stray bytes follow as their own item) or a truncated SOS whose bytes hold no other marker: our restart events cover every byte after the marker, as ExifTool lists them. Only an SOS at the divergence counts: the event there, the one before, or the run of events identical (name and size) to the divergent one ending there |
| trailers after a different EOI | 2 | follows one of the marker rules above |
| ExifTool reads XMP without its signature | 1 | ExifTool takes any APP1 starting `http` or holding `<?xpacket`; XMP Part 3 requires the namespace signature |
| MPF image that is not a complete JPEG stream | 1 | the MPF directory names the range; its bytes do not walk to an EOI, so they stay `UnknownTrailer` |
| appended JPEG inside an ExifTool unknown trailer | 1 | ExifTool names appended images only from MPF/EXIF pointers; we find them by walking. ExifTool's Unknown trailer must cover the whole stream |
| SPIFF with the T.84 header length | 1 | the 24-byte T.84 F.2.1 header; ExifTool identifies only its sample's 26-byte layout (seed) |
| ISO 21496-1 gain map metadata | 1 | ISO 21496-1 postdates ExifTool 12.40 (seed) |
| HasExtendedXMP outside rdf:Description | 1 | XMP Part 3 names the property, not its place; ExifTool reads a non-RDF `HasExtendedXMP` as another tag (`XHasExtendedXMP`) and never reads the extended packet (seed) |
| not a JPEG for either | 3 | |

MPImage and PreviewImage trailers must match an `EmbeddedJpeg` item in offset
and length. Signatures ExifTool 12.40 recognizes beyond the 6.99 table (QVCI,
FLIR, Parrot, Stim, JPS, SCALADO, NITF, TDHD, GoPro, PENTAX, HUAWEI,
Qualcomm, Media Jukebox, AROT, JPEG-HDR, Ocad, FPXR in APP4, preview images in
APP2-APP5) are `Unknown` here; the test lists them and settles them by rule,
and none occurs in the corpus. ExifTool also accepts up to four stray bytes
before `Exif\0` in APP1; we require the signature at the start (no corpus file
has the stray bytes).

No corpus file carries CIFF, Kodak Meta, Ricoh RMETA, EPPIM, SPIFF,
PhotoStudio, JUMBF, Adobe_CM, GraphicConverter, FPXR, ISO 21496-1 (in a
primary), or the AFCP, CanonVRD, FotoStation, PhotoMechanic and MIE trailers.
The coverage seeds carry each of them. Four seeds agree with ExifTool exactly
(`icc-and-limits`, `trailer-mie-afcp`, `trailer-mpf-pm-fotostation` with its
MPF image, `trailer-vrd-samsung`); `app-formats-a` and `app-formats-b` agree
except for the seed rules above, and their other segments are compared,
Adobe_CM's field value included; `resync` (not `FF D8 FF`) and
`truncated-segment` (ExifTool stops at once) are not cross-checked. So
ExifTool recognizes every one of those APP segments (the two seed rules
aside) and reports every seed trailer with the same kind, offset and length.
`app-formats-b` repeats each Ducky tag, so the oracle checks that the first
value of each is kept (ExifTool gives the three tags priority 0).
Field values beyond the compared tags are checked by unit tests built from
the cited readers.

A second test runs ExifTool's APP12 Picture Info regex in Perl and compares
its verdict with our linear-time matcher, over 414,716 generated inputs: every
string of up to 5 bytes over an 11-symbol structural alphabet, every string of
up to 4 bytes over a 19-symbol alphabet with one byte of each class the regex
or the matcher distinguishes, and 100,000 random longer strings over the
latter. 0 mismatches. The test reads the regex from the installed
`APP12.pm` and checks that it equals the copy it runs.

## Deliberate divergences from ExifTool

Not seen in the corpus, so not in the table above; each has a unit test.

| case | ExifTool 12.40 | here |
|---|---|---|
| Photoshop run starting with `Adobe_Photoshop2.5:` and continuing with `Photoshop 3.0` | applies the last segment's header length (14) to the joined buffer, reads inside the old 27-byte header, finds no resources | strips each segment's own header and returns the joined resources |
| `Exif\0` directly followed by a TIFF header (byte 5 dropped) | "Malformed APP1 EXIF segment" | TIFF from byte 5 |
| APP signatures | regexes in which `.` matches any byte (`Photoshop 3x0\0` is Photoshop) | literal bytes (`Photoshop 3x0\0` is `Unknown`) |
| extended XMP GUID selection | exact string comparison | hex digits compared without regard to case (the digest check accepts both) |
| extended XMP digest | not checked | checked (XMP Part 3); a mismatch is an error |
| extended XMP chunks that disagree on the full length | warns, reads | `ExtendedXmpLengthMismatch` |
| extended XMP chunk repeated at the same offset | reads | `ExtendedXmpDiscontinuous` |
| `HasExtendedXMP` GUID padded with whitespace (attribute or element) | "Ignored non-standard extended XMP" | trimmed, the packet is read |
| a prefixed `HasExtendedXMP=` or `HasExtendedXMP>` inside an XML comment or element text, after the real property | parses RDF, ignores it, reads the packet the real property names | the last occurrence wins, so the real packet is `ExtendedXmpMissing` |
| `y:HasExtendedXMP="..."` (another namespace) or `xmlns:HasExtendedXMP="..."` after the real property | reads `XMP-y:HasExtendedXMP` (or nothing) as a separate tag and the packet `xmpNote:HasExtendedXMP` names | the prefix is not resolved, so the last occurrence wins: `ExtendedXmpMissing` |

## Performance

`benches/metadata_walk.rs`, recorded in `benchmarks/metadata_walk_2026-10-11.txt`
(Ryzen 9 7950X under WSL2, shared with other build jobs; zenbench kept 4
clean rounds per group, so its confidence intervals are wide). The
bench installs zenbench's allocation profiler as the global allocator, so all
timings include its counting.

- Per corpus file, `JpegMetadata::parse` median of 15 runs, over 2208 files:
  p50 340 ns, p90 790 ns, p99 42 us, max 3.1 ms.
- Allocations per parse: one, the item list (2,816 bytes: 32 items reserved
  once) for every representative file. Over the whole corpus, 2,210
  allocations and 105 reallocations per 2,208 parses (files with more than 32
  items grow the list; two files carry vendor trailers).
- Reassembly allocates only what it returns: on `ycck.jpg` parse plus the
  accessors allocate 559,984 bytes = the 557,168-byte ICC profile (18 chunks)
  + the item list. On `mountains.jpg` (267-byte extended XMP) and a Pixel
  Ultra HDR sample (52,330-byte extended XMP) each packet is one chunk and is
  borrowed, so parse plus the accessors allocate only the 2,816-byte item
  list.
- On multi-megabyte files the full parse costs about what the bare
  `marker::iter`-to-EOI walk costs (`mountains.jpg`, medians: 250 us against
  236 us); on small files identification shows (Canon 40D, 8 KB: 273 ns
  against 78 ns).

## Limits and allocation

`MetadataLimits::with_max_items` (default 16384) bounds the item list;
`with_max_reassembled_bytes` (default 64 MiB) bounds each reassembled payload,
borrowed or allocated, and is checked before allocating. Walking never allocates payloads and never
decodes pixels; entropy-coded data is skipped with `memchr`.

Parsing allocates, fallibly: the item list, reserved once for up to 32 items
and grown by doubling past that (the item count never exceeds `max_items`;
the list's capacity can, for a limit that is not a power of two), and, when
the file ends in vendor trailers, a list of the trailers found (one entry per
trailer or FotoStation record, within the remaining item budget). When the
item limit stops the walk before the primary EOI, `primary_image_end()` is
`None`.

## Follow-up: one walker for metadata and decoding

The planned next step builds this module's items from a walker that follows
the decoder's marker loop (the inventory walker of PR #221), and retires
`walk.rs` (its tiling tests stay, as gates on the shared walker). Where the
two walkers disagree, the decoder's reading wins, and any change to the
published items goes in the CHANGELOG. Differences known today, from reading
that walker's source (information for the follow-up, not a contract; the
list is not complete):

- an SOS whose `Ls` differs from `6 + 2*Ns`: the decoder reads the scan
  header by its component count, so a marker segment inside the bytes `Ls`
  covers here can become an item there (15 of the 2,201 non-moving corpus
  files have such an SOS, measured 2026-10-11);
- `Unparsed` runs: where stray bytes and the bytes after a bad length word
  end, and what follows them (a restart marker after a bad length, for one);
- entropy-coded data: a run of two or more `0xFF` before a stuffed `00`, a
  restart marker or the end of the input (the gap above), and a restart
  marker the decoder does not expect, which ends the scan there;
- markers T.81 makes standalone that the decoder reads with a length word: an
  SOI after the first, and a TEM after the frame header;
- table segments the decoder reads by content: a DRI whose length word is
  below 4 (it always reads 4 bytes), a DAC whose length word is odd (it stops
  at the last whole entry);
- an SOS before the frame header;
- everything after the primary EOI: the decoder's walker places MPF images
  from the MPF directory and does not look for vendor-trailer footers.

The decoder's own metadata readers are then routed through this module's
identification and reassembly, and keep their current output:

1. **ICC.** `extract_icc_profile` accepts a chunk numbered 0
   (`amazed2k1424ktv.jpg`, one chunk "0 of 1"), which ICC.1 B.4 and
   libjpeg-turbo reject and ExifTool accepts. Routing it through the strict
   reassembly would drop that file's profile, so the routed decoder path keeps
   its lenient result (or the change is put to the user as a behavior
   change); `icc_profile()` stays strict.
2. **Extended XMP.** `DecodedExtras::xmp()` today returns the standard packet
   followed by every extended chunk sorted by offset, with no GUID, length or
   digest check. After routing it keeps that result so decoded output does
   not change; the verified packet is offered next to it (with whether the
   digest matched), not in place of it.
3. **MPF.** The decoder's MPF parser (`decode::extras::parse_mpf_directory`)
   scans for tag 0xB002 at non-12-byte spacing for some Android writers;
   `container::mpf` instead handles the out-of-line version quirk and falls
   back to IFD-relative entry offsets. Over the corpus the two agree on all 41
   files that decode and carry MPF
   (`tests/extended_metadata.rs::mpf_directories_agree_with_the_decoder_parser`).
   Before routing, `container::mpf` gains the 0xB002 scan, with a synthetic
   test of that layout (the corpus has none).


## Header and full probes (2026-10-25)

The zencodec header probe stops before the first SOS. Its ICC extractor now
requires a complete, consistently numbered profile within that header; a
partial profile never becomes usable ICC bytes. A complete legacy single chunk
numbered zero of one remains readable in the native compatibility path.

`JpegDecoderConfig::probe_full_metadata` now walks through the primary EOI,
without decoding coefficients or pixels. It reports recognized ICC, EXIF, XMP
and JFIF metadata as native preserved extras do, including carriers between or
after scans. It rejects a missing primary EOI and respects extraction filters.
It does not inspect trailing/nested image metadata, validate XML disclosure
policy, or establish that a late profile governed native color conversion.
Use structural inspection for a carrier inventory. The legacy optional
metadata fields do not distinguish absence from failed reassembly.

`just metadata-header-check` covers malformed/complete ICC chunks, the legacy
profile, baseline/progressive late metadata, missing EOI, trailer exclusion and
policy filtering. Retained full-probe segments are admitted before copying;
legacy reassembly caches and returned metadata also allocate, so this is not a
whole-operation memory guarantee.
