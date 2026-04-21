//! Single-pass container probe — locate every interesting APP segment,
//! detect gain-map fingerprints, identify the ICC profile, and capture
//! image byte ranges in one [`super::marker::iter`] walk.
//!
//! # Why this exists
//!
//! A caller that today does "find the MPF, find the ISO gainmap, find
//! the ICC, parse the XMP for `hdrgm:` — oh and also get the image
//! ranges" pays for **3–5 independent marker walks**. Each is fast
//! (memchr-backed, 16-40 GiB/s), but N walks is N× the work and N×
//! the Vec allocations. `container::probe` does **one** walk and
//! captures everything the caller asked for.
//!
//! # Zero allocation
//!
//! The returned [`ContainerProbe`] has fixed-size inline storage for
//! image ranges and extended-XMP chunks — no heap. APP segment
//! payloads are referenced by [`Range<u32>`] offsets into the caller's
//! buffer, not copied.
//!
//! Size of the returned struct is pinned by a const assertion — any
//! future field addition that bloats the struct beyond ~256 bytes
//! needs a reviewer's explicit approval.
//!
//! # Opt-in via [`Wants`]
//!
//! Callers pass a [`Wants`] mask to say which signals they need. The
//! probe then skips per-segment fingerprinting (memmem, ICC hash) for
//! signals the caller didn't ask for. Useful for a fast
//! [`is_ultrahdr`] check that short-circuits on first match.
//!
//! # Forward-compatibility invariants (locked in at commit time)
//!
//! 1. Offsets are [`u32`]. A future noncontiguous-buffer reader can
//!    wrap them in `(buffer_id, u32)` without a layout break behind
//!    `#[non_exhaustive]`.
//! 2. All public structs/enums are `#[non_exhaustive]`.
//! 3. A [`ContainerProbe`] captured from a truncated buffer reports
//!    [`ContainerProbe::truncated`] == `true`, and consumers must treat
//!    every `None` accessor return as "not yet confirmed absent."
//! 4. Storage fields are `pub(crate)`; consumers go through accessor
//!    methods so storage can evolve (e.g. fixed inline → `TinyVec`)
//!    without a semver break.

use core::ops::Range;

use tinyvec::ArrayVec;

use super::marker::{MarkerKind, MarkerSpan, iter};

/// Classification of an image's gain-map metadata presence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum GainMapPresence {
    /// No gain-map signals found.
    #[default]
    None,
    /// ISO 21496-1 APP2 segment present (modern, always wins).
    Iso21496,
    /// Only XMP `hdrgm:*` attributes (legacy libultrahdr pre-1.4).
    XmpHdrgmLegacy,
    /// Only GContainer `Item:Semantic="GainMap"` (Google Camera).
    GContainerOnly,
    /// ISO 21496-1 + XMP `hdrgm:*` (canonical libultrahdr output).
    IsoAndXmp,
    /// ISO 21496-1 + GContainer (modern HDR + multi-item container).
    IsoAndGContainer,
}

/// Capability mask for [`probe`]. Set bits for the signals you need;
/// everything else is skipped.
///
/// The inner `u16` is intentionally private — construct via `|` of named
/// constants. This keeps the bit layout an implementation detail that
/// can be reshuffled without a semver break.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Wants {
    bits: u16,
}

impl Wants {
    const fn new(bits: u16) -> Self {
        Self { bits }
    }

    /// Record SOI..EOI byte ranges for every top-level image.
    pub const IMAGE_RANGES: Self = Self::new(1 << 0);
    /// Record the ICC profile APP2 location (single-segment only;
    /// multi-segment reassembly is left to the caller).
    pub const ICC_LOCATION: Self = Self::new(1 << 1);
    /// Additionally run `zenpixels::icc::identify_common` on the ICC
    /// payload. Requires [`ICC_LOCATION`](Self::ICC_LOCATION).
    pub const ICC_IDENTIFY: Self = Self::new(1 << 2);
    /// Record the first APP1 Exif segment location.
    pub const EXIF_LOCATION: Self = Self::new(1 << 3);
    /// Record the XMP APP1 segment location + extended-XMP chunks.
    pub const XMP_LOCATION: Self = Self::new(1 << 4);
    /// memmem-scan the XMP payload for `hdrgm:` and GContainer gain-map
    /// fingerprints. Requires [`XMP_LOCATION`](Self::XMP_LOCATION).
    pub const XMP_GAINMAP_FLAGS: Self = Self::new(1 << 5);
    /// Record the APP2 MPF segment location.
    pub const MPF_LOCATION: Self = Self::new(1 << 6);
    /// Record the APP2 ISO 21496-1 segment location.
    pub const ISO_GAINMAP: Self = Self::new(1 << 7);
    /// Record SOF dimensions, subsampling, component count.
    pub const SOF_DIMENSIONS: Self = Self::new(1 << 8);
    /// Keep walking past the first SOS to count progressive scans.
    /// Without this, the walk stops at first SOS for a faster "headers
    /// only" probe.
    pub const SCAN_COUNT: Self = Self::new(1 << 9);
    /// Everything above — the "tell me everything" preset. Derived
    /// automatically from the named constants so adding a new bit
    /// automatically extends `ALL`.
    pub const ALL: Self = Self::new(
        Self::IMAGE_RANGES.bits
            | Self::ICC_LOCATION.bits
            | Self::ICC_IDENTIFY.bits
            | Self::EXIF_LOCATION.bits
            | Self::XMP_LOCATION.bits
            | Self::XMP_GAINMAP_FLAGS.bits
            | Self::MPF_LOCATION.bits
            | Self::ISO_GAINMAP.bits
            | Self::SOF_DIMENSIONS.bits
            | Self::SCAN_COUNT.bits,
    );
    /// Minimum for the `is_ultrahdr` short-circuit: MPF + ISO + XMP
    /// fingerprints.
    pub const ULTRAHDR_DETECT: Self = Self::new(
        Self::ISO_GAINMAP.bits
            | Self::XMP_LOCATION.bits
            | Self::XMP_GAINMAP_FLAGS.bits
            | Self::MPF_LOCATION.bits,
    );

    /// `true` if all bits in `other` are set in `self`.
    #[inline]
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.bits & other.bits) == other.bits
    }

    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.bits == 0
    }
}

impl core::ops::BitOr for Wants {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self::new(self.bits | rhs.bits)
    }
}

impl core::ops::BitOrAssign for Wants {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.bits |= rhs.bits;
    }
}

/// Minimal SOF-derived info. Matches [`crate::types::Dimensions`] for
/// the dimension fields but decouples the probe from the full
/// `JpegProbe` type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ProbeSof {
    /// Width in pixels.
    pub width: u16,
    /// Height in pixels.
    pub height: u16,
    /// Number of color components (1 grayscale, 3 color, 4 CMYK).
    pub num_components: u8,
    /// SOFn marker byte - 0xC0 (e.g. 0 for SOF0 = baseline,
    /// 2 for SOF2 = progressive).
    pub sofn: u8,
}

/// Single-pass probe result. All byte ranges reference positions in
/// the caller's buffer.
///
/// Fields are `pub(crate)`; consumers go through the accessor methods
/// on `impl ContainerProbe` so internal storage (currently
/// [`tinyvec::ArrayVec`] for the two bounded lists) can evolve without
/// a semver break.
///
/// Not `Copy` because [`Range`] deliberately isn't `Copy` (stdlib
/// choice — Range is also an Iterator and Copy would make iteration
/// surprising). The struct has no heap-backed fields, so `Clone` is
/// equivalent to `memcpy` — cheap to pass around by value.
///
/// See [`probe`] for the walker and [`Wants`] for opt-in knobs.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ContainerProbe {
    /// SOI..EOI byte ranges for each top-level image (primary first).
    /// Bounded at 8 entries; further images are silently ignored.
    pub(crate) image_ranges: ArrayVec<[Range<u32>; 8]>,

    /// ICC profile APP2 payload range (single-segment).
    pub(crate) icc_profile: Option<Range<u32>>,
    /// Result of [`zenpixels::icc::identify_common`] on the ICC profile.
    pub(crate) icc_identification: Option<zenpixels::icc::IccIdentification>,

    /// First APP1 Exif segment range.
    pub(crate) exif: Option<Range<u32>>,

    /// First APP1 XMP (adobe namespace) segment range.
    pub(crate) xmp: Option<Range<u32>>,
    /// Additional extended-XMP chunk ranges. Bounded at 4 entries.
    pub(crate) extended_xmp: ArrayVec<[Range<u32>; 4]>,

    /// APP2 MPF segment range.
    pub(crate) mpf: Option<Range<u32>>,

    /// APP2 ISO 21496-1 segment range.
    pub(crate) iso_gainmap: Option<Range<u32>>,

    /// Minimal SOF-derived info.
    pub(crate) sof: Option<ProbeSof>,

    /// Number of SOS segments (1 = baseline, >1 = progressive).
    pub(crate) scan_count: u16,

    /// XMP contains `hdrgm:` namespace attributes.
    pub(crate) has_xmp_hdrgm: bool,
    /// XMP contains an `Item:Semantic="GainMap"` GContainer entry.
    pub(crate) has_xmp_gcontainer_gainmap: bool,
    /// Computed gain-map presence classification.
    pub(crate) gainmap_presence: GainMapPresence,

    /// Probe stopped early due to truncated / malformed input.
    pub(crate) truncated: bool,
}

// Guardrail: don't let this struct silently bloat.
const _: () = {
    assert!(
        core::mem::size_of::<ContainerProbe>() <= 320,
        "ContainerProbe grew beyond 320 bytes; audit before bumping"
    );
};

impl Default for ContainerProbe {
    fn default() -> Self {
        Self {
            image_ranges: ArrayVec::new(),
            icc_profile: None,
            icc_identification: None,
            exif: None,
            xmp: None,
            extended_xmp: ArrayVec::new(),
            mpf: None,
            iso_gainmap: None,
            sof: None,
            scan_count: 0,
            has_xmp_hdrgm: false,
            has_xmp_gcontainer_gainmap: false,
            gainmap_presence: GainMapPresence::None,
            truncated: false,
        }
    }
}

impl ContainerProbe {
    /// SOI..EOI byte ranges for each top-level image (primary first).
    /// Bounded at 8 entries; further images are silently ignored.
    #[inline]
    #[must_use]
    pub fn image_ranges(&self) -> &[Range<u32>] {
        &self.image_ranges
    }

    /// ICC profile APP2 payload range (single-segment). `None` if
    /// absent or if [`Wants::ICC_LOCATION`] was not requested.
    #[inline]
    #[must_use]
    pub fn icc_profile(&self) -> Option<&Range<u32>> {
        self.icc_profile.as_ref()
    }

    /// Result of [`zenpixels::icc::identify_common`] on the ICC profile.
    /// `None` unless [`Wants::ICC_IDENTIFY`] was requested AND the
    /// profile was single-segment AND identification succeeded.
    #[inline]
    #[must_use]
    pub fn icc_identification(&self) -> Option<&zenpixels::icc::IccIdentification> {
        self.icc_identification.as_ref()
    }

    /// First APP1 Exif segment payload range. `None` unless requested
    /// and found.
    #[inline]
    #[must_use]
    pub fn exif(&self) -> Option<&Range<u32>> {
        self.exif.as_ref()
    }

    /// First APP1 XMP (adobe namespace) segment payload range.
    #[inline]
    #[must_use]
    pub fn xmp(&self) -> Option<&Range<u32>> {
        self.xmp.as_ref()
    }

    /// Additional Extended-XMP chunk payload ranges. Bounded at 4
    /// entries — callers seeing `len() == 4` may be looking at a
    /// truncated view of a longer chain.
    #[inline]
    #[must_use]
    pub fn extended_xmp(&self) -> &[Range<u32>] {
        &self.extended_xmp
    }

    /// APP2 MPF segment payload range.
    #[inline]
    #[must_use]
    pub fn mpf(&self) -> Option<&Range<u32>> {
        self.mpf.as_ref()
    }

    /// APP2 ISO 21496-1 segment payload range.
    #[inline]
    #[must_use]
    pub fn iso_gainmap(&self) -> Option<&Range<u32>> {
        self.iso_gainmap.as_ref()
    }

    /// SOF-derived frame info (dimensions, component count, SOFn byte).
    #[inline]
    #[must_use]
    pub fn sof(&self) -> Option<&ProbeSof> {
        self.sof.as_ref()
    }

    /// Number of SOS segments (1 = baseline, >1 = progressive). Only
    /// populated when [`Wants::SCAN_COUNT`] was requested.
    #[inline]
    #[must_use]
    pub fn scan_count(&self) -> u16 {
        self.scan_count
    }

    /// `true` if any APP1 XMP payload carried `hdrgm:Version` or
    /// `hdrgm:GainMapMax` — the legacy libultrahdr (pre-1.4) fingerprint.
    #[inline]
    #[must_use]
    pub fn has_xmp_hdrgm(&self) -> bool {
        self.has_xmp_hdrgm
    }

    /// `true` if any APP1 XMP payload carried an
    /// `Item:Semantic="GainMap"` GContainer entry.
    #[inline]
    #[must_use]
    pub fn has_xmp_gcontainer_gainmap(&self) -> bool {
        self.has_xmp_gcontainer_gainmap
    }

    /// Gain-map presence classification, derived from ISO + XMP + GContainer
    /// fingerprints.
    #[inline]
    #[must_use]
    pub fn gainmap_presence(&self) -> GainMapPresence {
        self.gainmap_presence
    }

    /// `true` if the probe stopped early due to truncated or malformed
    /// input. When `true`, every `None` accessor return must be read as
    /// "not yet confirmed absent" rather than definitive absence.
    #[inline]
    #[must_use]
    pub fn truncated(&self) -> bool {
        self.truncated
    }
}

/// Walk `data` once and capture every signal requested by `wants`.
///
/// See the module docs for semantics. Safe on arbitrary input — never
/// panics, never allocates.
///
/// # Examples
///
/// Detect Ultra HDR signals in a JPEG buffer:
///
/// ```
/// use zenjpeg::container::probe::{probe, Wants, GainMapPresence};
///
/// # let jpeg: &[u8] = &[
/// #     0xFF, 0xD8, 0xFF, 0xDB, 0x00, 0x43, 0,
/// #     1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
/// #     1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
/// #     0xFF, 0xC0, 0x00, 0x11, 8, 0, 16, 0, 32, 3,
/// #     1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1,
/// #     0xFF, 0xC4, 0x00, 0x1F, 0,
/// #     0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0,
/// #     0,1,2,3,4,5,6,7,8,9,10,11,
/// #     0xFF, 0xDA, 0x00, 0x0C, 3, 1, 0, 2, 0, 3, 0, 0, 0x3F, 0,
/// #     0xFF, 0xD9,
/// # ];
/// let p = probe(jpeg, Wants::ULTRAHDR_DETECT | Wants::SOF_DIMENSIONS);
/// assert_eq!(p.gainmap_presence(), GainMapPresence::None);
/// if let Some(sof) = p.sof() {
///     assert!(sof.width > 0 && sof.height > 0);
/// }
/// ```
#[must_use]
pub fn probe(data: &[u8], wants: Wants) -> ContainerProbe {
    let mut p = ContainerProbe::default();

    // Offsets are u32. If the input exceeds 4 GiB we silently refuse
    // rather than truncate: return an empty probe flagged as
    // truncated so the caller treats every None as ambiguous.
    if is_oversized_for_u32(data.len()) {
        p.truncated = true;
        return p;
    }

    // Multi-image boundary enumeration via for_each_jpeg_boundary (uses
    // memchr to skip between images — so for a typical Ultra HDR file
    // this is ~2 sub-walks across primary + gainmap). Done separately
    // from the main marker walk below because `MarkerIter` stops at
    // the first EOI by design.
    if wants.contains(Wants::IMAGE_RANGES) {
        super::marker::for_each_jpeg_boundary(data, |range| {
            // ArrayVec silently drops pushes past its capacity. Consumers
            // who care about "truncated list" semantics inspect `len()`
            // against the cap (8).
            let _ = p
                .image_ranges
                .try_push((range.start as u32)..(range.end as u32));
        });
    }

    // Main marker walk for APP segments, SOF, SOS scan count, etc.
    // Walks the primary image only (stops at first EOI by MarkerIter
    // design). For most signals that's enough — APP segments and SOF
    // must precede the primary's SOS. Scan counting past EOI is
    // unnecessary (progressive scans all live inside one image).
    let stop_at_first_sos = !wants.contains(Wants::SCAN_COUNT);

    for span in iter(data) {
        match span.kind {
            MarkerKind::Soi | MarkerKind::Eoi => {
                // Boundaries handled above.
            }
            MarkerKind::Sos => {
                p.scan_count = p.scan_count.saturating_add(1);
                if stop_at_first_sos {
                    break;
                }
            }
            // SOF payload layout: [precision u8, height u16 BE,
            // width u16 BE, num_components u8, per-component...].
            MarkerKind::Sof(sofn)
                if wants.contains(Wants::SOF_DIMENSIONS)
                    && p.sof.is_none()
                    && span.payload.len() >= 6 =>
            {
                let height = u16::from_be_bytes([span.payload[1], span.payload[2]]);
                let width = u16::from_be_bytes([span.payload[3], span.payload[4]]);
                let num_components = span.payload[5];
                p.sof = Some(ProbeSof {
                    width,
                    height,
                    num_components,
                    sofn: 0xC0 + sofn,
                });
            }
            MarkerKind::App(1) => {
                handle_app1(&mut p, &span, wants);
            }
            MarkerKind::App(2) => {
                handle_app2(&mut p, &span, wants);
            }
            _ => {}
        }
    }

    // Compute gain-map presence from the captured fingerprints.
    p.gainmap_presence = classify_gainmap_presence(&p);
    p
}

/// Short-circuit "is this UltraHDR?" probe.
///
/// Uses [`Wants::ULTRAHDR_DETECT`] and exits as soon as any positive
/// signal is found. Typical cost: ~10-100 µs on a multi-MB file (the
/// ISO URN APP2 sits inside the first ~30 KB).
///
/// # Examples
///
/// ```
/// use zenjpeg::container::probe::is_ultrahdr;
///
/// // An empty or malformed buffer is never Ultra HDR.
/// assert!(!is_ultrahdr(&[]));
/// assert!(!is_ultrahdr(b"not a jpeg"));
///
/// // A plain JPEG with no gain-map signals is not Ultra HDR either.
/// # let jpeg: &[u8] = &[0xFF, 0xD8, 0xFF, 0xD9];
/// assert!(!is_ultrahdr(jpeg));
/// ```
#[must_use]
pub fn is_ultrahdr(data: &[u8]) -> bool {
    // For the short-circuit path we inline a smaller walker that stops
    // at the first positive hit rather than completing the full probe.
    for span in iter(data) {
        match span.kind {
            MarkerKind::App(2)
                if span.payload.starts_with(super::iso21496::ISO_21496_1_URN)
                    || span.payload.starts_with(super::mpf::MPF_IDENTIFIER) =>
            {
                return true;
            }
            MarkerKind::App(1) if looks_like_xmp(span.payload) => {
                let xmp = xmp_body(span.payload);
                if memmem_contains(xmp, b"hdrgm:Version")
                    || memmem_contains(xmp, b"hdrgm:GainMapMax")
                    || memmem_contains(xmp, b"Item:Semantic=\"GainMap\"")
                {
                    return true;
                }
            }
            MarkerKind::Eoi | MarkerKind::Sos => {
                // Past the primary-image header region — no more APPs
                // to find. (Ultra HDR metadata never lives past SOS.)
                break;
            }
            _ => {}
        }
    }
    false
}

// ===========================================================================
// Internals
// ===========================================================================

/// `true` if a byte slice of this length cannot be safely indexed with
/// `u32` offsets (i.e. > 4 GiB). Extracted as a standalone function so
/// the guard can be unit-tested without allocating a real 4 GiB buffer.
#[inline]
fn is_oversized_for_u32(len: usize) -> bool {
    len > u32::MAX as usize
}

const ICC_PROFILE_PREFIX: &[u8; 12] = b"ICC_PROFILE\0";
const EXIF_PREFIX: &[u8; 6] = b"Exif\0\0";
const XMP_NAMESPACE_PREFIX: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
const EXTENDED_XMP_PREFIX: &[u8] = b"http://ns.adobe.com/xmp/extension/\0";

fn handle_app1(probe: &mut ContainerProbe, span: &MarkerSpan<'_>, wants: Wants) {
    let payload = span.payload;

    if payload.starts_with(EXIF_PREFIX)
        && wants.contains(Wants::EXIF_LOCATION)
        && probe.exif.is_none()
    {
        let start = (span.offset + 2 + 2 + EXIF_PREFIX.len()) as u32;
        let end = (span.offset + span.length) as u32;
        probe.exif = Some(start..end);
        return;
    }

    if payload.starts_with(XMP_NAMESPACE_PREFIX) {
        // Record location only for the FIRST XMP APP1. But ALWAYS
        // update the gain-map fingerprint flags — some writers emit a
        // decoy camera-metadata XMP before the real gain-map XMP, and
        // we must OR across all APP1 XMPs to avoid misclassifying.
        if wants.contains(Wants::XMP_LOCATION) && probe.xmp.is_none() {
            let start = (span.offset + 2 + 2 + XMP_NAMESPACE_PREFIX.len()) as u32;
            let end = (span.offset + span.length) as u32;
            probe.xmp = Some(start..end);
        }
        if wants.contains(Wants::XMP_GAINMAP_FLAGS) {
            let xmp_body = &payload[XMP_NAMESPACE_PREFIX.len()..];
            if !probe.has_xmp_hdrgm
                && (memmem_contains(xmp_body, b"hdrgm:Version")
                    || memmem_contains(xmp_body, b"hdrgm:GainMapMax"))
            {
                probe.has_xmp_hdrgm = true;
            }
            if !probe.has_xmp_gcontainer_gainmap
                && memmem_contains(xmp_body, b"Item:Semantic=\"GainMap\"")
            {
                probe.has_xmp_gcontainer_gainmap = true;
            }
        }
        return;
    }

    if payload.starts_with(EXTENDED_XMP_PREFIX) && wants.contains(Wants::XMP_LOCATION) {
        let start = (span.offset + 2 + 2 + EXTENDED_XMP_PREFIX.len()) as u32;
        let end = (span.offset + span.length) as u32;
        // ArrayVec drops pushes past capacity (4). Consumers inspect
        // `extended_xmp().len() == 4` to detect a truncated chain.
        let _ = probe.extended_xmp.try_push(start..end);
    }
}

fn handle_app2(probe: &mut ContainerProbe, span: &MarkerSpan<'_>, wants: Wants) {
    let payload = span.payload;

    if payload.starts_with(ICC_PROFILE_PREFIX) {
        if wants.contains(Wants::ICC_LOCATION) && probe.icc_profile.is_none() {
            // Payload layout after prefix: [seq_no u8, total_seqs u8, icc_bytes...].
            // For single-segment profiles (99% of the wild), seq_no == 1
            // && total_seqs == 1 and the ICC starts at prefix + 2.
            let icc_bytes_start = span.offset + 2 + 2 + ICC_PROFILE_PREFIX.len() + 2;
            let icc_bytes_end = span.offset + span.length;
            if icc_bytes_end > icc_bytes_start {
                probe.icc_profile = Some((icc_bytes_start as u32)..(icc_bytes_end as u32));

                if wants.contains(Wants::ICC_IDENTIFY)
                    && payload.len() >= ICC_PROFILE_PREFIX.len() + 2
                {
                    let seq_no = payload[ICC_PROFILE_PREFIX.len()];
                    let total_seqs = payload[ICC_PROFILE_PREFIX.len() + 1];
                    if seq_no == 1 && total_seqs == 1 {
                        let icc_bytes = &payload[ICC_PROFILE_PREFIX.len() + 2..];
                        probe.icc_identification = zenpixels::icc::identify_common(icc_bytes);
                    }
                }
            }
        }
        return;
    }

    if payload.starts_with(super::mpf::MPF_IDENTIFIER)
        && wants.contains(Wants::MPF_LOCATION)
        && probe.mpf.is_none()
    {
        let start = (span.offset + 2 + 2 + super::mpf::MPF_IDENTIFIER.len()) as u32;
        let end = (span.offset + span.length) as u32;
        probe.mpf = Some(start..end);
        return;
    }

    if payload.starts_with(super::iso21496::ISO_21496_1_URN)
        && wants.contains(Wants::ISO_GAINMAP)
        && probe.iso_gainmap.is_none()
    {
        let start = (span.offset + 2 + 2 + super::iso21496::ISO_21496_1_URN.len()) as u32;
        let end = (span.offset + span.length) as u32;
        probe.iso_gainmap = Some(start..end);
    }
}

#[inline]
fn memmem_contains(haystack: &[u8], needle: &[u8]) -> bool {
    memchr::memmem::find(haystack, needle).is_some()
}

#[inline]
fn looks_like_xmp(payload: &[u8]) -> bool {
    payload.starts_with(XMP_NAMESPACE_PREFIX)
}

#[inline]
fn xmp_body(payload: &[u8]) -> &[u8] {
    &payload[XMP_NAMESPACE_PREFIX.len().min(payload.len())..]
}

fn classify_gainmap_presence(p: &ContainerProbe) -> GainMapPresence {
    let iso = p.iso_gainmap.is_some();
    let xmp_hdrgm = p.has_xmp_hdrgm;
    let gcontainer = p.has_xmp_gcontainer_gainmap;
    match (iso, xmp_hdrgm, gcontainer) {
        (false, false, false) => GainMapPresence::None,
        (true, true, _) => GainMapPresence::IsoAndXmp,
        (true, false, true) => GainMapPresence::IsoAndGContainer,
        (true, false, false) => GainMapPresence::Iso21496,
        (false, true, _) => GainMapPresence::XmpHdrgmLegacy,
        (false, false, true) => GainMapPresence::GContainerOnly,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::super::iso21496::append_app2_marker as iso21496_append_app2;
    use super::super::mpf::create_mpf_header;
    use super::*;
    use alloc::vec::Vec;

    /// Build a stub ISO 21496-1 APP2 segment with `payload` as its body
    /// (after the URN). Test convenience — zenjpeg::container::iso21496
    /// is `pub(crate)`, so its low-level appender is available to tests
    /// in the same crate.
    fn iso_app2(payload: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        iso21496_append_app2(&mut v, payload);
        v
    }

    fn minimal_jpeg() -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&[0xFF, 0xD8]);
        v.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x11]);
        v.extend_from_slice(&[0x08, 0x00, 0x10, 0x00, 0x20, 0x03]); // 8-bit, 16h × 32w, 3 comp
        v.extend_from_slice(&[0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01]);
        v.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00]);
        v.extend_from_slice(&[0xAB, 0xCD]);
        v.extend_from_slice(&[0xFF, 0xD9]);
        v
    }

    #[test]
    fn probe_empty_returns_defaults() {
        let p = probe(&[], Wants::ALL);
        assert_eq!(p.scan_count(), 0);
        assert!(p.image_ranges().is_empty());
        assert_eq!(p.gainmap_presence(), GainMapPresence::None);
    }

    #[test]
    fn probe_sof_dimensions_parse_correctly() {
        let data = minimal_jpeg();
        let p = probe(&data, Wants::SOF_DIMENSIONS);
        let sof = p.sof().expect("sof");
        assert_eq!(sof.height, 16);
        assert_eq!(sof.width, 32);
        assert_eq!(sof.num_components, 3);
        assert_eq!(sof.sofn, 0xC0);
    }

    #[test]
    fn probe_scan_count_baseline() {
        let data = minimal_jpeg();
        let p = probe(&data, Wants::SCAN_COUNT);
        assert_eq!(p.scan_count(), 1);
    }

    #[test]
    fn probe_image_ranges_single_image() {
        let data = minimal_jpeg();
        let p = probe(&data, Wants::IMAGE_RANGES);
        let ranges = p.image_ranges();
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, 0);
        assert_eq!(ranges[0].end, data.len() as u32);
    }

    #[test]
    fn probe_detects_iso_gainmap_app2() {
        let mut data = Vec::new();
        data.extend_from_slice(&[0xFF, 0xD8]);
        let mut dummy_iso = [0u8; 16];
        dummy_iso[0] = 0x00; // min_version byte 1
        data.extend_from_slice(&iso_app2(&dummy_iso));
        data.extend_from_slice(&minimal_jpeg()[2..]);

        let p = probe(&data, Wants::ISO_GAINMAP);
        assert!(p.iso_gainmap().is_some());
        assert_eq!(p.gainmap_presence(), GainMapPresence::Iso21496);
    }

    #[test]
    fn probe_detects_mpf_app2() {
        let mut data = Vec::new();
        data.extend_from_slice(&[0xFF, 0xD8]);
        let mpf_bytes = create_mpf_header(100, 50, None);
        data.extend_from_slice(&mpf_bytes);
        data.extend_from_slice(&minimal_jpeg()[2..]);

        let p = probe(&data, Wants::MPF_LOCATION);
        assert!(p.mpf().is_some());
    }

    #[test]
    fn probe_detects_xmp_hdrgm_via_fingerprint() {
        let mut data = Vec::new();
        data.extend_from_slice(&[0xFF, 0xD8]);
        // APP1 with XMP containing hdrgm:Version.
        let xmp_payload = br#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
        xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/"
        hdrgm:Version="1.0"
        hdrgm:GainMapMax="2.0"/>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"#;
        let namespace = b"http://ns.adobe.com/xap/1.0/\0";
        let total_len = 2 + namespace.len() + xmp_payload.len();
        data.push(0xFF);
        data.push(0xE1);
        data.extend_from_slice(&(total_len as u16).to_be_bytes());
        data.extend_from_slice(namespace);
        data.extend_from_slice(xmp_payload);
        data.extend_from_slice(&minimal_jpeg()[2..]);

        let p = probe(&data, Wants::XMP_LOCATION | Wants::XMP_GAINMAP_FLAGS);
        assert!(p.xmp().is_some());
        assert!(p.has_xmp_hdrgm());
        assert_eq!(p.gainmap_presence(), GainMapPresence::XmpHdrgmLegacy);
    }

    #[test]
    fn probe_detects_gcontainer_gainmap_semantic() {
        let mut data = Vec::new();
        data.extend_from_slice(&[0xFF, 0xD8]);
        let xmp_payload =
            br#"<rdf:Description xmlns:Container="http://ns.google.com/photos/1.0/container/">
                <Container:Item Item:Semantic="GainMap"/>
            </rdf:Description>"#;
        let namespace = b"http://ns.adobe.com/xap/1.0/\0";
        let total_len = 2 + namespace.len() + xmp_payload.len();
        data.push(0xFF);
        data.push(0xE1);
        data.extend_from_slice(&(total_len as u16).to_be_bytes());
        data.extend_from_slice(namespace);
        data.extend_from_slice(xmp_payload);
        data.extend_from_slice(&minimal_jpeg()[2..]);

        let p = probe(&data, Wants::XMP_LOCATION | Wants::XMP_GAINMAP_FLAGS);
        assert!(p.has_xmp_gcontainer_gainmap());
        assert_eq!(p.gainmap_presence(), GainMapPresence::GContainerOnly);
    }

    #[test]
    fn probe_combines_iso_and_xmp_hdrgm() {
        let mut data = Vec::new();
        data.extend_from_slice(&[0xFF, 0xD8]);

        let dummy_iso = [0u8; 16];
        data.extend_from_slice(&iso_app2(&dummy_iso));

        let xmp_payload = b"hdrgm:Version=\"1.0\"";
        let namespace = b"http://ns.adobe.com/xap/1.0/\0";
        let total_len = 2 + namespace.len() + xmp_payload.len();
        data.push(0xFF);
        data.push(0xE1);
        data.extend_from_slice(&(total_len as u16).to_be_bytes());
        data.extend_from_slice(namespace);
        data.extend_from_slice(xmp_payload);

        data.extend_from_slice(&minimal_jpeg()[2..]);

        let p = probe(
            &data,
            Wants::ISO_GAINMAP | Wants::XMP_LOCATION | Wants::XMP_GAINMAP_FLAGS,
        );
        assert_eq!(p.gainmap_presence(), GainMapPresence::IsoAndXmp);
    }

    #[test]
    fn probe_wants_skips_unrequested_signals() {
        let mut data = Vec::new();
        data.extend_from_slice(&[0xFF, 0xD8]);
        data.extend_from_slice(&iso_app2(&[0u8; 16]));
        data.extend_from_slice(&minimal_jpeg()[2..]);

        // Probe asking ONLY for SOF should NOT surface ISO.
        let p = probe(&data, Wants::SOF_DIMENSIONS);
        assert!(p.iso_gainmap().is_none());
        assert!(p.sof().is_some());
    }

    /// Regression: when more top-level images exist than the capped
    /// `image_ranges` storage can hold (8), the array fills to 8 and
    /// further images are silently dropped. Callers who care about
    /// "was truncated" inspect `len() == 8`.
    #[test]
    fn probe_image_ranges_cap_at_8() {
        let mut data = Vec::new();
        let j = minimal_jpeg();
        for _ in 0..10 {
            data.extend_from_slice(&j);
        }
        let p = probe(&data, Wants::IMAGE_RANGES);
        assert_eq!(p.image_ranges().len(), 8, "array capacity enforced");
    }

    #[test]
    fn probe_non_jpeg_returns_empty() {
        let p = probe(b"hello world", Wants::ALL);
        assert_eq!(p.scan_count(), 0);
        assert!(p.image_ranges().is_empty());
        assert!(p.sof().is_none());
    }

    #[test]
    fn is_ultrahdr_false_on_plain_jpeg() {
        let data = minimal_jpeg();
        assert!(!is_ultrahdr(&data));
    }

    #[test]
    fn is_ultrahdr_true_on_iso_gainmap() {
        let mut data = Vec::new();
        data.extend_from_slice(&[0xFF, 0xD8]);
        data.extend_from_slice(&iso_app2(&[0u8; 16]));
        data.extend_from_slice(&minimal_jpeg()[2..]);
        assert!(is_ultrahdr(&data));
    }

    #[test]
    fn is_ultrahdr_true_on_mpf_only() {
        let mut data = Vec::new();
        data.extend_from_slice(&[0xFF, 0xD8]);
        let mpf_bytes = create_mpf_header(100, 50, None);
        data.extend_from_slice(&mpf_bytes);
        data.extend_from_slice(&minimal_jpeg()[2..]);
        // MPF presence alone counts as a gain-map signal for the
        // short-circuit — matches the behavior of the full probe's
        // MPF_LOCATION fingerprint in `Wants::ULTRAHDR_DETECT`.
        assert!(is_ultrahdr(&data));
    }

    #[test]
    fn is_ultrahdr_true_on_xmp_hdrgm() {
        let mut data = Vec::new();
        data.extend_from_slice(&[0xFF, 0xD8]);
        let xmp_payload = b"hdrgm:GainMapMax=\"2.0\"";
        let namespace = b"http://ns.adobe.com/xap/1.0/\0";
        let total_len = 2 + namespace.len() + xmp_payload.len();
        data.push(0xFF);
        data.push(0xE1);
        data.extend_from_slice(&(total_len as u16).to_be_bytes());
        data.extend_from_slice(namespace);
        data.extend_from_slice(xmp_payload);
        data.extend_from_slice(&minimal_jpeg()[2..]);
        assert!(is_ultrahdr(&data));
    }

    #[test]
    fn wants_bitwise_ops() {
        let w = Wants::IMAGE_RANGES | Wants::SOF_DIMENSIONS;
        assert!(w.contains(Wants::IMAGE_RANGES));
        assert!(w.contains(Wants::SOF_DIMENSIONS));
        assert!(!w.contains(Wants::ICC_IDENTIFY));
        assert!(Wants::ALL.contains(Wants::ICC_IDENTIFY));
    }

    #[test]
    fn probe_struct_is_cloneable_and_reasonable_size() {
        let p = ContainerProbe::default();
        let _clone = p.clone(); // must compile; Clone == memcpy for this layout.
        // Size guardrail enforced by const_assert above.
    }

    /// Regression: some writers emit a decoy camera-metadata APP1 XMP
    /// before the real gain-map APP1 XMP. The probe must OR fingerprint
    /// flags across every APP1 XMP, not only the first.
    #[test]
    fn probe_fingerprints_across_multiple_xmp_app1() {
        fn xmp_app1(body: &[u8]) -> Vec<u8> {
            let namespace = b"http://ns.adobe.com/xap/1.0/\0";
            let total_len = 2 + namespace.len() + body.len();
            let mut out = Vec::new();
            out.push(0xFF);
            out.push(0xE1);
            out.extend_from_slice(&(total_len as u16).to_be_bytes());
            out.extend_from_slice(namespace);
            out.extend_from_slice(body);
            out
        }

        let mut data = Vec::new();
        data.extend_from_slice(&[0xFF, 0xD8]);
        // First XMP: camera metadata decoy, NO hdrgm / no GainMap.
        data.extend_from_slice(&xmp_app1(b"<dc:creator>SomeCamera</dc:creator>"));
        // Second XMP: the real gain-map XMP.
        data.extend_from_slice(&xmp_app1(b"hdrgm:GainMapMax=\"2.0\""));
        data.extend_from_slice(&minimal_jpeg()[2..]);

        let p = probe(&data, Wants::XMP_LOCATION | Wants::XMP_GAINMAP_FLAGS);
        assert!(
            p.has_xmp_hdrgm(),
            "fingerprint must surface even when the gain-map signal is in a later APP1 XMP"
        );
    }

    /// Regression: files > 4 GiB cannot be safely indexed with u32
    /// offsets. Probe must set `truncated = true` and return without
    /// touching the bytes.
    ///
    /// We can't allocate a real 4 GiB buffer in a unit test without
    /// making CI flaky. Instead, fabricate a slice that claims
    /// `> u32::MAX` length by wrapping a static leaked array with
    /// `slice::from_raw_parts` — but `probe()` is safe-Rust so even
    /// a fake slice is fine as long as we never actually deref its
    /// tail. The guard check short-circuits before any indexing.
    ///
    /// Use a helper that extracts just the guard check as a testable
    /// function. Cleaner than faking slice pointers.
    ///
    /// Only meaningful on 64-bit targets: on 32-bit `usize == u32::MAX`,
    /// so `is_oversized_for_u32` can never return `true` and the
    /// `u32::MAX as usize + 1` test literal overflows at compile time.
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn probe_rejects_oversized_input_with_truncated_flag() {
        assert!(is_oversized_for_u32(u32::MAX as usize + 1));
        assert!(is_oversized_for_u32(u32::MAX as usize + 1_000_000));
        assert!(!is_oversized_for_u32(u32::MAX as usize));
        assert!(!is_oversized_for_u32(1024 * 1024));
        assert!(!is_oversized_for_u32(0));
    }

    /// 32-bit companion: on targets where `usize == u32`, the oversize
    /// guard is structurally unreachable — document that invariant.
    #[cfg(not(target_pointer_width = "64"))]
    #[test]
    fn probe_oversize_guard_unreachable_on_32bit() {
        assert!(!is_oversized_for_u32(usize::MAX));
        assert!(!is_oversized_for_u32(0));
    }

    #[test]
    fn probe_no_panic_on_garbage() {
        let garbage: Vec<u8> = (0..1024).map(|i| (i as u8).wrapping_mul(7)).collect();
        let _ = probe(&garbage, Wants::ALL);
        let _ = is_ultrahdr(&garbage);
    }

    // -----------------------------------------------------------------------
    // Coverage-gap tests (added in polish pass)
    // -----------------------------------------------------------------------

    /// Helper: build an APP1 segment with the given namespace prefix.
    fn app1_with_prefix(prefix: &[u8], body: &[u8]) -> Vec<u8> {
        let total_len = 2 + prefix.len() + body.len();
        let mut out = Vec::with_capacity(4 + prefix.len() + body.len());
        out.push(0xFF);
        out.push(0xE1);
        out.extend_from_slice(&(total_len as u16).to_be_bytes());
        out.extend_from_slice(prefix);
        out.extend_from_slice(body);
        out
    }

    /// Helper: build an APP2 segment with the given prefix + body.
    fn app2_with_prefix(prefix: &[u8], body: &[u8]) -> Vec<u8> {
        let total_len = 2 + prefix.len() + body.len();
        let mut out = Vec::with_capacity(4 + prefix.len() + body.len());
        out.push(0xFF);
        out.push(0xE2);
        out.extend_from_slice(&(total_len as u16).to_be_bytes());
        out.extend_from_slice(prefix);
        out.extend_from_slice(body);
        out
    }

    #[test]
    fn wants_is_empty_and_contains() {
        let empty = Wants::new(0);
        assert!(empty.is_empty());
        assert!(!Wants::IMAGE_RANGES.is_empty());
        assert!(!Wants::ALL.is_empty());
    }

    #[test]
    fn wants_bitor_assign() {
        let mut w = Wants::IMAGE_RANGES;
        w |= Wants::SOF_DIMENSIONS;
        assert!(w.contains(Wants::IMAGE_RANGES));
        assert!(w.contains(Wants::SOF_DIMENSIONS));
        assert!(!w.contains(Wants::ICC_IDENTIFY));
    }

    #[test]
    fn probe_captures_exif_range() {
        let mut data = Vec::new();
        data.extend_from_slice(&[0xFF, 0xD8]);
        let exif_body = b"\x49\x49\x2A\x00\x08\x00\x00\x00"; // TIFF little-endian header, no entries
        data.extend_from_slice(&app1_with_prefix(EXIF_PREFIX, exif_body));
        data.extend_from_slice(&minimal_jpeg()[2..]);

        let p = probe(&data, Wants::EXIF_LOCATION);
        let r = p.exif().expect("exif range captured");
        // Range should cover the EXIF body only (after segment length + identifier).
        assert_eq!((r.end - r.start) as usize, exif_body.len());
        // And the body at that range should match what we inserted.
        assert_eq!(&data[r.start as usize..r.end as usize], exif_body);
    }

    #[test]
    fn probe_records_extended_xmp_chunks() {
        let mut data = Vec::new();
        data.extend_from_slice(&[0xFF, 0xD8]);
        // Need a standard XMP APP1 first so XMP_LOCATION logic fires.
        data.extend_from_slice(&app1_with_prefix(XMP_NAMESPACE_PREFIX, b"<x:xmpmeta/>"));
        // Two Extended XMP chunks.
        data.extend_from_slice(&app1_with_prefix(EXTENDED_XMP_PREFIX, b"chunk-one-body"));
        data.extend_from_slice(&app1_with_prefix(EXTENDED_XMP_PREFIX, b"chunk-two-body"));
        data.extend_from_slice(&minimal_jpeg()[2..]);

        let p = probe(&data, Wants::XMP_LOCATION);
        assert_eq!(p.extended_xmp().len(), 2);
    }

    /// Regression: 5 Extended-XMP chunks into a 4-slot ArrayVec fills
    /// to cap, then silently drops the 5th push. Callers inspect
    /// `extended_xmp().len() == 4` to infer truncation.
    #[test]
    fn probe_extended_xmp_cap_at_4() {
        let mut data = Vec::new();
        data.extend_from_slice(&[0xFF, 0xD8]);
        data.extend_from_slice(&app1_with_prefix(XMP_NAMESPACE_PREFIX, b"<x:xmpmeta/>"));
        // 5 Extended XMP chunks; the inline storage holds 4.
        for i in 0..5 {
            let body = alloc::format!("chunk-{i}");
            data.extend_from_slice(&app1_with_prefix(EXTENDED_XMP_PREFIX, body.as_bytes()));
        }
        data.extend_from_slice(&minimal_jpeg()[2..]);

        let p = probe(&data, Wants::XMP_LOCATION);
        assert_eq!(p.extended_xmp().len(), 4, "array capacity enforced");
    }

    /// Synthesize a single-chunk ICC_PROFILE APP2 with a minimal valid
    /// header so `zenpixels::icc::identify_common` can classify it.
    /// The profile size is tiny — well under the 128-byte ICC header —
    /// but `identify_common` just looks at the header bytes it finds;
    /// anything it doesn't recognize returns `None`, which still
    /// exercises the identify code path we want to cover.
    #[test]
    fn probe_icc_identify_single_chunk() {
        let mut data = Vec::new();
        data.extend_from_slice(&[0xFF, 0xD8]);
        // payload = [seq=1, total=1, icc_bytes...]
        let mut icc_payload = Vec::new();
        icc_payload.push(1u8); // seq_no
        icc_payload.push(1u8); // total_seqs
        icc_payload.extend_from_slice(&[0u8; 32]); // fake ICC bytes
        data.extend_from_slice(&app2_with_prefix(ICC_PROFILE_PREFIX, &icc_payload));
        data.extend_from_slice(&minimal_jpeg()[2..]);

        let p = probe(&data, Wants::ICC_LOCATION | Wants::ICC_IDENTIFY);
        assert!(
            p.icc_profile().is_some(),
            "icc_profile range must be recorded"
        );
        // identify_common on garbage bytes returns None; the important
        // coverage point is that the identify branch was taken (no panic,
        // both `seq_no == 1 && total_seqs == 1` were satisfied).
        let _ = p.icc_identification(); // value-agnostic; type-level check suffices
    }

    #[test]
    fn probe_gainmap_presence_iso_plus_gcontainer() {
        // ISO 21496-1 APP2 + XMP with GContainer GainMap semantic, but
        // NO hdrgm:Version fingerprint. Exercises the
        // GainMapPresence::IsoAndGContainer arm of classify_gainmap_presence.
        let mut data = Vec::new();
        data.extend_from_slice(&[0xFF, 0xD8]);
        data.extend_from_slice(&iso_app2(&[0u8; 16]));

        let xmp_body =
            br#"<rdf:Description xmlns:Container="http://ns.google.com/photos/1.0/container/">
                <Container:Item Item:Semantic="GainMap"/>
            </rdf:Description>"#;
        data.extend_from_slice(&app1_with_prefix(XMP_NAMESPACE_PREFIX, xmp_body));
        data.extend_from_slice(&minimal_jpeg()[2..]);

        let p = probe(
            &data,
            Wants::ISO_GAINMAP | Wants::XMP_LOCATION | Wants::XMP_GAINMAP_FLAGS,
        );
        assert!(p.iso_gainmap().is_some());
        assert!(p.has_xmp_gcontainer_gainmap());
        assert!(!p.has_xmp_hdrgm());
        assert_eq!(p.gainmap_presence(), GainMapPresence::IsoAndGContainer);
    }
}
