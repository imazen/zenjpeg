//! `--strip`: remove metadata from an encoded JPEG without touching its scans.
//!
//! Every pipeline (lossless transform, restructure, re-encode) writes its
//! output first and then passes it through [`strip`], so a `--strip` choice
//! means the same bytes go on every path.

use anyhow::{Result, bail};
use zenjpeg::container::{self, MarkerKind};

const XMP_NS: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
const XMP_EXT_NS: &[u8] = b"http://ns.adobe.com/xmp/extension/\0";
const ICC_SIG: &[u8] = b"ICC_PROFILE\0";
const MPF_SIG: &[u8] = b"MPF\0";
const ISO_21496_URN: &[u8] = b"urn:iso:std:iso:ts:21496:-1\0";
/// The Ultra HDR gain-map namespace; a packet declaring it signals a gain map.
const HDRGM_NS: &[u8] = b"http://ns.adobe.com/hdr-gain-map/1.0/";

/// What to remove.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StripMask {
    pub exif: bool,
    pub icc: bool,
    pub xmp: bool,
    pub gain_maps: bool,
}

impl StripMask {
    pub fn is_empty(self) -> bool {
        self == Self::default()
    }
}

/// `jpeg` without the metadata `mask` names, plus notes about removals the
/// caller did not name explicitly (an XMP packet dropped with its gain map).
///
/// - EXIF: every APP1 `Exif\0` segment.
/// - ICC: every APP2 `ICC_PROFILE` chunk.
/// - XMP: the standard packet and every extended-XMP chunk.
/// - Gain maps: the MPF index, the ISO 21496-1 records, every image after the
///   primary, and an XMP packet that declares the Ultra HDR `hdrgm`
///   namespace (zenjpeg has no XMP editor, so the packet goes whole; the
///   note says so). Refused when the file carries other images after the
///   primary (an MPF index listing more than gain maps, or bytes after the
///   primary that no MPF entry covers): removing them would lose data the
///   caller did not ask to remove.
pub fn strip(jpeg: &[u8], mask: StripMask) -> Result<(Vec<u8>, Vec<&'static str>)> {
    let mut notes = Vec::new();
    if mask.is_empty() {
        return Ok((jpeg.to_vec(), notes));
    }
    let Some(primary) = container::primary_bounds(jpeg) else {
        bail!("cannot strip metadata: no complete primary JPEG");
    };
    let header_xmp_has_gain_map = mask.gain_maps
        && !mask.xmp
        && container::iter(jpeg).any(|s| {
            matches!(s.kind, MarkerKind::App(1))
                && s.payload.starts_with(XMP_NS)
                && contains(s.payload, HDRGM_NS)
        });
    if header_xmp_has_gain_map {
        notes.push("removed the XMP packet with the gain map (it declares the hdrgm namespace)");
    }
    let drop_xmp = mask.xmp || header_xmp_has_gain_map;
    if mask.gain_maps && primary.end < jpeg.len() {
        check_only_gain_maps_follow(jpeg, primary.end)?;
    }

    let mut out = Vec::with_capacity(primary.end);
    let mut at = 0;
    for span in container::iter(jpeg) {
        let end = span.offset + span.length;
        let drop = match span.kind {
            MarkerKind::App(1) if mask.exif && span.payload.starts_with(b"Exif\0") => true,
            MarkerKind::App(1)
                if drop_xmp
                    && (span.payload.starts_with(XMP_NS)
                        || span.payload.starts_with(XMP_EXT_NS)) =>
            {
                true
            }
            MarkerKind::App(2) if mask.icc && span.payload.starts_with(ICC_SIG) => true,
            MarkerKind::App(2)
                if mask.gain_maps
                    && (span.payload.starts_with(MPF_SIG)
                        || span.payload.starts_with(ISO_21496_URN)) =>
            {
                true
            }
            _ => false,
        };
        // Keep whatever sits between segments (fill bytes) with what follows.
        if drop {
            out.extend_from_slice(&jpeg[at..span.offset]);
        } else {
            out.extend_from_slice(&jpeg[at..end]);
        }
        at = end;
        if matches!(span.kind, MarkerKind::Eoi) {
            break;
        }
    }
    if at != primary.end {
        bail!(
            "cannot strip metadata: the marker walk ended at {at}, the primary at {}",
            primary.end
        );
    }
    if !mask.gain_maps {
        out.extend_from_slice(&jpeg[primary.end..]);
    }
    Ok((out, notes))
}

/// The bytes after the primary hold only MPF-listed gain maps.
fn check_only_gain_maps_follow(jpeg: &[u8], primary_end: usize) -> Result<()> {
    let Ok(index) = container::mpf::parse_mpf(jpeg) else {
        bail!(
            "--strip gainmaps: {} bytes follow the primary image and no MPF index describes \
             them; not removing data that may not be a gain map",
            jpeg.len() - primary_end
        );
    };
    let mut covered = primary_end;
    for entry in index.iter().skip(1) {
        // Ultra HDR marks its gain map with the MPF type Undefined.
        if entry.image_type != container::MpImageType::Undefined {
            bail!(
                "--strip gainmaps: the MPF index lists a secondary image that is not a gain map; \
                 not removing it"
            );
        }
        covered = covered.max(entry.offset.saturating_add(entry.size));
    }
    if covered < jpeg.len() {
        bail!(
            "--strip gainmaps: {} bytes after the gain map are not listed in the MPF index; \
             not removing them",
            jpeg.len() - covered
        );
    }
    Ok(())
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use enough::Unstoppable;
    use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, Exif, PixelLayout};

    const GAIN_MAP_XMP: &str = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/" hdrgm:Version="1.0"/></rdf:RDF></x:xmpmeta>"#;

    /// A 16x16 primary carrying EXIF, ICC and XMP (`xmp`).
    fn primary(xmp: &str) -> Vec<u8> {
        let px = vec![120u8; 16 * 16 * 3];
        let mut icc = vec![0u8; 3000];
        icc[0..4].copy_from_slice(&3000u32.to_be_bytes());
        icc[36..40].copy_from_slice(b"acsp");
        EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
            .request()
            .exif(Exif::build().copyright("zjpeg strip test"))
            .icc_profile(&icc)
            .xmp(xmp.as_bytes())
            .encode(bytemuck::cast_slice::<u8, rgb::RGB8>(&px), 16, 16)
            .unwrap()
    }

    fn gray(w: u32, h: u32) -> Vec<u8> {
        let px: Vec<u8> = (0..w * h).map(|i| (i * 37 % 256) as u8).collect();
        let mut e = EncoderConfig::grayscale(80.0)
            .encode_from_bytes(w, h, PixelLayout::Gray8Srgb)
            .unwrap();
        e.push_packed(&px, Unstoppable).unwrap();
        e.finish().unwrap()
    }

    /// `primary` with an MPF index right after SOI listing `map`, then `map`.
    fn with_gain_map(primary: &[u8], map: &[u8]) -> Vec<u8> {
        let probe = container::mpf::create_mpf_header(0, map.len(), Some(2));
        let mpf =
            container::mpf::create_mpf_header(primary.len() + probe.len(), map.len(), Some(2));
        [&primary[..2], &mpf, &primary[2..], map].concat()
    }

    /// The bytes from the first SOS to the primary's end: the image data.
    fn scans(jpeg: &[u8]) -> &[u8] {
        let sos = container::iter(jpeg)
            .find(|s| matches!(s.kind, MarkerKind::Sos))
            .unwrap()
            .offset;
        &jpeg[sos..container::primary_bounds(jpeg).unwrap().end]
    }

    fn count(jpeg: &[u8], needle: &[u8]) -> usize {
        jpeg.windows(needle.len()).filter(|w| *w == needle).count()
    }

    #[test]
    fn each_kind_goes_alone_and_the_scans_stay() {
        let src = primary("<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"/>");
        for (mask, gone) in [
            (
                StripMask {
                    exif: true,
                    ..Default::default()
                },
                b"Exif\0".as_slice(),
            ),
            (
                StripMask {
                    icc: true,
                    ..Default::default()
                },
                ICC_SIG,
            ),
            (
                StripMask {
                    xmp: true,
                    ..Default::default()
                },
                XMP_NS,
            ),
        ] {
            let (out, notes) = strip(&src, mask).unwrap();
            assert_eq!(count(&out, gone), 0, "{mask:?}");
            assert!(notes.is_empty());
            for kept in [b"Exif\0".as_slice(), ICC_SIG, XMP_NS] {
                if kept != gone {
                    assert_eq!(count(&out, kept), count(&src, kept), "{mask:?}");
                }
            }
            assert_eq!(scans(&out), scans(&src), "{mask:?}");
        }
    }

    #[test]
    fn gain_maps_go_with_their_index_and_signalling() {
        let src = with_gain_map(&primary(GAIN_MAP_XMP), &gray(8, 8));
        assert_eq!(container::mpf::parse_mpf(&src).unwrap().len(), 2);
        let mask = StripMask {
            gain_maps: true,
            ..Default::default()
        };
        let (out, notes) = strip(&src, mask).unwrap();
        assert_eq!(count(&out, MPF_SIG), 0);
        assert_eq!(count(&out, b"hdrgm"), 0);
        assert_eq!(container::find_jpeg_boundaries(&out).len(), 1);
        assert_eq!(notes.len(), 1, "the XMP packet goes with the gain map");
        assert_eq!(count(&out, b"Exif\0"), 1);
        assert_eq!(count(&out, ICC_SIG), 1);
        assert_eq!(scans(&out), scans(&src));
        // Kept when not asked: other kinds leave the map and its index alone.
        let (kept, _) = strip(
            &src,
            StripMask {
                exif: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(container::find_jpeg_boundaries(&kept).len(), 2);
        assert_eq!(container::mpf::parse_mpf(&kept).unwrap().len(), 2);
    }

    #[test]
    fn gain_map_strip_refuses_other_trailing_data() {
        // Bytes after the primary that no MPF entry lists (a Motion Photo's
        // video, say) are not a gain map to remove.
        let mut src = primary(GAIN_MAP_XMP);
        src.extend_from_slice(b"\0\0\0\x18ftypmp42 trailing video");
        let mask = StripMask {
            gain_maps: true,
            ..Default::default()
        };
        assert!(strip(&src, mask).is_err());
        let (exif_only, _) = strip(
            &src,
            StripMask {
                exif: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(exif_only.ends_with(b"trailing video"));
    }
}
