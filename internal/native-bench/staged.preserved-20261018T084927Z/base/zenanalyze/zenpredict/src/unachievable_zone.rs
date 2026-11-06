//! Size-discriminated **unachievable-zone** fallbacks for codec pickers.
//!
//! A codec picker is asked to hit a quality target `target_zq` for an image.
//! For some `(image size, target_zq)` combinations the target is **physically
//! unreachable** — e.g. a 1 MP photo simply cannot be encoded to SSIMULACRA2
//! 94 by JPEG-XL VarDCT no matter which knobset is chosen; the achievable
//! ceiling is ~93.9. In those zones the picker's normal "argmin predicted
//! bytes over the cells that reach the target" has **no reachable cell** — the
//! reach mask is empty — and what the codec does next is otherwise undefined
//! (a silent fall-through to whatever default the codec keeps).
//!
//! This module makes that boundary **declared data**, not an implicit skip.
//! The bake records, per size class, the achievable `ceiling_zq` and a
//! **fallback knobset** (the cell + scalar that achieves the ceiling — the
//! best-possible encode for that size). At inference the codec checks
//! [`UnachievableZones::resolve`] *before* the argmin: if `target_zq` exceeds
//! the ceiling for the image's size class, it returns the declared fallback
//! and the codec encodes that knobset directly. No empty-mask guessing.
//!
//! It is the deploy-side complement to the training pipeline's
//! `DATA_STARVED_SIZE` safety gate: that gate stops flagging a high-`zq`
//! `(size, zq)` cell as "starved" precisely *because* the bake declares it
//! unachievable here and ships a fallback — the skip is visible and tested,
//! not buried (the project's "no graceful skips" discipline).
//!
//! ## Size discriminant
//!
//! The picker is otherwise a feature-vector-in / output-vector-out function
//! with no notion of image dimensions. The discriminant added here is the
//! single **`feat_pixel_count`** input (raw pixel **area**, `width × height`):
//! [`pixels_feat_idx`](UnachievableZones::pixels_feat_idx) is its index into
//! the bake's feat_cols, and the zones partition the area axis with ascending
//! `pixel_upper` bounds (canonical buckets: tiny ≤ 4096, small ≤ 65536,
//! medium ≤ 1048576, large = +∞). The runtime reads the **raw** (pre-
//! transform) area from the feature vector — matching how the bake derived
//! the boundaries — so a `log1p`-transformed pixel feature is *not* used here.
//!
//! ## Wire format ([`UNACHIEVABLE_ZONES_KEY`], `value_type = bytes`)
//!
//! ```text
//! [2]  pixels_feat_idx: u16 (LE)   index of feat_pixel_count (raw area) into feat_cols
//! [1]  n_zones:         u8
//! repeat n_zones times (ascending by pixel_upper; last bound is +inf):
//!   [4] pixel_upper:    f32 (LE)   size class = pixel_count <= pixel_upper
//!   [4] ceiling_zq:     f32 (LE)   max achievable zq; target_zq > ceiling_zq => unachievable
//!   [1] fallback_cell:  u8         cell index to encode in this zone
//!   [4] fallback_scalar:f32 (LE)   scalar (e.g. effort) for the fallback cell;
//!                                   NaN sentinel = "use the picker's predicted scalar"
//! ```
//!
//! Each zone is 13 bytes; a 3-class table is 3 + 39 = 42 bytes. An absent
//! metadata entry (or empty blob) means "no zones" — the deploy path is
//! byte-identical to a bake that never carried the key (backward-compatible).
//!
//! ## Recommended codec composition
//!
//! ```rust
//! # use zenpredict::{UnachievableZones, ZoneFallback};
//! # fn run(zones: &UnachievableZones, features: &[f32], target_zq: f32) {
//! // Check the declared unachievable zones BEFORE the picker argmin.
//! if let Some(ZoneFallback { cell, scalar }) = zones.resolve(features, target_zq) {
//!     // target is physically unreachable for this size — encode the
//!     // declared best-achievable knobset directly (no argmin).
//!     let _ = (cell, scalar); // codec::encode(cell, scalar_or_predicted)
//!     return;
//! }
//! // ...otherwise: normal predict → reach-mask → argmin path.
//! # }
//! ```

use alloc::vec::Vec;

use crate::error::PredictError;
use crate::metadata::Metadata;

/// Metadata key carrying the packed unachievable-zone table.
///
/// Lives under the `zenpicker.*` namespace (the picker-selection reservation
/// in [`crate::keys`]'s module docs) because a zone constrains which targets
/// the picker can actually serve. Written by `bake_picker.py`,
/// `value_type = bytes`; read by [`parse_unachievable_zones`] /
/// [`Model::unachievable_zones`](crate::Model::unachievable_zones).
pub const UNACHIEVABLE_ZONES_KEY: &str = "zenpicker.unachievable_zones";

/// One size class's reachability boundary plus its fallback knobset.
///
/// A zone matches an image when `pixel_count <= pixel_upper` and the image
/// did not match an earlier (lower-`pixel_upper`) zone — i.e. zones partition
/// the pixel-area axis in ascending order, the last carrying `f32::INFINITY`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnachievableZone {
    /// Upper pixel-area bound (inclusive) of this size class. The final
    /// zone in a table carries `f32::INFINITY` (every larger image).
    pub pixel_upper: f32,
    /// Maximum achievable `zq` for this size class. A request with
    /// `target_zq > ceiling_zq` is physically unreachable → fallback.
    /// `target_zq == ceiling_zq` is *achievable* (the ceiling is reachable).
    pub ceiling_zq: f32,
    /// Cell index to encode when this zone's target is unreachable — the
    /// ceiling-achieving (best-quality) cell for the size class.
    pub fallback_cell: u8,
    /// Scalar (e.g. effort / distance) for [`Self::fallback_cell`]. A NaN
    /// value is the sentinel for "no declared scalar — use the picker's
    /// predicted scalar for this cell" (surfaced as [`ZoneFallback::scalar`]
    /// `= None`).
    pub fallback_scalar: f32,
}

/// Parsed unachievable-zone table: the size-discriminant feature index plus
/// the per-size-class [`UnachievableZone`]s (ascending `pixel_upper`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UnachievableZones {
    /// Index into the bake's feat_cols of the raw-pixel-area feature
    /// (`feat_pixel_count`). [`UnachievableZones::resolve`] reads the
    /// size signal from `features[pixels_feat_idx]`.
    pub pixels_feat_idx: u16,
    /// Size classes, ascending by `pixel_upper`. Empty ⇒ no zones.
    pub zones: Vec<UnachievableZone>,
}

/// The resolved fallback knobset for an unachievable `(size, target_zq)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZoneFallback {
    /// Cell index the codec should encode (the declared best-achievable
    /// cell for the image's size class).
    pub cell: usize,
    /// Scalar to apply to [`Self::cell`]. `None` means the bake declared
    /// no concrete scalar — the codec should use the picker's predicted
    /// scalar for the cell instead.
    pub scalar: Option<f32>,
}

/// Parse the packed unachievable-zone wire blob (see the [module docs](self)
/// for the byte layout) into an [`UnachievableZones`].
///
/// An empty input yields an empty table (the "no zones" case). Returns
/// [`PredictError::Truncated`] if the blob ends mid-record.
pub fn parse_unachievable_zones(bytes: &[u8]) -> Result<UnachievableZones, PredictError> {
    if bytes.is_empty() {
        return Ok(UnachievableZones::default());
    }
    let mut pos = 0usize;
    // [2] pixels_feat_idx
    let pixels_feat_idx = read_u16_le(bytes, &mut pos)?;
    // [1] n_zones
    let n_zones = read_u8(bytes, &mut pos)? as usize;

    let mut zones = Vec::with_capacity(n_zones);
    for _ in 0..n_zones {
        let pixel_upper = read_f32_le(bytes, &mut pos)?;
        let ceiling_zq = read_f32_le(bytes, &mut pos)?;
        let fallback_cell = read_u8(bytes, &mut pos)?;
        let fallback_scalar = read_f32_le(bytes, &mut pos)?;
        zones.push(UnachievableZone {
            pixel_upper,
            ceiling_zq,
            fallback_cell,
            fallback_scalar,
        });
    }
    Ok(UnachievableZones {
        pixels_feat_idx,
        zones,
    })
}

/// Read the unachievable-zone table from a parsed [`Metadata`] blob.
///
/// Returns an empty table when [`UNACHIEVABLE_ZONES_KEY`] is absent (a bake
/// from before the feature, or a picker whose every `(size, zq)` cell is
/// reachable).
pub fn unachievable_zones_from_metadata(
    meta: &Metadata<'_>,
) -> Result<UnachievableZones, PredictError> {
    match meta.get(UNACHIEVABLE_ZONES_KEY) {
        None => Ok(UnachievableZones::default()),
        Some(entry) => parse_unachievable_zones(entry.value),
    }
}

impl UnachievableZones {
    /// `true` when no zones are declared (every target is treated as
    /// reachable — `resolve` always returns `None`).
    pub fn is_empty(&self) -> bool {
        self.zones.is_empty()
    }

    /// Resolve the fallback knobset for a `(features, target_zq)` request, or
    /// `None` when the target is reachable (the normal picker path should run).
    ///
    /// Reads the raw pixel area from `features[pixels_feat_idx]`, finds the
    /// matching size class (first zone with `pixel_count <= pixel_upper`), and
    /// returns [`ZoneFallback`] iff `target_zq > ceiling_zq` for that class.
    ///
    /// Returns `None` (no override — defer to the picker) when:
    /// - there are no zones,
    /// - `pixels_feat_idx` is past the supplied feature vector,
    /// - the pixel feature is non-finite (NaN/∞ — size can't be determined), or
    /// - the target is at or below the size class's achievable ceiling.
    ///
    /// Panic-free: out-of-range indices defer rather than panic.
    pub fn resolve(&self, features: &[f32], target_zq: f32) -> Option<ZoneFallback> {
        if self.zones.is_empty() {
            return None;
        }
        let &pixels = features.get(self.pixels_feat_idx as usize)?;
        // A non-finite size signal can't be bucketed — defer to the picker
        // rather than risk forcing the smallest-class fallback on a NaN.
        if !pixels.is_finite() {
            return None;
        }
        // First zone (ascending pixel_upper) whose bound contains the area.
        let zone = self.zones.iter().find(|z| pixels <= z.pixel_upper)?;
        // The ceiling itself is achievable; only a strictly-higher target is
        // unreachable.
        if target_zq > zone.ceiling_zq {
            Some(ZoneFallback {
                cell: zone.fallback_cell as usize,
                scalar: if zone.fallback_scalar.is_nan() {
                    None
                } else {
                    Some(zone.fallback_scalar)
                },
            })
        } else {
            None
        }
    }
}

// --- little-endian cursor readers (panic-free, bounds-checked) ---
// Mirrors the readers in `knob_veto` — kept module-local so the two overlay
// parsers stay independent (neither can break the other under refactor).

#[inline]
fn read_u8(bytes: &[u8], pos: &mut usize) -> Result<u8, PredictError> {
    let b = *bytes.get(*pos).ok_or(PredictError::Truncated {
        offset: *pos,
        want: 1,
        have: bytes.len().saturating_sub(*pos),
    })?;
    *pos += 1;
    Ok(b)
}

#[inline]
fn read_u16_le(bytes: &[u8], pos: &mut usize) -> Result<u16, PredictError> {
    Ok(u16::from_le_bytes(read_array::<2>(bytes, pos)?))
}

#[inline]
fn read_f32_le(bytes: &[u8], pos: &mut usize) -> Result<f32, PredictError> {
    Ok(f32::from_le_bytes(read_array::<4>(bytes, pos)?))
}

#[inline]
fn read_array<const N: usize>(bytes: &[u8], pos: &mut usize) -> Result<[u8; N], PredictError> {
    let end = pos.checked_add(N).ok_or(PredictError::Truncated {
        offset: *pos,
        want: N,
        have: bytes.len().saturating_sub(*pos),
    })?;
    let slice = bytes.get(*pos..end).ok_or(PredictError::Truncated {
        offset: *pos,
        want: N,
        have: bytes.len().saturating_sub(*pos),
    })?;
    let arr: [u8; N] = slice.try_into().map_err(|_| PredictError::Truncated {
        offset: *pos,
        want: N,
        have: bytes.len().saturating_sub(*pos),
    })?;
    *pos = end;
    Ok(arr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::{Metadata, MetadataType};

    /// Encode the wire blob the way `bake_picker.py` does, so the tests
    /// exercise the exact byte layout the baker emits.
    fn encode(pixels_feat_idx: u16, zones: &[(f32, f32, u8, f32)]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&pixels_feat_idx.to_le_bytes());
        out.push(zones.len() as u8);
        for &(pixel_upper, ceiling_zq, fallback_cell, fallback_scalar) in zones {
            out.extend_from_slice(&pixel_upper.to_le_bytes());
            out.extend_from_slice(&ceiling_zq.to_le_bytes());
            out.push(fallback_cell);
            out.extend_from_slice(&fallback_scalar.to_le_bytes());
        }
        out
    }

    /// Canonical jxl-lossy-shaped table: tiny/small/medium/large with the
    /// measured ssim2 ceilings. feat_pixel_count at index 32 (raw area).
    fn canonical() -> UnachievableZones {
        parse_unachievable_zones(&encode(
            32,
            &[
                (4096.0, 91.7, 2, 9.0),        // tiny  → cell 2 (vd_zen) effort 9
                (65536.0, 91.8, 2, 9.0),       // small
                (1_048_576.0, 93.9, 2, 9.0),   // medium
                (f32::INFINITY, 95.0, 2, 9.0), // large
            ],
        ))
        .unwrap()
    }

    #[test]
    fn empty_blob_yields_no_zones() {
        let z = parse_unachievable_zones(&[]).unwrap();
        assert!(z.is_empty());
        assert!(z.resolve(&[0.0; 50], 99.0).is_none());
    }

    #[test]
    fn roundtrip_four_classes() {
        let z = canonical();
        assert_eq!(z.pixels_feat_idx, 32);
        assert_eq!(z.zones.len(), 4);
        assert_eq!(z.zones[0].pixel_upper, 4096.0);
        assert!((z.zones[2].ceiling_zq - 93.9).abs() < 1e-4);
        assert_eq!(z.zones[3].pixel_upper, f32::INFINITY);
    }

    #[test]
    fn medium_zq94_is_unachievable() {
        // A 1 MP (medium) image at target ssim2 94 → unreachable (ceiling 93.9).
        let z = canonical();
        let mut feats = [0.0f32; 50];
        feats[32] = 1_000_000.0; // medium
        let fb = z
            .resolve(&feats, 94.0)
            .expect("medium/zq94 is unachievable");
        assert_eq!(fb.cell, 2);
        assert_eq!(fb.scalar, Some(9.0));
    }

    #[test]
    fn medium_zq90_is_achievable() {
        // Same medium image at zq90 (< ceiling 93.9) → reachable → no override.
        let z = canonical();
        let mut feats = [0.0f32; 50];
        feats[32] = 1_000_000.0;
        assert!(z.resolve(&feats, 90.0).is_none());
    }

    #[test]
    fn ceiling_exactly_is_achievable() {
        // target_zq == ceiling_zq is reachable (strictly-greater is the test).
        let z = canonical();
        let mut feats = [0.0f32; 50];
        feats[32] = 1_000_000.0;
        assert!(z.resolve(&feats, 93.9).is_none());
        // A hair above → unreachable.
        assert!(z.resolve(&feats, 93.91).is_some());
    }

    #[test]
    fn size_class_boundary_is_inclusive() {
        // Exactly 4096 px (= 64×64) is tiny (<=), so its ceiling is the tiny one.
        let z = canonical();
        let mut feats = [0.0f32; 50];
        feats[32] = 4096.0; // tiny boundary
        // tiny ceiling 91.7: zq92 unreachable.
        assert!(z.resolve(&feats, 92.0).is_some());
        // 4097 px falls into small (ceiling 91.8): zq92 still unreachable but
        // via the small zone — just assert it resolves to a zone.
        feats[32] = 4097.0;
        assert!(z.resolve(&feats, 92.0).is_some());
    }

    #[test]
    fn large_image_uses_infinity_zone() {
        let z = canonical();
        let mut feats = [0.0f32; 50];
        feats[32] = 50_000_000.0; // huge → large zone (ceiling 95.0)
        assert!(z.resolve(&feats, 96.0).is_some());
        assert!(z.resolve(&feats, 94.0).is_none());
    }

    #[test]
    fn nan_pixel_defers_to_picker() {
        let z = canonical();
        let mut feats = [0.0f32; 50];
        feats[32] = f32::NAN;
        // Can't determine size → no override even at an absurd target.
        assert!(z.resolve(&feats, 99.0).is_none());
    }

    #[test]
    fn feat_idx_past_vector_defers() {
        let z =
            parse_unachievable_zones(&encode(99, &[(f32::INFINITY, 90.0, 0, f32::NAN)])).unwrap();
        // pixels_feat_idx 99 > supplied len → defer, no panic.
        assert!(z.resolve(&[1.0, 2.0], 99.0).is_none());
    }

    #[test]
    fn nan_fallback_scalar_means_use_predicted() {
        let z =
            parse_unachievable_zones(&encode(0, &[(f32::INFINITY, 50.0, 7, f32::NAN)])).unwrap();
        let fb = z.resolve(&[1.0], 80.0).unwrap();
        assert_eq!(fb.cell, 7);
        assert_eq!(fb.scalar, None); // NaN sentinel → use predicted
    }

    #[test]
    fn roundtrip_through_metadata_blob() {
        let blob = encode(32, &[(1_048_576.0, 93.9, 2, 9.0)]);
        let mut md = Vec::new();
        let key = UNACHIEVABLE_ZONES_KEY.as_bytes();
        md.push(key.len() as u8);
        md.extend_from_slice(key);
        md.push(0); // value_type = 0 (Bytes)
        md.extend_from_slice(&(blob.len() as u32).to_le_bytes());
        md.extend_from_slice(&blob);

        let meta = Metadata::parse(&md).unwrap();
        assert_eq!(
            meta.get(UNACHIEVABLE_ZONES_KEY).unwrap().kind,
            MetadataType::Bytes
        );
        let z = unachievable_zones_from_metadata(&meta).unwrap();
        assert_eq!(z.zones.len(), 1);
        assert_eq!(z.zones[0].fallback_cell, 2);
    }

    #[test]
    fn absent_metadata_key_is_no_zones() {
        let meta = Metadata::parse(&[]).unwrap();
        assert!(unachievable_zones_from_metadata(&meta).unwrap().is_empty());
    }

    #[test]
    fn truncated_blob_is_rejected() {
        // Header claims 1 zone but no record follows.
        let mut blob = Vec::new();
        blob.extend_from_slice(&32u16.to_le_bytes());
        blob.push(1); // n_zones = 1, but no zone bytes
        assert!(matches!(
            parse_unachievable_zones(&blob),
            Err(PredictError::Truncated { .. })
        ));
        // Mid-zone truncation (pixel_upper present, rest missing).
        blob.extend_from_slice(&4096.0f32.to_le_bytes());
        assert!(matches!(
            parse_unachievable_zones(&blob),
            Err(PredictError::Truncated { .. })
        ));
    }
}
