//! Lossless path detection and execution for the layout pipeline.
//!
//! A transform is lossless-eligible when ALL of these hold:
//! 1. Only orientation changes (no resize constraint)
//! 2. No crop (or MCU-aligned crop — not yet implemented)
//! 3. MCU-aligned dimensions for axis-swapping transforms

use crate::lossless::{EdgeHandling, LosslessTransform, TransformConfig};
use zenlayout::{Command, Constraint, Orientation};

/// Check if a set of layout commands can be handled losslessly.
///
/// Returns the composed `LosslessTransform` if all commands are orientation-only
/// (no resize, no crop, no padding). Returns `None` if lossy path is required.
pub(crate) fn detect_lossless(commands: &[Command]) -> Option<LosslessTransform> {
    let mut has_resize = false;
    let mut has_crop = false;
    let mut has_pad = false;
    let mut has_region = false;
    let mut orientation = Orientation::Identity;

    for cmd in commands {
        match cmd {
            Command::AutoOrient(exif) => {
                if let Some(o) = Orientation::from_exif(*exif) {
                    orientation = orientation.compose(o);
                }
            }
            Command::Rotate(r) => {
                let o = match r {
                    zenlayout::Rotation::Rotate90 => Orientation::Rotate90,
                    zenlayout::Rotation::Rotate180 => Orientation::Rotate180,
                    zenlayout::Rotation::Rotate270 => Orientation::Rotate270,
                };
                orientation = orientation.compose(o);
            }
            Command::Flip(axis) => {
                let o = match axis {
                    zenlayout::FlipAxis::Horizontal => Orientation::FlipH,
                    zenlayout::FlipAxis::Vertical => Orientation::FlipV,
                };
                orientation = orientation.compose(o);
            }
            Command::Constrain(c) => {
                has_resize = constraint_may_resize(c);
            }
            Command::Crop(_) => has_crop = true,
            Command::Region(_) => has_region = true,
            Command::Pad(_) => has_pad = true,
            _ => return None, // Unknown command — conservative: not lossless
        }
    }

    if has_resize || has_crop || has_pad || has_region {
        return None;
    }

    if orientation.is_identity() {
        return Some(LosslessTransform::None);
    }

    orientation_to_lossless(orientation)
}

/// Map zenlayout Orientation to zenjpeg LosslessTransform.
fn orientation_to_lossless(o: Orientation) -> Option<LosslessTransform> {
    Some(match o {
        Orientation::Identity => LosslessTransform::None,
        Orientation::FlipH => LosslessTransform::FlipHorizontal,
        Orientation::FlipV => LosslessTransform::FlipVertical,
        Orientation::Rotate180 => LosslessTransform::Rotate180,
        Orientation::Rotate90 => LosslessTransform::Rotate90,
        Orientation::Rotate270 => LosslessTransform::Rotate270,
        Orientation::Transpose => LosslessTransform::Transpose,
        Orientation::Transverse => LosslessTransform::Transverse,
        _ => return None, // Unknown orientation variant
    })
}

/// Execute the lossless transform path.
pub(crate) fn execute_lossless(
    jpeg_data: &[u8],
    transform: LosslessTransform,
    edge_handling: EdgeHandling,
    stop: &dyn enough::Stop,
) -> crate::error::Result<Vec<u8>> {
    if transform == LosslessTransform::None {
        // No transform needed — return input unchanged.
        return Ok(jpeg_data.to_vec());
    }

    let config = TransformConfig {
        transform,
        edge_handling,
    };
    crate::lossless::transform(jpeg_data, &config, stop)
}

/// Execute lossless restructure: convert to baseline sequential with restart
/// markers for fast parallel decoding. Optionally applies a spatial transform
/// at the same time (one decode/encode pass for both).
pub(crate) fn execute_restructure(
    jpeg_data: &[u8],
    transform: LosslessTransform,
    edge_handling: EdgeHandling,
    stop: &dyn enough::Stop,
) -> crate::error::Result<Vec<u8>> {
    use crate::lossless::{OutputMode, RestartInterval, RestructureConfig};

    let transform_config = if transform == LosslessTransform::None {
        None
    } else {
        Some(TransformConfig {
            transform,
            edge_handling,
        })
    };

    let config = RestructureConfig {
        output_mode: OutputMode::Sequential,
        restart_interval: RestartInterval::EveryMcuRows(4),
        transform: transform_config,
    };
    crate::lossless::restructure(jpeg_data, &config, stop)
}

/// Check if EXIF auto-orient can be applied losslessly without trimming any pixels.
///
/// Uses per-transform trimming rules from `coeff_transform.rs`:
/// - FlipHorizontal: trims if width not MCU-aligned
/// - FlipVertical: trims if height not MCU-aligned
/// - Rotate180: trims if either dimension not MCU-aligned
/// - Rotate90: trims if height not MCU-aligned
/// - Rotate270: trims if width not MCU-aligned
/// - Transpose: NEVER trims
/// - Transverse: trims if either dimension not MCU-aligned
///
/// Returns `Some(exif_value)` if orientation should be applied, `None` if it
/// would require trimming or if orientation is already normal.
pub(crate) fn safe_auto_orient(info: &crate::decode::JpegInfo) -> Option<u8> {
    let exif_orient = info
        .exif
        .as_ref()
        .and_then(|e| crate::lossless::parse_exif_orientation(e))?;

    if exif_orient == 1 {
        return None; // Already normal
    }

    let transform = LosslessTransform::from_exif_orientation(exif_orient)?;

    // Compute MCU dimensions from subsampling
    let (mcu_w, mcu_h) = mcu_dimensions(info.subsampling);
    let w = info.dimensions.width;
    let h = info.dimensions.height;

    let w_aligned = w % mcu_w == 0;
    let h_aligned = h % mcu_h == 0;

    // Per-transform trimming rules (mirrors coeff_transform.rs logic)
    let would_trim = match transform {
        LosslessTransform::None => false,
        LosslessTransform::FlipHorizontal => !w_aligned,
        LosslessTransform::FlipVertical => !h_aligned,
        LosslessTransform::Rotate180 => !w_aligned || !h_aligned,
        LosslessTransform::Rotate90 => !h_aligned,
        LosslessTransform::Rotate270 => !w_aligned,
        LosslessTransform::Transpose => false, // Never trims
        LosslessTransform::Transverse => !w_aligned || !h_aligned,
    };

    if would_trim {
        None
    } else {
        Some(exif_orient)
    }
}

/// Get MCU dimensions for a given subsampling mode.
fn mcu_dimensions(subsampling: crate::types::Subsampling) -> (u32, u32) {
    use crate::types::Subsampling;
    match subsampling {
        Subsampling::S444 => (8, 8),
        Subsampling::S422 => (16, 8),
        Subsampling::S420 => (16, 16),
        Subsampling::S440 => (8, 16),
    }
}

/// Reset EXIF orientation tag to 1 (Normal) in a JPEG byte stream.
///
/// Scans for the APP1 EXIF segment and modifies the orientation tag in-place.
/// Returns true if the tag was found and reset.
pub(crate) fn reset_exif_orientation_in_jpeg(jpeg_data: &mut Vec<u8>) -> bool {
    // Find APP1 marker (0xFF 0xE1) with "Exif\0\0" prefix
    let mut i = 0;
    while i + 1 < jpeg_data.len() {
        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xE1 {
            // APP1 found — read segment length
            if i + 3 >= jpeg_data.len() {
                break;
            }
            let seg_len = u16::from_be_bytes([jpeg_data[i + 2], jpeg_data[i + 3]]) as usize;
            let seg_start = i + 2; // After 0xFF 0xE1
            let seg_end = seg_start + seg_len;

            if seg_end > jpeg_data.len() {
                break;
            }

            // Check for "Exif\0\0" prefix (at offset +2 from segment length)
            let data_start = i + 4; // After marker + length
            if data_start + 6 <= jpeg_data.len()
                && jpeg_data[data_start..data_start + 6] == *b"Exif\0\0"
            {
                // Modify the EXIF data in-place (including prefix)
                crate::lossless::set_exif_orientation(&mut jpeg_data[data_start..seg_end], 1);
                return true;
            }
        }
        i += 1;
    }
    false
}

/// Check if a constraint might cause a resize.
/// Conservative: returns true unless we can prove it won't resize.
fn constraint_may_resize(c: &Constraint) -> bool {
    // If no target dimensions specified, it won't resize
    if c.width.is_none() && c.height.is_none() {
        return false;
    }

    // Any constraint with dimensions conservatively means lossy
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_identity() {
        let commands = vec![];
        assert_eq!(detect_lossless(&commands), Some(LosslessTransform::None));
    }

    #[test]
    fn detect_exif_rotate90() {
        let commands = vec![Command::AutoOrient(6)]; // EXIF 6 = Rotate90
        assert_eq!(
            detect_lossless(&commands),
            Some(LosslessTransform::Rotate90)
        );
    }

    #[test]
    fn detect_composed_orientation() {
        use zenlayout::{FlipAxis, Rotation};

        let commands = vec![
            Command::Rotate(Rotation::Rotate90),
            Command::Flip(FlipAxis::Horizontal),
        ];
        let result = detect_lossless(&commands);
        assert!(result.is_some());
    }

    #[test]
    fn detect_resize_is_lossy() {
        use zenlayout::ConstraintMode;
        let commands = vec![
            Command::AutoOrient(6),
            Command::Constrain(Constraint::new(ConstraintMode::Fit, 800, 600)),
        ];
        assert_eq!(detect_lossless(&commands), None);
    }

    #[test]
    fn detect_crop_is_lossy() {
        let commands = vec![Command::Crop(zenlayout::SourceCrop::pixels(
            10, 10, 100, 100,
        ))];
        assert_eq!(detect_lossless(&commands), None);
    }
}
