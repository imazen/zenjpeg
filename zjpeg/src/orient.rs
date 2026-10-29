//! EXIF orientation for the lossy pipeline: read the source's tag, compose it
//! with `--rotate` / `--flip`, and reset the tag once the pixels are upright.

use zenjpeg::lossless::LosslessTransform;
use zenresize::Orientation;

/// The EXIF orientation (1..=8) in an APP1 EXIF payload (with or without the
/// `Exif\0\0` header), or `None` when there is no usable tag.
pub fn exif_orientation(exif: &[u8]) -> Option<u8> {
    let (tiff, at, big) = orientation_value(exif)?;
    let v = read_u16(&tiff[at..], big)?;
    (1..=8).contains(&v).then_some(v as u8)
}

/// Overwrite the orientation tag in place. `false` when there is none.
pub fn set_exif_orientation(exif: &mut [u8], value: u8) -> bool {
    let Some((_, at, big)) = orientation_value(exif) else {
        return false;
    };
    let start = exif.len() - tiff_of(exif).len() + at;
    let bytes = if big {
        u16::from(value).to_be_bytes()
    } else {
        u16::from(value).to_le_bytes()
    };
    exif[start..start + 2].copy_from_slice(&bytes);
    true
}

/// `t` as an orientation (the EXIF value whose correction it is).
pub fn orientation_of(t: LosslessTransform) -> Orientation {
    let exif = match t {
        LosslessTransform::None => 1,
        LosslessTransform::FlipHorizontal => 2,
        LosslessTransform::Rotate180 => 3,
        LosslessTransform::FlipVertical => 4,
        LosslessTransform::Transpose => 5,
        LosslessTransform::Rotate90 => 6,
        LosslessTransform::Transverse => 7,
        LosslessTransform::Rotate270 => 8,
    };
    Orientation::from_exif(exif).unwrap_or_default()
}

/// The transform that applies `o`.
pub fn transform_of(o: Orientation) -> LosslessTransform {
    LosslessTransform::from_exif_orientation(o.to_exif()).unwrap_or(LosslessTransform::None)
}

fn tiff_of(exif: &[u8]) -> &[u8] {
    exif.strip_prefix(b"Exif\0\0").unwrap_or(exif)
}

/// The TIFF block, the offset of the orientation value in it, and whether it
/// is big-endian. Only a SHORT tag with count 1 in IFD0 is used.
fn orientation_value(exif: &[u8]) -> Option<(&[u8], usize, bool)> {
    let tiff = tiff_of(exif);
    let big = match tiff.get(..2)? {
        b"MM" => true,
        b"II" => false,
        _ => return None,
    };
    if read_u16(tiff.get(2..)?, big)? != 42 {
        return None;
    }
    let ifd = read_u32(tiff.get(4..)?, big)? as usize;
    let count = read_u16(tiff.get(ifd..)?, big)? as usize;
    (0..count).find_map(|i| {
        let entry = ifd.checked_add(2 + 12 * i)?;
        let e = tiff.get(entry..entry + 12)?;
        (read_u16(e, big)? == 0x0112
            && read_u16(&e[2..], big)? == 3
            && read_u32(&e[4..], big)? == 1)
            .then_some((tiff, entry + 8, big))
    })
}

fn read_u16(b: &[u8], big: bool) -> Option<u16> {
    let a = [*b.first()?, *b.get(1)?];
    Some(if big {
        u16::from_be_bytes(a)
    } else {
        u16::from_le_bytes(a)
    })
}

fn read_u32(b: &[u8], big: bool) -> Option<u32> {
    let a: [u8; 4] = b.get(..4)?.try_into().ok()?;
    Some(if big {
        u32::from_be_bytes(a)
    } else {
        u32::from_le_bytes(a)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exif(big: bool, o: u16) -> Vec<u8> {
        let mut t = b"Exif\0\0".to_vec();
        let (w16, w32): (fn(u16) -> [u8; 2], fn(u32) -> [u8; 4]) = if big {
            (u16::to_be_bytes, u32::to_be_bytes)
        } else {
            (u16::to_le_bytes, u32::to_le_bytes)
        };
        t.extend_from_slice(if big { b"MM" } else { b"II" });
        t.extend_from_slice(&w16(42));
        t.extend_from_slice(&w32(8));
        t.extend_from_slice(&w16(1));
        t.extend_from_slice(&w16(0x0112));
        t.extend_from_slice(&w16(3));
        t.extend_from_slice(&w32(1));
        t.extend_from_slice(&w16(o));
        t.extend_from_slice(&[0; 6]);
        t
    }

    #[test]
    fn reads_and_resets_both_byte_orders() {
        for big in [false, true] {
            let mut e = exif(big, 6);
            assert_eq!(exif_orientation(&e), Some(6));
            assert!(set_exif_orientation(&mut e, 1));
            assert_eq!(exif_orientation(&e), Some(1));
            assert_eq!(e.len(), exif(big, 1).len());
            assert_eq!(e, exif(big, 1));
        }
    }

    #[test]
    fn transforms_round_trip_through_orientations() {
        for t in [
            LosslessTransform::None,
            LosslessTransform::FlipHorizontal,
            LosslessTransform::FlipVertical,
            LosslessTransform::Rotate90,
            LosslessTransform::Rotate180,
            LosslessTransform::Rotate270,
            LosslessTransform::Transpose,
            LosslessTransform::Transverse,
        ] {
            assert_eq!(transform_of(orientation_of(t)), t);
        }
    }
}
