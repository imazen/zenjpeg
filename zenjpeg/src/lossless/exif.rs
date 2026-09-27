//! EXIF orientation parsing and rewriting for lossless transforms.
//!
//! Parsing delegates to [`zencodec::helpers::parse_exif_orientation`].
//! Writing uses minimal TIFF IFD parsing to modify the Orientation tag (0x0112)
//! in raw EXIF segment data in-place.

use super::coeff_transform::LosslessTransform;

/// EXIF Orientation tag number.
const TAG_ORIENTATION: u16 = 0x0112;

/// Size of `Exif\0\0` prefix in APP1 segment data.
const EXIF_PREFIX_LEN: usize = 6;

/// Minimum TIFF header size: byte order (2) + magic (2) + IFD offset (4).
const TIFF_HEADER_LEN: usize = 8;

/// Parse the EXIF orientation value from raw APP1 segment data.
///
/// The input `exif_data` is the full APP1 segment payload including the `Exif\0\0` prefix.
/// Returns `Some(1..=8)` if the orientation tag is found, `None` otherwise.
///
/// Delegates to [`zencodec::helpers::parse_exif_orientation`] and converts the
/// [`zenpixels::Orientation`] result to a raw EXIF `u8` value.
pub fn parse_exif_orientation(exif_data: &[u8]) -> Option<u8> {
    zencodec::helpers::parse_exif_orientation(exif_data).map(|o| o.to_exif())
}

/// Set the EXIF orientation value in raw APP1 segment data.
///
/// Overwrites the orientation tag value in-place. If the tag doesn't exist,
/// the data is returned unchanged (we don't insert new tags).
///
/// Returns `true` if the tag was found and modified, `false` otherwise.
pub fn set_exif_orientation(exif_data: &mut [u8], orientation: u8) -> bool {
    if exif_data.len() < EXIF_PREFIX_LEN + TIFF_HEADER_LEN {
        return false;
    }

    if &exif_data[..6] != b"Exif\0\0" {
        return false;
    }

    let tiff_start = EXIF_PREFIX_LEN;
    let tiff = &exif_data[tiff_start..];

    let big_endian = match &tiff[0..2] {
        b"MM" => true,
        b"II" => false,
        _ => return false,
    };

    let magic = read_u16(tiff, 2, big_endian);
    if magic != 42 {
        return false;
    }

    let ifd_offset = read_u32(tiff, 4, big_endian) as usize;
    if ifd_offset + 2 > tiff.len() {
        return false;
    }

    let entry_count = read_u16(tiff, ifd_offset, big_endian) as usize;
    let entries_start = ifd_offset + 2;

    for i in 0..entry_count {
        let entry_offset = entries_start + i * 12;
        if entry_offset + 12 > tiff.len() {
            break;
        }

        let tag = read_u16(tiff, entry_offset, big_endian);
        if tag == TAG_ORIENTATION {
            // Write the new orientation value at the value/offset field (offset +8)
            let abs_offset = tiff_start + entry_offset + 8;
            if abs_offset + 2 > exif_data.len() {
                return false;
            }
            write_u16(exif_data, abs_offset, orientation as u16, big_endian);
            return true;
        }

        if tag > TAG_ORIENTATION {
            break;
        }
    }

    false
}

impl LosslessTransform {
    /// Map an EXIF orientation value (1-8) to the corresponding lossless transform.
    ///
    /// Returns `None` for invalid orientation values (0 or >8).
    ///
    /// | EXIF | Meaning         | Transform    |
    /// |------|-----------------|--------------|
    /// | 1    | Normal          | None         |
    /// | 2    | Flip horizontal | FlipHorizontal |
    /// | 3    | Rotate 180      | Rotate180    |
    /// | 4    | Flip vertical   | FlipVertical |
    /// | 5    | Transpose       | Transpose    |
    /// | 6    | Rotate 90 CW    | Rotate90     |
    /// | 7    | Transverse      | Transverse   |
    /// | 8    | Rotate 270 CW   | Rotate270    |
    #[must_use]
    pub fn from_exif_orientation(orientation: u8) -> Option<Self> {
        match orientation {
            1 => Some(Self::None),
            2 => Some(Self::FlipHorizontal),
            3 => Some(Self::Rotate180),
            4 => Some(Self::FlipVertical),
            5 => Some(Self::Transpose),
            6 => Some(Self::Rotate90),
            7 => Some(Self::Transverse),
            8 => Some(Self::Rotate270),
            _ => None,
        }
    }
}

fn read_u16(data: &[u8], offset: usize, big_endian: bool) -> u16 {
    if big_endian {
        u16::from_be_bytes([data[offset], data[offset + 1]])
    } else {
        u16::from_le_bytes([data[offset], data[offset + 1]])
    }
}

fn read_u32(data: &[u8], offset: usize, big_endian: bool) -> u32 {
    if big_endian {
        u32::from_be_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ])
    } else {
        u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ])
    }
}

fn write_u16(data: &mut [u8], offset: usize, value: u16, big_endian: bool) {
    let bytes = if big_endian {
        value.to_be_bytes()
    } else {
        value.to_le_bytes()
    };
    data[offset] = bytes[0];
    data[offset + 1] = bytes[1];
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(big: bool, kind: u16, unsorted: bool) -> Vec<u8> {
        let mut bytes = b"Exif\0\0".to_vec();
        let u16_bytes = |v: u16| {
            if big {
                v.to_be_bytes()
            } else {
                v.to_le_bytes()
            }
        };
        let u32_bytes = |v: u32| {
            if big {
                v.to_be_bytes()
            } else {
                v.to_le_bytes()
            }
        };
        bytes.extend_from_slice(if big { b"MM" } else { b"II" });
        bytes.extend_from_slice(&u16_bytes(42));
        bytes.extend_from_slice(&u32_bytes(8));
        bytes.extend_from_slice(&u16_bytes(if unsorted { 2 } else { 1 }));
        if unsorted {
            bytes.extend_from_slice(&u16_bytes(0x0131)); // Software before Orientation
            bytes.extend_from_slice(&u16_bytes(2));
            bytes.extend_from_slice(&u32_bytes(4));
            bytes.extend_from_slice(b"abc\0");
        }
        bytes.extend_from_slice(&u16_bytes(0x0112));
        bytes.extend_from_slice(&u16_bytes(kind));
        bytes.extend_from_slice(&u32_bytes(1));
        if kind == 3 {
            bytes.extend_from_slice(&u16_bytes(6));
            bytes.extend_from_slice(&[0, 0]);
        } else {
            bytes.extend_from_slice(&u32_bytes(6));
        }
        bytes.extend_from_slice(&u32_bytes(0));
        bytes.extend_from_slice(b"opaque-offset-sensitive-bytes");
        bytes
    }

    #[test]
    fn rewrite_orientation_preserves_every_other_byte() {
        for big in [false, true] {
            for kind in [3, 4] {
                for unsorted in [false, true] {
                    let original = fixture(big, kind, unsorted);
                    assert_eq!(parse_exif_orientation(&original), Some(6));
                    let mut rewritten = original.clone();
                    assert!(set_exif_orientation(&mut rewritten, 1));
                    assert_eq!(parse_exif_orientation(&rewritten), Some(1));
                    let value_start = 6 + 8 + 2 + usize::from(unsorted) * 12 + 8;
                    let size = if kind == 3 { 2 } else { 4 };
                    assert_eq!(&rewritten[..value_start], &original[..value_start]);
                    assert_eq!(
                        &rewritten[value_start + size..],
                        &original[value_start + size..]
                    );
                    assert!(set_exif_orientation(&mut rewritten, 6));
                    assert_eq!(rewritten, original);
                }
            }
        }
    }

    #[test]
    fn rewrite_orientation_rejects_invalid_input_without_mutation() {
        for orientation in [0, 9, 255] {
            let original = fixture(true, 4, false);
            let mut rewritten = original.clone();
            assert!(!set_exif_orientation(&mut rewritten, orientation));
            assert_eq!(rewritten, original);
        }
        let original = fixture(false, 2, false); // ASCII is not an orientation
        let mut rewritten = original.clone();
        assert!(!set_exif_orientation(&mut rewritten, 1));
        assert_eq!(rewritten, original);
        for len in 0..28 {
            let original = fixture(false, 4, false)[..len].to_vec();
            let mut rewritten = original.clone();
            assert!(!set_exif_orientation(&mut rewritten, 1));
            assert_eq!(rewritten, original);
        }
    }
}
