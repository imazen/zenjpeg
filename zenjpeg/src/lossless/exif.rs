//! EXIF orientation parsing and rewriting for lossless transforms.
//!
//! Provides minimal TIFF IFD parsing to read and modify the Orientation tag (0x0112)
//! in raw EXIF segment data. This avoids pulling in a full EXIF library for the
//! single tag we need.

use super::coeff_transform::LosslessTransform;

/// EXIF Orientation tag number.
const TAG_ORIENTATION: u16 = 0x0112;

/// TIFF type SHORT (unsigned 16-bit).
const TIFF_TYPE_SHORT: u16 = 3;

/// Size of `Exif\0\0` prefix in APP1 segment data.
const EXIF_PREFIX_LEN: usize = 6;

/// Minimum TIFF header size: byte order (2) + magic (2) + IFD offset (4).
const TIFF_HEADER_LEN: usize = 8;

/// Parse the EXIF orientation value from raw APP1 segment data.
///
/// The input `exif_data` is the full APP1 segment payload including the `Exif\0\0` prefix.
/// Returns `Some(1..=8)` if the orientation tag is found, `None` otherwise.
pub fn parse_exif_orientation(exif_data: &[u8]) -> Option<u8> {
    // Need at least the Exif prefix + TIFF header
    if exif_data.len() < EXIF_PREFIX_LEN + TIFF_HEADER_LEN {
        return None;
    }

    // Verify Exif\0\0 prefix
    if &exif_data[..6] != b"Exif\0\0" {
        return None;
    }

    let tiff = &exif_data[EXIF_PREFIX_LEN..];

    // Determine byte order
    let big_endian = match &tiff[0..2] {
        b"MM" => true,
        b"II" => false,
        _ => return None,
    };

    // Verify TIFF magic (42)
    let magic = read_u16(tiff, 2, big_endian);
    if magic != 42 {
        return None;
    }

    // Read IFD0 offset
    let ifd_offset = read_u32(tiff, 4, big_endian) as usize;
    if ifd_offset + 2 > tiff.len() {
        return None;
    }

    // Read number of IFD entries
    let entry_count = read_u16(tiff, ifd_offset, big_endian) as usize;
    let entries_start = ifd_offset + 2;

    // Scan IFD entries for orientation tag
    for i in 0..entry_count {
        let entry_offset = entries_start + i * 12;
        if entry_offset + 12 > tiff.len() {
            break;
        }

        let tag = read_u16(tiff, entry_offset, big_endian);
        if tag == TAG_ORIENTATION {
            let type_ = read_u16(tiff, entry_offset + 2, big_endian);
            if type_ != TIFF_TYPE_SHORT {
                return None;
            }
            let value = read_u16(tiff, entry_offset + 8, big_endian);
            if (1..=8).contains(&value) {
                return Some(value as u8);
            }
            return None;
        }

        // IFD entries are sorted by tag — if we've passed 0x0112, stop early
        if tag > TAG_ORIENTATION {
            break;
        }
    }

    None
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
