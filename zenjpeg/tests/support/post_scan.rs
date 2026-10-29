//! Shared fixture placement for preservation and independent-decoder checks.

use crate::jpeg_writer;

/// `jpeg` with `seg` inserted right before its second SOS (after the first
/// scan's entropy-coded data).
pub fn insert_before_second_sos(jpeg: &[u8], seg: &[u8]) -> Vec<u8> {
    let mut sos_seen = 0;
    let mut p = 2;
    while p + 4 <= jpeg.len() {
        if jpeg[p] == 0xFF && jpeg[p + 1] == 0xDA {
            sos_seen += 1;
            if sos_seen == 2 {
                return [&jpeg[..p], seg, &jpeg[p..]].concat();
            }
            p += 2 + jpeg_writer::be16(jpeg, p + 2);
            while p + 1 < jpeg.len()
                && !(jpeg[p] == 0xFF
                    && jpeg[p + 1] != 0x00
                    && !(0xD0..=0xD7).contains(&jpeg[p + 1]))
            {
                p += 1;
            }
            continue;
        }
        if jpeg[p] == 0xFF && jpeg[p + 1] != 0xFF && jpeg[p + 1] != 0x00 {
            p += 2 + jpeg_writer::be16(jpeg, p + 2);
        } else {
            p += 1;
        }
    }
    panic!("no second SOS");
}
