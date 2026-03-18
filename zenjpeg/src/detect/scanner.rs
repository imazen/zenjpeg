//! Lightweight JPEG marker scanner for header-only probing.
//!
//! Parses marker headers without entropy decoding. Extracts DQT tables,
//! SOF parameters, DHT symbol counts, APP markers, and SOS count.

use alloc::vec::Vec;

use crate::foundation::consts::{
    JPEG_NATURAL_ORDER, MARKER_APP0, MARKER_APP2, MARKER_DQT, MARKER_EOI, MARKER_SOF0, MARKER_SOF1,
    MARKER_SOF2, MARKER_SOF9, MARKER_SOF10, MARKER_SOI, MARKER_SOS,
};

/// Raw data extracted from JPEG headers.
#[derive(Debug)]
pub(crate) struct ScanResult {
    /// DQT tables in natural (row-major) order, indexed by table ID (0-3).
    pub dqt_tables: [Option<DqtEntry>; 4],
    /// SOF parameters (None if no SOF found).
    pub sof: Option<SofInfo>,
    /// Total AC Huffman symbol count (sum across all AC tables).
    pub total_ac_symbols: u16,
    /// Number of DHT tables found.
    pub dht_count: u8,
    /// Whether a JFIF APP0 marker was found.
    pub has_jfif: bool,
    /// Whether an ICC_PROFILE APP2 marker was found.
    pub has_icc_profile: bool,
    /// Whether an Adobe APP14 marker was found.
    pub has_adobe: bool,
    /// Whether a Photoshop 3.0 APP13 (IPTC) marker was found.
    pub has_photoshop_iptc: bool,
    /// Number of SOS markers found.
    pub sos_count: u16,
}

/// A single DQT table entry.
#[derive(Debug, Clone)]
pub(crate) struct DqtEntry {
    /// Quantization values in natural (row-major) order.
    pub values: [u16; 64],
    /// Precision: 0 = 8-bit, 1 = 16-bit.
    pub precision: u8,
}

/// SOF (Start of Frame) parameters.
#[derive(Debug, Clone)]
pub(crate) struct SofInfo {
    /// SOF marker type (0xC0, 0xC1, 0xC2, 0xC9, 0xCA).
    pub marker: u8,
    /// Image width in pixels.
    pub width: u16,
    /// Image height in pixels.
    pub height: u16,
    /// Number of components.
    pub num_components: u8,
    /// Per-component info: (id, h_samp, v_samp, quant_table_idx).
    pub components: Vec<(u8, u8, u8, u8)>,
}

/// Errors from scanning.
#[derive(Debug)]
pub(crate) enum ScanError {
    TooShort,
    NotJpeg,
    Truncated,
}

/// Scan JPEG headers and extract structural information.
///
/// Reads only marker headers — no entropy decoding. Stops fully parsing
/// after finding SOS but continues scanning for additional SOS markers
/// to count scans in progressive files.
pub(crate) fn scan_headers(data: &[u8]) -> Result<ScanResult, ScanError> {
    if data.len() < 2 {
        return Err(ScanError::TooShort);
    }

    // Check SOI marker
    if data[0] != 0xFF || data[1] != MARKER_SOI {
        return Err(ScanError::NotJpeg);
    }

    let mut result = ScanResult {
        dqt_tables: [const { None }; 4],
        sof: None,
        total_ac_symbols: 0,
        dht_count: 0,
        has_jfif: false,
        has_icc_profile: false,
        has_adobe: false,
        has_photoshop_iptc: false,
        sos_count: 0,
    };

    let mut pos = 2;

    loop {
        // Find next marker
        pos = match find_marker(data, pos) {
            Some(p) => p,
            None => break,
        };

        if pos + 1 >= data.len() {
            break;
        }

        let marker = data[pos + 1];
        pos += 2; // Skip 0xFF and marker byte

        match marker {
            MARKER_EOI => break,

            MARKER_DQT => {
                pos = parse_dqt(data, pos, &mut result)?;
            }

            m if is_sof_marker(m) => {
                pos = parse_sof(data, pos, m, &mut result)?;
            }

            // DHT (0xC4)
            0xC4 => {
                pos = parse_dht(data, pos, &mut result)?;
            }

            MARKER_SOS => {
                result.sos_count += 1;
                // Skip the SOS header
                if pos + 1 >= data.len() {
                    break;
                }
                let len = read_u16(data, pos) as usize;
                pos += len;
                // Skip entropy data until next marker
                pos = skip_entropy_data(data, pos);
            }

            // APP0 (JFIF check)
            MARKER_APP0 => {
                pos = parse_app0(data, pos, &mut result)?;
            }

            // APP2 (ICC profile check)
            MARKER_APP2 => {
                pos = parse_app2(data, pos, &mut result)?;
            }

            // APP13 (IPTC / Photoshop 3.0 check)
            0xED => {
                pos = parse_app13(data, pos, &mut result)?;
            }

            // APP14 (Adobe check, 0xEE)
            0xEE => {
                pos = parse_app14(data, pos, &mut result)?;
            }

            // Restart markers (0xD0-0xD7) — no length field
            0xD0..=0xD7 => {
                // No payload, continue
            }

            // All other markers have a length field
            _ => {
                if pos + 1 >= data.len() {
                    break;
                }
                let len = read_u16(data, pos) as usize;
                if len < 2 || pos + len > data.len() {
                    break;
                }
                pos += len;
            }
        }
    }

    Ok(result)
}

/// Find the next 0xFF marker byte, skipping any padding 0xFF bytes.
fn find_marker(data: &[u8], mut pos: usize) -> Option<usize> {
    while pos < data.len() {
        if data[pos] == 0xFF {
            // Skip padding 0xFF bytes
            while pos + 1 < data.len() && data[pos + 1] == 0xFF {
                pos += 1;
            }
            if pos + 1 < data.len() && data[pos + 1] != 0x00 {
                return Some(pos);
            }
        }
        pos += 1;
    }
    None
}

/// Check if a marker byte is a SOF marker we care about.
fn is_sof_marker(m: u8) -> bool {
    matches!(
        m,
        MARKER_SOF0 | MARKER_SOF1 | MARKER_SOF2 | MARKER_SOF9 | MARKER_SOF10
    )
}

/// Parse DQT marker segment. Can contain multiple tables.
fn parse_dqt(data: &[u8], pos: usize, result: &mut ScanResult) -> Result<usize, ScanError> {
    if pos + 1 >= data.len() {
        return Err(ScanError::Truncated);
    }
    let seg_len = read_u16(data, pos) as usize;
    if seg_len < 2 || pos + seg_len > data.len() {
        return Err(ScanError::Truncated);
    }

    let seg_end = pos + seg_len;
    let mut p = pos + 2; // Skip length field

    while p < seg_end {
        if p >= data.len() {
            return Err(ScanError::Truncated);
        }

        let pq_tq = data[p];
        let precision = (pq_tq >> 4) & 0x0F; // 0 = 8-bit, 1 = 16-bit
        let table_id = (pq_tq & 0x0F) as usize;
        p += 1;

        if table_id > 3 {
            // Invalid table ID, skip
            let value_bytes = if precision == 0 { 64 } else { 128 };
            p += value_bytes;
            continue;
        }

        let mut values_zigzag = [0u16; 64];

        if precision == 0 {
            // 8-bit values
            if p + 64 > data.len() {
                return Err(ScanError::Truncated);
            }
            for i in 0..64 {
                values_zigzag[i] = data[p + i] as u16;
            }
            p += 64;
        } else {
            // 16-bit values
            if p + 128 > data.len() {
                return Err(ScanError::Truncated);
            }
            for i in 0..64 {
                values_zigzag[i] = read_u16(data, p + i * 2);
            }
            p += 128;
        }

        // Convert from zigzag to natural (row-major) order
        let mut values_natural = [0u16; 64];
        for zigzag_idx in 0..64 {
            let natural_idx = JPEG_NATURAL_ORDER[zigzag_idx] as usize;
            values_natural[natural_idx] = values_zigzag[zigzag_idx];
        }

        result.dqt_tables[table_id] = Some(DqtEntry {
            values: values_natural,
            precision,
        });
    }

    Ok(seg_end)
}

/// Parse SOF marker segment.
fn parse_sof(
    data: &[u8],
    pos: usize,
    marker: u8,
    result: &mut ScanResult,
) -> Result<usize, ScanError> {
    if pos + 1 >= data.len() {
        return Err(ScanError::Truncated);
    }
    let seg_len = read_u16(data, pos) as usize;
    if seg_len < 8 || pos + seg_len > data.len() {
        return Err(ScanError::Truncated);
    }

    let precision = data[pos + 2];
    let _ = precision; // We don't need sample precision for detection
    let height = read_u16(data, pos + 3);
    let width = read_u16(data, pos + 5);
    let num_components = data[pos + 7];

    let expected_len = 8 + num_components as usize * 3;
    if seg_len < expected_len || pos + expected_len > data.len() {
        return Err(ScanError::Truncated);
    }

    let mut components = Vec::new();
    for c in 0..num_components as usize {
        let offset = pos + 8 + c * 3;
        let id = data[offset];
        let sampling = data[offset + 1];
        let h_samp = (sampling >> 4) & 0x0F;
        let v_samp = sampling & 0x0F;
        let quant_table_idx = data[offset + 2];
        components.push((id, h_samp, v_samp, quant_table_idx));
    }

    result.sof = Some(SofInfo {
        marker,
        width,
        height,
        num_components,
        components,
    });

    Ok(pos + seg_len)
}

/// Parse DHT marker segment. Counts total AC symbols across all tables.
fn parse_dht(data: &[u8], pos: usize, result: &mut ScanResult) -> Result<usize, ScanError> {
    if pos + 1 >= data.len() {
        return Err(ScanError::Truncated);
    }
    let seg_len = read_u16(data, pos) as usize;
    if seg_len < 2 || pos + seg_len > data.len() {
        return Err(ScanError::Truncated);
    }

    let seg_end = pos + seg_len;
    let mut p = pos + 2;

    while p < seg_end {
        if p >= data.len() {
            return Err(ScanError::Truncated);
        }

        let tc_th = data[p];
        let table_class = (tc_th >> 4) & 0x0F; // 0 = DC, 1 = AC
        p += 1;

        if p + 16 > data.len() {
            return Err(ScanError::Truncated);
        }

        // Count symbols from the 16-byte `bits` array
        let mut total_symbols: u16 = 0;
        for i in 0..16 {
            total_symbols += data[p + i] as u16;
        }
        p += 16;

        // Track AC symbol counts for Huffman discrimination
        if table_class == 1 {
            result.total_ac_symbols += total_symbols;
        }

        result.dht_count += 1;

        // Skip symbol values
        let sym_count = total_symbols as usize;
        if p + sym_count > data.len() {
            return Err(ScanError::Truncated);
        }
        p += sym_count;
    }

    Ok(seg_end)
}

/// Parse APP0 marker — check for JFIF identifier.
fn parse_app0(data: &[u8], pos: usize, result: &mut ScanResult) -> Result<usize, ScanError> {
    if pos + 1 >= data.len() {
        return Err(ScanError::Truncated);
    }
    let seg_len = read_u16(data, pos) as usize;
    if seg_len < 2 || pos + seg_len > data.len() {
        return Err(ScanError::Truncated);
    }

    // Check for "JFIF\0" identifier at offset 2
    if seg_len >= 7 && pos + 6 < data.len() {
        let id = &data[pos + 2..pos + 7];
        if id == b"JFIF\0" {
            result.has_jfif = true;
        }
    }

    Ok(pos + seg_len)
}

/// Parse APP2 marker — check for ICC_PROFILE identifier.
fn parse_app2(data: &[u8], pos: usize, result: &mut ScanResult) -> Result<usize, ScanError> {
    if pos + 1 >= data.len() {
        return Err(ScanError::Truncated);
    }
    let seg_len = read_u16(data, pos) as usize;
    if seg_len < 2 || pos + seg_len > data.len() {
        return Err(ScanError::Truncated);
    }

    // Check for "ICC_PROFILE\0" identifier at offset 2
    if seg_len >= 14 && pos + 14 <= data.len() {
        let id = &data[pos + 2..pos + 14];
        if id == b"ICC_PROFILE\0" {
            result.has_icc_profile = true;
        }
    }

    Ok(pos + seg_len)
}

/// Parse APP13 marker — check for Photoshop 3.0 (IPTC) identifier.
fn parse_app13(data: &[u8], pos: usize, result: &mut ScanResult) -> Result<usize, ScanError> {
    if pos + 1 >= data.len() {
        return Err(ScanError::Truncated);
    }
    let seg_len = read_u16(data, pos) as usize;
    if seg_len < 2 || pos + seg_len > data.len() {
        return Err(ScanError::Truncated);
    }

    // Check for "Photoshop 3.0\0" identifier at offset 2
    if seg_len >= 16 && pos + 16 <= data.len() {
        let id = &data[pos + 2..pos + 16];
        if id == b"Photoshop 3.0\0" {
            result.has_photoshop_iptc = true;
        }
    }

    Ok(pos + seg_len)
}

/// Parse APP14 marker — check for Adobe identifier.
fn parse_app14(data: &[u8], pos: usize, result: &mut ScanResult) -> Result<usize, ScanError> {
    if pos + 1 >= data.len() {
        return Err(ScanError::Truncated);
    }
    let seg_len = read_u16(data, pos) as usize;
    if seg_len < 2 || pos + seg_len > data.len() {
        return Err(ScanError::Truncated);
    }

    // Check for "Adobe" identifier at offset 2
    if seg_len >= 7 && pos + 7 <= data.len() {
        let id = &data[pos + 2..pos + 7];
        if id == b"Adobe" {
            result.has_adobe = true;
        }
    }

    Ok(pos + seg_len)
}

/// Skip entropy-coded data after an SOS marker.
/// Looks for the next 0xFF byte that's not followed by 0x00 or a restart marker.
fn skip_entropy_data(data: &[u8], mut pos: usize) -> usize {
    while pos < data.len() {
        if data[pos] == 0xFF {
            if pos + 1 >= data.len() {
                return pos;
            }
            let next = data[pos + 1];
            if next == 0x00 {
                // Byte-stuffed 0xFF in data stream — skip both bytes
                pos += 2;
                continue;
            }
            if (0xD0..=0xD7).contains(&next) {
                // Restart marker — skip and continue entropy data
                pos += 2;
                continue;
            }
            // Found a real marker — back up so find_marker can see it
            return pos;
        }
        pos += 1;
    }
    pos
}

/// Read a big-endian u16 from data at the given position.
#[inline]
fn read_u16(data: &[u8], pos: usize) -> u16 {
    (data[pos] as u16) << 8 | data[pos + 1] as u16
}
