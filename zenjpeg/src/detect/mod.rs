//! JPEG encoder detection and quality estimation.
//!
//! Identifies which encoder produced a JPEG file, estimates its quality level,
//! and extracts structural metadata — all from header-only parsing (~500 bytes, <1µs).
//!
//! # Example
//!
//! ```rust,ignore
//! use zenjpeg::detect::{probe, EncoderFamily, QualityScale};
//!
//! let jpeg_data = std::fs::read("photo.jpg").unwrap();
//! let info = probe(&jpeg_data).unwrap();
//!
//! match info.encoder {
//!     EncoderFamily::Mozjpeg => println!("Encoded with mozjpeg"),
//!     EncoderFamily::CjpegliYcbcr => println!("Encoded with jpegli (YCbCr)"),
//!     _ => println!("Encoder: {:?}", info.encoder),
//! }
//!
//! match info.quality.scale {
//!     QualityScale::IjgQuality => println!("Quality: {:.0}", info.quality.value),
//!     QualityScale::ButteraugliDistance => println!("Distance: {:.2}", info.quality.value),
//!     _ => {}
//! }
//! ```

#[doc(hidden)]
pub mod content;
mod fingerprint;
mod quality;
mod reencode;
mod scanner;

pub use fingerprint::EncoderFamily;
#[cfg(test)]
pub(crate) use fingerprint::generate_ijg_table;
pub use quality::{Confidence, QualityEstimate, QualityScale};
pub use reencode::{ReencodeError, ReencodeSettings};

use crate::foundation::consts::{MARKER_SOF0, MARKER_SOF2};
use crate::types::{Dimensions, JpegMode, Subsampling};

use alloc::vec::Vec;

/// Result of probing a JPEG file.
#[derive(Debug, Clone)]
pub struct JpegProbe {
    /// Identified encoder family.
    pub encoder: EncoderFamily,
    /// Estimated quality level.
    pub quality: QualityEstimate,
    /// Image dimensions.
    pub dimensions: Dimensions,
    /// Chroma subsampling mode.
    pub subsampling: Subsampling,
    /// Baseline or progressive.
    pub mode: JpegMode,
    /// Number of color components (1 = grayscale, 3 = color, 4 = CMYK).
    pub num_components: u8,
    /// Number of SOS markers (1 for baseline, 4-16 for progressive).
    pub scan_count: u16,
    /// DQT tables in natural (row-major) order. Typically 1-3 tables.
    pub dqt_tables: Vec<DqtTable>,
}

/// A quantization table extracted from a JPEG file, in natural (row-major) order.
#[derive(Debug, Clone)]
pub struct DqtTable {
    /// Table index (0-3) as stored in the JPEG.
    pub index: u8,
    /// 64 quantization values in natural (row-major) order.
    pub values: [u16; 64],
    /// Precision: 0 = 8-bit, 1 = 16-bit.
    pub precision: u8,
}

/// Errors that can occur during probing.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProbeError {
    /// Data is too short to be a JPEG.
    TooShort,
    /// Missing SOI marker — not a JPEG file.
    NotJpeg,
    /// No DQT tables found before scan data.
    NoQuantTables,
    /// Hit EOF during header parsing.
    Truncated,
}

impl core::fmt::Display for ProbeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooShort => write!(f, "data too short to be a JPEG"),
            Self::NotJpeg => write!(f, "missing SOI marker, not a JPEG file"),
            Self::NoQuantTables => write!(f, "no quantization tables found"),
            Self::Truncated => write!(f, "truncated JPEG data"),
        }
    }
}

impl std::error::Error for ProbeError {}

/// Probe a JPEG from its raw bytes.
///
/// Reads only headers (~500 bytes), no entropy decoding.
/// Returns encoder identification, quality estimate, and structural metadata.
pub fn probe(data: &[u8]) -> Result<JpegProbe, ProbeError> {
    let scan = scanner::scan_headers(data).map_err(|e| match e {
        scanner::ScanError::TooShort => ProbeError::TooShort,
        scanner::ScanError::NotJpeg => ProbeError::NotJpeg,
        scanner::ScanError::Truncated => ProbeError::Truncated,
    })?;

    // Must have at least one DQT table
    if scan.dqt_tables.iter().all(|t| t.is_none()) {
        return Err(ProbeError::NoQuantTables);
    }

    // Extract SOF info
    let sof = scan.sof.as_ref();

    let dimensions = sof.map_or(Dimensions::new(0, 0), |s| {
        Dimensions::new(s.width as u32, s.height as u32)
    });

    let num_components = sof.map_or(0, |s| s.num_components);

    let mode = match sof.map(|s| s.marker) {
        Some(MARKER_SOF0) => JpegMode::Baseline,
        Some(MARKER_SOF2) => JpegMode::Progressive,
        Some(0xC9) => JpegMode::ArithmeticSequential,
        Some(0xCA) => JpegMode::ArithmeticProgressive,
        _ => JpegMode::Baseline,
    };

    let subsampling = detect_subsampling(sof);

    // Identify encoder
    let encoder = fingerprint::identify_encoder(&scan);

    // Estimate quality
    let quality = quality::estimate_quality(&scan, &encoder);

    // Collect DQT tables for output
    let mut dqt_tables = Vec::new();
    for (idx, table) in scan.dqt_tables.iter().enumerate() {
        if let Some(t) = table {
            dqt_tables.push(DqtTable {
                index: idx as u8,
                values: t.values,
                precision: t.precision,
            });
        }
    }

    Ok(JpegProbe {
        encoder,
        quality,
        dimensions,
        subsampling,
        mode,
        num_components,
        scan_count: scan.sos_count,
        dqt_tables,
    })
}

impl zencodec::SourceEncodingDetails for JpegProbe {
    fn source_generic_quality(&self) -> Option<f32> {
        // Map to generic 0-100 scale. IJG quality is already 0-100.
        // For other scales, approximate.
        match self.quality.scale {
            QualityScale::IjgQuality | QualityScale::MozjpegQuality => Some(self.quality.value),
            QualityScale::ButteraugliDistance => {
                // Butteraugli: lower = better. ~1.0 = visually lossless.
                // Rough mapping: distance 0.0 → q100, 1.0 → q90, 3.0 → q75, 10.0 → q50
                Some((100.0 - self.quality.value * 5.0).clamp(0.0, 100.0))
            }
        }
    }
}

/// Detect chroma subsampling from SOF component sampling factors.
fn detect_subsampling(sof: Option<&scanner::SofInfo>) -> Subsampling {
    let sof = match sof {
        Some(s) => s,
        None => return Subsampling::S420,
    };

    if sof.num_components < 3 {
        return Subsampling::S444; // Grayscale — no subsampling
    }

    // Compare luma sampling to chroma sampling
    let (_, luma_h, luma_v, _) = sof.components[0];
    let (_, cb_h, cb_v, _) = sof.components[1];

    if luma_h == cb_h && luma_v == cb_v {
        Subsampling::S444
    } else if luma_h == 2 * cb_h && luma_v == 2 * cb_v {
        Subsampling::S420
    } else if luma_h == 2 * cb_h && luma_v == cb_v {
        Subsampling::S422
    } else if luma_h == cb_h && luma_v == 2 * cb_v {
        Subsampling::S440
    } else {
        Subsampling::S420 // Fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::consts::{
        MARKER_APP0, MARKER_DQT, MARKER_EOI, MARKER_SOF0, MARKER_SOF2, MARKER_SOI, MARKER_SOS,
    };

    #[test]
    fn test_probe_error_display() {
        assert_eq!(
            ProbeError::TooShort.to_string(),
            "data too short to be a JPEG"
        );
        assert_eq!(
            ProbeError::NotJpeg.to_string(),
            "missing SOI marker, not a JPEG file"
        );
    }

    #[test]
    fn test_probe_too_short() {
        assert_eq!(probe(&[]).unwrap_err(), ProbeError::TooShort);
        assert_eq!(probe(&[0xFF]).unwrap_err(), ProbeError::TooShort);
        assert_eq!(probe(&[0xFF, 0xD8]).unwrap_err(), ProbeError::NoQuantTables);
    }

    #[test]
    fn test_probe_not_jpeg() {
        assert_eq!(
            probe(&[0x00, 0x00, 0x00, 0x00]).unwrap_err(),
            ProbeError::NotJpeg
        );
        assert_eq!(probe(b"PNG\r\n").unwrap_err(), ProbeError::NotJpeg);
    }

    #[test]
    fn test_probe_minimal_jpeg() {
        // Build a minimal valid JPEG: SOI + DQT + SOF0 + DHT + SOS + EOI
        let jpeg = build_minimal_jpeg(75, false);
        let result = probe(&jpeg).unwrap();

        assert_eq!(result.encoder, EncoderFamily::LibjpegTurbo);
        assert_eq!(result.quality.value, 75.0);
        assert_eq!(result.quality.confidence, Confidence::Exact);
        assert_eq!(result.quality.scale, QualityScale::IjgQuality);
        assert_eq!(result.dimensions.width, 8);
        assert_eq!(result.dimensions.height, 8);
        assert_eq!(result.mode, JpegMode::Baseline);
        assert_eq!(result.num_components, 3);
        assert_eq!(result.scan_count, 1);
        assert_eq!(result.dqt_tables.len(), 2);
    }

    #[test]
    fn test_probe_quality_sweep_ijg() {
        // Test a range of IJG qualities round-trip through probe
        for q in [10, 25, 50, 75, 85, 90, 95, 100] {
            let jpeg = build_minimal_jpeg(q, false);
            let result = probe(&jpeg).unwrap();

            assert_eq!(
                result.quality.value, q as f32,
                "IJG Q{q} probe failed: got {:.0}",
                result.quality.value
            );
            assert_eq!(result.quality.confidence, Confidence::Exact);
        }
    }

    #[test]
    fn test_probe_optimized_huffman_detected_as_imagemagick() {
        // Build a JPEG with IJG tables but non-standard Huffman (not 162 AC symbols)
        let jpeg = build_minimal_jpeg_with_custom_huffman(75);
        let result = probe(&jpeg).unwrap();

        assert_eq!(result.encoder, EncoderFamily::ImageMagick);
    }

    /// Build a minimal synthetic JPEG for testing.
    ///
    /// Creates: SOI + DQT(luma) + DQT(chroma) + SOF0 + DHT(standard) + SOS + EOI
    fn build_minimal_jpeg(quality: u8, progressive: bool) -> Vec<u8> {
        use super::fingerprint::generate_ijg_table;
        use crate::foundation::consts::MARKER_DHT;

        let mut data = Vec::new();

        // SOI
        data.extend_from_slice(&[0xFF, MARKER_SOI]);

        // JFIF APP0
        let jfif = [
            0xFF,
            MARKER_APP0,
            0x00,
            0x10, // Length = 16
            b'J',
            b'F',
            b'I',
            b'F',
            0x00, // Identifier
            0x01,
            0x01, // Version 1.1
            0x00, // Aspect ratio units
            0x00,
            0x01, // X density
            0x00,
            0x01, // Y density
            0x00,
            0x00, // No thumbnail
        ];
        data.extend_from_slice(&jfif);

        // DQT table 0 (luminance) - written in zigzag order as JPEG expects
        let luma_natural = generate_ijg_table(quality, false);
        data.extend_from_slice(&[0xFF, MARKER_DQT]);
        data.extend_from_slice(&[0x00, 0x43]); // Length = 67
        data.push(0x00); // Precision 0 (8-bit), table ID 0
        // JPEG_NATURAL_ORDER[z] maps zigzag position z → natural index
        for z in 0..64 {
            let natural_idx = crate::foundation::consts::JPEG_NATURAL_ORDER[z] as usize;
            data.push(luma_natural[natural_idx] as u8);
        }

        // DQT table 1 (chrominance)
        let chroma_natural = generate_ijg_table(quality, true);
        data.extend_from_slice(&[0xFF, MARKER_DQT]);
        data.extend_from_slice(&[0x00, 0x43]); // Length = 67
        data.push(0x01); // Precision 0, table ID 1
        for z in 0..64 {
            let natural_idx = crate::foundation::consts::JPEG_NATURAL_ORDER[z] as usize;
            data.push(chroma_natural[natural_idx] as u8);
        }

        // SOF0 (baseline) or SOF2 (progressive)
        let sof_marker = if progressive {
            MARKER_SOF2
        } else {
            MARKER_SOF0
        };
        data.extend_from_slice(&[0xFF, sof_marker]);
        data.extend_from_slice(&[0x00, 0x11]); // Length = 17
        data.push(0x08); // 8-bit precision
        data.extend_from_slice(&[0x00, 0x08]); // Height = 8
        data.extend_from_slice(&[0x00, 0x08]); // Width = 8
        data.push(0x03); // 3 components
        // Component 1 (Y): ID=1, H=1, V=1, QT=0
        data.extend_from_slice(&[0x01, 0x11, 0x00]);
        // Component 2 (Cb): ID=2, H=1, V=1, QT=1
        data.extend_from_slice(&[0x02, 0x11, 0x01]);
        // Component 3 (Cr): ID=3, H=1, V=1, QT=1
        data.extend_from_slice(&[0x03, 0x11, 0x01]);

        // DHT - standard Huffman tables (AC with 162 symbols each)
        // We need two DC and two AC tables. For testing, we'll create
        // minimal tables with the standard symbol counts.
        // DC table 0
        data.extend_from_slice(&[0xFF, MARKER_DHT]);
        data.extend_from_slice(&[0x00, 0x1F]); // Length = 31
        data.push(0x00); // DC table 0
        // Standard DC luminance bits
        data.extend_from_slice(&[
            0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ]);
        // 12 symbols
        data.extend_from_slice(&[
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
        ]);

        // AC table 0 (standard: 162 symbols)
        data.extend_from_slice(&[0xFF, MARKER_DHT]);
        // Length = 2 + 1 + 16 + 162 = 181
        data.extend_from_slice(&[0x00, 0xB5]);
        data.push(0x10); // AC table 0
        // Standard AC luminance bits (sum = 162)
        data.extend_from_slice(&[
            0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00,
            0x01, 0x7D,
        ]);
        // 162 symbols (just fill with sequential values for test)
        for i in 0..162u8 {
            data.push(i);
        }

        // DC table 1
        data.extend_from_slice(&[0xFF, MARKER_DHT]);
        data.extend_from_slice(&[0x00, 0x1F]); // Length = 31
        data.push(0x01); // DC table 1
        data.extend_from_slice(&[
            0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ]);
        data.extend_from_slice(&[
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
        ]);

        // AC table 1 (standard: 162 symbols)
        data.extend_from_slice(&[0xFF, MARKER_DHT]);
        data.extend_from_slice(&[0x00, 0xB5]); // Length = 181
        data.push(0x11); // AC table 1
        // Standard AC chrominance bits (sum = 162)
        data.extend_from_slice(&[
            0x00, 0x02, 0x01, 0x02, 0x04, 0x04, 0x03, 0x04, 0x07, 0x05, 0x04, 0x04, 0x00, 0x01,
            0x02, 0x77,
        ]);
        for i in 0..162u8 {
            data.push(i);
        }

        // SOS
        data.extend_from_slice(&[0xFF, MARKER_SOS]);
        data.extend_from_slice(&[0x00, 0x0C]); // Length = 12
        data.push(0x03); // 3 components
        data.extend_from_slice(&[0x01, 0x00]); // Y: DC=0, AC=0
        data.extend_from_slice(&[0x02, 0x11]); // Cb: DC=1, AC=1
        data.extend_from_slice(&[0x03, 0x11]); // Cr: DC=1, AC=1
        data.extend_from_slice(&[0x00, 0x3F, 0x00]); // Ss=0, Se=63, Ah=0, Al=0

        // Minimal entropy data (just one byte that's not a marker)
        data.push(0x00);

        // EOI
        data.extend_from_slice(&[0xFF, MARKER_EOI]);

        data
    }

    /// Build a minimal JPEG with non-standard Huffman tables (not 162 AC symbols).
    fn build_minimal_jpeg_with_custom_huffman(quality: u8) -> Vec<u8> {
        use super::fingerprint::generate_ijg_table;
        use crate::foundation::consts::MARKER_DHT;

        let mut data = Vec::new();

        // SOI
        data.extend_from_slice(&[0xFF, MARKER_SOI]);

        // JFIF APP0
        let jfif = [
            0xFF,
            MARKER_APP0,
            0x00,
            0x10,
            b'J',
            b'F',
            b'I',
            b'F',
            0x00,
            0x01,
            0x01,
            0x00,
            0x00,
            0x01,
            0x00,
            0x01,
            0x00,
            0x00,
        ];
        data.extend_from_slice(&jfif);

        // DQT tables (IJG formula)
        let luma_natural = generate_ijg_table(quality, false);
        data.extend_from_slice(&[0xFF, MARKER_DQT, 0x00, 0x43, 0x00]);
        for z in 0..64 {
            let natural_idx = crate::foundation::consts::JPEG_NATURAL_ORDER[z] as usize;
            data.push(luma_natural[natural_idx] as u8);
        }

        let chroma_natural = generate_ijg_table(quality, true);
        data.extend_from_slice(&[0xFF, MARKER_DQT, 0x00, 0x43, 0x01]);
        for z in 0..64 {
            let natural_idx = crate::foundation::consts::JPEG_NATURAL_ORDER[z] as usize;
            data.push(chroma_natural[natural_idx] as u8);
        }

        // SOF0 (baseline, 4:4:4)
        data.extend_from_slice(&[0xFF, MARKER_SOF0]);
        data.extend_from_slice(&[0x00, 0x11, 0x08]);
        data.extend_from_slice(&[0x00, 0x08, 0x00, 0x08]);
        data.push(0x03);
        data.extend_from_slice(&[0x01, 0x11, 0x00]);
        data.extend_from_slice(&[0x02, 0x11, 0x01]);
        data.extend_from_slice(&[0x03, 0x11, 0x01]);

        // DHT with non-standard AC symbol counts (100 symbols instead of 162)
        // DC table 0
        data.extend_from_slice(&[0xFF, MARKER_DHT, 0x00, 0x1F, 0x00]);
        data.extend_from_slice(&[
            0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ]);
        data.extend_from_slice(&[
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
        ]);

        // AC table 0 (optimized: 100 symbols, not 162)
        let ac_sym_count: u16 = 100;
        let ac_len = 2 + 1 + 16 + ac_sym_count;
        data.extend_from_slice(&[0xFF, MARKER_DHT]);
        data.push((ac_len >> 8) as u8);
        data.push((ac_len & 0xFF) as u8);
        data.push(0x10); // AC table 0
        // Bits array summing to 100
        data.extend_from_slice(&[
            0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00,
            0x01, 0x3F,
        ]);
        for i in 0..100u8 {
            data.push(i);
        }

        // DC table 1
        data.extend_from_slice(&[0xFF, MARKER_DHT, 0x00, 0x1F, 0x01]);
        data.extend_from_slice(&[
            0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ]);
        data.extend_from_slice(&[
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
        ]);

        // AC table 1 (optimized: 100 symbols)
        data.extend_from_slice(&[0xFF, MARKER_DHT]);
        data.push((ac_len >> 8) as u8);
        data.push((ac_len & 0xFF) as u8);
        data.push(0x11); // AC table 1
        // Bits array summing to 100 (same as AC table 0)
        data.extend_from_slice(&[
            0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00,
            0x01, 0x3F,
        ]);
        for i in 0..100u8 {
            data.push(i);
        }

        // SOS
        data.extend_from_slice(&[0xFF, MARKER_SOS]);
        data.extend_from_slice(&[0x00, 0x0C, 0x03]);
        data.extend_from_slice(&[0x01, 0x00, 0x02, 0x11, 0x03, 0x11]);
        data.extend_from_slice(&[0x00, 0x3F, 0x00]);
        data.push(0x00);

        // EOI
        data.extend_from_slice(&[0xFF, MARKER_EOI]);

        data
    }
}
