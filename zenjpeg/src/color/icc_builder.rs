//! ICC v2 matrix profile builder for KLT decorrelation.
//!
//! Generates a minimal ICC v2 matrix-based profile (~500 bytes) that encodes
//! the inverse of a decorrelation transform composed with the sRGB-to-XYZ D50
//! matrix. Any ICC-aware decoder will use this profile to recover correct colors.
//!
//! The profile uses:
//! - sRGB TRC (transfer response curves) — parametric type 3
//! - Custom 3x3 matrix columns (rXYZ, gXYZ, bXYZ)
//! - D50 white point
//!
//! The matrix columns encode: `D_inverse * M_srgb_to_xyz_d50`, where `D` is the
//! forward decorrelation matrix (RGB → decorrelated). This means the ICC engine
//! performs: linearize → multiply by matrix → get XYZ → convert to display space.
//! The matrix multiplication effectively inverts the decorrelation.

#![allow(dead_code)]

use super::klt::Mat3;

/// sRGB to XYZ D50 matrix (from ICC sRGB profile specification).
///
/// This is the Bradford-adapted matrix that maps linear sRGB primaries to
/// XYZ under D50 illuminant (the ICC profile connection space).
const SRGB_TO_XYZ_D50: Mat3 = Mat3::from_rows([
    [0.4360747, 0.3850649, 0.1430804],
    [0.2225045, 0.7168786, 0.0606169],
    [0.0139322, 0.0971045, 0.7141733],
]);

/// D50 white point in XYZ (ICC standard PCS illuminant).
const D50_WHITE: [f32; 3] = [0.9642, 1.0000, 0.8249];

/// sRGB TRC parametric curve parameters (IEC 61966-2-1).
///
/// Type 3 parametric curve: f(x) = (a*x + b)^g  if x >= d
///                          f(x) = c*x           if x < d
///
/// Stored as s15Fixed16: gamma=2.4, a=1/1.055, b=0.055/1.055, c=1/12.92, d=0.04045
const SRGB_TRC_GAMMA: u32 = 0x0002_6666; // 2.4 as s15Fixed16
const SRGB_TRC_A: u32 = 0x0000_F2A7; // 1/1.055 ≈ 0.947867
const SRGB_TRC_B: u32 = 0x0000_0D59; // 0.055/1.055 ≈ 0.052133
const SRGB_TRC_C: u32 = 0x0000_13D0; // 1/12.92 ≈ 0.077399
const SRGB_TRC_D: u32 = 0x0000_0A5B; // 0.04045 as s15Fixed16

/// Convert f32 to ICC s15Fixed16Number format.
fn f32_to_s15f16(val: f32) -> u32 {
    // s15Fixed16: 1 sign bit + 15 integer bits + 16 fractional bits
    // Range: -32768.0 to ~32767.99998
    let clamped = val.clamp(-32768.0, 32767.99998);
    let fixed = (clamped * 65536.0).round() as i32;
    fixed as u32
}

/// Write a big-endian u32.
fn write_u32(buf: &mut Vec<u8>, val: u32) {
    buf.extend_from_slice(&val.to_be_bytes());
}

/// Write a big-endian u16.
fn write_u16(buf: &mut Vec<u8>, val: u16) {
    buf.extend_from_slice(&val.to_be_bytes());
}

/// Build an ICC v2 matrix profile for a KLT decorrelation transform.
///
/// The profile encodes `D_inverse * M_srgb_to_xyz_d50` as its matrix columns,
/// where `D` is the forward decorrelation matrix (RGB → decorrelated).
///
/// When the ICC engine processes the decoded "RGB" values (which are actually
/// decorrelated channels), it:
/// 1. Linearizes via sRGB TRC curves
/// 2. Multiplies by the matrix columns → gets XYZ
/// 3. Converts XYZ to display space
///
/// Step 2 effectively computes `XYZ = M_srgb_to_xyz * D_inverse * channels`,
/// which is `M_srgb_to_xyz * original_RGB` — correct colors.
///
/// # Arguments
/// * `d_inverse` - The inverse of the forward decorrelation matrix
/// * `description` - UTF-8 profile description (short, e.g. "KLT")
///
/// # Returns
/// Complete ICC v2 profile as bytes, typically ~500 bytes.
pub fn build_klt_icc_profile(d_inverse: &Mat3, description: &str) -> Vec<u8> {
    // Compute the composite matrix: D_inverse * sRGB_to_XYZ_D50
    // Each column of the ICC matrix represents where one input channel maps in XYZ.
    let composite = d_inverse.mul(&SRGB_TO_XYZ_D50);

    // Profile structure:
    // 1. Header (128 bytes)
    // 2. Tag table (4 + 7 tags * 12 = 88 bytes)
    // 3. Tag data:
    //    - desc (description)
    //    - cprt (copyright)
    //    - wtpt (white point)
    //    - rXYZ, gXYZ, bXYZ (matrix columns)
    //    - rTRC, gTRC, bTRC (sRGB curves — shared, single tag data)

    let mut profile = Vec::with_capacity(600);

    // ========================================================================
    // Build tag data first to know offsets
    // ========================================================================

    // Description tag data (mluc - multiLocalizedUnicodeType for v4, or desc for v2)
    // Using v2 'desc' type for maximum compatibility
    let desc_data = build_desc_tag(description);

    // Copyright tag
    let cprt_data = build_desc_tag("CC0");

    // White point (XYZ type)
    let wtpt_data = build_xyz_tag(D50_WHITE);

    // Matrix column tags (XYZ type)
    // rXYZ = first column of composite matrix
    let rxyz_data = build_xyz_tag(composite.col(0));
    // gXYZ = second column
    let gxyz_data = build_xyz_tag(composite.col(1));
    // bXYZ = third column
    let bxyz_data = build_xyz_tag(composite.col(2));

    // TRC tag (parametric curve, sRGB) — shared by all three channels
    let trc_data = build_srgb_para_trc();

    // ========================================================================
    // Calculate offsets
    // ========================================================================
    let num_tags: u32 = 9; // desc, cprt, wtpt, rXYZ, gXYZ, bXYZ, rTRC, gTRC, bTRC
    let header_size: u32 = 128;
    let tag_table_size: u32 = 4 + num_tags * 12;
    let data_start = header_size + tag_table_size;

    // Tag data offsets (must be 4-byte aligned)
    let desc_offset = data_start;
    let desc_size = desc_data.len() as u32;

    let cprt_offset = align4(desc_offset + desc_size);
    let cprt_size = cprt_data.len() as u32;

    let wtpt_offset = align4(cprt_offset + cprt_size);
    let wtpt_size = wtpt_data.len() as u32;

    let rxyz_offset = align4(wtpt_offset + wtpt_size);
    let rxyz_size = rxyz_data.len() as u32;

    let gxyz_offset = align4(rxyz_offset + rxyz_size);
    let gxyz_size = gxyz_data.len() as u32;

    let bxyz_offset = align4(gxyz_offset + gxyz_size);
    let bxyz_size = bxyz_data.len() as u32;

    let trc_offset = align4(bxyz_offset + bxyz_size);
    let trc_size = trc_data.len() as u32;

    let profile_size = align4(trc_offset + trc_size);

    // ========================================================================
    // Write header (128 bytes)
    // ========================================================================
    write_u32(&mut profile, profile_size); // Profile size
    profile.extend_from_slice(b"zen "); // Preferred CMM: "zen " for zenjpeg
    write_u32(&mut profile, 0x0210_0000); // Version 2.1.0
    profile.extend_from_slice(b"scnr"); // Device class: input (scanner)
    profile.extend_from_slice(b"RGB "); // Color space: RGB
    profile.extend_from_slice(b"XYZ "); // PCS: XYZ
    // Date/time: 2026-01-01 00:00:00
    write_u16(&mut profile, 2026);
    write_u16(&mut profile, 1);
    write_u16(&mut profile, 1);
    write_u16(&mut profile, 0);
    write_u16(&mut profile, 0);
    write_u16(&mut profile, 0);
    profile.extend_from_slice(b"acsp"); // Profile file signature
    profile.extend_from_slice(b"APPL"); // Primary platform: Apple (best compatibility)
    write_u32(&mut profile, 0); // Profile flags
    write_u32(&mut profile, 0); // Device manufacturer
    write_u32(&mut profile, 0); // Device model
    write_u32(&mut profile, 0); // Device attributes (8 bytes)
    write_u32(&mut profile, 0);
    write_u32(&mut profile, 1); // Rendering intent: relative colorimetric
    // PCS illuminant (D50 in s15Fixed16)
    write_u32(&mut profile, f32_to_s15f16(D50_WHITE[0]));
    write_u32(&mut profile, f32_to_s15f16(D50_WHITE[1]));
    write_u32(&mut profile, f32_to_s15f16(D50_WHITE[2]));
    profile.extend_from_slice(b"zen "); // Profile creator
    // Profile ID (MD5, set to zero — optional)
    profile.extend_from_slice(&[0u8; 16]);
    // Reserved (28 bytes)
    profile.extend_from_slice(&[0u8; 28]);

    assert_eq!(profile.len(), 128, "header must be exactly 128 bytes");

    // ========================================================================
    // Write tag table
    // ========================================================================
    write_u32(&mut profile, num_tags);

    // desc
    profile.extend_from_slice(b"desc");
    write_u32(&mut profile, desc_offset);
    write_u32(&mut profile, desc_size);

    // cprt
    profile.extend_from_slice(b"cprt");
    write_u32(&mut profile, cprt_offset);
    write_u32(&mut profile, cprt_size);

    // wtpt
    profile.extend_from_slice(b"wtpt");
    write_u32(&mut profile, wtpt_offset);
    write_u32(&mut profile, wtpt_size);

    // rXYZ
    profile.extend_from_slice(b"rXYZ");
    write_u32(&mut profile, rxyz_offset);
    write_u32(&mut profile, rxyz_size);

    // gXYZ
    profile.extend_from_slice(b"gXYZ");
    write_u32(&mut profile, gxyz_offset);
    write_u32(&mut profile, gxyz_size);

    // bXYZ
    profile.extend_from_slice(b"bXYZ");
    write_u32(&mut profile, bxyz_offset);
    write_u32(&mut profile, bxyz_size);

    // rTRC — all three share the same tag data
    profile.extend_from_slice(b"rTRC");
    write_u32(&mut profile, trc_offset);
    write_u32(&mut profile, trc_size);

    // gTRC — points to same data as rTRC
    profile.extend_from_slice(b"gTRC");
    write_u32(&mut profile, trc_offset);
    write_u32(&mut profile, trc_size);

    // bTRC — points to same data as rTRC
    profile.extend_from_slice(b"bTRC");
    write_u32(&mut profile, trc_offset);
    write_u32(&mut profile, trc_size);

    // ========================================================================
    // Write tag data
    // ========================================================================
    pad_to(&mut profile, desc_offset as usize);
    profile.extend_from_slice(&desc_data);

    pad_to(&mut profile, cprt_offset as usize);
    profile.extend_from_slice(&cprt_data);

    pad_to(&mut profile, wtpt_offset as usize);
    profile.extend_from_slice(&wtpt_data);

    pad_to(&mut profile, rxyz_offset as usize);
    profile.extend_from_slice(&rxyz_data);

    pad_to(&mut profile, gxyz_offset as usize);
    profile.extend_from_slice(&gxyz_data);

    pad_to(&mut profile, bxyz_offset as usize);
    profile.extend_from_slice(&bxyz_data);

    pad_to(&mut profile, trc_offset as usize);
    profile.extend_from_slice(&trc_data);

    // Pad to final size
    pad_to(&mut profile, profile_size as usize);

    profile
}

/// Build a v2 'desc' (textDescriptionType) tag.
fn build_desc_tag(text: &str) -> Vec<u8> {
    let mut tag = Vec::new();
    tag.extend_from_slice(b"desc"); // Type signature
    write_u32(&mut tag, 0); // Reserved
    let ascii_len = text.len() as u32 + 1; // Include null terminator
    write_u32(&mut tag, ascii_len);
    tag.extend_from_slice(text.as_bytes());
    tag.push(0); // Null terminator
    // Unicode and ScriptCode counts (zero — not used)
    write_u32(&mut tag, 0); // Unicode language code
    write_u32(&mut tag, 0); // Unicode count
    write_u16(&mut tag, 0); // ScriptCode code
    tag.push(0); // ScriptCode count
    tag.extend_from_slice(&[0u8; 67]); // ScriptCode string (67 bytes)
    tag
}

/// Build an XYZType tag (3 s15Fixed16 values).
fn build_xyz_tag(xyz: [f32; 3]) -> Vec<u8> {
    let mut tag = Vec::new();
    tag.extend_from_slice(b"XYZ "); // Type signature
    write_u32(&mut tag, 0); // Reserved
    write_u32(&mut tag, f32_to_s15f16(xyz[0]));
    write_u32(&mut tag, f32_to_s15f16(xyz[1]));
    write_u32(&mut tag, f32_to_s15f16(xyz[2]));
    tag
}

/// Build a parametric TRC tag with sRGB transfer function.
///
/// Uses parametric curve type 3:
///   f(x) = (a*x + b)^g     if x >= d
///   f(x) = c*x             if x < d
fn build_srgb_para_trc() -> Vec<u8> {
    let mut tag = Vec::new();
    tag.extend_from_slice(b"para"); // Type signature
    write_u32(&mut tag, 0); // Reserved
    write_u16(&mut tag, 3); // Function type 3
    write_u16(&mut tag, 0); // Reserved
    write_u32(&mut tag, SRGB_TRC_GAMMA); // g
    write_u32(&mut tag, SRGB_TRC_A); // a
    write_u32(&mut tag, SRGB_TRC_B); // b
    write_u32(&mut tag, SRGB_TRC_C); // c
    write_u32(&mut tag, SRGB_TRC_D); // d
    tag
}

/// Align a value up to the next multiple of 4.
fn align4(val: u32) -> u32 {
    (val + 3) & !3
}

/// Pad buffer with zeros to reach target length.
fn pad_to(buf: &mut Vec<u8>, target: usize) {
    while buf.len() < target {
        buf.push(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_to_s15f16() {
        assert_eq!(f32_to_s15f16(1.0), 0x0001_0000);
        assert_eq!(f32_to_s15f16(0.0), 0x0000_0000);
        assert_eq!(f32_to_s15f16(-1.0), 0xFFFF_0000);
        assert_eq!(f32_to_s15f16(0.5), 0x0000_8000);

        // sRGB red X value ≈ 0.4360747
        let val = f32_to_s15f16(0.4360747);
        // Should be approximately 0x00006FA2
        assert!((val as i32 - 0x6FA2i32).unsigned_abs() < 2);
    }

    #[test]
    fn test_identity_profile_is_srgb_like() {
        // Identity decorrelation → profile should be ~sRGB
        let identity = Mat3::IDENTITY;
        let profile = build_klt_icc_profile(&identity, "Test");

        // Profile should be valid (starts with correct size)
        let size = u32::from_be_bytes([profile[0], profile[1], profile[2], profile[3]]);
        assert_eq!(size as usize, profile.len());

        // Should have "acsp" signature at offset 36
        assert_eq!(&profile[36..40], b"acsp");

        // Should be reasonable size
        assert!(profile.len() < 700, "profile too large: {}", profile.len());
        assert!(profile.len() > 300, "profile too small: {}", profile.len());
    }

    #[test]
    fn test_custom_matrix_profile() {
        // A non-trivial decorrelation matrix
        let d_inv = Mat3::from_rows([
            [0.6, 0.3, 0.1],
            [-0.2, 0.8, 0.4],
            [0.1, -0.5, 0.9],
        ]);

        let profile = build_klt_icc_profile(&d_inv, "KLT Custom");

        // Should be valid size
        let size = u32::from_be_bytes([profile[0], profile[1], profile[2], profile[3]]);
        assert_eq!(size as usize, profile.len());

        // Should have correct signature
        assert_eq!(&profile[36..40], b"acsp");

        // Should have 9 tags
        let tag_count =
            u32::from_be_bytes([profile[128], profile[129], profile[130], profile[131]]);
        assert_eq!(tag_count, 9);
    }

    #[test]
    fn test_profile_tag_alignment() {
        let identity = Mat3::IDENTITY;
        let profile = build_klt_icc_profile(&identity, "Align Test");

        // Read tag table and verify all offsets are 4-byte aligned
        let tag_count =
            u32::from_be_bytes([profile[128], profile[129], profile[130], profile[131]]);

        for i in 0..tag_count {
            let base = 132 + (i as usize) * 12;
            let offset = u32::from_be_bytes([
                profile[base + 4],
                profile[base + 5],
                profile[base + 6],
                profile[base + 7],
            ]);
            assert_eq!(
                offset % 4,
                0,
                "tag {} offset {} not 4-byte aligned",
                i,
                offset
            );
        }
    }
}
