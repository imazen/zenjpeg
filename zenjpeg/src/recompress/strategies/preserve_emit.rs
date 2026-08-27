//! Minimal coefficient-domain JPEG re-emitter for the Preserve strategy.
//!
//! Takes a [`DecodedCoefficients`] (obtained via zenjpeg's public
//! `DecodeConfig::decode_coefficients`), optionally re-quantizes the
//! coefficients against new quant tables, optionally applies a
//! per-block AC zero-bias mask, recomputes optimal Huffman tables, and
//! emits a sequential JPEG (baseline SOF0, or extended SOF1 with 16-bit
//! DQT tables when the source/target quant values exceed 255).
//!
//! All of zenjpeg's primitives used here are *public*: the marker
//! constants under [`crate::foundation::consts`], the [`BitWriter`]
//! under [`crate::foundation::bitstream`], the [`HuffmanTableSet`]
//! and [`OptimizedTable`] under [`crate::huffman::optimize`], and
//! [`crate::entropy::encode_blocks_mcu_order`].
//!
//! Entropy layout is a pure rate decision once the coefficients are fixed,
//! so this emitter gets the same "Smallest" treatment as the main encoder
//! (#143 item 2): it serializes sequentially and, when that lands at or
//! below [`crate::encode::ENTROPY_TRIAL_MAX_BYTES`], also serializes the
//! SAME edited coefficients progressively through
//! `lossless::restructure::encode_progressive_from_coefficients` and ships
//! the shorter stream. Coefficients are preserved by construction either
//! way; only the scan structure differs.
//!
//! What this module deliberately does **not** do:
//! - Trellis rewinding — this is a straight lossless coefficient re-emit.
//! - Restart markers (`restart_interval = 0`).
//!
//! Metadata (ICC / EXIF / XMP / JFIF / Adobe / COM) IS carried through
//! verbatim via the `EmitConfig::preserved_segments` chain, written after
//! SOI before the frame header.
//!
//! This is a coefficient-domain edit: there is **no IDCT/FDCT round
//! trip**. Generation loss is bounded by the rounding error in the
//! re-quantize step `round(coeff * old_q / new_q)`.

use crate::decode::{DecodedCoefficients, PreservedSegment};
use crate::entropy::{StreamingEntropyState, encode_blocks_mcu_order};
use crate::foundation::bitstream::BitWriter;
use crate::foundation::consts::{
    MARKER_DHT, MARKER_DQT, MARKER_EOI, MARKER_SOF0, MARKER_SOF1, MARKER_SOI, MARKER_SOS,
};
use crate::huffman::optimize::{FrequencyCounter, HuffmanTableSet, OptimizedTable};
use crate::types::Subsampling;

use crate::recompress::error::Error;

/// JPEG zigzag-scan order: `ZIGZAG[zz_idx]` is the corresponding
/// natural-order (row-major) index in an 8×8 block.
const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Robidoux base quantization table (natural / row-major order), the
/// mozjpeg and ImageMagick default. Nicolas Robidoux's psychovisually
/// optimized table for high-frequency detail preservation. mozjpeg uses
/// the same table for luma and chroma. Values mirror zenjpeg's
/// `encode::tables::robidoux::ROBIDOUX_LUMINANCE` (which is `pub(crate)`,
/// hence embedded here with provenance rather than imported).
const ROBIDOUX_BASE: [u16; 64] = [
    16, 16, 16, 18, 25, 37, 56, 85, 16, 17, 20, 27, 34, 40, 53, 75, 16, 20, 24, 31, 43, 62, 91,
    135, 18, 27, 31, 40, 53, 74, 106, 156, 25, 34, 43, 53, 69, 94, 131, 189, 37, 40, 62, 74, 94,
    124, 169, 238, 56, 53, 91, 106, 131, 169, 226, 311, 85, 75, 135, 156, 189, 238, 311, 418,
];

/// libjpeg `jpeg_quality_scaling`: quality (1–100) → scale percentage.
/// `q < 50 → 5000/q`, `q >= 50 → 200 - 2q`. mozjpeg and ImageMagick
/// both use this curve over the Robidoux base table.
fn libjpeg_quality_scale(quality: u8) -> u32 {
    let q = quality.clamp(1, 100) as u32;
    if q < 50 { 5000 / q } else { 200 - q * 2 }
}

/// Scale a base quant table by a percentage (libjpeg formula):
/// `scaled[i] = (base[i] * scale + 50) / 100`, clamped to `[1, 255]`
/// (baseline JPEG / SOF0).
fn scale_base_table(base: &[u16; 64], scale: u32) -> [u16; 64] {
    let mut out = [0u16; 64];
    for i in 0..64 {
        let v = (base[i] as u32 * scale + 50) / 100;
        out[i] = v.clamp(1, 255) as u16;
    }
    out
}

/// How the emitter computes the new quantization tables.
#[derive(Debug, Clone, Copy)]
pub enum QuantStrategy {
    /// Multiply every position in the source's quant table by a single
    /// scalar (per component class). Preserves the source's per-
    /// frequency perceptual weighting. Generation loss bound per
    /// coefficient is `scale × old_quant / 2`.
    ///
    /// Best when the source was encoded with high-quality
    /// perceptually-tuned tables (e.g. jpegli) and the target is a
    /// modest tightening — the source's table shape is at least as
    /// good as any retargeted one.
    UniformScale(QuantScale),

    /// Generate new quant tables from the JPEG Annex K standard
    /// (ITU-T T.81 K.1 / K.2) at `target_ijg_q`, then requantize each
    /// position individually with the position-specific old/new ratio.
    /// Decouples the new table's shape from the source's, which gives
    /// better results when the source's per-frequency weighting is
    /// already aggressive (e.g. low-quality libjpeg-turbo outputs).
    /// Generation loss bound per coefficient is `new_qt[i] / 2` in
    /// dequantized amplitude.
    TargetQuality { target_ijg_q: u8 },

    /// Generate new quant tables from the **Robidoux** base table
    /// (mozjpeg / ImageMagick default) at `target_quality` on the
    /// libjpeg 1–100 scale, then requantize per position. For sources
    /// that were themselves encoded with Robidoux-shaped tables
    /// (mozjpeg, ImageMagick), this is *same-family* requantization:
    /// the new table's per-frequency shape matches the source's, so the
    /// per-position `old/new` ratio is nearly the single scalar
    /// `scale_target / scale_source` everywhere. That makes the requant
    /// behave like a clean uniform scale to the *exact* target quality —
    /// the per-position rounding error is minimized because we never
    /// reshape the spectrum (which is what craters cross-family IJG-std
    /// retargeting of a Robidoux source). Generation loss bound per
    /// coefficient is `new_qt[i] / 2` in dequantized amplitude.
    RobidouxTargetQuality { target_quality: u8 },
}

impl QuantStrategy {
    /// Identity (no quant change). For round-trip tests.
    pub const IDENTITY: Self = Self::UniformScale(QuantScale::IDENTITY);
}

/// One quant-table scale (per component-class). 1.0 = identity (no edit).
/// Values > 1.0 increase quantization (lower quality, smaller bytes).
#[derive(Debug, Clone, Copy)]
pub struct QuantScale {
    pub luma: f32,
    pub chroma: f32,
}

impl QuantScale {
    pub const IDENTITY: Self = Self {
        luma: 1.0,
        chroma: 1.0,
    };
}

/// AQ-domain zero-bias mask: for each luma block, the AC indices that
/// should be force-zeroed *if non-zero*. None = no mask.
pub type AqMask = Vec<u64>;

/// `build_new_quant_tables` output: the per-table-slot new quant tables
/// (indexed by `quant_table_idx`, `None` for unreferenced slots) and the
/// per-component natural-order `old/new` requant ratios.
type NewQuantTables = (Vec<Option<[u16; 64]>>, Vec<[f32; 64]>);

/// Configuration for [`emit_preserved`].
///
/// Construct via [`EmitConfig::uniform_scale`] or
/// [`EmitConfig::target_quality`], optionally setting `aq_mask`.
/// New `#[non_exhaustive]` so adding strategies later is non-breaking.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EmitConfig {
    pub quant_strategy: QuantStrategy,
    pub aq_mask: Option<AqMask>,
    /// APPn / COM segments decoded from the source, written verbatim
    /// after SOI so the recompressed file keeps the source's ICC
    /// profile, EXIF (orientation), XMP, JFIF density, and Adobe color
    /// transform. Coefficient-domain recompression is metadata-
    /// transparent — dropping these would silently change the decoded
    /// colors (ICC) and display orientation (EXIF). MPF segments are
    /// excluded by the caller (their byte offsets reference embedded
    /// images that recompression invalidates).
    pub preserved_segments: Vec<PreservedSegment>,
}

// The constructors/builders are the `recompress-expert` surface for
// `emit_preserved`; in-crate callers build `EmitConfig` literally (#143).
#[cfg_attr(not(feature = "recompress-expert"), allow(dead_code))]
impl EmitConfig {
    /// Uniform per-component scaling of the source's quant tables.
    pub fn uniform_scale(scale: QuantScale) -> Self {
        Self {
            quant_strategy: QuantStrategy::UniformScale(scale),
            aq_mask: None,
            preserved_segments: Vec::new(),
        }
    }

    /// Target-quality tables generated from ITU-T T.81 K.1/K.2 at
    /// `target_ijg_q`. Tends to win on per-position rounding accuracy
    /// for sources whose own quant tables don't track standard JPEG.
    pub fn target_quality(target_ijg_q: u8) -> Self {
        Self {
            quant_strategy: QuantStrategy::TargetQuality { target_ijg_q },
            aq_mask: None,
            preserved_segments: Vec::new(),
        }
    }

    /// Target-quality tables generated from the Robidoux base table
    /// (mozjpeg / ImageMagick default) at `target_quality`. Use for
    /// sources detected as mozjpeg or ImageMagick — same-family
    /// retargeting keeps the spectral shape and minimizes per-position
    /// requant rounding.
    pub fn robidoux_target_quality(target_quality: u8) -> Self {
        Self {
            quant_strategy: QuantStrategy::RobidouxTargetQuality { target_quality },
            aq_mask: None,
            preserved_segments: Vec::new(),
        }
    }

    /// Set the AQ zero-bias mask. Returns `self` for builder chaining.
    pub fn with_aq_mask(mut self, aq_mask: Option<AqMask>) -> Self {
        self.aq_mask = aq_mask;
        self
    }

    /// Set the source APPn/COM segments to carry through (ICC, EXIF,
    /// XMP, JFIF, Adobe, COM). Returns `self` for builder chaining.
    pub fn with_preserved_segments(mut self, segments: Vec<PreservedSegment>) -> Self {
        self.preserved_segments = segments;
        self
    }
}

impl Default for EmitConfig {
    fn default() -> Self {
        Self {
            quant_strategy: QuantStrategy::IDENTITY,
            aq_mask: None,
            preserved_segments: Vec::new(),
        }
    }
}

/// Apply edits to coefficients in-place and emit a baseline-sequential
/// JPEG. The output is exactly the source's structure with new quant
/// tables, requantized coefficients, optimized Huffman, and no other
/// changes.
pub fn emit_preserved(
    coeffs: &DecodedCoefficients,
    subsampling: Subsampling,
    config: &EmitConfig,
) -> Result<Vec<u8>, Error> {
    if coeffs.components.is_empty() {
        return Err(Error::Internal("DecodedCoefficients has no components"));
    }

    // Sanity: the decoder stores MCU-padded block arrays. For
    // partial-MCU sources, `luma.blocks_wide * 8` may exceed
    // `coeffs.width` — that's normal. We pass the MCU-aligned width
    // to `encode_blocks_mcu_order` so its computed `y_blocks_w`
    // matches our actual block stride; the SOF segment we write
    // separately carries the unpadded image dimensions so the
    // decoder crops correctly.
    let (h_samp, v_samp) = match subsampling {
        Subsampling::S444 => (1usize, 1usize),
        Subsampling::S422 => (2, 1),
        Subsampling::S420 => (2, 2),
        Subsampling::S440 => (1, 2),
    };
    let luma = &coeffs.components[0];
    let blocks_wide = luma.blocks_wide;
    let blocks_high = luma.blocks_high;
    if !blocks_wide.is_multiple_of(h_samp) || !blocks_high.is_multiple_of(v_samp) {
        // Should never happen because the decoder pads to MCU; bail
        // out conservatively rather than feed misaligned data to the
        // scan emitter.
        return Err(Error::Internal(
            "preserve: decoder produced non-MCU-aligned block array",
        ));
    }

    // 1. Build new (edited) quant tables + per-position scale arrays.
    let (new_quant_tables, position_scale_per_component) =
        build_new_quant_tables(coeffs, config.quant_strategy)?;

    // 2. Re-quantize coefficients per component + apply AQ mask.
    let edited_components = edit_coefficients(
        coeffs,
        &position_scale_per_component,
        config.aq_mask.as_ref(),
    );

    // 3. Compute optimized Huffman tables from the edited coefficient
    // distributions.
    let out = emit_sequential(
        coeffs,
        subsampling,
        &edited_components,
        &new_quant_tables,
        &config.preserved_segments,
    )?;

    // Smallest-trial (#143 item 2): same coefficients, progressive scan
    // structure, ship whichever is shorter. Gated on the sequential size
    // like the main encoder's trials, so large outputs never pay for a
    // second serialization.
    if out.len() <= crate::encode::ENTROPY_TRIAL_MAX_BYTES
        && let Some(prog) = emit_progressive_trial(
            coeffs,
            edited_components,
            new_quant_tables,
            &config.preserved_segments,
        )
        && prog.len() < out.len()
    {
        return Ok(prog);
    }
    Ok(out)
}

/// Serialize the edited coefficients as one baseline-sequential scan.
fn emit_sequential(
    coeffs: &DecodedCoefficients,
    subsampling: Subsampling,
    edited_components: &[EditedComponent],
    new_quant_tables: &[Option<[u16; 64]>],
    preserved_segments: &[PreservedSegment],
) -> Result<Vec<u8>, Error> {
    let is_color = coeffs.components.len() >= 3;
    let huff = optimize_huffman_tables(edited_components, is_color)?;

    // 4. Emit JPEG bytes.
    let mut out = Vec::with_capacity(64 * 1024);
    write_marker_segment(&mut out, MARKER_SOI, &[]); // SOI has no payload
    // Carry the source's APPn/COM metadata (ICC, EXIF, XMP, JFIF, Adobe,
    // COM) verbatim, right after SOI and before DQT — JPEG requires APPn
    // segments before the frame header. Written in their original decoded
    // order (JFIF/APP0 first in well-formed files). `seg.data` is the raw
    // payload excluding the 2 length bytes, exactly what
    // `write_marker_segment` expects. MPF is filtered out by the caller.
    for seg in preserved_segments {
        write_marker_segment(&mut out, seg.marker, &seg.data);
    }
    let needs_sof1 = write_dqt(&mut out, new_quant_tables)?;
    write_sof(&mut out, coeffs, subsampling, edited_components, needs_sof1)?;
    write_dht(&mut out, &huff, is_color)?;
    write_sos(&mut out, edited_components, is_color)?;
    // Pass the MCU-aligned width to the scan emitter so its computed
    // `y_blocks_w` matches our actual block array stride. The SOF we
    // wrote above already carries the unpadded image dimensions, so
    // the decoder crops correctly after reconstruction.
    let mcu_aligned_width = edited_components[0].blocks_wide * 8;
    write_scan_data(
        &mut out,
        edited_components,
        &huff,
        is_color,
        subsampling,
        mcu_aligned_width,
    )?;
    out.push(0xFF);
    out.push(MARKER_EOI);
    Ok(out)
}

/// Serialize the SAME edited coefficients progressively via the lossless
/// pipeline's emitter (jpegli scan script, optimized per-scan Huffman
/// tables, natural-order quant tables — the layout `DecodedCoefficients`
/// already uses). Consumes the edited components so no coefficient plane
/// is cloned. `None` if the emitter declines (the sequential stream is
/// then shipped unchanged).
fn emit_progressive_trial(
    coeffs: &DecodedCoefficients,
    edited_components: Vec<EditedComponent>,
    new_quant_tables: Vec<Option<[u16; 64]>>,
    preserved_segments: &[PreservedSegment],
) -> Option<Vec<u8>> {
    use crate::decode::ComponentCoefficients;
    use crate::lossless::TransformedCoefficients;
    let components = edited_components
        .into_iter()
        .zip(&coeffs.components)
        .map(|(e, src)| ComponentCoefficients {
            id: src.id,
            coeffs: e.coeffs,
            blocks_wide: e.blocks_wide,
            blocks_high: e.blocks_high,
            h_samp: e.h_samp,
            v_samp: e.v_samp,
            quant_table_idx: e.quant_table_idx,
        })
        .collect();
    let tc = TransformedCoefficients {
        width: coeffs.width,
        height: coeffs.height,
        components,
        quant_tables: new_quant_tables,
    };
    crate::lossless::restructure::encode_progressive_from_coefficients(
        &tc,
        Some(preserved_segments),
        0,
        &enough::Unstoppable,
    )
    .ok()
}

/// Per-component requantized data (in zigzag order, length =
/// num_blocks * 64).
struct EditedComponent {
    coeffs: Vec<i16>,
    blocks_wide: usize,
    blocks_high: usize,
    h_samp: u8,
    v_samp: u8,
    quant_table_idx: u8,
}

fn build_new_quant_tables(
    coeffs: &DecodedCoefficients,
    strategy: QuantStrategy,
) -> Result<NewQuantTables, Error> {
    // `DecodedCoefficients.quant_tables` are in NATURAL (row-major)
    // order. The DQT marker writer applies the zigzag re-order on
    // emit. The requantize step (`edit_coefficients`) is given a
    // per-position scale array (also natural-order — same as the
    // coefficients' zigzag indexing).
    //
    // Returns a per-component `[f32; 64]` of `old_quant[i] /
    // new_quant[i]` ratios. Re-quantization is then `new_coeff[i] =
    // round(old_coeff[i] * ratio[i])` — when `ratio[i] = 1/k` (k
    // integer), this is exact for old_coeff divisible by k.
    // CRITICAL: build each *unique* quant table exactly once, always
    // reading from the ORIGINAL `coeffs.quant_tables`. turbo/mozjpeg use
    // a 2-table layout where Cb and Cr SHARE one chroma table; iterating
    // per-component and reading the in-progress `tables` would scale the
    // shared chroma table twice (once per chroma component) → chroma
    // over-quantized by scale². (jpegli's 3-separate-table layout hid
    // this — each table is referenced exactly once. This was the entire
    // turbo/mozjpeg Preserve "crater".)
    let original = &coeffs.quant_tables;
    let mut new_tables = original.clone();
    // Build each referenced table once, keyed by table_idx, using the
    // is_luma flag of the FIRST component that references it. (A table
    // is only ever shared among same-class components — luma tables by
    // luma, chroma tables by chroma — so the class is unambiguous.)
    let mut built = [false; 8];
    for comp in &coeffs.components {
        let idx = comp.quant_table_idx as usize;
        if idx >= 8 || built[idx] {
            continue;
        }
        let old = original
            .get(idx)
            .and_then(|t| t.as_ref())
            .copied()
            .ok_or(Error::Internal("missing quant table"))?;
        new_tables[idx] = Some(build_new_table(old, comp.id == 1, strategy));
        built[idx] = true;
    }
    // Per-component ratio = original_table[idx] / new_table[idx].
    let mut per_component_position_scale: Vec<[f32; 64]> =
        Vec::with_capacity(coeffs.components.len());
    for comp in &coeffs.components {
        let idx = comp.quant_table_idx as usize;
        let old = original
            .get(idx)
            .and_then(|t| t.as_ref())
            .copied()
            .ok_or(Error::Internal("missing quant table"))?;
        let new = new_tables[idx]
            .as_ref()
            .ok_or(Error::Internal("missing new quant table"))?;
        let mut ratio = [1.0f32; 64];
        for i in 0..64 {
            ratio[i] = old[i] as f32 / new[i] as f32;
        }
        per_component_position_scale.push(ratio);
    }
    Ok((new_tables, per_component_position_scale))
}

fn build_new_table(old: [u16; 64], is_luma: bool, strategy: QuantStrategy) -> [u16; 64] {
    let mut new_table = [0u16; 64];
    match strategy {
        QuantStrategy::UniformScale(scale) => {
            let scl = if is_luma { scale.luma } else { scale.chroma };
            for i in 0..64 {
                let scaled = (old[i] as f32 * scl).round() as i32;
                // Ceiling is `max(255, old)`, NOT a hard 255: a hard 255
                // clamp silently corrupts IDENTITY of a 16-bit-table source
                // (old 400 → 255 makes the old/new requant ratio ≠ 1.0,
                // requantizing coefficients that should pass through, and
                // loses precision). Capping at `max(255, old)` lets the
                // source's own 16-bit values pass through unchanged while
                // keeping the original baseline behavior (clamp scaled-up
                // 8-bit tables at 255) that the lossy recompress confidence
                // calibration relies on. `emit_preserved` emits a Pq=1 DQT
                // under SOF1 whenever any value exceeds 255.
                let ceiling = 255.max(old[i] as i32);
                new_table[i] = scaled.clamp(1, ceiling) as u16;
            }
        }
        QuantStrategy::TargetQuality { target_ijg_q } => {
            // Standard JPEG quality formula (ITU-T T.81 Annex K).
            let q = target_ijg_q.clamp(1, 100) as f32;
            let factor = if q < 50.0 {
                5000.0 / q
            } else {
                200.0 - 2.0 * q
            };
            let std = if is_luma {
                &crate::quant::STD_LUMINANCE_QUANT
            } else {
                &crate::quant::STD_CHROMINANCE_QUANT
            };
            for i in 0..64 {
                let v = (std[i] as f32 * factor / 100.0 + 0.5) as i32;
                // Never let the new table get FINER than the source's.
                // The source already discarded information at this
                // position — refining the table doesn't recover it
                // and just inflates the output. Pin to max(old, std-q).
                let new_v = v.clamp(1, 255).max(old[i] as i32);
                new_table[i] = new_v as u16;
            }
        }
        QuantStrategy::RobidouxTargetQuality { target_quality } => {
            // Same-family retarget for mozjpeg / ImageMagick sources:
            // scale the Robidoux base table (which both encoders use) to
            // the target quality with libjpeg's integer scaling. Because
            // the source's own table is also Robidoux-shaped, the
            // resulting `old[i]/new[i]` ratio is nearly constant across
            // positions, so the requant behaves like a clean uniform
            // scale to the exact target quality (no spectral reshape).
            // The same never-finer-than-source clamp applies: a coarser
            // target can only tighten quantization, never recover
            // already-discarded detail.
            let scale = libjpeg_quality_scale(target_quality);
            let scaled = scale_base_table(&ROBIDOUX_BASE, scale);
            let _ = is_luma; // mozjpeg uses one Robidoux table for both
            for i in 0..64 {
                let new_v = (scaled[i] as i32).clamp(1, 255).max(old[i] as i32);
                new_table[i] = new_v as u16;
            }
        }
    }
    new_table
}

fn edit_coefficients(
    coeffs: &DecodedCoefficients,
    position_scale_per_component: &[[f32; 64]],
    aq_mask: Option<&AqMask>,
) -> Vec<EditedComponent> {
    // Per-position requantize. `position_scale_per_component[c][i]`
    // is `old_quant[i] / new_quant[i]`. The new coefficient is
    // `round(old_coeff[i] * ratio[i])`. Note that the ratio is in
    // NATURAL order and so are the coefficients (decoder stores them
    // in zigzag — but actually wait, zenjpeg's docs say zigzag).
    //
    // CORRECTNESS NOTE on ordering: zenjpeg's `DecodedCoefficients`
    // docstring says "coeffs in zigzag order within each block".
    // The `quant_tables` field is in NATURAL order (verified by the
    // DQT byte-layout investigation that found the v0.2.1 zigzag bug).
    // To pair them correctly we'd zigzag-reorder one of them.
    //
    // For `UniformScale`, the per-position ratio is the same scalar
    // for every i, so ordering is irrelevant. For `TargetQuality`,
    // the ratio varies with position — and the position semantics
    // differ (zigzag-i vs natural-i) — so we need to be explicit.
    //
    // We build `ratio_in_zigzag[zigzag_i] = old_natural[zigzag_to_nat[zz_i]]
    // / new_natural[zigzag_to_nat[zz_i]]` (which equals
    // `position_scale_per_component[c][zigzag_to_nat[zz_i]]`).
    let mut out = Vec::with_capacity(coeffs.components.len());
    for (comp_idx, comp) in coeffs.components.iter().enumerate() {
        let ratio_nat = &position_scale_per_component[comp_idx];
        let mut ratio_zz = [1.0f32; 64];
        for zz_i in 0..64 {
            ratio_zz[zz_i] = ratio_nat[ZIGZAG[zz_i]];
        }
        let mut new_coeffs = Vec::with_capacity(comp.coeffs.len());
        let n_blocks = comp.num_blocks();
        for block_idx in 0..n_blocks {
            let block = comp.block(block_idx);
            let mut block_buf = [0i16; 64];
            for (i, &c) in block.iter().enumerate() {
                block_buf[i] = ((c as f32) * ratio_zz[i]).round().clamp(-1024.0, 1023.0) as i16;
            }
            // AQ zero-bias mask applies only to the luma component and
            // only to AC coefficients (indices 1..64). Match by POSITION
            // (`comp_idx == 0`), not component id: `build_aq_mask` builds
            // the mask from `components[0]`, and the luma component's id is
            // not guaranteed to be 1 (some encoders use 0-based or RGB ids).
            if comp_idx == 0
                && let Some(mask) = aq_mask
                && block_idx < mask.len()
            {
                let m = mask[block_idx];
                for (i, slot) in block_buf.iter_mut().enumerate().skip(1) {
                    if (m >> i) & 1 == 1 {
                        *slot = 0;
                    }
                }
            }
            new_coeffs.extend_from_slice(&block_buf);
        }
        out.push(EditedComponent {
            coeffs: new_coeffs,
            blocks_wide: comp.blocks_wide,
            blocks_high: comp.blocks_high,
            h_samp: comp.h_samp,
            v_samp: comp.v_samp,
            quant_table_idx: comp.quant_table_idx,
        });
    }
    out
}

fn optimize_huffman_tables(
    components: &[EditedComponent],
    is_color: bool,
) -> Result<HuffmanTableSet, Error> {
    let _ = (components, is_color);
    // Preserve emitter uses Annex K standard Huffman tables. The
    // frequency-counter-optimized path produced occasional
    // invalid-Huffman-code outputs in v0.1 testing; Annex K is
    // ~5-10% larger but is always correct, which beats a sometimes-
    // smaller-sometimes-broken optimizer. Returning to the
    // frequency-fitted path is a v0.2 task.
    //
    // ROOT CAUSE OF THE v0.1 BREAKAGE (diagnosed 2026-08-26, sweep issue
    // #197 / same class as #194): `_frequency_optimized_huffman_tables_v02_path`
    // below counts each component's DC diffs in RASTER/storage block order,
    // but emission goes through `encode_blocks_mcu_order`, which walks luma
    // MCU-INTERLEAVED for subsampled sources — a different DC-diff sequence,
    // so a category can appear at emit time that was never counted, get no
    // code, and be written as ZERO bits (silently corrupt output). Before
    // re-enabling: make the counting pass share the exact
    // `encode_blocks_mcu_order` traversal (per-block callback), the way
    // `lossless/pipeline.rs` unified its count+emit after #194.
    HuffmanTableSet::from_standard().map_err(|e| Error::Zenjpeg(format!("standard huffman: {e}")))
}

#[allow(dead_code)]
fn _frequency_optimized_huffman_tables_v02_path(
    components: &[EditedComponent],
    is_color: bool,
) -> Result<HuffmanTableSet, Error> {
    let mut fc_dc_luma = FrequencyCounter::new();
    let mut fc_ac_luma = FrequencyCounter::new();
    let mut fc_dc_chroma = FrequencyCounter::new();
    let mut fc_ac_chroma = FrequencyCounter::new();

    for (comp_idx, comp) in components.iter().enumerate() {
        let dc = if comp_idx == 0 {
            &mut fc_dc_luma
        } else {
            &mut fc_dc_chroma
        };
        let ac = if comp_idx == 0 {
            &mut fc_ac_luma
        } else {
            &mut fc_ac_chroma
        };
        let mut prev_dc = 0_i16;
        let n_blocks = comp.blocks_wide * comp.blocks_high;
        for b in 0..n_blocks {
            let block = &comp.coeffs[b * 64..(b + 1) * 64];
            // DC diff category
            let dc_diff = block[0].wrapping_sub(prev_dc);
            prev_dc = block[0];
            dc.count(crate::entropy::category(dc_diff));
            // AC run-length symbols
            let mut run = 0_u8;
            for &c in &block[1..64] {
                if c == 0 {
                    run += 1;
                } else {
                    while run >= 16 {
                        ac.count(0xF0);
                        run -= 16;
                    }
                    let ac_cat = crate::entropy::category(c);
                    ac.count((run << 4) | ac_cat);
                    run = 0;
                }
            }
            if run > 0 {
                ac.count(0x00); // EOB
            }
        }
    }
    let dc_luma = fc_dc_luma
        .generate_table_with_dht()
        .map_err(|e| Error::Zenjpeg(format!("DC luma huffman: {e}")))?;
    let ac_luma = fc_ac_luma
        .generate_table_with_dht()
        .map_err(|e| Error::Zenjpeg(format!("AC luma huffman: {e}")))?;
    let (dc_chroma, ac_chroma) = if is_color {
        (
            fc_dc_chroma
                .generate_table_with_dht()
                .map_err(|e| Error::Zenjpeg(format!("DC chroma huffman: {e}")))?,
            fc_ac_chroma
                .generate_table_with_dht()
                .map_err(|e| Error::Zenjpeg(format!("AC chroma huffman: {e}")))?,
        )
    } else {
        let std = HuffmanTableSet::from_standard()
            .map_err(|e| Error::Zenjpeg(format!("standard huffman: {e}")))?;
        (std.dc_chroma, std.ac_chroma)
    };
    Ok(HuffmanTableSet {
        dc_luma,
        ac_luma,
        dc_chroma,
        ac_chroma,
    })
}

fn write_marker_segment(out: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    out.push(0xFF);
    out.push(marker);
    if !payload.is_empty() {
        let len = (payload.len() + 2) as u16;
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(payload);
    }
}

/// Write the DQT segment. Each table is emitted at 8-bit precision (Pq=0)
/// when all its values fit a byte, or 16-bit precision (Pq=1, 2 bytes
/// big-endian per value) otherwise — exactly mirroring how the source was
/// decoded. Returns `true` if any table needed 16-bit precision, in which
/// case the frame header must be SOF1 (extended sequential): baseline SOF0
/// only permits Pq=0 tables.
fn write_dqt(out: &mut Vec<u8>, tables: &[Option<[u16; 64]>]) -> Result<bool, Error> {
    let mut payload = Vec::new();
    let mut any_16bit = false;
    for (idx, t) in tables.iter().enumerate() {
        if let Some(table) = t {
            // Pq=1 (16-bit) if any coefficient exceeds 255, else Pq=0.
            let pq: u8 = if table.iter().any(|&v| v > 255) { 1 } else { 0 };
            any_16bit |= pq == 1;
            payload.push((pq << 4) | (idx as u8 & 0x0F));
            // `table` is stored in NATURAL (row-major) order — that's
            // how zenjpeg's decoder hands it back. The DQT marker
            // requires ZIGZAG order. Apply the zigzag mapping on write.
            for &zz_to_natural in &ZIGZAG {
                let v = table[zz_to_natural];
                if pq == 1 {
                    payload.extend_from_slice(&v.to_be_bytes());
                } else {
                    payload.push(v as u8);
                }
            }
        }
    }
    write_marker_segment(out, MARKER_DQT, &payload);
    Ok(any_16bit)
}

fn write_sof(
    out: &mut Vec<u8>,
    coeffs: &DecodedCoefficients,
    _subsampling: Subsampling,
    components: &[EditedComponent],
    needs_sof1: bool,
) -> Result<(), Error> {
    // SOF1 (extended sequential) when any quant table is 16-bit; baseline
    // SOF0 otherwise. Both are sequential Huffman frames — the only
    // difference relevant here is that SOF0 forbids Pq=1 DQT tables.
    let marker = if needs_sof1 { MARKER_SOF1 } else { MARKER_SOF0 };
    let mut payload = Vec::new();
    payload.push(8); // sample precision
    payload.extend_from_slice(&(coeffs.height as u16).to_be_bytes());
    payload.extend_from_slice(&(coeffs.width as u16).to_be_bytes());
    payload.push(components.len() as u8);
    for (i, comp) in components.iter().enumerate() {
        let component_id = match i {
            0 => 1,
            1 => 2,
            2 => 3,
            _ => return Err(Error::Internal("too many components")),
        };
        payload.push(component_id);
        // sampling factors: high nibble = H, low nibble = V
        payload.push((comp.h_samp << 4) | (comp.v_samp & 0x0F));
        payload.push(comp.quant_table_idx);
    }
    write_marker_segment(out, marker, &payload);
    Ok(())
}

fn write_dht_table(payload: &mut Vec<u8>, class: u8, dest_id: u8, table: &OptimizedTable) {
    payload.push((class << 4) | (dest_id & 0x0F));
    payload.extend_from_slice(&table.bits);
    payload.extend_from_slice(&table.values);
}

fn write_dht(out: &mut Vec<u8>, huff: &HuffmanTableSet, is_color: bool) -> Result<(), Error> {
    let mut payload = Vec::new();
    write_dht_table(&mut payload, 0, 0, &huff.dc_luma);
    write_dht_table(&mut payload, 1, 0, &huff.ac_luma);
    if is_color {
        write_dht_table(&mut payload, 0, 1, &huff.dc_chroma);
        write_dht_table(&mut payload, 1, 1, &huff.ac_chroma);
    }
    write_marker_segment(out, MARKER_DHT, &payload);
    Ok(())
}

fn write_sos(
    out: &mut Vec<u8>,
    components: &[EditedComponent],
    is_color: bool,
) -> Result<(), Error> {
    let mut payload = Vec::new();
    payload.push(components.len() as u8);
    for (i, _comp) in components.iter().enumerate() {
        let component_id = (i + 1) as u8;
        // dc/ac table selector: high nibble = DC, low = AC
        // luma = table 0, chroma = table 1
        let td = if i == 0 { 0 } else { 1 };
        let ta = if i == 0 { 0 } else { 1 };
        payload.push(component_id);
        payload.push((td << 4) | (ta & 0x0F));
    }
    // Sequential: Ss=0, Se=63, Ah/Al=0
    payload.push(0);
    payload.push(63);
    payload.push(0);
    let _ = is_color;
    write_marker_segment(out, MARKER_SOS, &payload);
    Ok(())
}

fn write_scan_data(
    out: &mut Vec<u8>,
    components: &[EditedComponent],
    huff: &HuffmanTableSet,
    is_color: bool,
    subsampling: Subsampling,
    width: usize,
) -> Result<(), Error> {
    // Collect blocks per component as [[i16;64]; N].
    let y_blocks = component_to_blocks(&components[0]);
    let (cb_blocks, cr_blocks) = if is_color && components.len() >= 3 {
        (
            component_to_blocks(&components[1]),
            component_to_blocks(&components[2]),
        )
    } else {
        (Vec::new(), Vec::new())
    };

    let mut writer = BitWriter::new();
    let mut state = StreamingEntropyState {
        prev_dc: [0; 3],
        mcu_idx: 0,
        restart_count: 0,
    };

    // Total MCUs in image.
    let (h_samp, v_samp) = match subsampling {
        Subsampling::S444 => (1, 1),
        Subsampling::S422 => (2, 1),
        Subsampling::S420 => (2, 2),
        Subsampling::S440 => (1, 2),
    };
    let luma = &components[0];
    let mcus_x = luma.blocks_wide.div_ceil(h_samp);
    let mcus_y = luma.blocks_high.div_ceil(v_samp);
    let total_mcus = mcus_x * mcus_y;

    encode_blocks_mcu_order(
        &y_blocks,
        &cb_blocks,
        &cr_blocks,
        huff,
        &mut writer,
        is_color,
        &mut state,
        subsampling,
        width,
        0, // no restart markers
        total_mcus,
    )
    .map_err(|e| Error::Zenjpeg(format!("scan emit: {e}")))?;

    // `BitWriter::into_bytes` already applies 0xFF→0xFF 0x00 byte
    // stuffing internally per JPEG scan-data convention (see
    // `crate::foundation::bitstream`). Do NOT double-stuff.
    let scan_bytes = writer.into_bytes();
    out.extend_from_slice(&scan_bytes);
    Ok(())
}

fn component_to_blocks(comp: &EditedComponent) -> Vec<[i16; 64]> {
    let n = comp.blocks_wide * comp.blocks_high;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut b = [0i16; 64];
        b.copy_from_slice(&comp.coeffs[i * 64..(i + 1) * 64]);
        out.push(b);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A source encoded at mozjpeg/ImageMagick quality `q` has the
    /// Robidoux base table scaled by `libjpeg_quality_scale(q)`. When we
    /// retarget it to a coarser quality with `RobidouxTargetQuality`,
    /// the new table must equal `scale_base_table(ROBIDOUX_BASE,
    /// target_scale)` at every position where that's ≥ the source — the
    /// same-family property that makes the requant a clean uniform
    /// rescale rather than a spectral reshape.
    #[test]
    fn robidoux_retarget_matches_canonical_scaled_table() {
        let src_q = 90u8;
        let tgt_q = 70u8;
        let src = scale_base_table(&ROBIDOUX_BASE, libjpeg_quality_scale(src_q));
        let expected = scale_base_table(&ROBIDOUX_BASE, libjpeg_quality_scale(tgt_q));
        let got = build_new_table(
            src,
            true,
            QuantStrategy::RobidouxTargetQuality {
                target_quality: tgt_q,
            },
        );
        for i in 0..64 {
            // Coarser target ⇒ never finer than source ⇒ the clamp is a
            // no-op and the new table is exactly the canonical scaled
            // Robidoux table.
            assert!(
                expected[i] >= src[i],
                "pos {i}: target table finer than source (q{tgt_q} vs q{src_q})"
            );
            assert_eq!(
                got[i], expected[i],
                "pos {i}: robidoux retarget diverged from canonical scaled table"
            );
        }
    }

    /// The per-position `old/new` ratio for same-family Robidoux
    /// retargeting must be nearly constant across positions (within a
    /// tight band) — that's the property that makes the requant a clean
    /// uniform scale instead of the spectral reshape that craters
    /// cross-family IJG-std retargeting.
    #[test]
    fn robidoux_retarget_ratio_is_near_uniform() {
        let src = scale_base_table(&ROBIDOUX_BASE, libjpeg_quality_scale(90));
        let new = build_new_table(
            src,
            true,
            QuantStrategy::RobidouxTargetQuality { target_quality: 70 },
        );
        let mut ratios = Vec::new();
        for i in 0..64 {
            ratios.push(src[i] as f32 / new[i] as f32);
        }
        let mean = ratios.iter().sum::<f32>() / 64.0;
        // Robidoux@90→Robidoux@70 is a roughly 2.7× quant increase. The
        // per-position ratio is dominated by integer rounding of small
        // base values; require every position within ±35% of the mean
        // (cross-family IJG retargeting of a Robidoux source spreads far
        // wider, with low-frequency ratios near 1.0 and high near 0.3).
        for (i, r) in ratios.iter().enumerate() {
            assert!(
                (*r - mean).abs() <= 0.35 * mean,
                "pos {i}: ratio {r} too far from mean {mean} (not near-uniform)"
            );
        }
    }
}

#[cfg(test)]
mod smallest_trial_tests {
    use super::*;

    /// #143 item 2: `emit_preserved` must ship the shorter of the sequential
    /// and progressive serializations (when the sequential one is within the
    /// trial gate), and both candidates must carry IDENTICAL coefficients —
    /// the trial is a pure rate decision.
    #[test]
    fn preserve_emit_ships_the_smaller_of_sequential_and_progressive() {
        use crate::decoder::Decoder;
        use crate::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
        use enough::Unstoppable;

        fn smooth_blocks(w: u32, h: u32) -> Vec<u8> {
            // Flat 16×16 tiles with mild per-tile colour steps: mostly-zero AC,
            // where progressive EOB runs and DC-first scans pay off.
            let mut v = vec![0u8; (w * h * 3) as usize];
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let i = (y * w as usize + x) * 3;
                    let t = ((x / 16) * 7 + (y / 16) * 13) as u8;
                    v[i] = 90u8.wrapping_add(t);
                    v[i + 1] = 110u8.wrapping_add(t.wrapping_mul(2));
                    v[i + 2] = 130u8.wrapping_sub(t);
                }
            }
            v
        }
        fn noisy(w: u32, h: u32) -> Vec<u8> {
            let mut v = vec![0u8; (w * h * 3) as usize];
            let mut s = 0x1357u32;
            for b in v.iter_mut() {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                *b = (s >> 24) as u8;
            }
            v
        }
        fn jpeg(rgb: &[u8], w: u32, h: u32) -> Vec<u8> {
            let mut enc = EncoderConfig::ycbcr(75.0, ChromaSubsampling::Quarter)
                .progressive(false)
                .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
                .unwrap();
            enc.push_packed(rgb, Unstoppable).unwrap();
            enc.finish().unwrap()
        }
        fn coeff_planes(jpeg: &[u8]) -> Vec<(u8, Vec<i16>)> {
            let c = Decoder::new()
                .decode_coefficients(jpeg, Unstoppable)
                .expect("decode_coefficients");
            c.components
                .iter()
                .map(|comp| (comp.id, comp.coeffs.clone()))
                .collect()
        }

        let mut prog_won = 0usize;
        for (name, rgb, w, h) in [
            ("smooth", smooth_blocks(256, 192), 256u32, 192u32),
            ("noisy", noisy(96, 64), 96, 64),
            ("tiny", smooth_blocks(24, 24), 24, 24),
        ] {
            let src = jpeg(&rgb, w, h);
            let coeffs = Decoder::new()
                .decode_coefficients(&src, Unstoppable)
                .expect("source coefficients");
            let cfg = EmitConfig::default();

            // The two candidates, built exactly as emit_preserved builds them.
            let (tables, scales) = build_new_quant_tables(&coeffs, cfg.quant_strategy).unwrap();
            let edited = edit_coefficients(&coeffs, &scales, None);
            let seq = emit_sequential(&coeffs, Subsampling::S420, &edited, &tables, &[]).unwrap();
            let prog = emit_progressive_trial(&coeffs, edited, tables, &[])
                .expect("progressive trial must succeed on a plain 4:2:0 source");
            assert!(
                seq.len() <= crate::encode::ENTROPY_TRIAL_MAX_BYTES,
                "{name}: fixture must sit inside the trial gate ({} B)",
                seq.len()
            );

            // Pure rate decision: identical coefficients either way.
            assert_eq!(
                coeff_planes(&seq),
                coeff_planes(&prog),
                "{name}: candidates differ"
            );
            assert_eq!(
                coeff_planes(&seq),
                coeff_planes(&src),
                "{name}: not lossless vs source"
            );

            let shipped = emit_preserved(&coeffs, Subsampling::S420, &cfg).unwrap();
            let expect = if prog.len() < seq.len() { &prog } else { &seq };
            assert_eq!(
                shipped.len(),
                expect.len(),
                "{name}: shipped {} B, sequential {} B, progressive {} B",
                shipped.len(),
                seq.len(),
                prog.len()
            );
            assert_eq!(
                &shipped, expect,
                "{name}: shipped bytes are not the shorter candidate"
            );
            eprintln!(
                "{name}: sequential {} B, progressive {} B",
                seq.len(),
                prog.len()
            );
            if prog.len() < seq.len() {
                prog_won += 1;
            }
        }
        assert!(
            prog_won > 0,
            "no fixture where progressive wins — the trial gate is untested"
        );
    }
}
