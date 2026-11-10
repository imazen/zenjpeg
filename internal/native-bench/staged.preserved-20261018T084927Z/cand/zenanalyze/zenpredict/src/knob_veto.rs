//! Feature-gated knob-veto safety bounds for K=1 single-pick pickers.
//!
//! A K=1 (pure-argmin) codec picker occasionally mis-sets a categorical
//! toggle (chroma subsampling / bit-depth / quant-matrix) on a tiny
//! fraction of images, blowing the worst-case RD overhead far past the
//! safety gate. The training pipeline ([`zentrain`]) derives feature-gated
//! per-(categorical-axis-value) **vetoes** that bound that tail: a rule
//! "forbid value `V` on axis `A` when feature `F` `<`/`>` `threshold`".
//! Firing a rule removes the offending cells from the picker's reachable
//! set *without* touching the oracle, so the achievable optimum is
//! unchanged while the worst case is bounded.
//!
//! This module is the **deploy side** of that feature. The trainer embeds
//! the rules in the baked model's metadata under [`KNOB_VETOES_KEY`]; the
//! codec runtime parses them ([`Model::knob_vetoes`] / [`parse_knob_vetoes`])
//! and applies them as an extra masking pass ([`apply_knob_vetoes`]) over
//! the [`AllowedMask`] before [`argmin_masked`], so the deployed picker
//! enforces exactly the vetoes the bake-time safety gate evaluated.
//!
//! ## Wire format ([`KNOB_VETOES_KEY`], `value_type = bytes`)
//!
//! ```text
//! [1]  n_vetoes: u8
//! repeat n_vetoes times:
//!   [2]  feat_idx:  u16 (LE)   index into the bake's feat_cols
//!   [1]  op:        u8         0 = LessThan (<), 1 = GreaterThan (>)
//!   [4]  threshold: f32 (LE)
//!   [1]  n_cells:   u8
//!   [n_cells] cell_id: u8 each (cell indices to forbid when the rule fires)
//! ```
//!
//! An absent metadata entry (or an empty blob) means "no vetoes" — the
//! deploy path is byte-identical to a bake that never carried the key
//! (backward-compatible).
//!
//! ## NaN handling — matches the trainer
//!
//! The trainer evaluates the gate with `np.nan_to_num(features, nan=0.0)`
//! before the comparison (`build_veto_mask` / `_veto_feature_matrix` in
//! `zentrain/tools/train_hybrid.py`): a NaN feature is treated as **0.0**,
//! not as "never fires". [`apply_knob_vetoes`] replicates this so the
//! runtime mask is identical to what the bake gate scored.
//!
//! ## Never-strand — the picker loop's responsibility, NOT this helper's
//!
//! [`apply_knob_vetoes`] is a plain masking pass: it only ever sets
//! `allowed[cell] = false`. The trainer's *deployed picker* additionally
//! reverts the vetoes for a row if they would leave **no** reachable cell
//! (never strand a row — `evaluate_argmin_per_row`). Reproduce that in the
//! codec's argmin composition:
//!
//! ```rust
//! # use zenpredict::{AllowedMask, KnobVeto, ScoreTransform, apply_knob_vetoes, argmin_masked};
//! # let predictions = [1.0f32, 2.0];
//! # let features = [0.0f32];
//! # let vetoes: &[KnobVeto<'_>] = &[];
//! # let reach = [true, true];
//! // `reach` = reachable & caller constraints (the pre-veto AllowedMask data).
//! let mut allowed = reach;                 // snapshot the pre-veto mask
//! apply_knob_vetoes(&features, vetoes, &mut allowed);
//! let pick = match argmin_masked(&predictions, &AllowedMask::new(&allowed), ScoreTransform::Exp, None) {
//!     Some(p) => Some(p),
//!     // vetoes stranded the row → fall back to the un-vetoed reachable set.
//!     None => argmin_masked(&predictions, &AllowedMask::new(&reach), ScoreTransform::Exp, None),
//! };
//! # let _ = pick;
//! ```
//!
//! [`zentrain`]: https://github.com/imazen/zenanalyze/tree/main/zentrain
//! [`AllowedMask`]: crate::AllowedMask
//! [`argmin_masked`]: crate::argmin_masked
//! [`Model::knob_vetoes`]: crate::Model::knob_vetoes

use alloc::vec::Vec;

use crate::error::PredictError;
use crate::metadata::Metadata;

/// Metadata key carrying the packed knob-veto rules.
///
/// Lives under the `zenpicker.*` namespace (the picker-selection
/// reservation in [`crate::keys`]'s module docs) because a veto constrains
/// which picker *cells* are reachable. Written by `bake_picker.py`,
/// `value_type = bytes`; read by [`parse_knob_vetoes`] /
/// [`Model::knob_vetoes`](crate::Model::knob_vetoes).
pub const KNOB_VETOES_KEY: &str = "zenpicker.knob_vetoes";

/// Sentinel `feat_idx` marking a veto that gates on the caller's **target zq**
/// (the quality target) rather than an analyzed image feature — the `__zq__`
/// pseudo-feature the trainer derives for quality-conditioned catastrophe
/// bounds. It is `u16::MAX`, out of range for any real feature vector, so the
/// plain [`apply_knob_vetoes`] masking pass *skips* it (it can't see target_zq);
/// the picker-safety composition ([`crate::resolve_pre_argmin`]) recognizes the
/// sentinel and applies it against the `target_zq` it already holds. MUST match
/// `bake_picker.ZQ_VETO_SENTINEL`.
pub(crate) const ZQ_VETO_SENTINEL: u16 = 0xFFFF;

/// Comparison direction for a knob veto's feature gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum VetoOp {
    /// Wire code `0`. The veto fires when `feature < threshold`.
    LessThan,
    /// Wire code `1`. The veto fires when `feature > threshold`.
    GreaterThan,
}

impl VetoOp {
    /// Decode the wire byte. `0 = LessThan`, `1 = GreaterThan`; any other
    /// value is rejected (forward-incompatible op — fail loud rather than
    /// silently mis-apply a safety bound).
    fn from_byte(b: u8) -> Result<Self, PredictError> {
        match b {
            0 => Ok(Self::LessThan),
            1 => Ok(Self::GreaterThan),
            other => Err(PredictError::UnknownVetoOp { byte: other }),
        }
    }
}

/// One feature-gated safety veto: "when `features[feat_idx] {op} threshold`,
/// forbid every cell in `cells`."
///
/// `cells` borrows directly from the metadata blob (zero-copy), so the
/// struct carries the blob's lifetime. The trainer guarantees `feat_idx`
/// is a valid index into the bake's feat_cols and each `cell` is a valid
/// picker cell index; [`apply_knob_vetoes`] bounds-checks both anyway and
/// skips out-of-range indices rather than panicking.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KnobVeto<'a> {
    /// Index into the bake's feat_cols (the analyzed-feature order, i.e.
    /// the leading `feat_cols.len()` slots of the model input vector).
    pub feat_idx: u16,
    /// Comparison direction of the feature gate.
    pub op: VetoOp,
    /// Threshold the gated feature is compared against.
    pub threshold: f32,
    /// Picker cell indices forbidden when the gate fires.
    pub cells: &'a [u8],
}

/// Parse the packed knob-veto wire blob (see the [module docs](self) for
/// the byte layout) into a list of [`KnobVeto`] borrowing `bytes`.
///
/// An empty input yields an empty list (the "no vetoes" case). Returns
/// [`PredictError::Truncated`] if the blob ends mid-record and
/// [`PredictError::UnknownVetoOp`] for an unrecognized op byte.
pub fn parse_knob_vetoes(bytes: &[u8]) -> Result<Vec<KnobVeto<'_>>, PredictError> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    let mut pos = 0usize;
    let n_vetoes = bytes[0] as usize;
    pos += 1;

    let mut out = Vec::with_capacity(n_vetoes);
    for _ in 0..n_vetoes {
        // [2] feat_idx (LE)
        let feat_idx = read_u16_le(bytes, &mut pos)?;
        // [1] op
        let op_byte = read_u8(bytes, &mut pos)?;
        let op = VetoOp::from_byte(op_byte)?;
        // [4] threshold (LE)
        let threshold = read_f32_le(bytes, &mut pos)?;
        // [1] n_cells
        let n_cells = read_u8(bytes, &mut pos)? as usize;
        // [n_cells] cell ids
        let cells = read_slice(bytes, &mut pos, n_cells)?;
        out.push(KnobVeto {
            feat_idx,
            op,
            threshold,
            cells,
        });
    }
    Ok(out)
}

/// Read the knob-veto rules from a parsed [`Metadata`] blob.
///
/// Returns an empty list when [`KNOB_VETOES_KEY`] is absent (a bake from
/// before the feature, or a picker that needed no veto). The returned
/// [`KnobVeto`]s borrow the metadata's underlying bytes.
pub fn knob_vetoes_from_metadata<'a>(
    meta: &Metadata<'a>,
) -> Result<Vec<KnobVeto<'a>>, PredictError> {
    match meta.get(KNOB_VETOES_KEY) {
        None => Ok(Vec::new()),
        Some(entry) => parse_knob_vetoes(entry.value),
    }
}

/// Apply knob vetoes as a pre-argmin masking pass: for every veto whose
/// feature gate fires, set `allowed[cell] = false` for each of its cells.
///
/// `features` is indexed by the bake's feat_cols order (`feat_idx` is a
/// feat_cols index); pass at least the feat_cols-prefixed feature vector
/// the picker was built with. `allowed` is the picker's [`AllowedMask`]
/// backing slice — typically `reachable & caller-constraints` — and is
/// mutated in place. Out-of-range `feat_idx` (rule can't fire) and
/// out-of-range cell ids are skipped (panic-free).
///
/// A NaN feature is treated as `0.0` before the comparison, matching the
/// trainer's `np.nan_to_num(..., nan=0.0)` gate evaluation (see the
/// [module docs](self)). This helper only ever **denies** cells; the
/// never-strand fallback (revert if all cells become denied) is the
/// codec's to compose around the subsequent argmin — see the module docs
/// for the recommended shape.
///
/// [`AllowedMask`]: crate::AllowedMask
///
/// # Examples
///
/// ```
/// use zenpredict::{KnobVeto, VetoOp, apply_knob_vetoes};
///
/// // "forbid cells 1 and 3 when feature 0 > 9.0"
/// let veto = KnobVeto { feat_idx: 0, op: VetoOp::GreaterThan, threshold: 9.0, cells: &[1, 3] };
///
/// // Feature fires (10.0 > 9.0): cells 1 and 3 are denied.
/// let mut allowed = [true; 4];
/// apply_knob_vetoes(&[10.0], &[veto], &mut allowed);
/// assert_eq!(allowed, [true, false, true, false]);
///
/// // Feature does not fire (1.0 > 9.0 is false): mask unchanged.
/// let mut allowed = [true; 4];
/// apply_knob_vetoes(&[1.0], &[veto], &mut allowed);
/// assert_eq!(allowed, [true, true, true, true]);
/// ```
pub fn apply_knob_vetoes(features: &[f32], vetoes: &[KnobVeto<'_>], allowed: &mut [bool]) {
    for v in vetoes {
        let Some(&raw) = features.get(v.feat_idx as usize) else {
            // Feature index past the supplied vector — the rule can't be
            // evaluated, so it can't fire. Matches the trainer's
            // "no feature for this row -> no veto" graceful skip.
            continue;
        };
        // Match the trainer's NaN handling: a NaN feature reads as 0.0,
        // NOT as "comparison is false" (IEEE `NaN < x` / `NaN > x` are
        // both false, which would silently disagree with the bake gate).
        let fv = if raw.is_nan() { 0.0 } else { raw };
        let fires = match v.op {
            VetoOp::LessThan => fv < v.threshold,
            VetoOp::GreaterThan => fv > v.threshold,
        };
        if fires {
            for &c in v.cells {
                if let Some(slot) = allowed.get_mut(c as usize) {
                    *slot = false;
                }
            }
        }
    }
}

// --- little-endian cursor readers (panic-free, bounds-checked) ---

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
    let raw = read_array::<2>(bytes, pos)?;
    Ok(u16::from_le_bytes(raw))
}

#[inline]
fn read_f32_le(bytes: &[u8], pos: &mut usize) -> Result<f32, PredictError> {
    let raw = read_array::<4>(bytes, pos)?;
    Ok(f32::from_le_bytes(raw))
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
    // Infallible after the bounds check above; `?` keeps it panic-free
    // under any future refactor that drops the check.
    let arr: [u8; N] = slice.try_into().map_err(|_| PredictError::Truncated {
        offset: *pos,
        want: N,
        have: bytes.len().saturating_sub(*pos),
    })?;
    *pos = end;
    Ok(arr)
}

#[inline]
fn read_slice<'a>(bytes: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8], PredictError> {
    let end = pos.checked_add(n).ok_or(PredictError::Truncated {
        offset: *pos,
        want: n,
        have: bytes.len().saturating_sub(*pos),
    })?;
    let slice = bytes.get(*pos..end).ok_or(PredictError::Truncated {
        offset: *pos,
        want: n,
        have: bytes.len().saturating_sub(*pos),
    })?;
    *pos = end;
    Ok(slice)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::{Metadata, MetadataType};

    /// Encode the wire blob the way `bake_picker.py` does, so the tests
    /// exercise the exact byte layout the baker emits.
    fn encode(vetoes: &[(u16, u8, f32, &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(vetoes.len() as u8);
        for &(feat_idx, op, threshold, cells) in vetoes {
            out.extend_from_slice(&feat_idx.to_le_bytes());
            out.push(op);
            out.extend_from_slice(&threshold.to_le_bytes());
            out.push(cells.len() as u8);
            out.extend_from_slice(cells);
        }
        out
    }

    #[test]
    fn empty_blob_yields_no_vetoes() {
        assert!(parse_knob_vetoes(&[]).unwrap().is_empty());
    }

    #[test]
    fn roundtrip_two_avif_rules() {
        // The two real avif rules (resolved form): the qm=on veto on
        // feat 3 (< 0.0102435), 4 cells; the sub=420 veto on feat 33
        // (> 9.118219), 4 cells.
        let blob = encode(&[
            (3, 0, 0.010_243_5_f32, &[1, 3, 5, 7]),
            (33, 1, 9.118_219_f32, &[0, 1, 2, 3]),
        ]);
        let parsed = parse_knob_vetoes(&blob).unwrap();
        assert_eq!(parsed.len(), 2);

        assert_eq!(parsed[0].feat_idx, 3);
        assert_eq!(parsed[0].op, VetoOp::LessThan);
        assert!((parsed[0].threshold - 0.010_243_5).abs() < 1e-9);
        assert_eq!(parsed[0].cells, &[1, 3, 5, 7]);

        assert_eq!(parsed[1].feat_idx, 33);
        assert_eq!(parsed[1].op, VetoOp::GreaterThan);
        assert!((parsed[1].threshold - 9.118_219).abs() < 1e-5);
        assert_eq!(parsed[1].cells, &[0, 1, 2, 3]);
    }

    #[test]
    fn roundtrip_through_metadata_blob() {
        // Build a metadata TLV blob carrying just the knob_vetoes entry,
        // parse via Metadata, read the rules back, assert identical.
        let blob = encode(&[(33, 1, 9.118_219_f32, &[0, 1, 2, 3])]);
        let mut md = Vec::new();
        let key = KNOB_VETOES_KEY.as_bytes();
        md.push(key.len() as u8);
        md.extend_from_slice(key);
        md.push(0); // value_type = 0 (Bytes)
        md.extend_from_slice(&(blob.len() as u32).to_le_bytes());
        md.extend_from_slice(&blob);

        let meta = Metadata::parse(&md).unwrap();
        assert_eq!(meta.get(KNOB_VETOES_KEY).unwrap().kind, MetadataType::Bytes);
        let parsed = knob_vetoes_from_metadata(&meta).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].feat_idx, 33);
        assert_eq!(parsed[0].op, VetoOp::GreaterThan);
        assert_eq!(parsed[0].cells, &[0, 1, 2, 3]);
    }

    #[test]
    fn absent_metadata_key_is_no_vetoes() {
        let meta = Metadata::parse(&[]).unwrap();
        assert!(knob_vetoes_from_metadata(&meta).unwrap().is_empty());
    }

    #[test]
    fn apply_denies_cells_when_gate_fires() {
        // sub=420 veto: forbid cells {0,1,2,3} when feat_log_pixels > 9.118.
        let veto = KnobVeto {
            feat_idx: 33,
            op: VetoOp::GreaterThan,
            threshold: 9.118_219,
            cells: &[0, 1, 2, 3],
        };
        let mut features = [0.0f32; 50];
        features[33] = 9.5; // > threshold -> fires

        let mut allowed = [true; 8];
        apply_knob_vetoes(&features, &[veto], &mut allowed);
        assert_eq!(
            allowed,
            [false, false, false, false, true, true, true, true]
        );
    }

    #[test]
    fn apply_leaves_mask_when_gate_does_not_fire() {
        let veto = KnobVeto {
            feat_idx: 33,
            op: VetoOp::GreaterThan,
            threshold: 9.118_219,
            cells: &[0, 1, 2, 3],
        };
        let mut features = [0.0f32; 50];
        features[33] = 8.0; // < threshold -> does not fire

        let mut allowed = [true; 8];
        apply_knob_vetoes(&features, &[veto], &mut allowed);
        assert_eq!(allowed, [true; 8]);
    }

    #[test]
    fn apply_nan_feature_reads_as_zero() {
        // qm=on veto: forbid when feat < 0.0102. A NaN feature must read
        // as 0.0 (0.0 < 0.0102 -> fires), matching the trainer's
        // np.nan_to_num gate — NOT IEEE `NaN < x == false`.
        let veto = KnobVeto {
            feat_idx: 3,
            op: VetoOp::LessThan,
            threshold: 0.010_243_5,
            cells: &[1, 3, 5, 7],
        };
        let mut features = [0.5f32; 50];
        features[3] = f32::NAN;

        let mut allowed = [true; 8];
        apply_knob_vetoes(&features, &[veto], &mut allowed);
        // NaN -> 0.0 -> fires -> odd cells denied.
        assert_eq!(
            allowed,
            [true, false, true, false, true, false, true, false]
        );
    }

    #[test]
    fn apply_skips_out_of_range_feat_and_cells() {
        // feat_idx past the supplied vector -> can't fire (no panic).
        let veto_oob_feat = KnobVeto {
            feat_idx: 99,
            op: VetoOp::GreaterThan,
            threshold: 0.0,
            cells: &[0],
        };
        let mut allowed = [true; 4];
        apply_knob_vetoes(&[1.0f32, 2.0], &[veto_oob_feat], &mut allowed);
        assert_eq!(allowed, [true; 4]);

        // cell id past the mask -> skipped (no panic); in-range cell denied.
        let veto_oob_cell = KnobVeto {
            feat_idx: 0,
            op: VetoOp::GreaterThan,
            threshold: 0.0,
            cells: &[1, 250],
        };
        let mut allowed = [true; 4];
        apply_knob_vetoes(&[1.0f32], &[veto_oob_cell], &mut allowed);
        assert_eq!(allowed, [true, false, true, true]);
    }

    #[test]
    fn unknown_op_byte_is_rejected() {
        let blob = encode(&[(0, 2, 1.0, &[0])]); // op=2 is invalid
        assert_eq!(
            parse_knob_vetoes(&blob),
            Err(PredictError::UnknownVetoOp { byte: 2 })
        );
    }

    #[test]
    fn truncated_blob_is_rejected() {
        // Claims 1 veto but no record follows.
        assert!(matches!(
            parse_knob_vetoes(&[1]),
            Err(PredictError::Truncated { .. })
        ));
        // Claims 2 cells but only 1 byte present.
        let mut blob = encode(&[(0, 0, 1.0, &[])]);
        *blob.last_mut().unwrap() = 2; // n_cells = 2, but no cell bytes
        assert!(matches!(
            parse_knob_vetoes(&blob),
            Err(PredictError::Truncated { .. })
        ));
    }
}
