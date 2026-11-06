//! Safety + rescue summary embedded in every bake.
//!
//! Three small `#[repr(C, Pod)]` structures, all carried in the
//! metadata blob under the keys in [`crate::keys`]:
//!
//! - [`SafetyCompact`] (32 bytes) — bake-time safety gate result +
//!   calibration anchor + bake-side-tuned rescue thresholds. Codec
//!   reads once at startup; seeds [`crate::RescuePolicy::from_bake`].
//! - [`CellHint`] × `n_cells` (4 bytes each) — per-cell rescue
//!   guidance: which strategy to prefer, expected p99 shortfall,
//!   how often this cell needs rescue. Codec consults when the
//!   picker returns cell `c` and pass-0 verify undershoots.
//! - [`FallbackEntry`] × `n_zq_bands` (4 bytes each) — known-good
//!   `(target_zq → fallback_cell + quality bump)` table for the
//!   `KnownGoodFallback` strategy. Bake-prescribed safe configs;
//!   the picker is bypassed entirely on this path.
//!
//! Total typical size: ~200-400 bytes per bake. Comfortably under
//! the 500-byte budget for codecs with ≤ 80 cells × ≤ 50 zq bands.
//!
//! See [`zentrain/PRINCIPLES.md`](https://github.com/imazen/zenanalyze/blob/main/zentrain/PRINCIPLES.md)
//! § "Bake metadata triage" for what runtime needs vs what stays in
//! the sibling `manifest.json`.

use bytemuck::{Pod, Zeroable};

/// 32-byte fixed-shape header. Read once at codec startup;
/// hard-fail load when `passed == 0` unless the consumer explicitly
/// opts into `force_load`.
///
/// Field order is the wire layout — do not reorder. Future fields
/// go at the tail (the `_reserved` array carries 8 bytes of slack).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub struct SafetyCompact {
    /// Schema version of this struct. Bumps when the field layout
    /// changes (rare; new fields go in `_reserved` first).
    pub version: u16,
    /// `bit 0` = strict-mode certified. `bit 1` = bake covers HDR
    /// content. Future bits at the bake-side's discretion; runtime
    /// ignores unknown bits.
    pub flags: u16,

    /// Bake-time safety gate. `0` = failed (codec should refuse to
    /// load unless explicitly overridden). `1` = passed.
    pub passed: u8,
    /// Number of safety violations the trainer recorded. `0` when
    /// `passed = 1`. Above 0 is a strong signal even when `passed`
    /// might be flagged as 1 by `--allow-unsafe`.
    pub n_violations: u8,
    /// `0` = size_optimal, `1` = zensim_strict, future profiles get
    /// new codes. Codec advertises this to its caller verbatim.
    pub safety_profile: u8,
    /// Bumps when the trainer's safety-threshold table changes shape.
    pub threshold_set_version: u8,

    /// Stable hash of the `(corpus, sweep_grid, training_objective)`
    /// triple. Different bakes that share corpus + grid hash to the
    /// same value; useful for ops dashboards comparing across bakes.
    pub corpus_hash: u32,

    /// Bake-time held-out mean overhead, in percent. Surfaced in
    /// downstream encode-metrics logging by codec orchestrators.
    /// NaN if not measured.
    pub mean_overhead_pct: f32,
    /// Held-out p99 zensim shortfall vs target_zq, in points.
    /// Drives the `zensim_strict` advertise contract.
    pub p99_shortfall_pp: f32,
    /// Held-out argmin accuracy on the validation set (0..=1).
    pub argmin_acc: f32,

    /// Tail (8 bytes) — future field slot. Currently zero. Older
    /// runtimes read whatever's here as zero; new bake fields go in
    /// here first via a `version` bump.
    pub _reserved: [u8; 8],
}

const _: () = assert!(core::mem::size_of::<SafetyCompact>() == 32);

/// Per-cell rescue guidance. `4` bytes, indexed by output cell
/// number — the array length equals the bake's `n_cells`.
///
/// Codec consults when the picker returns cell `c` and a pass-0
/// verify undershoots. The `suggested_strategy` field nominates
/// which `RescueStrategy` matches THIS cell's failure mode based on
/// the trainer's per-cell post-mortem.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub struct CellHint {
    /// `0` = `ConservativeBump`, `1` = `SecondBestPick`, `2` =
    /// `KnownGoodFallback`. Picker's bake-time analysis of "what's
    /// most likely to recover when this cell undershoots."
    pub suggested_strategy: u8,
    /// Held-out p99 shortfall in zq points when this cell was
    /// chosen. Clamped to `0..=255`. Codec uses to decide whether
    /// the predicted shortfall is even worth rescuing — small p99
    /// values can be shipped without pass-1.
    pub p99_shortfall_pp: u8,
    /// Fraction of training-corpus picks for this cell that
    /// undershot target_zq, scaled to `0..=255` (= `0..=1.0`).
    /// High-rate cells are candidates for speculative rescue
    /// (run pass-1 in parallel from the start).
    pub expected_rescue_rate: u8,
    /// Bit 0 = dead cell (zero training rows; should never be
    /// selected). Bit 1 = degenerate (always picked → corpus
    /// skew warning). Bit 2 = high-variance (post-rescue
    /// behavior unpredictable; prefer `KnownGoodFallback`).
    pub flags: u8,
}

const _: () = assert!(core::mem::size_of::<CellHint>() == 4);

/// Known-good `(target_zq → fallback)` entry. The `KnownGoodFallback`
/// rescue strategy bypasses the picker and encodes the entry's cell
/// directly; no MLP forward pass on the rescue path.
///
/// Codec looks up the closest entry `<= target_zq`, encodes
/// `fallback_cell` with the listed quality bump.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub struct FallbackEntry {
    /// Target zq this entry is calibrated for, `0..=100`.
    pub zq_target: u8,
    /// Cell index to encode if this entry is selected.
    pub fallback_cell: u8,
    /// Signed override added to the codec's quality knob (e.g.
    /// jpegli-q + N points). Positive bumps quality; negative
    /// drops it (rare — used for time-budget rescue).
    pub quality_bump_pp: i8,
    /// Bit 0 = "verified by held-out reach >= 0.99 at this zq".
    /// Codec can refuse to use unverified entries in
    /// `Profile::Strict`.
    pub flags: u8,
}

const _: () = assert!(core::mem::size_of::<FallbackEntry>() == 4);

impl SafetyCompact {
    /// Convert the `safety_profile` byte to a typed enum. Returns
    /// `None` for unknown profile codes (forward-compat).
    pub fn profile(&self) -> Option<SafetyProfile> {
        match self.safety_profile {
            0 => Some(SafetyProfile::SizeOptimal),
            1 => Some(SafetyProfile::ZensimStrict),
            _ => None,
        }
    }

    /// `bit 0` of `flags`. Bake claims the strict-mode contract is
    /// honored on the held-out corpus.
    pub fn strict_certified(&self) -> bool {
        self.flags & 0x0001 != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SafetyProfile {
    SizeOptimal = 0,
    ZensimStrict = 1,
}

impl CellHint {
    /// Decode `suggested_strategy` to a typed enum. Falls back to
    /// the policy default (caller-supplied) if the byte is
    /// unrecognized.
    pub fn strategy(&self) -> Option<crate::RescueStrategy> {
        match self.suggested_strategy {
            0 => Some(crate::RescueStrategy::ConservativeBump),
            1 => Some(crate::RescueStrategy::SecondBestPick),
            2 => Some(crate::RescueStrategy::KnownGoodFallback),
            _ => None,
        }
    }

    pub fn is_dead(&self) -> bool {
        self.flags & 0x01 != 0
    }
    pub fn is_degenerate(&self) -> bool {
        self.flags & 0x02 != 0
    }
    pub fn is_high_variance(&self) -> bool {
        self.flags & 0x04 != 0
    }
}

impl FallbackEntry {
    pub fn is_verified(&self) -> bool {
        self.flags & 0x01 != 0
    }
}

/// Look up the closest fallback entry for a given target_zq.
/// Returns the entry whose `zq_target` is the largest value `≤ target_zq`,
/// or `None` if every entry's target exceeds the requested zq.
///
/// `table` MUST be sorted by `zq_target` ascending — bake-side
/// emits in that order. Linear scan; tables are small (≤ 50 entries).
pub fn fallback_for(target_zq: f32, table: &[FallbackEntry]) -> Option<&FallbackEntry> {
    let target = target_zq.clamp(0.0, 255.0) as u8;
    table
        .iter()
        .filter(|e| e.zq_target <= target)
        .max_by_key(|e| e.zq_target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::{cast_slice, pod_read_unaligned};

    #[test]
    fn safety_compact_round_trips_through_pod() {
        let s = SafetyCompact {
            version: 1,
            flags: 0x0001, // strict_certified
            passed: 1,
            n_violations: 0,
            safety_profile: 1,
            threshold_set_version: 3,
            corpus_hash: 0xdeadbeef,
            mean_overhead_pct: 2.33,
            p99_shortfall_pp: 1.5,
            argmin_acc: 0.85,
            _reserved: [0; 8],
        };
        let bytes = bytemuck::bytes_of(&s);
        assert_eq!(bytes.len(), 32);
        let recovered: SafetyCompact = pod_read_unaligned(bytes);
        assert_eq!(recovered, s);
        assert_eq!(recovered.profile(), Some(SafetyProfile::ZensimStrict));
        assert!(recovered.strict_certified());
    }

    #[test]
    fn cell_hints_round_trip_as_slice() {
        let hints = [
            CellHint {
                suggested_strategy: 0,
                p99_shortfall_pp: 3,
                expected_rescue_rate: 50,
                flags: 0,
            },
            CellHint {
                suggested_strategy: 2,
                p99_shortfall_pp: 12,
                expected_rescue_rate: 200,
                flags: 0x04, // high_variance
            },
        ];
        let bytes: &[u8] = cast_slice(&hints);
        assert_eq!(bytes.len(), 8);
        let recovered: &[CellHint] = cast_slice(bytes);
        assert_eq!(recovered, &hints);
        assert_eq!(
            recovered[0].strategy(),
            Some(crate::RescueStrategy::ConservativeBump)
        );
        assert_eq!(
            recovered[1].strategy(),
            Some(crate::RescueStrategy::KnownGoodFallback)
        );
        assert!(recovered[1].is_high_variance());
    }

    #[test]
    fn fallback_for_picks_closest_below_target() {
        let table = [
            FallbackEntry {
                zq_target: 50,
                fallback_cell: 1,
                quality_bump_pp: 0,
                flags: 1,
            },
            FallbackEntry {
                zq_target: 70,
                fallback_cell: 2,
                quality_bump_pp: 2,
                flags: 1,
            },
            FallbackEntry {
                zq_target: 85,
                fallback_cell: 3,
                quality_bump_pp: 5,
                flags: 1,
            },
            FallbackEntry {
                zq_target: 95,
                fallback_cell: 4,
                quality_bump_pp: 10,
                flags: 0,
            },
        ];
        // Exact match
        assert_eq!(fallback_for(85.0, &table).map(|e| e.fallback_cell), Some(3));
        // Between 70 and 85 → picks 70
        assert_eq!(fallback_for(80.0, &table).map(|e| e.fallback_cell), Some(2));
        // Above all → picks the highest (95)
        assert_eq!(fallback_for(99.0, &table).map(|e| e.fallback_cell), Some(4));
        // Below all → None
        assert_eq!(fallback_for(40.0, &table), None);
    }

    #[test]
    fn dead_cell_bit_works() {
        let h = CellHint {
            flags: 0x01,
            ..Default::default()
        };
        assert!(h.is_dead());
        assert!(!h.is_degenerate());
    }
}
