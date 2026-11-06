//! Debug-only instrumentation for boundary-RD refinement.
//!
//! When the `__bdrd-trace` feature is enabled, callers can install a
//! [`BoundaryRdTrace`] sink via [`install_trace`]; the BD-RD hot loop
//! will push one [`BdrdBlockEntry`] per luma block covering the default
//! candidate, the refined candidate (if any), and the coefficient delta
//! between them. The trace is collected in scan order and lives in a
//! thread-local to avoid threading through every encoder layer.
//!
//! This is for analysis only — not a shipping API. The feature is
//! gated out of default builds entirely.

#![cfg(feature = "__bdrd-trace")]

use std::cell::RefCell;

/// Per-block BD-RD instrumentation record.
#[derive(Debug, Clone, Copy)]
pub struct BdrdBlockEntry {
    /// Block column (0..blocks_w).
    pub bx: u16,
    /// Block row (0..blocks_h).
    pub by: u16,
    /// True iff the D_b threshold fired and at least one retry ran.
    pub triggered: bool,
    /// Number of retries actually executed. Always ≤ `max_retries`.
    pub retries_run: u8,
    /// D_b of the default quantize (before any refinement).
    pub db_default: f32,
    /// D_b of the finally-committed candidate.
    pub db_final: f32,
    /// Per-block AC DCT energy (the denominator of the trigger test).
    pub ac_energy: f32,
    /// AQ strength used for the finally-committed candidate.
    pub aq_final: f32,
    /// L1 norm of `(best_zigzag − zigzag_default)` in coefficient
    /// units. Zero means no refinement was picked up (either because
    /// it didn't trigger, or because none of the retries improved).
    pub coeff_delta_l1: u32,
}

/// Scan-order trace of BD-RD decisions for one encode.
#[derive(Debug, Default)]
pub struct BoundaryRdTrace {
    /// One entry per luma block, row-major (scan order).
    pub blocks: Vec<BdrdBlockEntry>,
}

impl BoundaryRdTrace {
    pub fn new() -> Self {
        Self { blocks: Vec::new() }
    }

    /// Derive `blocks_w` from `max(bx)+1` after collection.
    pub fn blocks_w(&self) -> u32 {
        self.blocks
            .iter()
            .map(|b| b.bx as u32)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0)
    }

    /// Derive `blocks_h` from `max(by)+1` after collection.
    pub fn blocks_h(&self) -> u32 {
        self.blocks
            .iter()
            .map(|b| b.by as u32)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0)
    }
}

thread_local! {
    static TRACE_SINK: RefCell<Option<BoundaryRdTrace>> = const { RefCell::new(None) };
}

/// Install an empty trace sink on the current thread, replacing any
/// prior sink. The next encode will populate it.
pub fn install_trace() {
    TRACE_SINK.with(|t| {
        *t.borrow_mut() = Some(BoundaryRdTrace::new());
    });
}

/// Take the trace sink off the current thread. Returns `None` if none
/// was installed. Subsequent encodes will not be traced until
/// [`install_trace`] is called again.
pub fn take_trace() -> Option<BoundaryRdTrace> {
    TRACE_SINK.with(|t| t.borrow_mut().take())
}

/// Append a block record to the active trace. No-op when no sink is
/// installed. Called from the BD-RD hot loop under cfg-gate.
#[inline]
pub(crate) fn push_block(entry: BdrdBlockEntry) {
    TRACE_SINK.with(|t| {
        if let Some(tr) = t.borrow_mut().as_mut() {
            tr.blocks.push(entry);
        }
    });
}
