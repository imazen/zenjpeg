//! Iteration / time budget bookkeeping.

use std::time::Instant;

use crate::recompress::api::Budget;

/// Tracks remaining work allowance for a single `recompress` call.
#[derive(Debug)]
pub struct BudgetState {
    pub budget: Budget,
    pub started: Instant,
    pub iterations_used: u32,
}

impl BudgetState {
    pub fn new(budget: Budget) -> Self {
        Self {
            budget,
            started: Instant::now(),
            iterations_used: 0,
        }
    }

    /// True if at least one IQA-measurement-capable iteration is still
    /// allowed. `OneShot` is always false (we never measure).
    pub fn may_measure(&self) -> bool {
        match self.budget {
            Budget::OneShot => false,
            Budget::MaxIterations(n) => self.iterations_used < n,
            Budget::MaxTime(d) => self.started.elapsed() < d,
        }
    }

    pub fn note_iteration(&mut self) {
        self.iterations_used = self.iterations_used.saturating_add(1);
    }
}
