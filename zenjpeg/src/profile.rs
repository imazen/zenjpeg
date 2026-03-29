//! Lightweight text-based profiling for encoder phases.
//!
//! Enable with `profile` feature. Zero-cost when disabled.
//!
//! Usage:
//! ```ignore
//! profile_scope!("dct");
//! // ... do DCT work ...
//! // automatically records time when scope ends
//!
//! // At end of encode:
//! ProfileStats::print_summary();
//! ```

#[cfg(feature = "__profile")]
use std::cell::RefCell;
#[cfg(feature = "__profile")]
use std::collections::HashMap;
#[cfg(feature = "__profile")]
use std::time::{Duration, Instant};

// Stub Duration for non-profile builds (only used in no-op functions)
#[cfg(not(feature = "__profile"))]
use core::time::Duration;

#[cfg(feature = "__profile")]
thread_local! {
    static STATS: RefCell<ProfileStats> = RefCell::new(ProfileStats::new());
    static STACK: RefCell<Vec<(&'static str, Instant)>> = const { RefCell::new(Vec::new()) };
}

#[cfg(feature = "__profile")]
#[derive(Default)]
pub struct ProfileStats {
    timings: HashMap<&'static str, Timing>,
    call_order: Vec<&'static str>,
}

#[cfg(not(feature = "__profile"))]
#[derive(Default)]
pub struct ProfileStats;

#[cfg(feature = "__profile")]
#[derive(Default, Clone)]
struct Timing {
    total: Duration,
    count: u64,
    min: Duration,
    max: Duration,
}

impl ProfileStats {
    #[cfg(feature = "__profile")]
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(not(feature = "__profile"))]
    pub fn new() -> Self {
        Self
    }

    #[cfg(feature = "__profile")]
    pub fn record(name: &'static str, elapsed: Duration) {
        STATS.with(|stats| {
            let mut stats = stats.borrow_mut();
            // Check if new entry, track call order separately to avoid borrow conflict
            let is_new = !stats.timings.contains_key(name);
            if is_new {
                stats.call_order.push(name);
            }
            let timing = stats.timings.entry(name).or_insert_with(|| Timing {
                min: Duration::MAX,
                ..Default::default()
            });
            timing.total += elapsed;
            timing.count += 1;
            timing.min = timing.min.min(elapsed);
            timing.max = timing.max.max(elapsed);
        });
    }

    #[cfg(feature = "__profile")]
    pub fn print_summary() {
        STATS.with(|stats| {
            let stats = stats.borrow();
            let total: Duration = stats.timings.values().map(|t| t.total).sum();

            eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
            eprintln!("║                    PROFILE SUMMARY                           ║");
            eprintln!("╠══════════════════════════════════════════════════════════════╣");
            eprintln!(
                "║ {:30} {:>8} {:>8} {:>6} {:>6} ║",
                "Phase", "Total", "Avg", "Calls", "%"
            );
            eprintln!("╠══════════════════════════════════════════════════════════════╣");

            for name in &stats.call_order {
                if let Some(timing) = stats.timings.get(name) {
                    let pct = if total.as_nanos() > 0 {
                        (timing.total.as_nanos() as f64 / total.as_nanos() as f64) * 100.0
                    } else {
                        0.0
                    };
                    let avg = if timing.count > 0 {
                        timing.total / timing.count as u32
                    } else {
                        Duration::ZERO
                    };

                    eprintln!(
                        "║ {:30} {:>8.2?} {:>8.2?} {:>6} {:>5.1}% ║",
                        name, timing.total, avg, timing.count, pct
                    );
                }
            }

            eprintln!("╠══════════════════════════════════════════════════════════════╣");
            eprintln!(
                "║ {:30} {:>8.2?} {:>8} {:>6} {:>6} ║",
                "TOTAL", total, "", "", "100%"
            );
            eprintln!("╚══════════════════════════════════════════════════════════════╝\n");
        });
    }

    #[cfg(feature = "__profile")]
    pub fn reset() {
        STATS.with(|stats| {
            *stats.borrow_mut() = ProfileStats::new();
        });
    }

    #[cfg(not(feature = "__profile"))]
    #[inline(always)]
    pub fn record(_name: &'static str, _elapsed: Duration) {}

    #[cfg(not(feature = "__profile"))]
    #[inline(always)]
    pub fn print_summary() {}

    #[cfg(not(feature = "__profile"))]
    #[inline(always)]
    pub fn reset() {}
}

/// RAII guard for timing a scope
#[cfg(feature = "__profile")]
pub struct ProfileGuard {
    name: &'static str,
    start: Instant,
}

#[cfg(not(feature = "__profile"))]
pub struct ProfileGuard;

impl ProfileGuard {
    #[cfg(feature = "__profile")]
    #[inline]
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            start: Instant::now(),
        }
    }

    #[cfg(not(feature = "__profile"))]
    #[inline(always)]
    pub fn new(_name: &'static str) -> Self {
        Self
    }
}

#[cfg(feature = "__profile")]
impl Drop for ProfileGuard {
    fn drop(&mut self) {
        ProfileStats::record(self.name, self.start.elapsed());
    }
}

#[cfg(not(feature = "__profile"))]
impl Drop for ProfileGuard {
    #[inline(always)]
    fn drop(&mut self) {}
}

/// Profile a scope. Zero-cost when `profile` feature is disabled.
#[macro_export]
macro_rules! profile_scope {
    ($name:expr) => {
        let _guard = $crate::profile::ProfileGuard::new($name);
    };
}

/// Profile a block and return its value.
#[macro_export]
macro_rules! profile_block {
    ($name:expr, $block:expr) => {{
        let _guard = $crate::profile::ProfileGuard::new($name);
        $block
    }};
}
