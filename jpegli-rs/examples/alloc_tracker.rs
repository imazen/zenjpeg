//! Memory usage validation: estimated vs actual vs locked expectations.
//!
//! This example validates that:
//! 1. Actual peak memory matches the estimate within tolerance
//! 2. Actual peak memory doesn't exceed locked regression thresholds
//!
//! Run with: cargo run --release --example alloc_tracker

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Tracking allocator that wraps System allocator
struct TrackingAllocator;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            let size = layout.size();
            let current = ALLOCATED.fetch_add(size, Ordering::SeqCst) + size;
            PEAK.fetch_max(current, Ordering::SeqCst);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        ALLOCATED.fetch_sub(layout.size(), Ordering::SeqCst);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = System.realloc(ptr, layout, new_size);
        if !new_ptr.is_null() {
            let old_size = layout.size();
            if new_size > old_size {
                let diff = new_size - old_size;
                let current = ALLOCATED.fetch_add(diff, Ordering::SeqCst) + diff;
                PEAK.fetch_max(current, Ordering::SeqCst);
            } else {
                ALLOCATED.fetch_sub(old_size - new_size, Ordering::SeqCst);
            }
        }
        new_ptr
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

fn reset_peak() {
    PEAK.store(ALLOCATED.load(Ordering::SeqCst), Ordering::SeqCst);
}

fn get_peak() -> usize {
    PEAK.load(Ordering::SeqCst)
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

/// Locked memory expectations (peak bytes).
/// These are regression thresholds - actual usage should not exceed these.
/// Update these values when intentionally changing memory behavior.
fn locked_expectation(width: usize, height: usize, subsampling: &str) -> Option<usize> {
    // Format: (width, height, subsampling) -> max allowed peak bytes
    // These are ~10% above measured values to allow for minor fluctuations
    match (width, height, subsampling) {
        (1920, 1080, "4:2:0") => Some(28 * 1024 * 1024), // ~25 MB measured
        (1920, 1080, "4:4:4") => Some(35 * 1024 * 1024), // ~32 MB measured
        (3840, 2160, "4:2:0") => Some(105 * 1024 * 1024), // ~95 MB measured
        (3840, 2160, "4:4:4") => Some(135 * 1024 * 1024), // ~122 MB measured
        (4000, 3000, "4:2:0") => Some(130 * 1024 * 1024), // ~118 MB measured
        (4000, 3000, "4:4:4") => Some(170 * 1024 * 1024), // ~155 MB measured
        _ => None,
    }
}

fn main() {
    use enough::Unstoppable;
    use jpegli::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

    println!("=== Memory Usage Validation ===\n");
    println!(
        "{:<20} {:<10} {:>12} {:>12} {:>12} {:>8} {:>8}",
        "Image", "Subsamp", "Estimated", "Actual", "Locked", "Est Δ", "Status"
    );
    println!("{}", "-".repeat(90));

    let test_cases = [(1920, 1080, "2K"), (3840, 2160, "4K"), (4000, 3000, "12MP")];

    let subsamplings = [
        (ChromaSubsampling::Quarter, "4:2:0"),
        (ChromaSubsampling::None, "4:4:4"),
    ];

    let mut all_passed = true;

    for (width, height, name) in &test_cases {
        for (subsampling, sub_name) in &subsamplings {
            // Create test image
            let input_size = width * height * 3;
            let mut rgb_data = vec![0u8; input_size];
            for y in 0..*height {
                for x in 0..*width {
                    let idx = (y * width + x) * 3;
                    rgb_data[idx] = (x * 255 / width) as u8;
                    rgb_data[idx + 1] = (y * 255 / height) as u8;
                    rgb_data[idx + 2] = 128;
                }
            }

            // Get estimate before encoding
            let config = EncoderConfig::ycbcr(85.0, *subsampling).optimize_huffman(true);

            let estimated = config.estimate_memory(*width as u32, *height as u32);

            // Measure actual usage
            reset_peak();

            let mut enc = config
                .encode_from_bytes(*width as u32, *height as u32, PixelLayout::Rgb8Srgb)
                .expect("encoder setup");
            enc.push_packed(&rgb_data, Unstoppable).expect("push");
            let _output = enc.finish().expect("encoding failed");

            let actual = get_peak();

            // Get locked expectation
            let locked = locked_expectation(*width, *height, sub_name);

            // Calculate estimate accuracy
            let est_diff_pct = ((actual as f64 - estimated as f64) / estimated as f64) * 100.0;

            // Check status
            let status = if let Some(max) = locked {
                if actual > max {
                    all_passed = false;
                    "REGRESS"
                } else if est_diff_pct.abs() > 20.0 {
                    "EST_OFF"
                } else {
                    "OK"
                }
            } else if est_diff_pct.abs() > 20.0 {
                "EST_OFF"
            } else {
                "OK"
            };

            println!(
                "{:<20} {:<10} {:>12} {:>12} {:>12} {:>+7.1}% {:>8}",
                name,
                sub_name,
                format_bytes(estimated),
                format_bytes(actual),
                locked.map(format_bytes).unwrap_or_else(|| "-".to_string()),
                est_diff_pct,
                status
            );

            // Clean up
            drop(_output);
            drop(rgb_data);
        }
    }

    println!();
    if all_passed {
        println!("✓ All memory checks passed");
    } else {
        println!("✗ Memory regression detected!");
        std::process::exit(1);
    }

    println!("\nLegend:");
    println!("  Estimated: EncoderConfig::estimate_memory() prediction");
    println!("  Actual:    Peak allocation measured via tracking allocator");
    println!("  Locked:    Regression threshold (actual must not exceed)");
    println!("  Est Δ:     (Actual - Estimated) / Estimated × 100%");
    println!("  Status:    OK | EST_OFF (>20% estimate error) | REGRESS (exceeds locked)");
}
