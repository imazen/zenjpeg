//! Memory usage matrix: estimated vs actual for encoder configurations.
//!
//! Tests a matrix of encoder configurations and compares estimated peak memory
//! with actual peak memory measured via a global allocator tracker.
//!
//! Run with: cargo run --release --example memory_matrix
//!
//! Use --detailed for per-allocation breakdown (slower):
//!   cargo run --release --example memory_matrix -- --detailed

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Tracking allocator that wraps System allocator
struct TrackingAllocator;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            let size = layout.size();
            let current = ALLOCATED.fetch_add(size, Ordering::SeqCst) + size;
            PEAK.fetch_max(current, Ordering::SeqCst);
            ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
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
            ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        }
        new_ptr
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

fn reset_stats() {
    let current = ALLOCATED.load(Ordering::SeqCst);
    PEAK.store(current, Ordering::SeqCst);
    ALLOC_COUNT.store(0, Ordering::SeqCst);
}

fn get_peak() -> usize {
    PEAK.load(Ordering::SeqCst)
}

fn get_alloc_count() -> usize {
    ALLOC_COUNT.load(Ordering::SeqCst)
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

/// A single encoder configuration to test
#[derive(Debug, Clone)]
struct EncoderTest {
    name: &'static str,
    width: u32,
    height: u32,
    subsampling: jpegli::encoder::ChromaSubsampling,
    optimize_huffman: bool,
    xyb_mode: bool,
    quality: f32,
}

/// Result of running an encoder test
struct TestResult {
    test: EncoderTest,
    estimated: usize,
    ceiling: usize,
    actual_peak: usize,
    alloc_count: usize,
    output_size: usize,
    error_pct: f64,
    ceiling_headroom_pct: f64,
    ceiling_violated: bool,
}

impl TestResult {
    fn status(&self) -> &'static str {
        if self.ceiling_violated {
            "CEIL!"
        } else if self.error_pct.abs() <= 10.0 {
            "OK"
        } else if self.error_pct.abs() <= 20.0 {
            "WARN"
        } else if self.error_pct.abs() <= 50.0 {
            "HIGH"
        } else {
            "BAD"
        }
    }
}

fn run_test(test: EncoderTest) -> TestResult {
    use enough::Unstoppable;
    use jpegli::encoder::{ColorMode, EncoderConfig, PixelLayout, XybSubsampling};

    // Create test image (gradient)
    let input_size = test.width as usize * test.height as usize * 3;
    let mut rgb_data = vec![0u8; input_size];
    for y in 0..test.height as usize {
        for x in 0..test.width as usize {
            let idx = (y * test.width as usize + x) * 3;
            rgb_data[idx] = (x * 255 / test.width as usize) as u8;
            rgb_data[idx + 1] = (y * 255 / test.height as usize) as u8;
            rgb_data[idx + 2] = 128;
        }
    }

    // Build config
    let mut config = EncoderConfig::new()
        .quality(test.quality)
        .optimize_huffman(test.optimize_huffman);

    if test.xyb_mode {
        config = config.color_mode(ColorMode::Xyb { subsampling: XybSubsampling::Full });
    } else {
        config = config.ycbcr(test.subsampling);
    }

    // Get estimate and ceiling
    let estimated = config.estimate_memory(test.width, test.height);
    let ceiling = config.estimate_memory_ceiling(test.width, test.height);

    // Reset peak tracking
    reset_stats();

    // Encode
    let mut enc = config
        .encode_from_bytes(test.width, test.height, PixelLayout::Rgb8Srgb)
        .expect("encoder setup");
    enc.push_packed(&rgb_data, Unstoppable).expect("push");
    let output = enc.finish().expect("encoding failed");

    let actual_peak = get_peak();
    let alloc_count = get_alloc_count();
    let output_size = output.len();

    // Clean up before calculating error
    drop(output);
    drop(rgb_data);

    let error_pct = ((actual_peak as f64 - estimated as f64) / estimated as f64) * 100.0;
    let ceiling_headroom_pct = ((ceiling as f64 - actual_peak as f64) / actual_peak as f64) * 100.0;
    let ceiling_violated = actual_peak > ceiling;

    TestResult {
        test,
        estimated,
        ceiling,
        actual_peak,
        alloc_count,
        output_size,
        error_pct,
        ceiling_headroom_pct,
        ceiling_violated,
    }
}

fn main() {
    use jpegli::encoder::ChromaSubsampling;

    let detailed = std::env::args().any(|a| a == "--detailed");

    println!("=== Memory Usage Matrix ===\n");

    // Define test matrix
    let dimensions = [
        (640, 480, "VGA"),
        (1280, 720, "HD"),
        (1920, 1080, "FHD"),
        (2560, 1440, "QHD"),
        (3840, 2160, "4K"),
        (4000, 3000, "12MP"),
    ];

    let subsamplings = [
        (ChromaSubsampling::Quarter, "4:2:0"),
        (ChromaSubsampling::HalfHorizontal, "4:2:2"),
        (ChromaSubsampling::Full, "4:4:4"),
    ];

    let mut tests: Vec<EncoderTest> = Vec::new();

    // YCbCr configurations
    for (w, h, name) in &dimensions {
        for (sub, _sub_name) in &subsamplings {
            // Without Huffman optimization
            tests.push(EncoderTest {
                name,
                width: *w,
                height: *h,
                subsampling: *sub,
                optimize_huffman: false,
                xyb_mode: false,
                quality: 85.0,
            });
            // With Huffman optimization
            tests.push(EncoderTest {
                name,
                width: *w,
                height: *h,
                subsampling: *sub,
                optimize_huffman: true,
                xyb_mode: false,
                quality: 85.0,
            });
        }
    }

    // XYB configurations (uses 4:4:4 internally)
    for (w, h, name) in &dimensions {
        tests.push(EncoderTest {
            name,
            width: *w,
            height: *h,
            subsampling: ChromaSubsampling::Full,
            optimize_huffman: true,
            xyb_mode: true,
            quality: 85.0,
        });
    }

    // Run tests
    let mut results: Vec<TestResult> = Vec::new();
    let total = tests.len();

    if detailed {
        println!("Running {} tests with detailed tracking...\n", total);
    } else {
        println!("Running {} tests...\n", total);
    }

    for (i, test) in tests.into_iter().enumerate() {
        if !detailed {
            eprint!("\rProgress: {}/{}", i + 1, total);
        }
        let result = run_test(test);
        results.push(result);
    }
    if !detailed {
        eprintln!();
    }

    // Print results table
    println!();
    println!(
        "{:<8} {:<7} {:>12} {:>12} {:>12} {:>7} {:>8} {:>6}",
        "Size", "Samp", "Estimated", "Actual", "Ceiling", "Est%", "Headrm", "Status"
    );
    println!("{}", "-".repeat(85));

    for r in &results {
        let subsamp = match r.test.subsampling {
            ChromaSubsampling::Quarter => "4:2:0",
            ChromaSubsampling::HalfHorizontal => "4:2:2",
            ChromaSubsampling::HalfVertical => "4:4:0",
            ChromaSubsampling::Full => "4:4:4",
            _ => "???",
        };

        println!(
            "{:<8} {:<7} {:>12} {:>12} {:>12} {:>+6.1}% {:>+6.0}% {:>6}",
            r.test.name,
            if r.test.xyb_mode { "XYB" } else { subsamp },
            format_bytes(r.estimated),
            format_bytes(r.actual_peak),
            format_bytes(r.ceiling),
            r.error_pct,
            r.ceiling_headroom_pct,
            r.status()
        );
    }

    // Summary statistics
    println!("\n=== Summary Statistics ===\n");

    let errors: Vec<f64> = results.iter().map(|r| r.error_pct).collect();
    let min_err = errors.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_err = errors.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let avg_err = errors.iter().sum::<f64>() / errors.len() as f64;

    let abs_errors: Vec<f64> = errors.iter().map(|e| e.abs()).collect();
    let avg_abs_err = abs_errors.iter().sum::<f64>() / abs_errors.len() as f64;

    println!("Error range: {:.1}% to {:.1}%", min_err, max_err);
    println!("Average error: {:+.1}%", avg_err);
    println!("Average absolute error: {:.1}%", avg_abs_err);

    // Count by status
    let ok_count = results.iter().filter(|r| r.status() == "OK").count();
    let warn_count = results.iter().filter(|r| r.status() == "WARN").count();
    let high_count = results.iter().filter(|r| r.status() == "HIGH").count();
    let bad_count = results.iter().filter(|r| r.status() == "BAD").count();
    let ceil_violations = results.iter().filter(|r| r.ceiling_violated).count();

    println!("\nEstimate accuracy breakdown:");
    println!("  OK (≤10%): {} tests", ok_count);
    println!("  WARN (10-20%): {} tests", warn_count);
    println!("  HIGH (20-50%): {} tests", high_count);
    println!("  BAD (>50%): {} tests", bad_count);

    // Ceiling analysis
    println!("\n=== Ceiling Analysis ===\n");

    let headrooms: Vec<f64> = results.iter().map(|r| r.ceiling_headroom_pct).collect();
    let min_headroom = headrooms.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_headroom = headrooms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let avg_headroom = headrooms.iter().sum::<f64>() / headrooms.len() as f64;

    println!("Ceiling violations: {}/{} tests", ceil_violations, results.len());
    if ceil_violations == 0 {
        println!("  ✓ All tests passed - actual never exceeded ceiling");
    } else {
        println!("  ✗ CEILING VIOLATED - actual exceeded ceiling!");
        for r in results.iter().filter(|r| r.ceiling_violated) {
            println!("    {} {:?}: actual {} > ceiling {}",
                r.test.name,
                r.test.subsampling,
                format_bytes(r.actual_peak),
                format_bytes(r.ceiling)
            );
        }
    }

    println!("\nHeadroom (ceiling - actual) / actual:");
    println!("  Min: {:.1}%", min_headroom);
    println!("  Max: {:.1}%", max_headroom);
    println!("  Avg: {:.1}%", avg_headroom);

    // Analyze patterns
    println!("\n=== Error Patterns ===\n");

    // By subsampling
    let mut by_samp: std::collections::HashMap<&str, Vec<f64>> = std::collections::HashMap::new();
    for r in &results {
        if r.test.xyb_mode {
            by_samp.entry("XYB").or_default().push(r.error_pct);
        } else {
            let s = match r.test.subsampling {
                ChromaSubsampling::Quarter => "4:2:0",
                ChromaSubsampling::HalfHorizontal => "4:2:2",
                ChromaSubsampling::HalfVertical => "4:4:0",
                ChromaSubsampling::Full => "4:4:4",
                _ => "???",
            };
            by_samp.entry(s).or_default().push(r.error_pct);
        }
    }
    println!("By color mode/subsampling:");
    for (s, errs) in &by_samp {
        let avg = errs.iter().sum::<f64>() / errs.len() as f64;
        println!("  {}: {:+.1}% avg error ({} tests)", s, avg, errs.len());
    }

    // By Huffman optimization
    let huff_yes: Vec<f64> = results.iter().filter(|r| r.test.optimize_huffman).map(|r| r.error_pct).collect();
    let huff_no: Vec<f64> = results.iter().filter(|r| !r.test.optimize_huffman).map(|r| r.error_pct).collect();
    println!("\nBy Huffman optimization:");
    if !huff_no.is_empty() {
        println!("  Without: {:+.1}% avg error", huff_no.iter().sum::<f64>() / huff_no.len() as f64);
    }
    if !huff_yes.is_empty() {
        println!("  With: {:+.1}% avg error", huff_yes.iter().sum::<f64>() / huff_yes.len() as f64);
    }

    // Find worst cases
    println!("\n=== Worst Cases (>20% error) ===\n");
    let mut worst: Vec<_> = results.iter().filter(|r| r.error_pct.abs() > 20.0).collect();
    worst.sort_by(|a, b| b.error_pct.abs().partial_cmp(&a.error_pct.abs()).unwrap());

    if worst.is_empty() {
        println!("None! All estimates within 20%.");
    } else {
        for r in worst.iter().take(10) {
            let subsamp = if r.test.xyb_mode {
                "XYB"
            } else {
                match r.test.subsampling {
                    ChromaSubsampling::Quarter => "4:2:0",
                    ChromaSubsampling::HalfHorizontal => "4:2:2",
                    ChromaSubsampling::HalfVertical => "4:4:0",
                    ChromaSubsampling::Full => "4:4:4",
                    _ => "???",
                }
            };
            println!(
                "  {}x{} {} huff={}: {:+.1}% (est {} vs actual {})",
                r.test.width,
                r.test.height,
                subsamp,
                if r.test.optimize_huffman { "Y" } else { "N" },
                r.error_pct,
                format_bytes(r.estimated),
                format_bytes(r.actual_peak)
            );
        }
    }

    // Memory efficiency analysis
    println!("\n=== Memory Efficiency Analysis ===\n");
    println!("Bytes per megapixel (actual peak / megapixels):");

    let mut by_mode: std::collections::HashMap<&str, Vec<(f64, f64)>> = std::collections::HashMap::new();
    for r in &results {
        let mp = (r.test.width as f64 * r.test.height as f64) / 1_000_000.0;
        let bytes_per_mp = r.actual_peak as f64 / mp;
        let key = if r.test.xyb_mode {
            "XYB"
        } else {
            match r.test.subsampling {
                ChromaSubsampling::Quarter => "4:2:0",
                ChromaSubsampling::HalfHorizontal => "4:2:2",
                ChromaSubsampling::HalfVertical => "4:4:0",
                ChromaSubsampling::Full => "4:4:4",
                _ => "???",
            }
        };
        by_mode.entry(key).or_default().push((mp, bytes_per_mp));
    }

    for (mode, data) in &by_mode {
        let avg_bytes_per_mp = data.iter().map(|(_, b)| b).sum::<f64>() / data.len() as f64;
        println!(
            "  {}: {:.2} MB/megapixel average",
            mode,
            avg_bytes_per_mp / (1024.0 * 1024.0)
        );
    }

    println!("\n=== Recommendations for Estimation Algorithm ===\n");

    // Calculate correction factors
    if avg_err > 5.0 {
        println!("Estimates are too LOW by {:.1}% on average.", avg_err);
        println!("Consider adding a {:.1}% safety margin.", avg_err.abs());
    } else if avg_err < -5.0 {
        println!("Estimates are too HIGH by {:.1}% on average.", avg_err.abs());
        println!("Consider reducing estimates by {:.1}%.", avg_err.abs());
    } else {
        println!("Estimates are reasonably accurate (within 5% average).");
    }

    // Check for systematic bias by mode
    for (mode, errs) in &by_samp {
        let avg = errs.iter().sum::<f64>() / errs.len() as f64;
        if avg.abs() > 10.0 {
            if avg > 0.0 {
                println!("\n{} mode: estimates are {:.1}% too low. Check:", mode, avg);
            } else {
                println!("\n{} mode: estimates are {:.1}% too high. Check:", mode, avg.abs());
            }
            println!("  - Strip buffer calculations");
            println!("  - DCT block count formulas");
            println!("  - Output buffer growth");
        }
    }
}
