//! Memory benchmark comparing strip-based vs full-plane encoding.
//!
//! This example measures peak memory usage for both approaches.
//!
//! Run with: cargo run --release --example strip_memory_bench

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use jpegli::encode::strip::{StripProcessor, StripProcessorOutput};
use jpegli::quant::{generate_quant_table, Quality, ZeroBiasParams};
use jpegli::types::{ColorSpace, PixelFormat, Subsampling};
use jpegli::Encoder;

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
        }
        new_ptr
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

fn reset_stats() {
    ALLOCATED.store(0, Ordering::SeqCst);
    PEAK.store(0, Ordering::SeqCst);
    ALLOC_COUNT.store(0, Ordering::SeqCst);
}

fn get_stats() -> (usize, usize, usize) {
    (
        ALLOCATED.load(Ordering::SeqCst),
        PEAK.load(Ordering::SeqCst),
        ALLOC_COUNT.load(Ordering::SeqCst),
    )
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Encode using the standard full-plane approach
fn encode_standard(rgb_data: &[u8], width: usize, height: usize) -> (usize, std::time::Duration) {
    reset_stats();
    let start = Instant::now();

    let encoder = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .jpegli_quality(Quality::Traditional(85.0))
        .pixel_format(PixelFormat::Rgb)
        .subsampling(Subsampling::S420)
        .optimize_huffman(true);

    let output = encoder.encode(rgb_data).expect("encoding failed");
    let elapsed = start.elapsed();

    // Keep output alive until we measure
    let output_size = output.len();
    drop(output);

    let (_, peak, _) = get_stats();
    (peak, elapsed)
}

/// Encode using the strip-based approach (full JPEG output)
fn encode_strip_jpeg(
    rgb_data: &[u8],
    width: usize,
    height: usize,
) -> (usize, std::time::Duration, usize) {
    reset_stats();
    let start = Instant::now();

    let encoder = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .jpegli_quality(Quality::Traditional(85.0))
        .pixel_format(PixelFormat::Rgb)
        .subsampling(Subsampling::S420)
        .optimize_huffman(true);

    let output = encoder
        .encode_strip_based(rgb_data)
        .expect("strip encoding failed");
    let elapsed = start.elapsed();

    let output_size = output.len();
    drop(output);

    let (_, peak, _) = get_stats();
    (peak, elapsed, output_size)
}

/// Encode using the strip-based approach (blocks only, for comparison)
fn encode_strip_blocks(
    rgb_data: &[u8],
    width: usize,
    height: usize,
) -> (usize, std::time::Duration, StripProcessorOutput) {
    reset_stats();
    let start = Instant::now();

    let mut processor =
        StripProcessor::new(width, height, Subsampling::S420, PixelFormat::Rgb).unwrap();

    // Generate quant tables
    let is_420 = true;
    let quality = Quality::Traditional(85.0);
    let y_quant = generate_quant_table(quality, 0, ColorSpace::YCbCr, false, is_420);
    let cb_quant = generate_quant_table(quality, 1, ColorSpace::YCbCr, false, is_420);
    let cr_quant = generate_quant_table(quality, 2, ColorSpace::YCbCr, false, is_420);

    // Compute zero bias params
    let effective_distance = jpegli::quant::quant_vals_to_distance(&y_quant, &cb_quant, &cr_quant);
    let y_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 0);
    let cb_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 1);
    let cr_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 2);

    processor.set_quant_tables(
        y_quant,
        cb_quant,
        cr_quant,
        y_zero_bias,
        cb_zero_bias,
        cr_zero_bias,
    );

    // Process in strips
    let strip_height = processor.strip_height();
    for strip_y in (0..height).step_by(strip_height) {
        let strip_end = (strip_y + strip_height).min(height);
        let strip_start = strip_y * width * 3;
        let strip_end_idx = strip_end * width * 3;
        let rgb_strip = &rgb_data[strip_start..strip_end_idx];

        processor.process_strip(rgb_strip, strip_y).unwrap();
    }

    let output = processor.finalize();
    let elapsed = start.elapsed();

    let (_, peak, _) = get_stats();
    (peak, elapsed, output)
}

fn main() {
    println!("=== Strip-Based vs Standard Encoder Memory Benchmark ===\n");

    // Test cases: (width, height, name)
    let test_cases = [
        (1920, 1080, "2K (1920×1080)"),
        (3840, 2160, "4K (3840×2160)"),
        (4000, 3000, "12MP (4000×3000)"),
    ];

    println!("=== Full JPEG Output (encode_strip_based) ===\n");
    println!(
        "{:<25} {:<15} {:<15} {:<12} {:<12} {:<12}",
        "Image", "Standard Peak", "Strip Peak", "Reduction", "Speed", "Output"
    );
    println!("{}", "-".repeat(95));

    for (width, height, name) in &test_cases {
        // Create test image (gradient)
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

        // Benchmark standard encoder
        let (standard_peak, standard_time) = encode_standard(&rgb_data, *width, *height);

        // Benchmark strip encoder (full JPEG)
        let (strip_peak, strip_time, jpeg_size) = encode_strip_jpeg(&rgb_data, *width, *height);

        // Calculate reduction
        let reduction = if standard_peak > 0 {
            ((standard_peak as f64 - strip_peak as f64) / standard_peak as f64) * 100.0
        } else {
            0.0
        };

        // Speed comparison
        let speed_ratio = standard_time.as_secs_f64() / strip_time.as_secs_f64();

        println!(
            "{:<25} {:<15} {:<15} {:<12} {:<12} {:<12}",
            name,
            format_bytes(standard_peak),
            format_bytes(strip_peak),
            format!("{:.1}%", reduction),
            format!("{:.2}x", speed_ratio),
            format_bytes(jpeg_size)
        );
    }

    println!("\n=== Block Processing Only (StripProcessor) ===\n");
    println!(
        "{:<25} {:<15} {:<15} {:<12} {:<12}",
        "Image", "Standard Peak", "Strip Peak", "Reduction", "Speed"
    );
    println!("{}", "-".repeat(80));

    for (width, height, name) in &test_cases {
        // Create test image (gradient)
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

        // Benchmark standard encoder
        let (standard_peak, standard_time) = encode_standard(&rgb_data, *width, *height);

        // Benchmark strip encoder (blocks only)
        let (strip_peak, strip_time, output) = encode_strip_blocks(&rgb_data, *width, *height);

        // Calculate reduction
        let reduction = if standard_peak > 0 {
            ((standard_peak as f64 - strip_peak as f64) / standard_peak as f64) * 100.0
        } else {
            0.0
        };

        // Speed comparison
        let speed_ratio = standard_time.as_secs_f64() / strip_time.as_secs_f64();

        println!(
            "{:<25} {:<15} {:<15} {:<12} {:.2}x",
            name,
            format_bytes(standard_peak),
            format_bytes(strip_peak),
            format!("{:.1}%", reduction),
            speed_ratio
        );

        // Print block counts for verification
        if *width == 4000 && *height == 3000 {
            println!("\n  Block counts for 12MP:");
            println!(
                "    Y blocks:  {} (expected: {})",
                output.y_blocks.len(),
                ((width + 7) / 8) * ((height + 7) / 8)
            );
            println!(
                "    Cb blocks: {} (expected: {})",
                output.cb_blocks.len(),
                ((width + 15) / 16) * ((height + 15) / 16)
            );
            println!("    Cr blocks: {}", output.cr_blocks.len());
            println!();
        }
    }

    println!();
    println!("Note: Strip encoder only produces quantized blocks (no JPEG output yet).");
    println!("      Memory savings come from not materializing full f32 planes.");

    // Detailed breakdown for 12MP
    println!("\n=== 12MP Memory Breakdown ===\n");

    let width = 4000usize;
    let height = 3000usize;

    println!("Theoretical memory for full-plane approach:");
    let y_plane_size = width * height * 4; // f32
    let cb_full_size = width * height * 4;
    let cr_full_size = width * height * 4;
    let cb_down_size = (width / 2) * (height / 2) * 4;
    let cr_down_size = (width / 2) * (height / 2) * 4;
    let y_blocks_size = ((width + 7) / 8) * ((height + 7) / 8) * 64 * 2; // i16
    let cb_blocks_size = ((width / 2 + 7) / 8) * ((height / 2 + 7) / 8) * 64 * 2;
    let cr_blocks_size = cb_blocks_size;

    println!("  Y plane (f32):           {}", format_bytes(y_plane_size));
    println!("  Cb plane (f32):          {}", format_bytes(cb_full_size));
    println!("  Cr plane (f32):          {}", format_bytes(cr_full_size));
    println!("  Cb downsampled (f32):    {}", format_bytes(cb_down_size));
    println!("  Cr downsampled (f32):    {}", format_bytes(cr_down_size));
    println!("  Y blocks (i16):          {}", format_bytes(y_blocks_size));
    println!(
        "  Cb blocks (i16):         {}",
        format_bytes(cb_blocks_size)
    );
    println!(
        "  Cr blocks (i16):         {}",
        format_bytes(cr_blocks_size)
    );
    println!("  ---");
    let total_full = y_plane_size
        + cb_full_size
        + cr_full_size
        + cb_down_size
        + cr_down_size
        + y_blocks_size
        + cb_blocks_size
        + cr_blocks_size;
    println!("  Total (theoretical):     {}", format_bytes(total_full));

    println!("\nTheoretical memory for strip-based approach:");
    let strip_height = 16usize;
    let strip_size = width * strip_height * 4 * 3; // 3 planes
    let blocks_size = y_blocks_size + cb_blocks_size + cr_blocks_size;
    let aq_size = ((width + 7) / 8) * ((height + 7) / 8) * 4;

    println!("  Strip buffers (reused):  {}", format_bytes(strip_size));
    println!("  Block storage (i16):     {}", format_bytes(blocks_size));
    println!("  AQ state:                {}", format_bytes(aq_size));
    println!("  ---");
    let total_strip = strip_size + blocks_size + aq_size;
    println!("  Total (theoretical):     {}", format_bytes(total_strip));

    println!(
        "\nTheoretical reduction: {:.1}x ({:.1}%)",
        total_full as f64 / total_strip as f64,
        (1.0 - total_strip as f64 / total_full as f64) * 100.0
    );
}
