//! Allocation tracker to measure peak memory usage during encoding.
//!
//! This example tracks all allocations through a custom global allocator
//! and reports peak memory usage for different image sizes.
//!
//! Run with: cargo run --release --example alloc_tracker

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Tracking allocator that wraps System allocator
struct TrackingAllocator;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static DEALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

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
        DEALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
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
    DEALLOC_COUNT.store(0, Ordering::SeqCst);
}

fn get_stats() -> (usize, usize, usize, usize) {
    (
        ALLOCATED.load(Ordering::SeqCst),
        PEAK.load(Ordering::SeqCst),
        ALLOC_COUNT.load(Ordering::SeqCst),
        DEALLOC_COUNT.load(Ordering::SeqCst),
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

fn main() {
    use enough::Never;
    use jpegli::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality};

    println!("=== jpegli-rs Allocation Tracking ===\n");

    // Test different image sizes
    let test_cases = [
        (1920, 1080, "2K (1920×1080)"),
        (3840, 2160, "4K (3840×2160)"),
        (4000, 3000, "12MP (4000×3000)"),
    ];

    let quality_levels = [75, 85, 95];
    let subsamplings = [(ChromaSubsampling::Quarter, "4:2:0"), (ChromaSubsampling::Full, "4:4:4")];

    println!(
        "{:<20} {:<10} {:<10} {:<15} {:<15} {:<12} {:<12}",
        "Image", "Quality", "Subsamp", "Input Size", "Peak Alloc", "Allocs", "Output"
    );
    println!("{}", "-".repeat(95));

    for (width, height, name) in &test_cases {
        for &quality in &quality_levels {
            for (subsampling, sub_name) in &subsamplings {
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

                // Reset stats before encoding
                reset_stats();

                // Create encoder using v2 API
                let config = EncoderConfig::new()
                    .quality(Quality::Traditional(quality as f32))
                    .ycbcr(*subsampling)
                    .optimize_huffman(true);

                let mut enc = config
                    .encode_from_bytes(*width as u32, *height as u32, PixelLayout::Rgb8Srgb)
                    .expect("encoder setup");
                enc.push_packed(&rgb_data, Never).expect("push");
                let output = enc.finish().expect("encoding failed");

                let (current, peak, alloc_count, _dealloc_count) = get_stats();

                println!(
                    "{:<20} {:<10} {:<10} {:<15} {:<15} {:<12} {:<12}",
                    name,
                    quality,
                    sub_name,
                    format_bytes(input_size),
                    format_bytes(peak),
                    alloc_count,
                    format_bytes(output.len())
                );

                // Give some time for deallocations
                drop(output);
                drop(rgb_data);

                // Small yield for memory tracking to settle
                std::thread::yield_now();
            }
        }
    }

    println!("\n=== Detailed Breakdown for 12MP @ q85 4:2:0 ===\n");

    // Detailed breakdown for one case
    let width = 4000usize;
    let height = 3000usize;
    let input_size = width * height * 3;

    // Create test image
    let mut rgb_data = vec![0u8; input_size];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            rgb_data[idx] = ((x + y) % 256) as u8;
            rgb_data[idx + 1] = ((x * 2 + y) % 256) as u8;
            rgb_data[idx + 2] = ((x + y * 2) % 256) as u8;
        }
    }

    reset_stats();

    let config = EncoderConfig::new()
        .quality(Quality::Traditional(85.0))
        .ycbcr(ChromaSubsampling::Quarter)
        .optimize_huffman(true);

    // Track pre-encode baseline
    let (pre_current, pre_peak, pre_allocs, _) = get_stats();
    println!("Pre-encode baseline:");
    println!("  Current: {}", format_bytes(pre_current));
    println!("  Peak:    {}", format_bytes(pre_peak));
    println!("  Allocs:  {}", pre_allocs);

    reset_stats();

    let mut enc = config
        .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
        .expect("encoder setup");
    enc.push_packed(&rgb_data, Never).expect("push");
    let output = enc.finish().expect("encoding failed");

    let (post_current, post_peak, post_allocs, post_deallocs) = get_stats();

    println!("\nDuring encode:");
    println!("  Peak:          {}", format_bytes(post_peak));
    println!("  Final current: {}", format_bytes(post_current));
    println!("  Allocations:   {}", post_allocs);
    println!("  Deallocations: {}", post_deallocs);
    println!("  Output size:   {}", format_bytes(output.len()));

    println!("\nExpected components (theoretical):");
    println!(
        "  RGB input (owned by caller):      {} (not counted)",
        format_bytes(input_size)
    );
    println!(
        "  Y plane (f32):                    {}",
        format_bytes(width * height * 4)
    );
    println!(
        "  Cb plane (f32):                   {}",
        format_bytes(width * height * 4)
    );
    println!(
        "  Cr plane (f32):                   {}",
        format_bytes(width * height * 4)
    );
    println!(
        "  Cb downsampled (f32):             {}",
        format_bytes((width / 2) * (height / 2) * 4)
    );
    println!(
        "  Cr downsampled (f32):             {}",
        format_bytes((width / 2) * (height / 2) * 4)
    );

    let blocks_y = ((width + 7) / 8) * ((height + 7) / 8);
    let blocks_c = ((width / 2 + 7) / 8) * (((height / 2) + 7) / 8);
    println!(
        "  Y blocks (i16×64):                {}",
        format_bytes(blocks_y * 64 * 2)
    );
    println!(
        "  Cb blocks (i16×64):               {}",
        format_bytes(blocks_c * 64 * 2)
    );
    println!(
        "  Cr blocks (i16×64):               {}",
        format_bytes(blocks_c * 64 * 2)
    );
    println!(
        "  AQ map:                           {}",
        format_bytes(blocks_y * 4)
    );

    let theoretical_total = width * height * 4 * 3 +  // YCbCr f32 planes
        (width/2) * (height/2) * 4 * 2 +  // downsampled chroma
        blocks_y * 64 * 2 + blocks_c * 64 * 2 * 2 +  // blocks
        blocks_y * 4; // AQ map

    println!(
        "\n  Theoretical total:                {}",
        format_bytes(theoretical_total)
    );
    println!(
        "  Actual peak:                      {}",
        format_bytes(post_peak)
    );
    println!(
        "  Ratio:                            {:.2}x theoretical",
        post_peak as f64 / theoretical_total as f64
    );
}
