//! Compare allocation patterns between full-plane and strip-based encoding.
//!
//! Key finding: Strip encoder provides significant peak memory savings:
//! 1. Incremental quantization: f32→i16 as soon as AQ strengths available
//! 2. Double-buffered pending blocks: only 2 iMCU rows of f32, not full image
//! 3. Strip buffers are small (16 rows) and reused
//!
//! Benefits: both cache locality AND reduced peak memory.

use jpegli::encode::strip::StripProcessor;
use jpegli::quant::{generate_quant_table, quant_vals_to_distance, ZeroBiasParams};
use jpegli::{ColorSpace, PixelFormat, Quality, Subsampling};

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

/// Calculate what each encoder actually allocates
fn calc_allocations(width: usize, height: usize, subsampling: Subsampling) -> (usize, usize) {
    let pixels = width * height;

    // Block counts
    let y_blocks_w = (width + 7) / 8;
    let y_blocks_h = (height + 7) / 8;
    let y_block_count = y_blocks_w * y_blocks_h;

    let c_block_count = match subsampling {
        Subsampling::S420 => ((width + 15) / 16) * ((height + 15) / 16),
        Subsampling::S422 => ((width + 15) / 16) * y_blocks_h,
        Subsampling::S440 => y_blocks_w * ((height + 15) / 16),
        Subsampling::S444 => y_block_count,
        _ => y_block_count,
    };

    // Chroma plane sizes
    let c_pixels = match subsampling {
        Subsampling::S420 => ((width + 1) / 2) * ((height + 1) / 2),
        Subsampling::S422 => ((width + 1) / 2) * height,
        Subsampling::S440 => width * ((height + 1) / 2),
        Subsampling::S444 => pixels,
        _ => pixels,
    };

    // =================
    // FULL-PLANE ENCODER
    // =================
    // 3 f32 planes (Y, Cb, Cr)
    let full_y_plane = pixels * 4;
    let full_cb_plane = c_pixels * 4;
    let full_cr_plane = c_pixels * 4;
    // i16 DCT blocks (after quantization)
    let full_y_blocks_i16 = y_block_count * 128; // [i16; 64] = 128 bytes
    let full_c_blocks_i16 = c_block_count * 2 * 128;
    // AQ map
    let full_aq = y_block_count * 4;

    let full_peak = full_y_plane
        + full_cb_plane
        + full_cr_plane
        + full_y_blocks_i16
        + full_c_blocks_i16
        + full_aq;

    // =================
    // STRIP ENCODER (with incremental quantization)
    // =================
    let strip_height = 16usize;
    let c_strip_height = match subsampling {
        Subsampling::S420 | Subsampling::S440 => 8,
        _ => strip_height,
    };
    let c_width = match subsampling {
        Subsampling::S420 | Subsampling::S422 => (width + 1) / 2,
        _ => width,
    };

    // Strip f32 buffers (small, 16 rows, reused each strip)
    // Note: cb_strip/cr_strip are FULL resolution before downsampling
    let strip_y = width * strip_height * 4;
    let strip_cb = width * strip_height * 4; // Full res, not c_width
    let strip_cr = width * strip_height * 4; // Full res, not c_width
                                             // Downsampled chroma temp buffers
    let strip_cb_down = c_width * c_strip_height * 4;
    let strip_cr_down = c_width * c_strip_height * 4;

    // Pending f32 DCT blocks - ONLY 2 iMCU rows (double-buffered)
    // For 4:2:0: one iMCU = 4 Y blocks + 1 Cb + 1 Cr = 6 blocks per column
    let y_blocks_per_imcu_row = y_blocks_w * 2; // 2 rows of 8x8 Y blocks
    let c_blocks_per_imcu_row = match subsampling {
        Subsampling::S420 => (width + 15) / 16, // 1 Cb + 1 Cr per 16x16
        Subsampling::S422 => (width + 15) / 16 * 2, // 2 rows of Cb/Cr
        Subsampling::S440 => y_blocks_w,        // 1 row of Cb/Cr
        _ => y_blocks_w * 2,
    };
    // Double-buffered: 2 buffers × blocks_per_row × 256 bytes (f32)
    let pending_y_f32 = 2 * y_blocks_per_imcu_row * 256;
    let pending_cb_f32 = 2 * c_blocks_per_imcu_row * 256;
    let pending_cr_f32 = 2 * c_blocks_per_imcu_row * 256;

    // Final i16 blocks (grows incrementally as blocks are quantized)
    let strip_y_blocks_i16 = y_block_count * 128;
    let strip_c_blocks_i16 = c_block_count * 2 * 128;

    // AQ strengths (one per Y block)
    let strip_aq = y_block_count * 4;

    // Peak memory: strip buffers + pending f32 (2 iMCU rows) + growing i16 storage
    // The i16 storage grows as we process, but pending f32 stays constant
    let strip_peak = strip_y
        + strip_cb
        + strip_cr
        + strip_cb_down
        + strip_cr_down
        + pending_y_f32
        + pending_cb_f32
        + pending_cr_f32
        + strip_y_blocks_i16
        + strip_c_blocks_i16
        + strip_aq;

    (full_peak, strip_peak)
}

/// Result of measuring strip allocations
struct StripMeasurement {
    peak_bytes: usize,
    by_context: Vec<(&'static str, usize)>,
}

fn measure_strip_allocs(
    width: usize,
    height: usize,
    subsampling: Subsampling,
    data: &[u8],
) -> Result<StripMeasurement, jpegli::Error> {
    let mut processor = StripProcessor::new(width, height, subsampling, PixelFormat::Rgb)?;

    let quality = Quality::from_quality(85.0);
    let is_420 = subsampling == Subsampling::S420;
    let y_quant = generate_quant_table(quality, 0, ColorSpace::YCbCr, false, is_420);
    let cb_quant = generate_quant_table(quality, 1, ColorSpace::YCbCr, false, is_420);
    let cr_quant = generate_quant_table(quality, 2, ColorSpace::YCbCr, false, is_420);

    let effective_distance = quant_vals_to_distance(&y_quant, &cb_quant, &cr_quant);
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
    )?;

    let strip_height = processor.strip_height();
    let bpp = 3;
    for strip_y in (0..height).step_by(strip_height) {
        let strip_end = (strip_y + strip_height).min(height);
        let strip_start = strip_y * width * bpp;
        let strip_end_idx = strip_end * width * bpp;
        processor.process_strip(&data[strip_start..strip_end_idx], strip_y)?;
    }

    let output = processor.finalize()?;
    Ok(StripMeasurement {
        peak_bytes: output.alloc_stats.peak_bytes,
        by_context: output.alloc_stats.by_context.clone(),
    })
}

fn main() {
    println!("Peak Allocation Comparison: Full-Plane vs Strip-Based Encoding");
    println!("================================================================\n");

    let subsampling = Subsampling::S420;

    // Standard resolutions
    println!("=== Standard Resolutions (4:2:0) ===\n");
    println!(
        "{:<20} {:>12} {:>12} {:>10}",
        "Resolution", "Estimated", "Measured", "Delta"
    );
    println!("{:-<58}", "");

    let resolutions = [
        ("1K (1920x1080)", 1920usize, 1080usize),
        ("2K (2560x1440)", 2560, 1440),
        ("4K (3840x2160)", 3840, 2160),
        ("8K (7680x4320)", 7680, 4320),
    ];

    for (name, width, height) in resolutions {
        let (_, strip_peak_calc) = calc_allocations(width, height, subsampling);

        let pixels = width * height;
        let data: Vec<u8> = (0..pixels * 3).map(|i| ((i * 17) % 256) as u8).collect();

        match measure_strip_allocs(width, height, subsampling, &data) {
            Ok(m) => {
                let delta = m.peak_bytes as i64 - strip_peak_calc as i64;
                let delta_pct = (delta as f64 / strip_peak_calc as f64) * 100.0;
                println!(
                    "{:<20} {:>12} {:>12} {:>+9.1}%",
                    name,
                    format_bytes(strip_peak_calc),
                    format_bytes(m.peak_bytes),
                    delta_pct
                );
            }
            Err(e) => println!("{:<20} err: {}", name, e),
        }
    }

    // Extreme aspect ratios
    println!("\n\n=== Extreme Aspect Ratios (4:2:0) ===\n");
    println!(
        "{:<20} {:>12} {:>12} {:>10}",
        "Resolution", "Estimated", "Measured", "Delta"
    );
    println!("{:-<58}", "");

    let extreme = [
        ("8000x1 (line)", 8000usize, 1usize),
        ("1x8000 (column)", 1, 8000),
        ("8000x16 (banner)", 8000, 16),
        ("16x8000 (tall)", 16, 8000),
        ("65000x1", 65000, 1),
        ("1x65000", 1, 65000),
        ("256x256", 256, 256),
        ("65500x1", 65500, 1), // Near max JPEG dimension
    ];

    for (name, width, height) in extreme {
        let (_, strip_peak_calc) = calc_allocations(width, height, subsampling);

        let pixels = width * height;
        let data: Vec<u8> = (0..pixels * 3).map(|i| ((i * 17) % 256) as u8).collect();

        match measure_strip_allocs(width, height, subsampling, &data) {
            Ok(m) => {
                let delta = m.peak_bytes as i64 - strip_peak_calc as i64;
                let delta_pct = if strip_peak_calc > 0 {
                    (delta as f64 / strip_peak_calc as f64) * 100.0
                } else {
                    0.0
                };
                println!(
                    "{:<20} {:>12} {:>12} {:>+9.1}%",
                    name,
                    format_bytes(strip_peak_calc),
                    format_bytes(m.peak_bytes),
                    delta_pct
                );
            }
            Err(e) => println!(
                "{:<20} {:>12} err: {}",
                name,
                format_bytes(strip_peak_calc),
                e
            ),
        }
    }

    // Show detailed breakdown for standard and extreme cases
    for (label, width, height) in [
        ("4K (3840x2160)", 3840usize, 2160usize),
        ("Wide (8000x1)", 8000, 1),
    ] {
        println!("\n\n=== Detailed Breakdown: {} ===\n", label);
        let pixels = width * height;
        let data: Vec<u8> = (0..pixels * 3).map(|i| ((i * 17) % 256) as u8).collect();

        let (_, estimated) = calc_allocations(width, height, subsampling);

        if let Ok(m) = measure_strip_allocs(width, height, subsampling, &data) {
            println!("Measured allocations by context:");
            println!("{:-<50}", "");

            // Group and sum by context
            let mut grouped: std::collections::HashMap<&str, usize> =
                std::collections::HashMap::new();
            for (ctx, bytes) in &m.by_context {
                *grouped.entry(*ctx).or_insert(0) += bytes;
            }

            // Sort by size descending
            let mut sorted: Vec<_> = grouped.into_iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(&a.1));

            for (ctx, bytes) in &sorted {
                println!("  {:<35} {:>12}", ctx, format_bytes(*bytes));
            }
            println!("{:-<50}", "");
            println!(
                "  {:<35} {:>12}",
                "MEASURED TOTAL",
                format_bytes(m.peak_bytes)
            );
            println!("  {:<35} {:>12}", "ESTIMATED", format_bytes(estimated));
            let delta = m.peak_bytes as i64 - estimated as i64;
            let sign = if delta >= 0 { "+" } else { "-" };
            println!(
                "  {:<35} {:>12}",
                "DELTA",
                format!("{}{}", sign, format_bytes(delta.unsigned_abs() as usize))
            );
            // Show input size for reference
            let input_rgb = width * height * 3;
            println!(
                "  {:<35} {:>12}",
                "(input RGB/BGR8 for reference)",
                format_bytes(input_rgb)
            );
        }
    }
}
