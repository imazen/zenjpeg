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
    let strip_y = width * strip_height * 4;
    let strip_cb = c_width * c_strip_height * 4;
    let strip_cr = c_width * c_strip_height * 4;
    // Downsampled chroma temp buffers
    let strip_cb_down = c_width * c_strip_height * 4;
    let strip_cr_down = c_width * c_strip_height * 4;

    // Pending f32 DCT blocks - ONLY 2 iMCU rows (double-buffered)
    // For 4:2:0: one iMCU = 4 Y blocks + 1 Cb + 1 Cr = 6 blocks per column
    let y_blocks_per_imcu_row = y_blocks_w * 2; // 2 rows of 8x8 Y blocks
    let c_blocks_per_imcu_row = match subsampling {
        Subsampling::S420 => (width + 15) / 16, // 1 Cb + 1 Cr per 16x16
        Subsampling::S422 => (width + 15) / 16 * 2, // 2 rows of Cb/Cr
        Subsampling::S440 => y_blocks_w, // 1 row of Cb/Cr
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

fn measure_strip_allocs(
    width: usize,
    height: usize,
    subsampling: Subsampling,
    data: &[u8],
) -> Result<usize, jpegli::Error> {
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

    let pre_finalize_peak = processor.allocation_stats().peak_bytes;
    let output = processor.finalize()?;
    Ok(pre_finalize_peak.max(output.alloc_stats.peak_bytes))
}

fn main() {
    println!("Peak Allocation Comparison: Full-Plane vs Strip-Based Encoding");
    println!("================================================================\n");

    let resolutions = [
        ("1K (1920x1080)", 1920usize, 1080usize),
        ("2K (2560x1440)", 2560, 1440),
        ("4K (3840x2160)", 3840, 2160),
        ("8K (7680x4320)", 7680, 4320),
    ];

    let subsampling = Subsampling::S420;

    println!("Subsampling: 4:2:0\n");
    println!(
        "{:<20} {:>15} {:>15} {:>15}",
        "Resolution", "Full-Plane", "Strip (calc)", "Strip (meas)"
    );
    println!("{:-<70}", "");

    for (name, width, height) in resolutions {
        let (full_peak, strip_peak_calc) = calc_allocations(width, height, subsampling);

        let pixels = width * height;
        let data: Vec<u8> = (0..pixels * 3).map(|i| ((i * 17) % 256) as u8).collect();

        let strip_peak_meas = measure_strip_allocs(width, height, subsampling, &data)
            .map(|p| format_bytes(p))
            .unwrap_or_else(|e| format!("err: {}", e));

        println!(
            "{:<20} {:>15} {:>15} {:>15}",
            name,
            format_bytes(full_peak),
            format_bytes(strip_peak_calc),
            strip_peak_meas,
        );
    }

    println!("\n\nWhy Strip Encoder Uses LESS Memory:");
    println!("====================================");
    println!("1. Incremental quantization: f32→i16 as AQ strengths become available");
    println!("2. Double-buffered pending: only 2 iMCU rows of f32, not full image");
    println!("3. Strip buffers: 16 rows, reused each iteration");
    println!("");
    println!("Strip encoder provides BOTH cache locality AND memory savings.");
    println!("The 16-row strips fit in L2/L3 cache during color conversion + DCT.");

    // Show the breakdown for 4K
    println!("\n\n4K Allocation Breakdown:");
    println!("========================");
    let (width, height) = (3840, 2160);
    let pixels = width * height;
    let y_blocks = ((width + 7) / 8) * ((height + 7) / 8);
    let c_blocks = ((width + 15) / 16) * ((height + 15) / 16);
    let c_pixels = ((width + 1) / 2) * ((height + 1) / 2);

    println!("\nFull-plane encoder:");
    println!("  Y plane (f32):      {}", format_bytes(pixels * 4));
    println!("  Cb/Cr planes (f32): {}", format_bytes(c_pixels * 4 * 2));
    println!("  Y blocks (i16):     {}", format_bytes(y_blocks * 128));
    println!("  Cb/Cr blocks (i16): {}", format_bytes(c_blocks * 2 * 128));
    println!("  AQ map:             {}", format_bytes(y_blocks * 4));
    println!(
        "  TOTAL:              {}",
        format_bytes(pixels * 4 + c_pixels * 8 + y_blocks * 128 + c_blocks * 256 + y_blocks * 4)
    );

    println!("\nStrip encoder:");
    let y_blocks_w = (width + 7) / 8;
    let y_blocks_per_imcu = y_blocks_w * 2;
    let c_blocks_per_imcu = (width + 15) / 16;
    println!("  Strip buffers:      {}", format_bytes(width * 16 * 4 * 5)); // y, cb, cr, cb_down, cr_down
    println!(
        "  Pending Y (f32):    {} (2 iMCU rows, double-buffered)",
        format_bytes(2 * y_blocks_per_imcu * 256)
    );
    println!(
        "  Pending Cb/Cr (f32):{} (2 iMCU rows, double-buffered)",
        format_bytes(2 * c_blocks_per_imcu * 2 * 256)
    );
    println!(
        "  Final i16 blocks:   {} (grows incrementally)",
        format_bytes(y_blocks * 128 + c_blocks * 256)
    );
    println!("  AQ map:             {}", format_bytes(y_blocks * 4));
    println!(
        "  TOTAL (peak):       {}",
        format_bytes(
            width * 16 * 4 * 5
                + 2 * y_blocks_per_imcu * 256
                + 2 * c_blocks_per_imcu * 2 * 256
                + y_blocks * 128
                + c_blocks * 256
                + y_blocks * 4
        )
    );
}
