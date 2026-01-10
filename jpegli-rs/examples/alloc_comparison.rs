//! Compare allocation patterns between full-plane and strip-based encoding.
//!
//! Key finding: Strip encoder currently provides NO peak memory savings because:
//! 1. It pre-allocates f32 DCT block storage for the entire image
//! 2. The strip buffers are in addition to this
//!
//! The strip encoder's benefit is cache locality during processing, not memory reduction.

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
    // STRIP ENCODER
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

    // Strip f32 buffers (small, 16 rows)
    let strip_y = width * strip_height * 4;
    let strip_cb = c_width * c_strip_height * 4;
    let strip_cr = c_width * c_strip_height * 4;
    // Downsampled chroma temp buffers
    let strip_cb_down = c_width * c_strip_height * 4;
    let strip_cr_down = c_width * c_strip_height * 4;

    // f32 DCT blocks - FULL IMAGE (the problem!)
    let strip_y_blocks_f32 = y_block_count * 256; // [f32; 64] = 256 bytes
    let strip_c_blocks_f32 = c_block_count * 2 * 256;

    // i16 blocks in finalize() - these replace f32 blocks
    let strip_y_blocks_i16 = y_block_count * 128;
    let strip_c_blocks_i16 = c_block_count * 2 * 128;

    // AQ map
    let strip_aq = y_block_count * 4;

    // Peak is when both f32 blocks and i16 blocks exist during finalize transition
    // Actually, finalize creates new i16 vecs while f32 still exists
    let strip_peak = strip_y
        + strip_cb
        + strip_cr
        + strip_cb_down
        + strip_cr_down
        + strip_y_blocks_f32
        + strip_c_blocks_f32
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
        y_quant, cb_quant, cr_quant, y_zero_bias, cb_zero_bias, cr_zero_bias,
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

    println!("\n\nWhy Strip Encoder Uses MORE Memory:");
    println!("====================================");
    println!("1. Full-plane: stores f32 planes + i16 blocks (after quantization)");
    println!("2. Strip: stores f32 blocks (pre-quant) + i16 blocks (post-quant)");
    println!("   - f32 blocks: 256 bytes each (double i16's 128 bytes)");
    println!("   - During finalize(), BOTH exist simultaneously");
    println!("");
    println!("Strip encoder's benefit is CACHE LOCALITY, not memory reduction.");
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
    println!("  Strip buffers:      {}", format_bytes(width * 16 * 4 * 5)); // y, cb, cr, cb_down, cr_down
    println!(
        "  Y blocks (f32):     {} (problem!)",
        format_bytes(y_blocks * 256)
    );
    println!(
        "  Cb/Cr blocks (f32): {} (problem!)",
        format_bytes(c_blocks * 2 * 256)
    );
    println!(
        "  + i16 blocks in finalize: {}",
        format_bytes(y_blocks * 128 + c_blocks * 256)
    );
    println!("  AQ map:             {}", format_bytes(y_blocks * 4));
}
