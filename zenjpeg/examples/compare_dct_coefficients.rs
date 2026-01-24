//! Compare DCT coefficients between Rust and C++ jpegli encoders.
//!
//! This tool helps identify the root cause of quality differences by
//! comparing the actual quantized DCT coefficients between the two encoders.
//!
//! IMPORTANT: Uses distance-based encoding (`jpegli_set_distance`) for both
//! encoders to ensure identical quant table configurations (3 tables).
//!
//! Run with: cargo run --release --example compare_dct_coefficients -- [image] [distance]
//! Default distance: 1.0 (roughly equivalent to q90)

use enough::Unstoppable;
use zenjpeg::decode::Decoder;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality};
use jpegli_bench_utils::{
    ChromaSubsampling as BenchChromaSubsampling, ColorMode, EncoderConfig as BenchEncoderConfig,
    EncoderImpl, ImageData, ScanMode,
};

fn load_test_image(path: &str) -> (Vec<u8>, u32, u32) {
    let file = std::fs::File::open(path).expect("open");
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("info");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("decode");

    // Convert to RGB if needed
    let pixels = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        png::ColorType::Grayscale => buf[..info.buffer_size()]
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect(),
        _ => panic!("unsupported color type"),
    };
    (pixels, info.width, info.height)
}

/// Encode using Rust jpegli with Butteraugli distance.
fn encode_rust(
    pixels: &[u8],
    width: u32,
    height: u32,
    distance: f32,
    progressive: bool,
) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(
        Quality::ApproxButteraugli(distance),
        ChromaSubsampling::Quarter,
    )
    .progressive(progressive)
    .optimize_huffman(true);
    let mut encoder = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder creation failed");
    encoder
        .push_packed(pixels, Unstoppable)
        .expect("push failed");
    encoder.finish().expect("finish failed")
}

/// Encode using C++ jpegli FFI with Butteraugli distance.
///
/// Uses `jpegli_set_distance` which sets `add_two_chroma_tables=true` (3 tables),
/// matching Rust's behavior. This is different from `jpeg_set_quality` which
/// uses 2 chroma tables (Cr shared for both Cb and Cr).
fn encode_cpp_ffi(
    pixels: &[u8],
    width: u32,
    height: u32,
    distance: f32,
    progressive: bool,
) -> Vec<u8> {
    let img = ImageData {
        name: "test".to_string(),
        pixels: pixels.to_vec(),
        width: width as usize,
        height: height as usize,
    };
    let scan_mode = if progressive {
        ScanMode::Progressive
    } else {
        ScanMode::Baseline
    };
    BenchEncoderConfig::new(EncoderImpl::CJpegli)
        .color(ColorMode::YCbCr)
        .scan(scan_mode)
        .subsampling(BenchChromaSubsampling::S420)
        .distance(distance) // Use distance, not quality!
        .encode(&img)
        .expect("C++ jpegli FFI encode failed")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let src_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "/home/lilith/work/codec-eval/codec-corpus/kodak/1.png".to_string()
    };

    // Distance (lower = better quality, ~1.0 is roughly q90)
    let distance: f32 = if args.len() > 2 {
        args[2].parse().unwrap_or(1.0)
    } else {
        1.0
    };

    let progressive = args.iter().any(|a| a == "--progressive" || a == "-p");

    println!("=== DCT Coefficient Comparison (Distance-based) ===");
    println!("Image: {}", src_path);
    println!("Distance: {} (Butteraugli)", distance);
    println!("Progressive: {}", progressive);
    println!();
    println!("NOTE: Using jpegli_set_distance for C++ (3 tables, add_two_chroma_tables=true)");
    println!("      and Quality::ApproxButteraugli for Rust (3 tables).");
    println!("      This ensures identical quant table configuration.");
    println!();

    // Load source image
    let (pixels, width, height) = load_test_image(&src_path);
    println!(
        "Loaded {}x{} image ({} bytes RGB)",
        width,
        height,
        pixels.len()
    );

    // Encode with both encoders using identical distance settings
    let rust_jpeg = encode_rust(&pixels, width, height, distance, progressive);
    let cpp_jpeg = encode_cpp_ffi(&pixels, width, height, distance, progressive);

    println!("Rust JPEG: {} bytes", rust_jpeg.len());
    println!("C++  JPEG: {} bytes", cpp_jpeg.len());
    println!(
        "Size diff: {:+.2}%",
        (rust_jpeg.len() as f64 / cpp_jpeg.len() as f64 - 1.0) * 100.0
    );
    println!();

    // Decode both and extract coefficients
    let decoder = Decoder::new();
    let rust_coeffs = decoder
        .decode_coefficients(&rust_jpeg)
        .expect("decode rust coefficients");
    let cpp_coeffs = decoder
        .decode_coefficients(&cpp_jpeg)
        .expect("decode cpp coefficients");

    // Print quant tables for verification
    println!("=== Quantization Tables ===");
    for i in 0..4 {
        match (
            &rust_coeffs.quant_tables.get(i),
            &cpp_coeffs.quant_tables.get(i),
        ) {
            (Some(Some(rq)), Some(Some(cq))) => {
                let dc_match = rq[0] == cq[0];
                println!(
                    "Table {}: DC rust={}, cpp={} {}",
                    i,
                    rq[0],
                    cq[0],
                    if dc_match { "(MATCH)" } else { "(DIFFERENT!)" }
                );
            }
            (Some(Some(rq)), _) => println!("Table {}: rust={} (C++ missing)", i, rq[0]),
            (_, Some(Some(cq))) => println!("Table {}: cpp={} (Rust missing)", i, cq[0]),
            _ => {}
        }
    }
    println!();

    println!("Rust coeffs: {} components", rust_coeffs.num_components());
    for (i, c) in rust_coeffs.components.iter().enumerate() {
        println!(
            "  Component {}: {}x{} blocks ({} total), id={}, h_samp={}, v_samp={}",
            i,
            c.blocks_wide,
            c.blocks_high,
            c.num_blocks(),
            c.id,
            c.h_samp,
            c.v_samp
        );
    }
    println!();

    // Compare coefficients
    let comparison = rust_coeffs.compare(&cpp_coeffs);

    println!("=== Coefficient Comparison Results ===");
    println!("Total blocks compared: {}", comparison.total_blocks);
    println!(
        "Blocks with differences: {} ({:.2}%)",
        comparison.differing_blocks,
        comparison.diff_block_pct()
    );
    println!("Max absolute difference: {}", comparison.max_diff);
    println!(
        "Total differing coefficients: {}",
        comparison.total_diff_coeffs
    );
    println!(
        "DC coefficients differing: {} ({:.2}%)",
        comparison.diff_by_position[0],
        comparison.dc_diff_pct()
    );
    println!();

    // Analyze difference distribution by position
    println!("=== Difference Distribution by Zigzag Position ===");
    println!("{:>4}  {:>10}  {:>8}", "Pos", "Count", "Pct");
    let mut ac_diffs = 0u64;
    for (pos, count) in comparison.diff_by_position.iter().enumerate() {
        if *count > 0 {
            let pct = 100.0 * *count as f64 / comparison.total_blocks as f64;
            if pos == 0 {
                println!("{:>4}  {:>10}  {:>7.2}%  (DC)", pos, count, pct);
            } else {
                println!("{:>4}  {:>10}  {:>7.2}%", pos, count, pct);
                ac_diffs += count;
            }
        }
    }
    println!();
    println!(
        "Total AC differences: {} ({:.2}% of blocks)",
        ac_diffs,
        100.0 * ac_diffs as f64 / (comparison.total_blocks * 63) as f64
    );
    println!();

    // Find blocks with largest differences
    println!("=== Blocks with Largest Differences ===");
    let mut max_diff_blocks: Vec<(usize, usize, usize, usize, i16, i16)> = Vec::new(); // (comp, bx, by, pos, rust, cpp)

    for comp_idx in 0..rust_coeffs
        .components
        .len()
        .min(cpp_coeffs.components.len())
    {
        let rc = &rust_coeffs.components[comp_idx];
        let cc = &cpp_coeffs.components[comp_idx];
        let num_blocks = rc.num_blocks().min(cc.num_blocks());

        for block_idx in 0..num_blocks {
            let rb = rc.block(block_idx);
            let cb = cc.block(block_idx);
            let bx = block_idx % rc.blocks_wide;
            let by = block_idx / rc.blocks_wide;

            for i in 0..64 {
                let diff = (rb[i] as i32 - cb[i] as i32).abs() as i16;
                if diff >= 3 {
                    // Show blocks with diff >= 3
                    max_diff_blocks.push((comp_idx, bx, by, i, rb[i], cb[i]));
                }
            }
        }
    }

    // Sort by absolute difference descending
    max_diff_blocks.sort_by(|a, b| {
        let diff_a = (a.4 as i32 - a.5 as i32).abs();
        let diff_b = (b.4 as i32 - b.5 as i32).abs();
        diff_b.cmp(&diff_a)
    });

    for (comp, bx, by, pos, rust_val, cpp_val) in max_diff_blocks.iter().take(20) {
        let diff = *rust_val as i32 - *cpp_val as i32;
        let comp_name = match comp {
            0 => "Y ",
            1 => "Cb",
            2 => "Cr",
            _ => "??",
        };
        println!(
            "{} block ({:3},{:3}) zigzag[{:2}]: rust={:5}, cpp={:5}, diff={:+4}",
            comp_name, bx, by, pos, rust_val, cpp_val, diff
        );
    }
    println!();

    // Detailed analysis of first few differing blocks
    println!("=== First 10 Differing Blocks (detailed) ===");
    let mut shown = 0;
    'outer: for comp_idx in 0..rust_coeffs
        .components
        .len()
        .min(cpp_coeffs.components.len())
    {
        let rc = &rust_coeffs.components[comp_idx];
        let cc = &cpp_coeffs.components[comp_idx];
        let num_blocks = rc.num_blocks().min(cc.num_blocks());

        for block_idx in 0..num_blocks {
            let rb = rc.block(block_idx);
            let cb = cc.block(block_idx);

            // Check for differences
            let mut diffs: Vec<(usize, i16, i16)> = Vec::new();
            for i in 0..64 {
                if rb[i] != cb[i] {
                    diffs.push((i, rb[i], cb[i]));
                }
            }

            if !diffs.is_empty() {
                let bx = block_idx % rc.blocks_wide;
                let by = block_idx / rc.blocks_wide;
                println!(
                    "Block (comp={}, x={}, y={}) - {} differences:",
                    comp_idx,
                    bx,
                    by,
                    diffs.len()
                );
                for (pos, rust_val, cpp_val) in diffs.iter().take(5) {
                    let diff = *rust_val as i32 - *cpp_val as i32;
                    println!(
                        "  zigzag[{}]: rust={}, cpp={}, diff={:+}",
                        pos, rust_val, cpp_val, diff
                    );
                }
                if diffs.len() > 5 {
                    println!("  ... and {} more differences", diffs.len() - 5);
                }
                println!();
                shown += 1;
                if shown >= 10 {
                    break 'outer;
                }
            }
        }
    }

    if comparison.differing_blocks == 0 {
        println!("No differences found! Coefficients are identical.");
    }

    // Analyze per-component statistics
    println!("=== Per-Component Analysis ===");
    for comp_idx in 0..rust_coeffs
        .components
        .len()
        .min(cpp_coeffs.components.len())
    {
        let rc = &rust_coeffs.components[comp_idx];
        let cc = &cpp_coeffs.components[comp_idx];
        let num_blocks = rc.num_blocks().min(cc.num_blocks());

        let mut comp_diff_blocks = 0;
        let mut comp_max_diff: i16 = 0;
        let mut comp_sum_abs_diff: i64 = 0;
        let mut comp_dc_diffs = 0;

        for block_idx in 0..num_blocks {
            let rb = rc.block(block_idx);
            let cb = cc.block(block_idx);
            let mut has_diff = false;

            for i in 0..64 {
                let diff = (rb[i] as i32 - cb[i] as i32).abs() as i16;
                if diff != 0 {
                    has_diff = true;
                    comp_sum_abs_diff += diff as i64;
                    comp_max_diff = comp_max_diff.max(diff);
                    if i == 0 {
                        comp_dc_diffs += 1;
                    }
                }
            }
            if has_diff {
                comp_diff_blocks += 1;
            }
        }

        let comp_name = match comp_idx {
            0 => "Y ",
            1 => "Cb",
            2 => "Cr",
            _ => "??",
        };
        println!(
            "Component {} (id={}): {}/{} blocks differ ({:.2}%), max_diff={}, DC_diffs={}, sum_abs={}",
            comp_name,
            rc.id,
            comp_diff_blocks,
            num_blocks,
            100.0 * comp_diff_blocks as f64 / num_blocks as f64,
            comp_max_diff,
            comp_dc_diffs,
            comp_sum_abs_diff
        );
    }
}
