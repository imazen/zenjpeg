//! Test edge MCU handling parity between Rust and C++ jpegli.
//!
//! This example amplifies edge-case bugs in rightmost MCU handling by tiling
//! just the partial MCU columns across a test image. Instead of affecting <1%
//! of blocks, the tiled image exercises the edge case in 100% of blocks.
//!
//! Usage:
//! ```bash
//! cargo run --release --example edge_mcu_parity
//! cargo run --release --example edge_mcu_parity -- --edge-width 1
//! ```

use enough::Unstoppable;
use jpegli::encoder::{
    ChromaSubsampling as JpegliChromaSubsampling, EncoderConfig as JpegliEncoderConfig, PixelLayout,
};
use jpegli_bench_utils::{
    create_edge_test_image, ChromaSubsampling, ColorMode, EdgeReplicationMode, EdgeTestConfig,
    EncoderConfig, EncoderImpl, ImageData, McuEdgeInfo, ScanMode,
};
use std::fs;
use std::path::PathBuf;

fn load_png(path: &std::path::Path) -> Option<(Vec<rgb::RGB8>, u32, u32)> {
    let file = fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;

    let width = info.width;
    let height = info.height;

    let rgb: Vec<rgb::RGB8> = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()]
            .chunks(3)
            .map(|c| rgb::RGB8::new(c[0], c[1], c[2]))
            .collect(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .map(|c| rgb::RGB8::new(c[0], c[1], c[2]))
            .collect(),
        png::ColorType::Grayscale => buf[..info.buffer_size()]
            .iter()
            .map(|&g| rgb::RGB8::new(g, g, g))
            .collect(),
        _ => return None,
    };

    Some((rgb, width, height))
}

fn encode_rust(pixels: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    let config = JpegliEncoderConfig::new(quality as f32, JpegliChromaSubsampling::Quarter)
        .progressive(true)
        .optimize_huffman(true);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder setup");
    enc.push_packed(pixels, Unstoppable).expect("push");
    enc.finish().expect("Rust encode failed")
}

fn encode_cpp(image: &ImageData, quality: u8) -> Vec<u8> {
    let config = EncoderConfig::new(EncoderImpl::CJpegli)
        .color(ColorMode::YCbCr)
        .scan(ScanMode::Progressive)
        .subsampling(ChromaSubsampling::S420)
        .quality(quality);
    config.encode(image).expect("C++ encode failed")
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode().expect("JPEG decode failed")
}

fn compute_dssim(orig: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    let attr = dssim::Dssim::new();
    let orig_rgba: Vec<rgb::RGBA8> = orig
        .chunks(3)
        .map(|c| rgb::RGBA8::new(c[0], c[1], c[2], 255))
        .collect();
    let dec_rgba: Vec<rgb::RGBA8> = decoded
        .chunks(3)
        .map(|c| rgb::RGBA8::new(c[0], c[1], c[2], 255))
        .collect();
    let orig_img = attr.create_image_rgba(&orig_rgba, width, height).unwrap();
    let dec_img = attr.create_image_rgba(&dec_rgba, width, height).unwrap();
    let (dssim, _) = attr.compare(&orig_img, dec_img);
    dssim.into()
}

fn find_frymire() -> Option<PathBuf> {
    let paths = [
        PathBuf::from("jpegli-rs/tests/images/frymire.png"),
        PathBuf::from("tests/images/frymire.png"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/images/frymire.png"),
    ];
    paths.into_iter().find(|p| p.exists())
}

fn main() {
    // Parse arguments
    let args: Vec<String> = std::env::args().collect();
    let edge_width: Option<usize> = args
        .iter()
        .position(|a| a == "--edge-width")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());
    let edge_height: Option<usize> = args
        .iter()
        .position(|a| a == "--edge-height")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());
    let mode_arg = args.iter().find(|a| a.starts_with("--mode="));

    println!("=== Edge MCU Parity Test ===\n");
    println!("Usage: edge_mcu_parity [--edge-width N] [--edge-height N] [--mode=right|bottom|both|auto]\n");

    // Load frymire
    let frymire_path = match find_frymire() {
        Some(p) => p,
        None => {
            eprintln!("Could not find frymire.png");
            return;
        }
    };

    let (rgb_pixels, width, height) = match load_png(&frymire_path) {
        Some(data) => data,
        None => {
            eprintln!("Failed to load frymire.png");
            return;
        }
    };

    let info = McuEdgeInfo::analyze(width as usize, height as usize);
    println!("Source image: {}x{}", width, height);
    println!(
        "  Partial MCU width:  {} columns ({:.2}% of blocks)",
        info.partial_mcu_width, info.width_affected_pct
    );
    println!(
        "  Partial MCU height: {} rows ({:.2}% of blocks)",
        info.partial_mcu_height, info.height_affected_pct
    );
    println!("  Total affected:     {:.2}%\n", info.total_affected_pct);

    // Convert to imgref for tiling
    let source_img = imgref::ImgVec::new(rgb_pixels.clone(), width as usize, height as usize);

    // Determine mode
    let mode = match mode_arg.map(|s| s.trim_start_matches("--mode=")) {
        Some("right") => EdgeReplicationMode::Right,
        Some("bottom") => EdgeReplicationMode::Bottom,
        Some("both") => EdgeReplicationMode::Both,
        _ => EdgeReplicationMode::Auto,
    };

    // Determine edge dimensions
    let test_edge_width = edge_width.unwrap_or(info.partial_mcu_width).max(1);
    let test_edge_height = edge_height.unwrap_or(info.partial_mcu_height).max(1);

    // Create edge-tiled image
    // Use 64 MCUs + partial for each dimension
    let target_width = 64 * 8 + test_edge_width;
    let target_height = 64 * 8 + test_edge_height;

    let config = EdgeTestConfig {
        mode,
        right_edge_width: Some(test_edge_width),
        bottom_edge_height: Some(test_edge_height),
        target_width: Some(target_width),
        target_height: Some(target_height),
    };

    let tiled = match create_edge_test_image(&source_img, config) {
        Some(img) => img,
        None => {
            println!("No edge replication needed (dimensions are 8-aligned)");
            return;
        }
    };
    let tiled_width = tiled.width();
    let tiled_height = tiled.height();

    println!(
        "Edge-tiled image: {}x{} (mode={:?}, right={}, bottom={})",
        tiled_width, tiled_height, mode, test_edge_width, test_edge_height
    );
    println!("  Now 100% of content comes from edge strips\n");

    // Convert tiled image to bytes
    let tiled_bytes: Vec<u8> = tiled.buf().iter().flat_map(|p| [p.r, p.g, p.b]).collect();
    let tiled_image_data = ImageData {
        name: "edge_tiled".to_string(),
        pixels: tiled_bytes.clone(),
        width: target_width,
        height: tiled_height,
    };

    // Test at multiple quality levels
    let qualities = [50, 75, 90, 95];

    println!(
        "{:>7} | {:>10} {:>10} {:>8} | {:>10} {:>10} {:>10}",
        "Quality", "Rust Size", "C++ Size", "Size Δ%", "Rust DSSIM", "C++ DSSIM", "DSSIM Δ%"
    );
    println!("{}", "-".repeat(85));

    for quality in qualities {
        // Encode
        let rust_jpeg = encode_rust(
            &tiled_bytes,
            target_width as u32,
            tiled_height as u32,
            quality,
        );
        let cpp_jpeg = encode_cpp(&tiled_image_data, quality);

        // Decode
        let rust_decoded = decode_jpeg(&rust_jpeg);
        let cpp_decoded = decode_jpeg(&cpp_jpeg);

        // Compute metrics
        let rust_dssim = compute_dssim(&tiled_bytes, &rust_decoded, target_width, tiled_height);
        let cpp_dssim = compute_dssim(&tiled_bytes, &cpp_decoded, target_width, tiled_height);

        let size_diff =
            (rust_jpeg.len() as f64 - cpp_jpeg.len() as f64) / cpp_jpeg.len() as f64 * 100.0;
        let dssim_diff = if cpp_dssim > 0.0 {
            (rust_dssim - cpp_dssim) / cpp_dssim * 100.0
        } else {
            0.0
        };

        println!(
            "{:>7} | {:>10} {:>10} {:>+7.2}% | {:>10.6} {:>10.6} {:>+9.2}%",
            format!("q{}", quality),
            rust_jpeg.len(),
            cpp_jpeg.len(),
            size_diff,
            rust_dssim,
            cpp_dssim,
            dssim_diff
        );
    }

    println!("\n=== Summary ===");
    println!("If the fix is working correctly, Size Δ% and DSSIM Δ% should be small.");
    println!("Large differences indicate bugs in partial MCU handling.");
}
