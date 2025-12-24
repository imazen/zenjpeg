//! Comprehensive XYB vs YCbCr vs mozjpeg comparison
//!
//! This compares jpegli (XYB mode), jpegli (YCbCr mode), and mozjpeg
//! using DSSIM and off-by-N statistics.

use dssim::Dssim;
use jpegli::{Encoder, Quality};
use std::fs;
use std::path::Path;

/// Statistics for comparing two images
#[derive(Debug, Default)]
struct DiffStats {
    pixels: usize,
    values: usize,  // pixels * 3 for RGB
    values_differing: usize,
    values_off_by_1: usize,
    values_off_by_2: usize,
    values_off_by_3_plus: usize,
    max_diff: u8,
    sum_abs_diff: u64,
}

impl DiffStats {
    fn from_pixels(a: &[u8], b: &[u8]) -> Self {
        assert_eq!(a.len(), b.len());
        let mut stats = DiffStats {
            pixels: a.len() / 3,
            values: a.len(),
            ..Default::default()
        };

        for (av, bv) in a.iter().zip(b.iter()) {
            let diff = (*av as i16 - *bv as i16).unsigned_abs() as u8;
            if diff > 0 {
                stats.values_differing += 1;
                stats.sum_abs_diff += diff as u64;
                stats.max_diff = stats.max_diff.max(diff);

                match diff {
                    1 => stats.values_off_by_1 += 1,
                    2 => stats.values_off_by_2 += 1,
                    _ => stats.values_off_by_3_plus += 1,
                }
            }
        }
        stats
    }

    fn report(&self, name: &str) {
        let pct_diff = 100.0 * self.values_differing as f64 / self.values as f64;
        let pct_off_1 = 100.0 * self.values_off_by_1 as f64 / self.values as f64;
        let pct_off_2 = 100.0 * self.values_off_by_2 as f64 / self.values as f64;
        let pct_off_3 = 100.0 * self.values_off_by_3_plus as f64 / self.values as f64;
        let avg_diff = if self.values_differing > 0 {
            self.sum_abs_diff as f64 / self.values_differing as f64
        } else {
            0.0
        };

        println!("{name}:");
        println!("  Total values: {}", self.values);
        println!("  Differing: {} ({:.2}%)", self.values_differing, pct_diff);
        println!("  Off by 1: {} ({:.2}%)", self.values_off_by_1, pct_off_1);
        println!("  Off by 2: {} ({:.2}%)", self.values_off_by_2, pct_off_2);
        println!("  Off by 3+: {} ({:.2}%)", self.values_off_by_3_plus, pct_off_3);
        println!("  Max diff: {}", self.max_diff);
        println!("  Avg diff (when differing): {:.2}", avg_diff);
    }
}

fn calculate_dssim(orig: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    let attr = Dssim::new();

    // Convert to RGBA for dssim
    let orig_rgba: Vec<_> = orig.chunks(3)
        .map(|c| rgb::RGBA8::new(c[0], c[1], c[2], 255))
        .collect();
    let decoded_rgba: Vec<_> = decoded.chunks(3)
        .map(|c| rgb::RGBA8::new(c[0], c[1], c[2], 255))
        .collect();

    let orig_img = attr.create_image_rgba(&orig_rgba, width, height).unwrap();
    let decoded_img = attr.create_image_rgba(&decoded_rgba, width, height).unwrap();

    let (dssim, _) = attr.compare(&orig_img, decoded_img);
    dssim.into()
}

fn load_png(path: &Path) -> (Vec<u8>, u32, u32) {
    let decoder = png::Decoder::new(fs::File::open(path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();

    let bytes = &buf[..info.buffer_size()];
    let rgb = match info.color_type {
        png::ColorType::Rgb => bytes.to_vec(),
        png::ColorType::Rgba => bytes.chunks(4).flat_map(|c| [c[0], c[1], c[2]]).collect(),
        _ => panic!("Unsupported color type"),
    };

    (rgb, info.width, info.height)
}

fn encode_mozjpeg(data: &[u8], width: u32, height: u32, quality: f32) -> Vec<u8> {
    let mut comp = mozjpeg::Compress::new(mozjpeg::ColorSpace::JCS_RGB);
    comp.set_size(width as usize, height as usize);
    comp.set_quality(quality);

    let mut output = Vec::new();
    let mut comp = comp.start_compress(&mut output).unwrap();

    // Write all scanlines at once
    comp.write_scanlines(data);

    comp.finish().unwrap();
    output
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    let decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(data));
    let mut decoder = decoder;
    let pixels = decoder.decode().unwrap();
    pixels
}

fn main() {
    println!("=== jpegli XYB vs YCbCr vs mozjpeg Comparison ===\n");

    // Test with flower_small image if available
    let test_images = [
        "/home/lilith/work/jpegli/testdata/jxl/flower/flower_small.rgb.png",
        "/mnt/v/work/corpus/CID22-512/0001.png",
    ];

    let quality = 90.0;

    for path_str in &test_images {
        let path = Path::new(path_str);
        if !path.exists() {
            println!("Skipping {} (not found)\n", path_str);
            continue;
        }

        println!("Testing: {}", path.file_name().unwrap().to_str().unwrap());
        println!("Quality: {}", quality);
        println!();

        let (original, width, height) = load_png(path);
        println!("Image size: {}x{} ({} pixels)\n", width, height, width * height);

        // Encode with each method
        let jpegli_xyb = Encoder::new()
            .width(width)
            .height(height)
            .quality(Quality::Traditional(quality))
            .use_xyb(true)
            .encode(&original)
            .unwrap();

        let jpegli_ycbcr = Encoder::new()
            .width(width)
            .height(height)
            .quality(Quality::Traditional(quality))
            .encode(&original)
            .unwrap();

        let mozjpeg_out = encode_mozjpeg(&original, width, height, quality);

        println!("File sizes:");
        println!("  jpegli XYB:   {} bytes ({:.2} bpp)",
            jpegli_xyb.len(),
            8.0 * jpegli_xyb.len() as f64 / (width * height) as f64);
        println!("  jpegli YCbCr: {} bytes ({:.2} bpp)",
            jpegli_ycbcr.len(),
            8.0 * jpegli_ycbcr.len() as f64 / (width * height) as f64);
        println!("  mozjpeg:      {} bytes ({:.2} bpp)",
            mozjpeg_out.len(),
            8.0 * mozjpeg_out.len() as f64 / (width * height) as f64);
        println!();

        // Decode and compare
        // Note: XYB JPEG needs ICC-aware decoder for proper colors
        // For raw comparison, we use standard decoder which gives XYB values

        let ycbcr_decoded = decode_jpeg(&jpegli_ycbcr);
        let mozjpeg_decoded = decode_jpeg(&mozjpeg_out);

        // For XYB, the decoded values are in XYB space, not sRGB
        // We'll compare file sizes and note that XYB needs ICC profile support

        // Compare jpegli YCbCr vs original
        let ycbcr_stats = DiffStats::from_pixels(&original, &ycbcr_decoded);
        let ycbcr_dssim = calculate_dssim(&original, &ycbcr_decoded, width as usize, height as usize);

        // Compare mozjpeg vs original
        let mozjpeg_stats = DiffStats::from_pixels(&original, &mozjpeg_decoded);
        let mozjpeg_dssim = calculate_dssim(&original, &mozjpeg_decoded, width as usize, height as usize);

        println!("--- jpegli YCbCr vs Original ---");
        println!("DSSIM: {:.6}", ycbcr_dssim);
        ycbcr_stats.report("Pixel diff");
        println!();

        println!("--- mozjpeg vs Original ---");
        println!("DSSIM: {:.6}", mozjpeg_dssim);
        mozjpeg_stats.report("Pixel diff");
        println!();

        // Compare jpegli vs mozjpeg directly
        let jpegli_vs_moz = DiffStats::from_pixels(&ycbcr_decoded, &mozjpeg_decoded);
        println!("--- jpegli YCbCr vs mozjpeg (decoded) ---");
        jpegli_vs_moz.report("Pixel diff");
        println!();

        // Summary
        println!("=== Summary ===");
        println!("DSSIM (lower = better): jpegli={:.6}, mozjpeg={:.6}", ycbcr_dssim, mozjpeg_dssim);
        if ycbcr_dssim < mozjpeg_dssim {
            println!("  → jpegli has {:.1}% better DSSIM", 100.0 * (1.0 - ycbcr_dssim / mozjpeg_dssim));
        } else {
            println!("  → mozjpeg has {:.1}% better DSSIM", 100.0 * (1.0 - mozjpeg_dssim / ycbcr_dssim));
        }

        println!("\nFile size:");
        if jpegli_ycbcr.len() < mozjpeg_out.len() {
            println!("  → jpegli is {:.1}% smaller",
                100.0 * (1.0 - jpegli_ycbcr.len() as f64 / mozjpeg_out.len() as f64));
        } else {
            println!("  → mozjpeg is {:.1}% smaller",
                100.0 * (1.0 - mozjpeg_out.len() as f64 / jpegli_ycbcr.len() as f64));
        }

        println!("\nXYB mode note: XYB JPEG ({} bytes) requires ICC-aware decoder for comparison.",
            jpegli_xyb.len());

        println!("\n{}\n", "=".repeat(60));
    }
}
