//! Compare encoders using Butteraugli quality metric
//!
//! Shows bpp (bits per pixel) vs Butteraugli score for:
//! - zenjpeg mozjpeg strategy
//! - zenjpeg jpegli strategy
//! - reference jpegli
//! - reference mozjpeg

use butteraugli_oxide::{compute_butteraugli, ButteraugliParams};

fn main() {
    // Create synthetic test image
    let width = 512usize;
    let height = 512usize;
    let pixels = width * height;
    let mut rgb_data = Vec::with_capacity(pixels * 3);
    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;
            let r = ((fx * 200.0 + fy * 50.0) % 256.0) as u8;
            let g = ((fy * 180.0 + (x * 3) as f32 % 100.0) as u32 % 256) as u8;
            let b = (((x + y) % 64) as f32 / 64.0 * 128.0 + 64.0) as u8;
            rgb_data.push(r);
            rgb_data.push(g);
            rgb_data.push(b);
        }
    }

    println!("=== Butteraugli Pareto Comparison ({}x{}) ===\n", width, height);
    println!("Lower Butteraugli = better quality, Lower bpp = better compression");
    println!("Target: within 5% file size of reference implementations\n");

    // Header
    println!("{:>4} | {:>12} {:>8} | {:>12} {:>8} | {:>12} {:>8} | {:>12} {:>8}",
        "Q", "zen/moz bpp", "butter", "zen/jpegli", "butter", "ref jpegli", "butter", "mozjpeg", "butter");
    println!("{}", "-".repeat(105));

    for q in [30, 40, 50, 60, 70, 75, 80, 85, 90, 95] {
        // zenjpeg with mozjpeg strategy
        let zen_moz = zenjpeg::Encoder::new()
            .quality(zenjpeg::Quality::Standard(q))
            .strategy(zenjpeg::EncodingStrategy::Mozjpeg)
            .encode_rgb(&rgb_data, width, height)
            .unwrap();

        // zenjpeg with jpegli strategy
        let zen_jpegli = zenjpeg::Encoder::new()
            .quality(zenjpeg::Quality::Standard(q))
            .strategy(zenjpeg::EncodingStrategy::Jpegli)
            .encode_rgb(&rgb_data, width, height)
            .unwrap();

        // jpegli (Rust port)
        let jpegli_result = jpegli::Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .pixel_format(jpegli::PixelFormat::Rgb)
            .quality(jpegli::Quality::Traditional(q as f32))
            .encode(&rgb_data);

        let jpegli_bytes = match jpegli_result {
            Ok(b) => b,
            Err(e) => {
                println!("Q{}: jpegli failed: {:?}", q, e);
                continue;
            }
        };

        // mozjpeg-oxide
        let moz = mozjpeg_oxide::Encoder::new()
            .quality(q)
            .subsampling(mozjpeg_oxide::Subsampling::S444)
            .encode_rgb(&rgb_data, width as u32, height as u32)
            .unwrap();

        // Decode all
        let zen_moz_dec = jpeg_decoder::Decoder::new(&zen_moz[..]).decode().unwrap();
        let zen_jpegli_dec = jpeg_decoder::Decoder::new(&zen_jpegli[..]).decode().unwrap();
        let jpegli_dec = jpeg_decoder::Decoder::new(&jpegli_bytes[..]).decode().unwrap();
        let moz_dec = jpeg_decoder::Decoder::new(&moz[..]).decode().unwrap();

        // Calculate Butteraugli scores
        let params = ButteraugliParams::default();
        let butter_zen_moz = compute_butteraugli(&rgb_data, &zen_moz_dec, width, height, &params)
            .map(|r| r.score).unwrap_or(f64::NAN);
        let butter_zen_jpegli = compute_butteraugli(&rgb_data, &zen_jpegli_dec, width, height, &params)
            .map(|r| r.score).unwrap_or(f64::NAN);
        let butter_jpegli = compute_butteraugli(&rgb_data, &jpegli_dec, width, height, &params)
            .map(|r| r.score).unwrap_or(f64::NAN);
        let butter_moz = compute_butteraugli(&rgb_data, &moz_dec, width, height, &params)
            .map(|r| r.score).unwrap_or(f64::NAN);

        // Calculate bpp
        let bits = 8.0;
        let bpp_zen_moz = (zen_moz.len() as f64 * bits) / pixels as f64;
        let bpp_zen_jpegli = (zen_jpegli.len() as f64 * bits) / pixels as f64;
        let bpp_jpegli = (jpegli_bytes.len() as f64 * bits) / pixels as f64;
        let bpp_moz = (moz.len() as f64 * bits) / pixels as f64;

        println!("{:>4} | {:>12.4} {:>8.4} | {:>12.4} {:>8.4} | {:>12.4} {:>8.4} | {:>12.4} {:>8.4}",
            q, bpp_zen_moz, butter_zen_moz, bpp_zen_jpegli, butter_zen_jpegli,
            bpp_jpegli, butter_jpegli, bpp_moz, butter_moz);
    }

    println!("\n=== Pareto Analysis ===\n");
    println!("Comparing at similar bpp levels:\n");

    // Collect all data points for Pareto analysis
    let mut all_points: Vec<(&str, u8, f64, f64)> = Vec::new();

    for q in [30, 40, 50, 60, 70, 75, 80, 85, 90, 95] {
        let zen_jpegli = zenjpeg::Encoder::new()
            .quality(zenjpeg::Quality::Standard(q))
            .strategy(zenjpeg::EncodingStrategy::Jpegli)
            .encode_rgb(&rgb_data, width, height)
            .unwrap();

        let jpegli_bytes = jpegli::Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .pixel_format(jpegli::PixelFormat::Rgb)
            .quality(jpegli::Quality::Traditional(q as f32))
            .encode(&rgb_data)
            .unwrap();

        let moz = mozjpeg_oxide::Encoder::new()
            .quality(q)
            .subsampling(mozjpeg_oxide::Subsampling::S444)
            .encode_rgb(&rgb_data, width as u32, height as u32)
            .unwrap();

        let zen_jpegli_dec = jpeg_decoder::Decoder::new(&zen_jpegli[..]).decode().unwrap();
        let jpegli_dec = jpeg_decoder::Decoder::new(&jpegli_bytes[..]).decode().unwrap();
        let moz_dec = jpeg_decoder::Decoder::new(&moz[..]).decode().unwrap();

        let params = ButteraugliParams::default();
        let butter_zen_jpegli = compute_butteraugli(&rgb_data, &zen_jpegli_dec, width, height, &params)
            .map(|r| r.score).unwrap_or(f64::NAN);
        let butter_jpegli = compute_butteraugli(&rgb_data, &jpegli_dec, width, height, &params)
            .map(|r| r.score).unwrap_or(f64::NAN);
        let butter_moz = compute_butteraugli(&rgb_data, &moz_dec, width, height, &params)
            .map(|r| r.score).unwrap_or(f64::NAN);

        let bpp_zen_jpegli = (zen_jpegli.len() as f64 * 8.0) / pixels as f64;
        let bpp_jpegli = (jpegli_bytes.len() as f64 * 8.0) / pixels as f64;
        let bpp_moz = (moz.len() as f64 * 8.0) / pixels as f64;

        all_points.push(("zen/jpegli", q, bpp_zen_jpegli, butter_zen_jpegli));
        all_points.push(("ref jpegli", q, bpp_jpegli, butter_jpegli));
        all_points.push(("mozjpeg", q, bpp_moz, butter_moz));
    }

    // Find comparable points (similar bpp, compare quality)
    println!("At similar file sizes, which has better quality?\n");
    println!("{:>15} vs {:>15} | {:>8} | {:>10} vs {:>10} | Winner",
        "Encoder A", "Encoder B", "~bpp", "Butter A", "Butter B");
    println!("{}", "-".repeat(85));

    // Compare zen/jpegli vs ref jpegli at each Q
    for q in [60, 75, 85, 95] {
        let zen = all_points.iter().find(|p| p.0 == "zen/jpegli" && p.1 == q).unwrap();
        let jpegli = all_points.iter().find(|p| p.0 == "ref jpegli" && p.1 == q).unwrap();

        let bpp_diff_pct = ((zen.2 - jpegli.2) / jpegli.2 * 100.0).abs();
        let winner = if zen.3 < jpegli.3 { "zen/jpegli" } else { "ref jpegli" };

        println!("{:>15} vs {:>15} | {:>8.3} | {:>10.4} vs {:>10.4} | {} (bpp diff: {:.1}%)",
            format!("zen/jpegli Q{}", q), format!("ref jpegli Q{}", q),
            (zen.2 + jpegli.2) / 2.0, zen.3, jpegli.3, winner, bpp_diff_pct);
    }
}
