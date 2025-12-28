//! Compare DCT and quantization coefficients between zenjpeg and mozjpeg-oxide
//!
//! This helps identify where the quality divergence occurs.

use zenjpeg::{Encoder, Quality};

fn main() {
    // Test on larger, more complex image (512x512 synthetic)
    let width = 512usize;
    let height = 512usize;
    let mut rgb_data = Vec::with_capacity(width * height * 3);
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

    // Encode with both encoders at Q80
    let q = 80;

    // zenjpeg with default settings
    let zen = Encoder::new()
        .quality(Quality::Standard(q))
        .encode_rgb(&rgb_data, width, height)
        .unwrap();

    // zenjpeg with trellis disabled (fastest mode uses no trellis optimization)
    let zen_fast = Encoder::fastest()
        .quality(Quality::Standard(q))
        .encode_rgb(&rgb_data, width, height)
        .unwrap();

    // mozjpeg-oxide
    let moz = mozjpeg_oxide::Encoder::new()
        .quality(q)
        .subsampling(mozjpeg_oxide::Subsampling::S444)
        .encode_rgb(&rgb_data, width as u32, height as u32)
        .unwrap();

    println!("=== JPEG Size Comparison ({}x{}) ===", width, height);
    println!("zenjpeg (trellis):  {} bytes", zen.len());
    println!("zenjpeg (fastest):  {} bytes", zen_fast.len());
    println!("mozjpeg-oxide:      {} bytes", moz.len());
    println!();

    // Decode both JPEGs and compare
    let zen_dec = jpeg_decoder::Decoder::new(&zen[..]).decode().unwrap();
    let zen_fast_dec = jpeg_decoder::Decoder::new(&zen_fast[..]).decode().unwrap();
    let moz_dec = jpeg_decoder::Decoder::new(&moz[..]).decode().unwrap();

    // Calculate total error for all 3
    let mut zen_total_err = 0i64;
    let mut zen_fast_total_err = 0i64;
    let mut moz_total_err = 0i64;
    let mut zen_max_err = 0i64;
    let mut zen_fast_max_err = 0i64;
    let mut moz_max_err = 0i64;

    for i in 0..rgb_data.len() {
        let orig = rgb_data[i] as i64;
        let zen_val = zen_dec[i] as i64;
        let zen_fast_val = zen_fast_dec[i] as i64;
        let moz_val = moz_dec[i] as i64;
        let zen_e = (orig - zen_val).abs();
        let zen_fast_e = (orig - zen_fast_val).abs();
        let moz_e = (orig - moz_val).abs();
        zen_total_err += zen_e;
        zen_fast_total_err += zen_fast_e;
        moz_total_err += moz_e;
        zen_max_err = zen_max_err.max(zen_e);
        zen_fast_max_err = zen_fast_max_err.max(zen_fast_e);
        moz_max_err = moz_max_err.max(moz_e);
    }

    let n = rgb_data.len() as f64;
    println!("=== Error Metrics ===");
    println!("Mean Absolute Error:");
    println!("  zenjpeg (trellis): {:.3}", zen_total_err as f64 / n);
    println!("  zenjpeg (fastest): {:.3}", zen_fast_total_err as f64 / n);
    println!("  mozjpeg-oxide:     {:.3}", moz_total_err as f64 / n);
    println!("Max Error:");
    println!("  zenjpeg (trellis): {}", zen_max_err);
    println!("  zenjpeg (fastest): {}", zen_fast_max_err);
    println!("  mozjpeg-oxide:     {}", moz_max_err);

    // Parse quantization tables from the JPEG data
    println!();
    println!("=== Quantization Tables ===");

    let zen_qtable = extract_qtable(&zen);
    let moz_qtable = extract_qtable(&moz);

    if let (Some(zen_q), Some(moz_q)) = (&zen_qtable, &moz_qtable) {
        println!("First 16 luma quant values:");
        println!("         zen  |  moz");
        let mut diffs = 0;
        for i in 0..64 {
            let zen_v = zen_q[i];
            let moz_v = moz_q[i];
            if zen_v != moz_v {
                diffs += 1;
            }
            if i < 16 {
                let diff = if zen_v != moz_v { " <--" } else { "" };
                println!("  [{:2}]: {:4} | {:4}{}", i, zen_v, moz_v, diff);
            }
        }
        println!("Total differences in quant table: {}", diffs);
    }

    // Check if zenjpeg is smaller but lower quality
    if zen.len() < moz.len() && zen_total_err > moz_total_err {
        println!();
        println!(
            "*** ISSUE: zenjpeg is {} bytes SMALLER but has {:.1}% MORE error ***",
            moz.len() - zen.len(),
            (zen_total_err as f64 / moz_total_err as f64 - 1.0) * 100.0
        );
        println!("This suggests over-aggressive quantization or trellis.");
    }
}

/// Extract first quantization table from JPEG data
fn extract_qtable(jpeg: &[u8]) -> Option<[u16; 64]> {
    // Find DQT marker (0xFF 0xDB)
    for i in 0..jpeg.len() - 2 {
        if jpeg[i] == 0xFF && jpeg[i + 1] == 0xDB {
            // Skip marker and length
            let table_start = i + 5; // marker(2) + length(2) + table_id(1)
            if table_start + 64 <= jpeg.len() {
                let mut table = [0u16; 64];
                // Read in zigzag order (as stored in JPEG)
                for j in 0..64 {
                    table[j] = jpeg[table_start + j] as u16;
                }
                return Some(table);
            }
        }
    }
    None
}
