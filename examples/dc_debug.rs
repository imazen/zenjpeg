//! Debug DC coefficient encoding
//!
//! Analyzes DCT and quantization behavior without accessing private modules.

fn main() {
    println!("=== DC Coefficient Debug ===\n");

    // Simulate DCT and quantization as our code does it
    println!("=== DC Dequantization Analysis ===");
    println!("For uniform block at pixel P:");
    println!("  1. Level shift: P - 128");
    println!("  2. DCT DC = (P - 128) * 8 (scaled by 8)");
    println!("  3. Descale: DC / 8 = P - 128");
    println!("  4. Quantize: round(DC / Q) where Q is DC quant value");
    println!("  5. Dequant: DC_q * Q");
    println!("  6. Inverse DCT: DC_dq + 128 = reconstructed pixel");
    println!();

    let dc_quant = 6i32; // Q80 DC quantization value
    println!("DC quant value at Q80: {}", dc_quant);
    println!();

    println!("Pixel | Level  | DCT DC | Descale | Quantized | Dequant | Reconstructed | Error");
    println!("{}", "-".repeat(90));

    for p in 0i32..=20 {
        let level_shifted = p - 128;
        let dct_dc = level_shifted * 8; // DCT of uniform block (scaled by 8)
        let descaled = (dct_dc + 4) >> 3; // Descale with rounding

        // Quantize with symmetric rounding
        let quantized = if descaled >= 0 {
            (descaled + dc_quant / 2) / dc_quant
        } else {
            (descaled - dc_quant / 2) / dc_quant
        };

        let dequant = quantized * dc_quant;
        let reconstructed = dequant + 128;
        let error = (p - reconstructed).abs();

        println!(
            "{:5} | {:6} | {:6} | {:7} | {:9} | {:7} | {:13} | {:5}",
            p, level_shifted, dct_dc, descaled, quantized, dequant, reconstructed, error
        );
    }

    // Now test what mozjpeg would do
    println!();
    println!("=== Checking Trellis Path ===");
    println!("In trellis, qtable is multiplied by 8:");
    println!();

    println!("Pixel | Level  | DCT DC | q*8 | Quantized | Dequant | Reconstructed | Error");
    println!("{}", "-".repeat(90));

    let q8 = dc_quant * 8; // Trellis multiplies qtable by 8
    for p in 0i32..=20 {
        let level_shifted = p - 128;
        let dct_dc = level_shifted * 8; // DCT scaled by 8

        // Trellis quantize: uses scaled q
        let quantized = if dct_dc >= 0 {
            (dct_dc + q8 / 2) / q8
        } else {
            (dct_dc - q8 / 2) / q8
        };

        // Dequant for reconstruction: multiplied by original q, not q*8
        // Because JPEG standard stores just the quantized value
        let dequant = quantized * dc_quant;
        let reconstructed = dequant + 128;
        let error = (p - reconstructed).abs();

        println!(
            "{:5} | {:6} | {:6} | {:3} | {:9} | {:7} | {:13} | {:5}",
            p, level_shifted, dct_dc, q8, quantized, dequant, reconstructed, error
        );
    }

    // Compare with actual encoded/decoded output
    println!();
    println!("=== Actual Encode/Decode Test (Gray) ===");

    // Create a single 8x8 block with uniform gray value
    for test_value in [0u8, 10, 20, 128] {
        let rgb: Vec<u8> = vec![test_value; 64 * 3];

        let zen = zenjpeg::Encoder::new()
            .quality(zenjpeg::Quality::Standard(80))
            .encode_rgb(&rgb, 8, 8)
            .unwrap();

        let zen_dec = jpeg_decoder::Decoder::new(&zen[..]).decode().unwrap();

        // Get first pixel
        let decoded = zen_dec[0];

        println!(
            "Gray {}: decoded as {} (error: {})",
            test_value,
            decoded,
            (test_value as i32 - decoded as i32).abs()
        );
    }

    // Test with RED color (R varies, G=B=0)
    println!();
    println!("=== Actual Encode/Decode Test (RED) ===");

    for test_red in [0u8, 10, 20, 50, 100, 200] {
        // Create uniform red block: (R, 0, 0)
        let mut rgb = Vec::with_capacity(64 * 3);
        for _ in 0..64 {
            rgb.push(test_red); // R
            rgb.push(0); // G
            rgb.push(0); // B
        }

        let zen = zenjpeg::Encoder::new()
            .quality(zenjpeg::Quality::Standard(80))
            .encode_rgb(&rgb, 8, 8)
            .unwrap();

        let moz = mozjpeg_oxide::Encoder::new()
            .quality(80)
            .subsampling(mozjpeg_oxide::Subsampling::S444)
            .encode_rgb(&rgb, 8, 8)
            .unwrap();

        let zen_dec = jpeg_decoder::Decoder::new(&zen[..]).decode().unwrap();
        let moz_dec = jpeg_decoder::Decoder::new(&moz[..]).decode().unwrap();

        println!(
            "Red {:3}: zen=({:3},{:3},{:3}) moz=({:3},{:3},{:3}) | zen_err={} moz_err={}",
            test_red,
            zen_dec[0],
            zen_dec[1],
            zen_dec[2],
            moz_dec[0],
            moz_dec[1],
            moz_dec[2],
            (test_red as i32 - zen_dec[0] as i32).abs(),
            (test_red as i32 - moz_dec[0] as i32).abs()
        );
    }

    // Calculate YCbCr for red values
    println!();
    println!("=== YCbCr Values for Red ===");
    println!("Red | Y calc | Cb calc | Cr calc");
    for test_red in [0u8, 10, 20, 50, 100, 200] {
        let r = test_red as f32;
        let y = 0.299 * r;
        let cb = -0.168736 * r + 128.0;
        let cr = 0.5 * r + 128.0;
        println!("{:3} | {:6.1} | {:7.1} | {:7.1}", test_red, y, cb, cr);
    }

    // Test with GRADIENT block (non-uniform - has AC coefficients)
    println!();
    println!("=== Gradient Block Test ===");

    // Create horizontal gradient: 0 to 248 across block
    let mut rgb_grad = Vec::with_capacity(64 * 3);
    for row in 0..8 {
        for col in 0..8 {
            let v = ((row * 8 + col) * 4) as u8;
            rgb_grad.push(v); // R
            rgb_grad.push(v); // G
            rgb_grad.push(v); // B
        }
    }

    let zen = zenjpeg::Encoder::new()
        .quality(zenjpeg::Quality::Standard(80))
        .encode_rgb(&rgb_grad, 8, 8)
        .unwrap();

    let moz = mozjpeg_oxide::Encoder::new()
        .quality(80)
        .subsampling(mozjpeg_oxide::Subsampling::S444)
        .encode_rgb(&rgb_grad, 8, 8)
        .unwrap();

    let zen_dec = jpeg_decoder::Decoder::new(&zen[..]).decode().unwrap();
    let moz_dec = jpeg_decoder::Decoder::new(&moz[..]).decode().unwrap();

    println!("First row of gradient block:");
    println!("Orig | zen | moz | zen_err | moz_err");
    let mut zen_total = 0i64;
    let mut moz_total = 0i64;
    for i in 0..8 {
        let orig = rgb_grad[i * 3];
        let zen_v = zen_dec[i * 3];
        let moz_v = moz_dec[i * 3];
        let zen_e = (orig as i32 - zen_v as i32).abs();
        let moz_e = (orig as i32 - moz_v as i32).abs();
        println!(
            "{:4} | {:3} | {:3} | {:7} | {:7}",
            orig, zen_v, moz_v, zen_e, moz_e
        );
    }

    for i in 0..rgb_grad.len() {
        zen_total += (rgb_grad[i] as i64 - zen_dec[i] as i64).abs();
        moz_total += (rgb_grad[i] as i64 - moz_dec[i] as i64).abs();
    }
    println!(
        "\nTotal gradient block error: zen={} moz={}",
        zen_total, moz_total
    );
    println!("File size: zen={} moz={}", zen.len(), moz.len());

    // Test multi-block image to see where errors accumulate
    println!();
    println!("=== Multi-block Test (16x16) ===");

    // 16x16 = 4 blocks total
    let width = 16usize;
    let height = 16usize;
    let mut rgb16 = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let v = ((y * width + x) * 1) as u8; // 0-255
            rgb16.push(v);
            rgb16.push(v);
            rgb16.push(v);
        }
    }

    let zen16 = zenjpeg::Encoder::new()
        .quality(zenjpeg::Quality::Standard(80))
        .encode_rgb(&rgb16, width, height)
        .unwrap();

    let zen16_fast = zenjpeg::Encoder::fastest()
        .quality(zenjpeg::Quality::Standard(80))
        .encode_rgb(&rgb16, width, height)
        .unwrap();

    let moz16 = mozjpeg_oxide::Encoder::new()
        .quality(80)
        .subsampling(mozjpeg_oxide::Subsampling::S444)
        .encode_rgb(&rgb16, width as u32, height as u32)
        .unwrap();

    let moz16_fast = mozjpeg_oxide::Encoder::fastest()
        .quality(80)
        .subsampling(mozjpeg_oxide::Subsampling::S444)
        .encode_rgb(&rgb16, width as u32, height as u32)
        .unwrap();

    let zen16_dec = jpeg_decoder::Decoder::new(&zen16[..]).decode().unwrap();
    let zen16_fast_dec = jpeg_decoder::Decoder::new(&zen16_fast[..])
        .decode()
        .unwrap();
    let moz16_dec = jpeg_decoder::Decoder::new(&moz16[..]).decode().unwrap();
    let moz16_fast_dec = jpeg_decoder::Decoder::new(&moz16_fast[..])
        .decode()
        .unwrap();

    let mut zen16_err = 0i64;
    let mut zen16_fast_err = 0i64;
    let mut moz16_err = 0i64;
    let mut moz16_fast_err = 0i64;
    for i in 0..rgb16.len() {
        zen16_err += (rgb16[i] as i64 - zen16_dec[i] as i64).abs();
        zen16_fast_err += (rgb16[i] as i64 - zen16_fast_dec[i] as i64).abs();
        moz16_err += (rgb16[i] as i64 - moz16_dec[i] as i64).abs();
        moz16_fast_err += (rgb16[i] as i64 - moz16_fast_dec[i] as i64).abs();
    }
    println!(
        "16x16 error: zen={} zen_fast={} moz={} moz_fast={}",
        zen16_err, zen16_fast_err, moz16_err, moz16_fast_err
    );
    println!(
        "16x16 size:  zen={} zen_fast={} moz={} moz_fast={}",
        zen16.len(),
        zen16_fast.len(),
        moz16.len(),
        moz16_fast.len()
    );

    // 64x64 test
    println!();
    println!("=== Larger Test (64x64) ===");

    let width = 64usize;
    let height = 64usize;
    let mut rgb64 = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let v = ((x + y) % 256) as u8;
            rgb64.push(v);
            rgb64.push(v);
            rgb64.push(v);
        }
    }

    let zen64 = zenjpeg::Encoder::new()
        .quality(zenjpeg::Quality::Standard(80))
        .encode_rgb(&rgb64, width, height)
        .unwrap();

    let moz64 = mozjpeg_oxide::Encoder::new()
        .quality(80)
        .subsampling(mozjpeg_oxide::Subsampling::S444)
        .encode_rgb(&rgb64, width as u32, height as u32)
        .unwrap();

    let zen64_dec = jpeg_decoder::Decoder::new(&zen64[..]).decode().unwrap();
    let moz64_dec = jpeg_decoder::Decoder::new(&moz64[..]).decode().unwrap();

    let mut zen64_err = 0i64;
    let mut moz64_err = 0i64;
    let mut zen64_max = 0i64;
    let mut moz64_max = 0i64;
    for i in 0..rgb64.len() {
        let ze = (rgb64[i] as i64 - zen64_dec[i] as i64).abs();
        let me = (rgb64[i] as i64 - moz64_dec[i] as i64).abs();
        zen64_err += ze;
        moz64_err += me;
        zen64_max = zen64_max.max(ze);
        moz64_max = moz64_max.max(me);
    }
    let n = (width * height * 3) as f64;
    println!(
        "64x64 MAE: zen={:.3} moz={:.3}",
        zen64_err as f64 / n,
        moz64_err as f64 / n
    );
    println!("64x64 max: zen={} moz={}", zen64_max, moz64_max);
    println!("64x64 size: zen={} moz={}", zen64.len(), moz64.len());
}
