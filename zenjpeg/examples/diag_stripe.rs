//! Diagnostic: dump per-row diffs at MCU boundaries to detect stripe artifacts.
//!
//! Compares zenjpeg (full decode, scanline, coefficient paths) against zune-jpeg,
//! jpeg-decoder, and mozjpeg-sys to identify systematic boundary bias.
//!
//! Tests MCU-aligned, non-aligned, progressive, and externally-encoded JPEGs.
//!
//! Run: cargo run --release -p zenjpeg --example diag_stripe

use enough::Unstoppable;
use zenjpeg::decode::{ChromaUpsampling, Decoder};
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn encode_420(pixels: &[u8], w: u32, h: u32, q: f32, progressive: bool) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
        .progressive(progressive)
        .allow_16bit_quant_tables(false);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(pixels, Unstoppable).unwrap();
    enc.finish().unwrap()
}

/// Encode using mozjpeg-sys (libjpeg-turbo with NASM SIMD).
unsafe fn encode_mozjpeg(pixels: &[u8], w: u32, h: u32, q: i32) -> Vec<u8> {
    unsafe {
        use mozjpeg_sys::*;
        use std::mem;

        let mut err: jpeg_error_mgr = mem::zeroed();
        jpeg_std_error(&mut err);

        let mut cinfo: jpeg_compress_struct = mem::zeroed();
        cinfo.common.err = &mut err;
        jpeg_create_compress(&mut cinfo);

        let mut outbuf: *mut u8 = std::ptr::null_mut();
        let mut outsize: u64 = 0;
        jpeg_mem_dest(&mut cinfo, &mut outbuf, &mut outsize);

        cinfo.image_width = w;
        cinfo.image_height = h;
        cinfo.input_components = 3;
        cinfo.in_color_space = J_COLOR_SPACE::JCS_RGB;

        jpeg_set_defaults(&mut cinfo);
        jpeg_set_quality(&mut cinfo, q, true as boolean);
        // Force 4:2:0
        (*cinfo.comp_info.offset(0)).h_samp_factor = 2;
        (*cinfo.comp_info.offset(0)).v_samp_factor = 2;
        (*cinfo.comp_info.offset(1)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(1)).v_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).h_samp_factor = 1;
        (*cinfo.comp_info.offset(2)).v_samp_factor = 1;

        jpeg_start_compress(&mut cinfo, true as boolean);

        let row_stride = w as usize * 3;
        while cinfo.next_scanline < h {
            let offset = cinfo.next_scanline as usize * row_stride;
            let row_ptr = pixels[offset..].as_ptr();
            jpeg_write_scanlines(&mut cinfo, &row_ptr, 1);
        }

        jpeg_finish_compress(&mut cinfo);
        let result = std::slice::from_raw_parts(outbuf, outsize as usize).to_vec();
        jpeg_destroy_compress(&mut cinfo);
        // Leak outbuf — short-lived diagnostic, no libc dep needed
        result
    }
}

fn decode_zen_full(jpeg: &[u8], up: ChromaUpsampling) -> Vec<u8> {
    Decoder::new()
        .chroma_upsampling(up)
        .auto_orient(false)
        .num_threads(1)
        .decode(jpeg, Unstoppable)
        .unwrap()
        .into_pixels_u8()
        .unwrap()
}

fn decode_zen_scanline(jpeg: &[u8], up: ChromaUpsampling) -> Vec<u8> {
    let mut reader = Decoder::new()
        .chroma_upsampling(up)
        .auto_orient(false)
        .num_threads(1)
        .scanline_reader(jpeg)
        .unwrap();
    let w = reader.width() as usize;
    let h = reader.height() as usize;
    let stride = w * 3;
    let mut px = vec![0u8; stride * h];
    let mut total = 0;
    while !reader.is_finished() {
        let rem = h - total;
        let output = imgref::ImgRefMut::new(&mut px[total * stride..], stride, rem);
        total += reader.read_rows_rgb8(output).unwrap();
    }
    px
}

fn decode_zune(jpeg: &[u8]) -> Vec<u8> {
    use zune_core::bytestream::ZCursor;
    let mut dec = zune_jpeg::JpegDecoder::new(ZCursor::new(jpeg));
    dec.decode().unwrap()
}

#[allow(dead_code)]
fn decode_jpd(jpeg: &[u8]) -> Vec<u8> {
    let mut dec = jpeg_decoder::Decoder::new(jpeg);
    dec.decode().unwrap()
}

/// Decode using mozjpeg-sys (libjpeg-turbo with NASM SIMD).
unsafe fn decode_mozjpeg(data: &[u8]) -> Vec<u8> {
    unsafe {
        use mozjpeg_sys::*;
        use std::mem;

        let mut err: jpeg_error_mgr = mem::zeroed();
        jpeg_std_error(&mut err);

        let mut cinfo: jpeg_decompress_struct = mem::zeroed();
        cinfo.common.err = &mut err;
        jpeg_create_decompress(&mut cinfo);

        jpeg_mem_src(&mut cinfo, data.as_ptr(), data.len() as _);
        jpeg_read_header(&mut cinfo, true as boolean);
        cinfo.out_color_space = J_COLOR_SPACE::JCS_RGB;
        jpeg_start_decompress(&mut cinfo);

        let width = cinfo.output_width as usize;
        let height = cinfo.output_height as usize;
        let components = cinfo.output_components as usize;
        let row_stride = width * components;

        let mut output = vec![0u8; height * row_stride];

        while (cinfo.output_scanline as usize) < height {
            let offset = cinfo.output_scanline as usize * row_stride;
            let mut row_ptr = output[offset..].as_mut_ptr();
            jpeg_read_scanlines(&mut cinfo, &mut row_ptr, 1);
        }

        jpeg_finish_decompress(&mut cinfo);
        jpeg_destroy_decompress(&mut cinfo);

        output
    }
}

fn row_max_diff(a: &[u8], b: &[u8], y: usize, w: usize) -> u32 {
    let start = y * w * 3;
    let end = start + w * 3;
    a[start..end]
        .iter()
        .zip(b[start..end].iter())
        .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs())
        .max()
        .unwrap_or(0)
}

fn row_mean_abs_diff(a: &[u8], b: &[u8], y: usize, w: usize) -> f64 {
    let start = y * w * 3;
    let end = start + w * 3;
    let sum: u64 = a[start..end]
        .iter()
        .zip(b[start..end].iter())
        .map(|(&a, &b)| (a as i64 - b as i64).unsigned_abs())
        .sum();
    sum as f64 / (w * 3) as f64
}

/// Count how many pixels in a row differ by more than `thresh`.
fn row_diff_count(a: &[u8], b: &[u8], y: usize, w: usize, thresh: u32) -> usize {
    let start = y * w * 3;
    let end = start + w * 3;
    a[start..end]
        .iter()
        .zip(b[start..end].iter())
        .filter(|&(&a, &b)| (a as i32 - b as i32).unsigned_abs() > thresh)
        .count()
}

#[allow(dead_code)]
struct AnalyzeResult {
    boundary_max: u32,
    interior_max: u32,
    stripe_detected: bool,
}

fn analyze(
    name: &str,
    w: usize,
    h: usize,
    zen: &[u8],
    zen_label: &str,
    ref_pixels: &[u8],
    ref_label: &str,
    verbose: bool,
) -> AnalyzeResult {
    let mut boundary_max = 0u32;
    let mut interior_max = 0u32;
    let mut boundary_mean = 0.0f64;
    let mut interior_mean = 0.0f64;
    let mut boundary_rows = 0u32;
    let mut interior_rows = 0u32;

    for y in 0..h {
        let in_mcu = y % 16;
        let is_boundary = in_mcu == 0 || in_mcu == 15;

        let max_d = row_max_diff(zen, ref_pixels, y, w);
        let mean_d = row_mean_abs_diff(zen, ref_pixels, y, w);

        if is_boundary {
            boundary_max = boundary_max.max(max_d);
            boundary_mean += mean_d;
            boundary_rows += 1;
        } else {
            interior_max = interior_max.max(max_d);
            interior_mean += mean_d;
            interior_rows += 1;
        }

        if verbose {
            let cnt_d = row_diff_count(zen, ref_pixels, y, w, 0);
            let note = match in_mcu {
                0 => " <-- TOP",
                15 => " <-- BOT",
                1 => " top+1",
                14 => " bot-1",
                _ => "",
            };
            let show = in_mcu <= 1 || in_mcu >= 14 || max_d > 1;
            if show {
                let flag = if max_d > interior_max.max(1) {
                    " ***"
                } else {
                    ""
                };
                println!("  {y:3} | max={max_d:2} mean={mean_d:.2} cnt={cnt_d:4} |{note}{flag}",);
            }
        }
    }

    if boundary_rows > 0 {
        boundary_mean /= boundary_rows as f64;
    }
    if interior_rows > 0 {
        interior_mean /= interior_rows as f64;
    }

    let stripe_detected = boundary_max > interior_max;
    let status = if stripe_detected {
        "*** STRIPE ***"
    } else {
        "OK"
    };
    println!(
        "  {name} {zen_label} vs {ref_label}: bnd_max={boundary_max} int_max={interior_max} bnd_mean={boundary_mean:.3} int_mean={interior_mean:.3} [{status}]"
    );

    AnalyzeResult {
        boundary_max,
        interior_max,
        stripe_detected,
    }
}

fn make_test_image(w: u32, h: u32) -> Vec<u8> {
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let idx = (y * w as usize + x) * 3;
            let bx = (x / 8) as u32;
            let by = (y / 8) as u32;
            let block_hash = bx
                .wrapping_mul(2654435761)
                .wrapping_add(by.wrapping_mul(40503));
            let block_type = block_hash % 4;
            let px = x as u32;
            let py = y as u32;
            let mut h_val = px
                .wrapping_mul(374761393)
                .wrapping_add(py.wrapping_mul(668265263));
            h_val = (h_val ^ (h_val >> 13)).wrapping_mul(1274126177);
            let n = (h_val >> 24) as u8;
            match block_type {
                0 => {
                    // High chroma contrast blocks
                    if (by % 2) == 0 {
                        pixels[idx] = 220;
                        pixels[idx + 1] = 50;
                        pixels[idx + 2] = 30;
                    } else {
                        pixels[idx] = 30;
                        pixels[idx + 1] = 50;
                        pixels[idx + 2] = 220;
                    }
                }
                1 => {
                    pixels[idx] = ((x * 255) / w as usize) as u8;
                    pixels[idx + 1] = ((y * 255) / h as usize) as u8;
                    pixels[idx + 2] = n >> 2;
                }
                2 => {
                    let edge = if (x % 8 < 4) ^ (y % 8 < 4) {
                        200u8
                    } else {
                        55u8
                    };
                    pixels[idx] = edge;
                    pixels[idx + 1] = edge.wrapping_add(n >> 4);
                    pixels[idx + 2] = 255 - edge;
                }
                _ => {
                    pixels[idx] = n;
                    pixels[idx + 1] = n.wrapping_mul(3);
                    pixels[idx + 2] = n.wrapping_mul(7);
                }
            }
        }
    }
    pixels
}

fn test_config(
    label: &str,
    jpeg: &[u8],
    w: usize,
    h: usize,
    up: ChromaUpsampling,
    verbose: bool,
) -> bool {
    println!("\n--- {label} ({w}x{h}, {:?}) ---", up);

    let zune = decode_zune(jpeg);
    let moz = unsafe { decode_mozjpeg(jpeg) };

    // Check reference agreement
    let ref_max = (0..h)
        .map(|y| row_max_diff(&zune, &moz, y, w))
        .max()
        .unwrap();
    println!("  Reference agreement: zune-moz max={ref_max}");

    let mut any_stripe = false;

    // Full decode (streaming for baseline, coefficient for progressive)
    let zen_full = decode_zen_full(jpeg, up);
    let r = analyze(label, w, h, &zen_full, "full", &zune, "zune", verbose);
    any_stripe |= r.stripe_detected;
    let r = analyze(label, w, h, &zen_full, "full", &moz, "moz", verbose);
    any_stripe |= r.stripe_detected;

    // Scanline decode
    let zen_scan = decode_zen_scanline(jpeg, up);
    let r = analyze(label, w, h, &zen_scan, "scanline", &zune, "zune", false);
    any_stripe |= r.stripe_detected;

    // Check full vs scanline consistency
    let fs_max: u32 = (0..h)
        .map(|y| row_max_diff(&zen_full, &zen_scan, y, w))
        .max()
        .unwrap();
    if fs_max > 0 {
        println!("  *** full vs scanline: max_diff={fs_max} ***");
        any_stripe = true;
    }

    any_stripe
}

fn main() {
    let mut total_stripes = 0u32;
    let mut total_tests = 0u32;

    // ======== MCU-aligned dimensions ========
    println!("\n========== MCU-ALIGNED DIMENSIONS ==========");
    for (w, h) in [(128, 128), (256, 256), (512, 512)] {
        let pixels = make_test_image(w, h);
        let jpeg = encode_420(&pixels, w, h, 85.0, false);
        total_tests += 1;
        if test_config(
            "zen-enc baseline",
            &jpeg,
            w as usize,
            h as usize,
            ChromaUpsampling::Triangle,
            false,
        ) {
            total_stripes += 1;
        }
    }

    // ======== Non-MCU-aligned dimensions (CRITICAL) ========
    println!("\n\n========== NON-MCU-ALIGNED DIMENSIONS ==========");
    for (w, h) in [
        (100, 100),
        (127, 127),
        (129, 129),
        (255, 255),
        (100, 200),
        (200, 100),
        (97, 63),
    ] {
        let pixels = make_test_image(w, h);
        let jpeg = encode_420(&pixels, w, h, 85.0, false);
        total_tests += 1;
        if test_config(
            "zen-enc baseline",
            &jpeg,
            w as usize,
            h as usize,
            ChromaUpsampling::Triangle,
            w <= 129,
        ) {
            total_stripes += 1;
        }
    }

    // ======== Progressive 4:2:0 ========
    println!("\n\n========== PROGRESSIVE 4:2:0 ==========");
    for (w, h) in [(128, 128), (255, 255)] {
        let pixels = make_test_image(w, h);
        let jpeg = encode_420(&pixels, w, h, 85.0, true);
        total_tests += 1;
        if test_config(
            "zen-enc progressive",
            &jpeg,
            w as usize,
            h as usize,
            ChromaUpsampling::Triangle,
            true,
        ) {
            total_stripes += 1;
        }
    }

    // ======== mozjpeg-encoded JPEGs decoded by zenjpeg ========
    println!("\n\n========== MOZJPEG-ENCODED (external encoder) ==========");
    for (w, h) in [(128, 128), (255, 255), (512, 512)] {
        let pixels = make_test_image(w, h);
        let jpeg = unsafe { encode_mozjpeg(&pixels, w, h, 85) };
        total_tests += 1;
        if test_config(
            "moz-enc baseline",
            &jpeg,
            w as usize,
            h as usize,
            ChromaUpsampling::Triangle,
            w <= 255,
        ) {
            total_stripes += 1;
        }
    }

    // ======== LibjpegCompat upsampling mode ========
    println!("\n\n========== LIBJPEG-COMPAT UPSAMPLING ==========");
    for (w, h) in [(128, 128), (255, 255)] {
        let pixels = make_test_image(w, h);
        let jpeg = encode_420(&pixels, w, h, 85.0, false);
        total_tests += 1;
        if test_config(
            "zen-enc baseline",
            &jpeg,
            w as usize,
            h as usize,
            ChromaUpsampling::Triangle,
            true,
        ) {
            total_stripes += 1;
        }
    }

    // ======== Low quality (higher quant = bigger chroma changes) ========
    println!("\n\n========== LOW QUALITY (Q50) ==========");
    for (w, h) in [(128, 128), (255, 255)] {
        let pixels = make_test_image(w, h);
        let jpeg = encode_420(&pixels, w, h, 50.0, false);
        total_tests += 1;
        if test_config(
            "zen-enc Q50",
            &jpeg,
            w as usize,
            h as usize,
            ChromaUpsampling::Triangle,
            false,
        ) {
            total_stripes += 1;
        }
    }

    // ======== Summary ========
    println!("\n\n========== FINAL SUMMARY ==========");
    println!("Tests run: {total_tests}");
    println!("Stripe patterns detected: {total_stripes}");
    if total_stripes > 0 {
        println!("*** STRIPE BUG IS PRESENT ***");
    } else {
        println!("No stripe patterns detected across all test configurations.");
    }
}
