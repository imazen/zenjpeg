//! Diagnose the root cause of zenjpeg vs reference decoder pixel differences.
//!
//! KEY FINDING: decode_to_ycbcr_f32 (coefficient path) has max diff = 1 vs mozjpeg
//! while default decode (streaming path) has max diff = 144. The bug is in the
//! streaming path's dequantization or data flow.
//!
//! Run: cargo test --release -p zenjpeg --test diagnose_decoder_diff -- --nocapture --ignored

use enough::Unstoppable;
use zenjpeg::foundation::consts::{DCT_BLOCK_SIZE, JPEG_NATURAL_ORDER};
use zenjpeg::quant::{dequantize_block_i32, dequantize_unzigzag_i32_into_partial};

/// Test that both dequantization approaches produce identical output.
/// If they differ, the streaming path's dequantization is the root cause.
#[test]
fn test_dequant_equivalence() {
    // Create test coefficients in zigzag order (simulating Huffman decoder output)
    let mut zigzag_coeffs = [0i16; 64];
    // DC coefficient
    zigzag_coeffs[0] = -50;
    // Some AC coefficients at various zigzag positions
    zigzag_coeffs[1] = 30;  // zigzag pos 1
    zigzag_coeffs[2] = -20; // zigzag pos 2
    zigzag_coeffs[3] = 15;  // zigzag pos 3
    zigzag_coeffs[10] = -8; // zigzag pos 10
    zigzag_coeffs[20] = 5;  // zigzag pos 20
    zigzag_coeffs[30] = -3; // zigzag pos 30

    // Create a quant table in natural order (typical values)
    let quant_natural: [u16; 64] = [
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99,
    ];
    let coeff_count = 31u8;

    // APPROACH 1: F32 path (unzigzag first, then dequantize)
    // This is what output.rs does for the coefficient path
    let mut natural_coeffs = [0i16; 64];
    for (i, &zi) in JPEG_NATURAL_ORDER[..64].iter().enumerate() {
        natural_coeffs[zi as usize] = zigzag_coeffs[i];
    }
    let dequant_f32_path = dequantize_block_i32(&natural_coeffs, &quant_natural);

    // APPROACH 2: Streaming path (combined dequantize + unzigzag)
    let mut dequant_streaming = [0i32; 64];
    dequantize_unzigzag_i32_into_partial(
        &zigzag_coeffs,
        &quant_natural,
        &mut dequant_streaming,
        coeff_count,
    );

    // Compare
    let mut max_diff = 0i32;
    let mut diff_count = 0;
    for i in 0..64 {
        let diff = (dequant_f32_path[i] - dequant_streaming[i]).abs();
        if diff > 0 {
            println!("  pos {i}: f32_path={}, streaming={}, diff={diff}",
                dequant_f32_path[i], dequant_streaming[i]);
            diff_count += 1;
        }
        max_diff = max_diff.max(diff);
    }

    if diff_count == 0 {
        println!("Dequantization approaches are IDENTICAL for coeff_count={coeff_count}");
    } else {
        println!("DIVERGENCE: {diff_count} positions differ, max_diff={max_diff}");
    }

    // Also test with full 64 coefficients
    let mut full_zigzag = [0i16; 64];
    let mut state = 42u64;
    for c in full_zigzag.iter_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *c = ((state >> 33) as i16) % 200;
    }

    let mut natural_full = [0i16; 64];
    for (i, &zi) in JPEG_NATURAL_ORDER[..64].iter().enumerate() {
        natural_full[zi as usize] = full_zigzag[i];
    }
    let dequant_f32_full = dequantize_block_i32(&natural_full, &quant_natural);

    let mut dequant_stream_full = [0i32; 64];
    dequantize_unzigzag_i32_into_partial(
        &full_zigzag,
        &quant_natural,
        &mut dequant_stream_full,
        64,
    );

    let mut max_diff_full = 0i32;
    for i in 0..64 {
        let diff = (dequant_f32_full[i] - dequant_stream_full[i]).abs();
        if diff > 0 {
            println!("  FULL pos {i}: f32_path={}, streaming={}, diff={diff}",
                dequant_f32_full[i], dequant_stream_full[i]);
        }
        max_diff_full = max_diff_full.max(diff);
    }

    if max_diff_full == 0 {
        println!("Full-block dequantization is IDENTICAL");
    } else {
        println!("Full-block DIVERGENCE: max_diff={max_diff_full}");
    }

    assert_eq!(max_diff, 0, "Dequantization approaches differ");
    assert_eq!(max_diff_full, 0, "Full dequantization approaches differ");
}

/// Compare `decode()` RGB output with scanline_reader RGB output on local test images.
/// If these differ, the streaming path (decode_baseline_streaming_rgb) is confirmed buggy.
#[test]
fn compare_decode_vs_scanline_rgb() {
    use imgref::ImgRefMut;

    let test_files = [
        ("444", "internal/jpegli-cpp/testdata/jxl/flower/flower.png.im_q85_444.jpg"),
        ("420", "internal/jpegli-cpp/testdata/jxl/flower/flower.png.im_q85_420.jpg"),
        ("422", "internal/jpegli-cpp/testdata/jxl/flower/flower.png.im_q85_422.jpg"),
        ("444_1x2", "internal/jpegli-cpp/testdata/jxl/flower/flower.png.im_q85_444_1x2.jpg"),
    ];

    for (label, rel_path) in &test_files {
        // Try to find the file relative to project root
        let path = format!("/home/lilith/work/zenjpeg/{rel_path}");
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Skipping {label}: {path}");
                continue;
            }
        };

        // Path 1: decode() — uses streaming path for baseline 3-component
        let d1 = zenjpeg::decoder::Decoder::new();
        let result1 = d1.decode(&data, Unstoppable).unwrap();
        let rgb1 = result1.pixels_u8().unwrap();
        let w = result1.width() as usize;
        let h = result1.height() as usize;

        // Path 2: scanline_reader → read_rows_rgb8
        let mut reader = zenjpeg::decoder::Decoder::new()
            .scanline_reader(&data)
            .unwrap();
        let sw = reader.width() as usize;
        let sh = reader.height() as usize;
        assert_eq!((w, h), (sw, sh), "{label}: dimension mismatch");

        let row_bytes = sw * 3;
        let mut rgb2 = vec![0u8; sh * row_bytes];
        let mut rows_read = 0;
        while rows_read < sh {
            let remaining = sh - rows_read;
            let batch = remaining.min(8);
            let buf = &mut rgb2[rows_read * row_bytes..(rows_read + batch) * row_bytes];
            let img = ImgRefMut::new(buf, row_bytes, batch);
            let got = reader.read_rows_rgb8(img).unwrap();
            if got == 0 { break; }
            rows_read += got;
        }

        // Compare per-channel
        let mut max_diff = [0i32; 3];
        let mut sum_diff = [0u64; 3];
        let mut worst_pos = [(0usize, 0usize); 3];
        let ch_names = ["R", "G", "B"];

        for py in 0..h {
            for px in 0..w {
                let idx = (py * w + px) * 3;
                for ch in 0..3 {
                    let v1 = rgb1[idx + ch] as i32;
                    let v2 = rgb2[idx + ch] as i32;
                    let d = (v1 - v2).abs();
                    sum_diff[ch] += d as u64;
                    if d > max_diff[ch] {
                        max_diff[ch] = d;
                        worst_pos[ch] = (px, py);
                    }
                }
            }
        }

        let total = (w * h) as f64;
        println!("{label} ({w}x{h}):");
        let mut any_diff = false;
        for ch in 0..3 {
            let mean = sum_diff[ch] as f64 / total;
            if max_diff[ch] > 1 {
                any_diff = true;
                println!("  {}: max={} mean={:.4} worst@({},{})",
                    ch_names[ch], max_diff[ch], mean, worst_pos[ch].0, worst_pos[ch].1);
            }
        }
        if !any_diff {
            println!("  MATCH (max diff <= 1 on all channels)");
        }

        // Dump first mismatched pixel for debugging
        if max_diff.iter().any(|&d| d > 5) {
            // Find first pixel with diff > 5
            'outer: for py in 0..h {
                for px in 0..w {
                    let idx = (py * w + px) * 3;
                    let dr = (rgb1[idx] as i32 - rgb2[idx] as i32).abs();
                    let dg = (rgb1[idx+1] as i32 - rgb2[idx+1] as i32).abs();
                    let db = (rgb1[idx+2] as i32 - rgb2[idx+2] as i32).abs();
                    if dr > 5 || dg > 5 || db > 5 {
                        println!("  First big diff at ({px},{py}): decode=({},{},{}) scanline=({},{},{})",
                            rgb1[idx], rgb1[idx+1], rgb1[idx+2],
                            rgb2[idx], rgb2[idx+1], rgb2[idx+2]);
                        break 'outer;
                    }
                }
            }
        }
    }
}

/// Decode with mozjpeg in RGB mode
fn decode_mozjpeg_rgb(data: &[u8]) -> (u32, u32, Vec<u8>) {
    use mozjpeg_sys::*;
    use std::mem;
    unsafe {
        let mut err: jpeg_error_mgr = mem::zeroed();
        jpeg_std_error(&mut err);
        let mut ci: jpeg_decompress_struct = mem::zeroed();
        ci.common.err = &mut err;
        jpeg_create_decompress(&mut ci);
        jpeg_mem_src(&mut ci, data.as_ptr(), data.len() as _);
        assert_eq!(jpeg_read_header(&mut ci, 1), 1);
        ci.out_color_space = J_COLOR_SPACE::JCS_RGB;
        jpeg_start_decompress(&mut ci);
        let w = ci.output_width as u32;
        let h = ci.output_height as u32;
        let stride = w as usize * ci.output_components as usize;
        let mut out = vec![0u8; h as usize * stride];
        while (ci.output_scanline as u32) < h {
            let off = ci.output_scanline as usize * stride;
            let mut p = out[off..].as_mut_ptr();
            jpeg_read_scanlines(&mut ci, &mut p, 1);
        }
        jpeg_finish_decompress(&mut ci);
        jpeg_destroy_decompress(&mut ci);
        (w, h, out)
    }
}

/// Compare decode(), scanline_reader, and mozjpeg for the 444_1x2 case
#[test]
fn compare_444_1x2_all_decoders() {
    use imgref::ImgRefMut;

    let path = "/home/lilith/work/zenjpeg/internal/jpegli-cpp/testdata/jxl/flower/flower.png.im_q85_444_1x2.jpg";
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Cannot read {path}: {e}");
            return;
        }
    };

    // Decode with mozjpeg (reference)
    let (mw, mh, moz_rgb) = decode_mozjpeg_rgb(&data);
    let w = mw as usize;
    let h = mh as usize;

    // Decode with zenjpeg decode()
    let d1 = zenjpeg::decoder::Decoder::new();
    let result1 = d1.decode(&data, Unstoppable).unwrap();
    let zen_rgb = result1.pixels_u8().unwrap();

    // Decode with zenjpeg scanline reader
    let mut reader = zenjpeg::decoder::Decoder::new()
        .scanline_reader(&data)
        .unwrap();
    let row_bytes = w * 3;
    let mut scan_rgb = vec![0u8; h * row_bytes];
    let mut rows_read = 0;
    while rows_read < h {
        let remaining = h - rows_read;
        let batch = remaining.min(8);
        let buf = &mut scan_rgb[rows_read * row_bytes..(rows_read + batch) * row_bytes];
        let img = ImgRefMut::new(buf, row_bytes, batch);
        let got = reader.read_rows_rgb8(img).unwrap();
        if got == 0 { break; }
        rows_read += got;
    }

    // Compare each decoder with mozjpeg
    for (name, test_rgb) in [("decode()", zen_rgb), ("scanline", &scan_rgb)] {
        let mut max_diff = [0i32; 3];
        let ch_names = ["R", "G", "B"];
        for py in 0..h {
            for px in 0..w {
                let idx = (py * w + px) * 3;
                for ch in 0..3 {
                    let d = (test_rgb[idx + ch] as i32 - moz_rgb[idx + ch] as i32).abs();
                    max_diff[ch] = max_diff[ch].max(d);
                }
            }
        }
        println!("{name} vs mozjpeg: R_max={} G_max={} B_max={}",
            max_diff[0], max_diff[1], max_diff[2]);
    }
}

/// Decode with mozjpeg in raw YCbCr mode
fn decode_mozjpeg_ycbcr(data: &[u8]) -> (u32, u32, Vec<u8>) {
    use mozjpeg_sys::*;
    use std::mem;
    unsafe {
        let mut err: jpeg_error_mgr = mem::zeroed();
        jpeg_std_error(&mut err);
        let mut ci: jpeg_decompress_struct = mem::zeroed();
        ci.common.err = &mut err;
        jpeg_create_decompress(&mut ci);
        jpeg_mem_src(&mut ci, data.as_ptr(), data.len() as _);
        assert_eq!(jpeg_read_header(&mut ci, 1), 1);
        ci.out_color_space = J_COLOR_SPACE::JCS_YCbCr;
        jpeg_start_decompress(&mut ci);
        let w = ci.output_width as u32;
        let h = ci.output_height as u32;
        let stride = w as usize * ci.output_components as usize;
        let mut out = vec![0u8; h as usize * stride];
        while (ci.output_scanline as u32) < h {
            let off = ci.output_scanline as usize * stride;
            let mut p = out[off..].as_mut_ptr();
            jpeg_read_scanlines(&mut ci, &mut p, 1);
        }
        jpeg_finish_decompress(&mut ci);
        jpeg_destroy_decompress(&mut ci);
        (w, h, out)
    }
}

/// Compare the coefficient path (correct) with the scanline path (buggy?)
/// using the native i16 YCbCr output.
#[test]
#[ignore = "requires corpus"]
fn compare_streaming_vs_coefficient_ycbcr() {
    let path = "/mnt/v/output/corpus-builder/wide-gamut/adobe-rgb/flickr_841c1e16a9a5484a.jpg";
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Cannot read {path}: {e}");
            return;
        }
    };

    println!("\n=== Comparing coefficient vs scanline path (adobe1) ===");

    // Path 1: Coefficient path (known correct — max diff 1 vs mozjpeg)
    let d = zenjpeg::decoder::Decoder::new();
    let mut ycbcr_f32 = d.decode_to_ycbcr_f32(&data, Unstoppable).unwrap();
    ycbcr_f32.shift_to_jpeg_range();
    let w = ycbcr_f32.width as usize;
    let h = ycbcr_f32.height as usize;

    // Path 2: Scanline reader path (streaming, same as default decode)
    let mut reader = zenjpeg::decoder::Decoder::new()
        .scanline_reader(&data)
        .unwrap();
    let info = reader.info();
    let sw = info.dimensions.width as usize;
    let sh = info.dimensions.height as usize;
    assert_eq!((w, h), (sw, sh));

    // Read all rows using scanline reader's native i16 output
    let y_stride = (sw + 15) & !15;
    let c_stride = y_stride; // 4:4:4
    let mcu_rows = (sh + 7) / 8;
    let mut y_buf = vec![0i16; y_stride * sh];
    let mut cb_buf = vec![0i16; c_stride * sh];
    let mut cr_buf = vec![0i16; c_stride * sh];

    let mut total_y_rows = 0;
    while total_y_rows < sh {
        let remaining = (sh - total_y_rows + 7) / 8;
        let (y_rows, _c_rows) = reader.read_rows_ycbcr_native_i16(
            &mut y_buf[total_y_rows * y_stride..],
            y_stride,
            &mut cb_buf[total_y_rows * c_stride..],
            &mut cr_buf[total_y_rows * c_stride..],
            c_stride,
            remaining.min(4),
        ).unwrap();
        if y_rows == 0 { break; }
        total_y_rows += y_rows;
    }

    // Compare coefficient path (f32) vs scanline path (i16) for each plane
    for (name, f32_plane, i16_buf, stride) in [
        ("Y", &ycbcr_f32.y, &y_buf, y_stride),
        ("Cb", &ycbcr_f32.cb, &cb_buf, c_stride),
        ("Cr", &ycbcr_f32.cr, &cr_buf, c_stride),
    ] {
        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f64;
        let mut worst_pos = (0usize, 0usize);
        let mut diff_count = 0u64;
        let total = (w * h) as f64;

        for py in 0..h {
            for px in 0..w {
                let f32_idx = py * w + px;
                let i16_idx = py * stride + px;
                let f32_val = f32_plane[f32_idx];
                let i16_val = i16_buf[i16_idx] as f32;
                let d = (f32_val - i16_val).abs();
                sum_diff += d as f64;
                if d > 0.5 { diff_count += 1; }
                if d > max_diff {
                    max_diff = d;
                    worst_pos = (px, py);
                }
            }
        }

        println!("{name}: max_diff={max_diff:.1} mean_diff={:.3} diff_pixels={diff_count} worst@({},{})",
            sum_diff / total, worst_pos.0, worst_pos.1);

        // Dump the worst block if diff > 2
        if max_diff > 2.0 {
            let bx = worst_pos.0 / 8;
            let by = worst_pos.1 / 8;
            println!("  Worst block ({bx},{by}):");
            for row in 0..8 {
                let py = by * 8 + row;
                if py >= h { break; }
                print!("    ");
                for col in 0..8 {
                    let px = bx * 8 + col;
                    if px >= w { break; }
                    let f32_idx = py * w + px;
                    let i16_idx = py * stride + px;
                    let fv = f32_plane[f32_idx];
                    let iv = i16_buf[i16_idx] as f32;
                    let d = fv - iv;
                    if d.abs() < 0.5 {
                        print!("{fv:>6.0}  ");
                    } else {
                        print!("{fv:>6.0}/{iv:>3.0} ");
                    }
                }
                println!();
            }
        }
    }

    // Also compare with mozjpeg YCbCr
    let (_, _, moz_ycbcr) = decode_mozjpeg_ycbcr(&data);
    println!("\nMozjpeg comparison of scanline i16 path:");
    for (name, i16_buf, stride, ch_off) in [
        ("Y", &y_buf, y_stride, 0),
        ("Cb", &cb_buf, c_stride, 1),
        ("Cr", &cr_buf, c_stride, 2),
    ] {
        let mut max_diff = 0i32;
        let mut sum_diff = 0u64;
        for py in 0..h {
            for px in 0..w {
                let i16_idx = py * stride + px;
                let moz_idx = (py * w + px) * 3 + ch_off;
                let iv = i16_buf[i16_idx] as i32;
                let mv = moz_ycbcr[moz_idx] as i32;
                let d = (iv - mv).abs();
                sum_diff += d as u64;
                max_diff = max_diff.max(d);
            }
        }
        let mean = sum_diff as f64 / (w * h) as f64;
        println!("  scanline {name} vs moz: max={max_diff} mean={mean:.3}");
    }
}

// =============================================================================
// Non-standard sampling factor tests
// =============================================================================

/// Helper: compare zenjpeg decode() and scanline_reader RGB output against mozjpeg.
/// Returns (decode_max, scanline_max) per channel.
fn compare_all_paths_rgb(data: &[u8]) -> ([i32; 3], [i32; 3]) {
    use imgref::ImgRefMut;

    // Reference: mozjpeg
    let (mw, mh, moz_rgb) = decode_mozjpeg_rgb(data);
    let w = mw as usize;
    let h = mh as usize;

    // Path 1: decode() (coefficient-based for non-streaming cases)
    let d = zenjpeg::decoder::Decoder::new();
    let result = d.decode(data, Unstoppable).unwrap();
    let dec_rgb = result.pixels_u8().unwrap();
    assert_eq!(result.width() as usize, w);
    assert_eq!(result.height() as usize, h);

    // Path 2: scanline reader
    let mut reader = zenjpeg::decoder::Decoder::new()
        .scanline_reader(data)
        .unwrap();
    assert_eq!(reader.width() as usize, w);
    assert_eq!(reader.height() as usize, h);

    let row_bytes = w * 3;
    let mut scan_rgb = vec![0u8; h * row_bytes];
    let mut rows_read = 0;
    while rows_read < h {
        let remaining = h - rows_read;
        let batch = remaining.min(8);
        let buf = &mut scan_rgb[rows_read * row_bytes..(rows_read + batch) * row_bytes];
        let img = ImgRefMut::new(buf, row_bytes, batch);
        let got = reader.read_rows_rgb8(img).unwrap();
        if got == 0 {
            break;
        }
        rows_read += got;
    }

    // Compute max diffs
    let mut dec_max = [0i32; 3];
    let mut scan_max = [0i32; 3];

    for py in 0..h {
        for px in 0..w {
            let idx = (py * w + px) * 3;
            for ch in 0..3 {
                let moz = moz_rgb[idx + ch] as i32;
                let d1 = (dec_rgb[idx + ch] as i32 - moz).abs();
                let d2 = (scan_rgb[idx + ch] as i32 - moz).abs();
                dec_max[ch] = dec_max[ch].max(d1);
                scan_max[ch] = scan_max[ch].max(d2);
            }
        }
    }

    (dec_max, scan_max)
}

/// Test all non-standard sampling modes found in the jpegli test corpus.
/// Verifies both decode() and scanline_reader produce correct output (max ≤ 4 vs mozjpeg).
///
/// Sampling modes tested:
/// - 444_1x2: all components (1,2) — 4:4:4 with 8×16 MCUs (previously max=113 bug)
/// - 440: Y(1,2) Cb(1,1) Cr(1,1) — vertical-only chroma subsampling
/// - asymmetric: Y(2,2) Cb(2,1) Cr(1,2) — all different factors
/// - luma_subsample: Y(1,1) Cb(2,2) Cr(2,2) — inverted subsampling (chroma > luma)
/// - rgb_subsample_blue: R(2,2) G(2,2) B(1,1) — RGB mode with blue subsampled
#[test]
fn test_nonstandard_sampling_all_paths() {
    let base = "/home/lilith/work/zenjpeg/internal/jpegli-cpp/testdata/jxl/flower";
    let test_cases: &[(&str, &str, i32)] = &[
        // (label, filename, max_allowed_diff)
        ("444_1x2", "flower.png.im_q85_444_1x2.jpg", 4),
        ("440", "flower.png.im_q85_440.jpg", 4),
        ("asymmetric", "flower.png.im_q85_asymmetric.jpg", 4),
        ("luma_subsample", "flower.png.im_q85_luma_subsample.jpg", 4),
        ("rgb_subsample_blue", "flower.png.im_q85_rgb_subsample_blue.jpg", 4),
    ];

    let mut all_passed = true;
    for (label, filename, max_allowed) in test_cases {
        let path = format!("{base}/{filename}");
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("  SKIP {label}: {path} not found");
                continue;
            }
        };

        let (dec_max, scan_max) = compare_all_paths_rgb(&data);
        let dec_worst = *dec_max.iter().max().unwrap();
        let scan_worst = *scan_max.iter().max().unwrap();

        let dec_ok = dec_worst <= *max_allowed;
        let scan_ok = scan_worst <= *max_allowed;
        let status = if dec_ok && scan_ok { "OK" } else { "FAIL" };

        println!(
            "{label:>20}: decode max={dec_worst:>3}  scanline max={scan_worst:>3}  [{status}]"
        );

        if !dec_ok {
            eprintln!(
                "  FAIL: {label} decode() max={dec_worst} exceeds threshold {max_allowed} (R={} G={} B={})",
                dec_max[0], dec_max[1], dec_max[2]
            );
            all_passed = false;
        }
        if !scan_ok {
            eprintln!(
                "  FAIL: {label} scanline max={scan_worst} exceeds threshold {max_allowed} (R={} G={} B={})",
                scan_max[0], scan_max[1], scan_max[2]
            );
            all_passed = false;
        }
    }

    assert!(all_passed, "Some non-standard sampling modes exceeded max diff threshold");
}

/// Test that decode() and scanline_reader produce consistent output for all standard
/// AND non-standard sampling modes (internal consistency, no mozjpeg reference needed).
#[test]
fn test_decode_scanline_consistency() {
    use imgref::ImgRefMut;

    let base = "/home/lilith/work/zenjpeg/internal/jpegli-cpp/testdata/jxl/flower";
    let test_cases: &[(&str, &str)] = &[
        // Standard modes
        ("444", "flower.png.im_q85_444.jpg"),
        ("420", "flower.png.im_q85_420.jpg"),
        ("422", "flower.png.im_q85_422.jpg"),
        // Non-standard same-sampling
        ("444_1x2", "flower.png.im_q85_444_1x2.jpg"),
        // Standard subsampled
        ("440", "flower.png.im_q85_440.jpg"),
        // Exotic
        ("asymmetric", "flower.png.im_q85_asymmetric.jpg"),
        ("luma_subsample", "flower.png.im_q85_luma_subsample.jpg"),
    ];

    let mut all_passed = true;
    for (label, filename) in test_cases {
        let path = format!("{base}/{filename}");
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("  SKIP {label}: not found");
                continue;
            }
        };

        // Path 1: decode()
        let d = zenjpeg::decoder::Decoder::new();
        let result = d.decode(&data, Unstoppable).unwrap();
        let dec_rgb = result.pixels_u8().unwrap();
        let w = result.width() as usize;
        let h = result.height() as usize;

        // Path 2: scanline reader
        let mut reader = zenjpeg::decoder::Decoder::new()
            .scanline_reader(&data)
            .unwrap();
        let row_bytes = w * 3;
        let mut scan_rgb = vec![0u8; h * row_bytes];
        let mut rows_read = 0;
        while rows_read < h {
            let remaining = h - rows_read;
            let batch = remaining.min(8);
            let buf = &mut scan_rgb[rows_read * row_bytes..(rows_read + batch) * row_bytes];
            let img = ImgRefMut::new(buf, row_bytes, batch);
            let got = reader.read_rows_rgb8(img).unwrap();
            if got == 0 {
                break;
            }
            rows_read += got;
        }

        // Compare
        let mut max_diff = 0i32;
        for i in 0..dec_rgb.len().min(scan_rgb.len()) {
            let d = (dec_rgb[i] as i32 - scan_rgb[i] as i32).abs();
            max_diff = max_diff.max(d);
        }

        // Allow ≤ 4 for upsampling filter differences between paths
        let ok = max_diff <= 4;
        let status = if ok { "OK" } else { "FAIL" };
        println!("{label:>20}: decode↔scanline max_diff={max_diff:>3}  [{status}]");

        if !ok {
            eprintln!("  FAIL: {label} decode↔scanline max_diff={max_diff} exceeds threshold 4");
            all_passed = false;
        }
    }

    assert!(all_passed, "Some modes have inconsistent decode vs scanline output");
}

/// Regression test: 444_1x2 scanline reader must match mozjpeg within IDCT rounding (max ≤ 3).
/// This was the original bug: scanline reader produced max=113 error for all-1x2 sampling
/// because StripProcessor classified it as S440 with undersized chroma buffers.
#[test]
fn regression_444_1x2_scanline_accuracy() {
    use imgref::ImgRefMut;

    let path = "/home/lilith/work/zenjpeg/internal/jpegli-cpp/testdata/jxl/flower/flower.png.im_q85_444_1x2.jpg";
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("SKIP: test image not found");
            return;
        }
    };

    let (mw, mh, moz_rgb) = decode_mozjpeg_rgb(&data);
    let w = mw as usize;
    let h = mh as usize;

    let mut reader = zenjpeg::decoder::Decoder::new()
        .scanline_reader(&data)
        .unwrap();
    let row_bytes = w * 3;
    let mut scan_rgb = vec![0u8; h * row_bytes];
    let mut rows_read = 0;
    while rows_read < h {
        let remaining = h - rows_read;
        let batch = remaining.min(8);
        let buf = &mut scan_rgb[rows_read * row_bytes..(rows_read + batch) * row_bytes];
        let img = ImgRefMut::new(buf, row_bytes, batch);
        let got = reader.read_rows_rgb8(img).unwrap();
        if got == 0 {
            break;
        }
        rows_read += got;
    }

    let mut max_diff = 0i32;
    for i in 0..moz_rgb.len().min(scan_rgb.len()) {
        let d = (moz_rgb[i] as i32 - scan_rgb[i] as i32).abs();
        max_diff = max_diff.max(d);
    }

    assert!(
        max_diff <= 3,
        "444_1x2 scanline reader regression: max_diff={max_diff} (expected ≤ 3, was 113 before fix)"
    );
}

/// Test the native i16 YCbCr output path for non-standard sampling.
/// Verifies that chroma planes have correct dimensions and plausible values.
#[test]
fn test_nonstandard_sampling_ycbcr_i16() {
    let base = "/home/lilith/work/zenjpeg/internal/jpegli-cpp/testdata/jxl/flower";

    // 444_1x2: all components same sampling → chroma should be full resolution
    let path = format!("{base}/flower.png.im_q85_444_1x2.jpg");
    let data = match std::fs::read(&path) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("SKIP: 444_1x2 not found");
            return;
        }
    };

    let mut reader = zenjpeg::decoder::Decoder::new()
        .scanline_reader(&data)
        .unwrap();
    let w = reader.width() as usize;
    let h = reader.height() as usize;

    // For 444_1x2: chroma is same resolution as luma
    let y_stride = (w + 31) & !31;
    let c_stride = y_stride;
    let mut y_buf = vec![0i16; y_stride * h];
    let mut cb_buf = vec![0i16; c_stride * h];
    let mut cr_buf = vec![0i16; c_stride * h];

    let mut total_y_rows = 0;
    let mut total_c_rows = 0;
    while total_y_rows < h {
        let remaining_mcu = ((h - total_y_rows) + 15) / 16; // MCU height = 16 for v_samp=2
        let (y_rows, c_rows) = reader
            .read_rows_ycbcr_native_i16(
                &mut y_buf[total_y_rows * y_stride..],
                y_stride,
                &mut cb_buf[total_c_rows * c_stride..],
                &mut cr_buf[total_c_rows * c_stride..],
                c_stride,
                remaining_mcu.min(2),
            )
            .unwrap();
        if y_rows == 0 {
            break;
        }
        total_y_rows += y_rows;
        total_c_rows += c_rows;
    }

    // Chroma rows should match luma rows for 444_1x2 (no subsampling)
    assert_eq!(
        total_y_rows, total_c_rows,
        "444_1x2: chroma rows ({total_c_rows}) should match luma rows ({total_y_rows})"
    );
    assert_eq!(total_y_rows, h, "Should decode all rows");

    // Verify chroma values are in plausible range [0, 255]
    // and not all zeros (which would indicate the buffer wasn't written)
    let mut cb_nonzero = 0u64;
    let mut cr_nonzero = 0u64;
    for py in 0..h {
        for px in 0..w {
            let idx = py * c_stride + px;
            let cb = cb_buf[idx];
            let cr = cr_buf[idx];
            assert!(
                (0..=255).contains(&cb),
                "Cb out of range at ({px},{py}): {cb}"
            );
            assert!(
                (0..=255).contains(&cr),
                "Cr out of range at ({px},{py}): {cr}"
            );
            if cb != 128 {
                cb_nonzero += 1;
            }
            if cr != 128 {
                cr_nonzero += 1;
            }
        }
    }

    let total = (w * h) as f64;
    assert!(
        cb_nonzero as f64 / total > 0.5,
        "Cb is mostly 128 ({cb_nonzero}/{} non-128) — likely not written",
        w * h
    );
    assert!(
        cr_nonzero as f64 / total > 0.5,
        "Cr is mostly 128 ({cr_nonzero}/{} non-128) — likely not written",
        w * h
    );
}
