//! Diagnose the root cause of zenjpeg vs reference decoder pixel differences.
//!
//! KEY FINDING: decode_to_ycbcr_f32 (coefficient path) has max diff = 1 vs mozjpeg
//! while default decode (streaming path) has max diff = 144. The bug is in the
//! streaming path's dequantization or data flow.
//!
//! Run: cargo test --release -p zenjpeg --test diagnose_decoder_diff -- --nocapture --ignored

use enough::Unstoppable;
use zenjpeg::foundation::consts::JPEG_NATURAL_ORDER;
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
    zigzag_coeffs[1] = 30; // zigzag pos 1
    zigzag_coeffs[2] = -20; // zigzag pos 2
    zigzag_coeffs[3] = 15; // zigzag pos 3
    zigzag_coeffs[10] = -8; // zigzag pos 10
    zigzag_coeffs[20] = 5; // zigzag pos 20
    zigzag_coeffs[30] = -3; // zigzag pos 30

    // Create a quant table in natural order (typical values)
    let quant_natural: [u16; 64] = [
        16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69,
        56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81,
        104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
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
            println!(
                "  pos {i}: f32_path={}, streaming={}, diff={diff}",
                dequant_f32_path[i], dequant_streaming[i]
            );
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
            println!(
                "  FULL pos {i}: f32_path={}, streaming={}, diff={diff}",
                dequant_f32_full[i], dequant_stream_full[i]
            );
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
        (
            "444",
            "internal/jpegli-cpp/testdata/jxl/flower/flower.png.im_q85_444.jpg",
        ),
        (
            "420",
            "internal/jpegli-cpp/testdata/jxl/flower/flower.png.im_q85_420.jpg",
        ),
        (
            "422",
            "internal/jpegli-cpp/testdata/jxl/flower/flower.png.im_q85_422.jpg",
        ),
        (
            "444_1x2",
            "internal/jpegli-cpp/testdata/jxl/flower/flower.png.im_q85_444_1x2.jpg",
        ),
    ];

    for (label, rel_path) in &test_files {
        // Try to find the file relative to project root
        let path = zenjpeg_bench_utils::workspace_root()
            .join(rel_path)
            .display()
            .to_string();
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
            if got == 0 {
                break;
            }
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
                println!(
                    "  {}: max={} mean={:.4} worst@({},{})",
                    ch_names[ch], max_diff[ch], mean, worst_pos[ch].0, worst_pos[ch].1
                );
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
                    let dg = (rgb1[idx + 1] as i32 - rgb2[idx + 1] as i32).abs();
                    let db = (rgb1[idx + 2] as i32 - rgb2[idx + 2] as i32).abs();
                    if dr > 5 || dg > 5 || db > 5 {
                        println!(
                            "  First big diff at ({px},{py}): decode=({},{},{}) scanline=({},{},{})",
                            rgb1[idx],
                            rgb1[idx + 1],
                            rgb1[idx + 2],
                            rgb2[idx],
                            rgb2[idx + 1],
                            rgb2[idx + 2]
                        );
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
        let w = ci.output_width;
        let h = ci.output_height;
        let stride = w as usize * ci.output_components as usize;
        let mut out = vec![0u8; h as usize * stride];
        while ci.output_scanline < h {
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

    let path =
        zenjpeg_bench_utils::jpegli_testdata_dir().join("jxl/flower/flower.png.im_q85_444_1x2.jpg");
    let data = match std::fs::read(&path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Cannot read {}: {e}", path.display());
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
        if got == 0 {
            break;
        }
        rows_read += got;
    }

    // Compare each decoder with mozjpeg
    for (name, test_rgb) in [("decode()", zen_rgb), ("scanline", &scan_rgb)] {
        let mut max_diff = [0i32; 3];
        for py in 0..h {
            for px in 0..w {
                let idx = (py * w + px) * 3;
                for ch in 0..3 {
                    let d = (test_rgb[idx + ch] as i32 - moz_rgb[idx + ch] as i32).abs();
                    max_diff[ch] = max_diff[ch].max(d);
                }
            }
        }
        println!(
            "{name} vs mozjpeg: R_max={} G_max={} B_max={}",
            max_diff[0], max_diff[1], max_diff[2]
        );
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
        let w = ci.output_width;
        let h = ci.output_height;
        let stride = w as usize * ci.output_components as usize;
        let mut out = vec![0u8; h as usize * stride];
        while ci.output_scanline < h {
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
    let path = zenjpeg_bench_utils::corpus_builder_dir()
        .join("wide-gamut/adobe-rgb/flickr_841c1e16a9a5484a.jpg");
    let path = path.to_str().unwrap();
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
    let mut y_buf = vec![0i16; y_stride * sh];
    let mut cb_buf = vec![0i16; c_stride * sh];
    let mut cr_buf = vec![0i16; c_stride * sh];

    let mut total_y_rows = 0;
    while total_y_rows < sh {
        let remaining = (sh - total_y_rows + 7) / 8;
        let (y_rows, _c_rows) = reader
            .read_rows_ycbcr_native_i16(
                &mut y_buf[total_y_rows * y_stride..],
                y_stride,
                &mut cb_buf[total_y_rows * c_stride..],
                &mut cr_buf[total_y_rows * c_stride..],
                c_stride,
                remaining.min(4),
            )
            .unwrap();
        if y_rows == 0 {
            break;
        }
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
                if d > 0.5 {
                    diff_count += 1;
                }
                if d > max_diff {
                    max_diff = d;
                    worst_pos = (px, py);
                }
            }
        }

        println!(
            "{name}: max_diff={max_diff:.1} mean_diff={:.3} diff_pixels={diff_count} worst@({},{})",
            sum_diff / total,
            worst_pos.0,
            worst_pos.1
        );

        // Dump the worst block if diff > 2
        if max_diff > 2.0 {
            let bx = worst_pos.0 / 8;
            let by = worst_pos.1 / 8;
            println!("  Worst block ({bx},{by}):");
            for row in 0..8 {
                let py = by * 8 + row;
                if py >= h {
                    break;
                }
                print!("    ");
                for col in 0..8 {
                    let px = bx * 8 + col;
                    if px >= w {
                        break;
                    }
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

/// Compare streaming decode RGB output vs manually-computed RGB from scanline YCbCr.
/// This isolates whether the bug is in SIMD color conversion or in the streaming IDCT data.
#[test]
#[ignore = "requires corpus"]
fn compare_streaming_rgb_vs_manual_scalar_rgb() {
    let path = zenjpeg_bench_utils::corpus_builder_dir()
        .join("wide-gamut/adobe-rgb/flickr_841c1e16a9a5484a.jpg");
    let data = match std::fs::read(&path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Cannot read {}: {e}", path.display());
            return;
        }
    };

    println!("\n=== Comparing streaming RGB vs manual scalar RGB ===");

    // Path 1: decode() — uses streaming path with SIMD color conversion
    let d = zenjpeg::decoder::Decoder::new();
    let result = d.decode(&data, Unstoppable).unwrap();
    let streaming_rgb = result.pixels_u8().unwrap();
    let w = result.width() as usize;
    let h = result.height() as usize;

    // Path 2: scanline reader → native i16 YCbCr → manual scalar RGB
    let mut reader = zenjpeg::decoder::Decoder::new()
        .scanline_reader(&data)
        .unwrap();
    let sw = reader.width() as usize;
    let sh = reader.height() as usize;
    assert_eq!((w, h), (sw, sh));

    let y_stride = (sw + 15) & !15;
    let c_stride = y_stride;
    let mut y_buf = vec![0i16; y_stride * sh];
    let mut cb_buf = vec![0i16; c_stride * sh];
    let mut cr_buf = vec![0i16; c_stride * sh];

    let mut total_y_rows = 0;
    while total_y_rows < sh {
        let remaining = (sh - total_y_rows + 7) / 8;
        let (y_rows, _c_rows) = reader
            .read_rows_ycbcr_native_i16(
                &mut y_buf[total_y_rows * y_stride..],
                y_stride,
                &mut cb_buf[total_y_rows * c_stride..],
                &mut cr_buf[total_y_rows * c_stride..],
                c_stride,
                remaining.min(4),
            )
            .unwrap();
        if y_rows == 0 {
            break;
        }
        total_y_rows += y_rows;
    }

    // Manual scalar RGB conversion using zenjpeg's exact constants
    let mut manual_rgb = vec![0u8; w * h * 3];
    for py in 0..h {
        for px in 0..w {
            let yi = py * y_stride + px;
            let ci = py * c_stride + px;
            let y_val = y_buf[yi] as i32;
            let cb_val = cb_buf[ci] as i32 - 128;
            let cr_val = cr_buf[ci] as i32 - 128;
            let y_scaled = y_val * 16384 + 8192;
            let r = ((y_scaled + cr_val * 22970) >> 14).clamp(0, 255) as u8;
            let g = ((y_scaled + cr_val * -11700 + cb_val * -5638) >> 14).clamp(0, 255) as u8;
            let b = ((y_scaled + cb_val * 29032) >> 14).clamp(0, 255) as u8;
            let o = (py * w + px) * 3;
            manual_rgb[o] = r;
            manual_rgb[o + 1] = g;
            manual_rgb[o + 2] = b;
        }
    }

    // Also get mozjpeg RGB for reference
    let (_, _, moz_rgb) = decode_mozjpeg_rgb(&data);

    // Compare streaming vs manual
    let mut max_stream_manual = 0i32;
    let mut max_stream_moz = 0i32;
    let mut max_manual_moz = 0i32;
    let mut worst_stream_manual_pos = (0usize, 0usize);
    for py in 0..h {
        for px in 0..w {
            let i = (py * w + px) * 3;
            for ch in 0..3 {
                let s = streaming_rgb[i + ch] as i32;
                let m = manual_rgb[i + ch] as i32;
                let z = moz_rgb[i + ch] as i32;
                let d_sm = (s - m).abs();
                let d_sz = (s - z).abs();
                let d_mz = (m - z).abs();
                if d_sm > max_stream_manual {
                    max_stream_manual = d_sm;
                    worst_stream_manual_pos = (px, py);
                }
                max_stream_moz = max_stream_moz.max(d_sz);
                max_manual_moz = max_manual_moz.max(d_mz);
            }
        }
    }

    println!("streaming RGB vs manual scalar RGB: max={max_stream_manual}");
    println!("streaming RGB vs mozjpeg RGB:       max={max_stream_moz}");
    println!("manual scalar vs mozjpeg RGB:       max={max_manual_moz}");

    // Dump worst pixel
    let (wpx, wpy) = worst_stream_manual_pos;
    let i = (wpy * w + wpx) * 3;
    let yi = wpy * y_stride + wpx;
    let ci = wpy * c_stride + wpx;
    println!(
        "\nWorst pixel at ({wpx},{wpy}): Y={} Cb={} Cr={}",
        y_buf[yi], cb_buf[ci], cr_buf[ci]
    );
    println!(
        "  streaming: ({}, {}, {})",
        streaming_rgb[i],
        streaming_rgb[i + 1],
        streaming_rgb[i + 2]
    );
    println!(
        "  manual:    ({}, {}, {})",
        manual_rgb[i],
        manual_rgb[i + 1],
        manual_rgb[i + 2]
    );
    println!(
        "  mozjpeg:   ({}, {}, {})",
        moz_rgb[i],
        moz_rgb[i + 1],
        moz_rgb[i + 2]
    );
}

/// Test the SIMD color conversion in isolation by passing correct i16 YCbCr through it.
#[test]
#[ignore = "requires corpus"]
fn test_simd_color_conversion_in_isolation() {
    use zenjpeg::color::ycbcr::ycbcr_planes_i16_to_rgb_u8;

    let path = zenjpeg_bench_utils::corpus_builder_dir()
        .join("wide-gamut/adobe-rgb/flickr_841c1e16a9a5484a.jpg");
    let data = match std::fs::read(&path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Cannot read {}: {e}", path.display());
            return;
        }
    };

    println!("\n=== Testing SIMD color conversion in isolation ===");

    // Get correct YCbCr from scanline reader
    let mut reader = zenjpeg::decoder::Decoder::new()
        .scanline_reader(&data)
        .unwrap();
    let w = reader.width() as usize;
    let h = reader.height() as usize;

    let stride = (w + 15) & !15;
    let mut y_buf = vec![0i16; stride * h];
    let mut cb_buf = vec![0i16; stride * h];
    let mut cr_buf = vec![0i16; stride * h];

    let mut total_rows = 0;
    while total_rows < h {
        let remaining = (h - total_rows + 7) / 8;
        let (y_rows, _) = reader
            .read_rows_ycbcr_native_i16(
                &mut y_buf[total_rows * stride..],
                stride,
                &mut cb_buf[total_rows * stride..],
                &mut cr_buf[total_rows * stride..],
                stride,
                remaining.min(4),
            )
            .unwrap();
        if y_rows == 0 {
            break;
        }
        total_rows += y_rows;
    }

    // Convert using SIMD path (ycbcr_planes_i16_to_rgb_u8)
    let mut simd_rgb = vec![0u8; w * h * 3];
    for py in 0..h {
        let src_off = py * stride;
        let dst_off = py * w * 3;
        ycbcr_planes_i16_to_rgb_u8(
            &y_buf[src_off..src_off + w],
            &cb_buf[src_off..src_off + w],
            &cr_buf[src_off..src_off + w],
            &mut simd_rgb[dst_off..dst_off + w * 3],
        );
    }

    // Convert using manual scalar
    let mut scalar_rgb = vec![0u8; w * h * 3];
    for py in 0..h {
        for px in 0..w {
            let si = py * stride + px;
            let y_val = y_buf[si] as i32;
            let cb_val = cb_buf[si] as i32 - 128;
            let cr_val = cr_buf[si] as i32 - 128;
            let y_scaled = y_val * 16384 + 8192;
            let r = ((y_scaled + cr_val * 22970) >> 14).clamp(0, 255) as u8;
            let g = ((y_scaled + cr_val * -11700 + cb_val * -5638) >> 14).clamp(0, 255) as u8;
            let b = ((y_scaled + cb_val * 29032) >> 14).clamp(0, 255) as u8;
            let o = (py * w + px) * 3;
            scalar_rgb[o] = r;
            scalar_rgb[o + 1] = g;
            scalar_rgb[o + 2] = b;
        }
    }

    // Compare SIMD vs scalar
    let mut max_diff = 0i32;
    let mut worst_pos = (0usize, 0usize);
    for py in 0..h {
        for px in 0..w {
            let i = (py * w + px) * 3;
            for ch in 0..3 {
                let d = (simd_rgb[i + ch] as i32 - scalar_rgb[i + ch] as i32).abs();
                if d > max_diff {
                    max_diff = d;
                    worst_pos = (px, py);
                }
            }
        }
    }
    println!("SIMD vs scalar color conversion: max={max_diff}");

    // Compare SIMD vs mozjpeg RGB
    let (_, _, moz_rgb) = decode_mozjpeg_rgb(&data);
    let mut max_simd_moz = 0i32;
    for i in 0..simd_rgb.len() {
        let d = (simd_rgb[i] as i32 - moz_rgb[i] as i32).abs();
        max_simd_moz = max_simd_moz.max(d);
    }
    println!("SIMD(correct YCbCr) vs mozjpeg RGB: max={max_simd_moz}");

    if max_diff > 1 {
        let (wpx, wpy) = worst_pos;
        let i = (wpy * w + wpx) * 3;
        let si = wpy * stride + wpx;
        println!(
            "Worst at ({wpx},{wpy}): Y={} Cb={} Cr={}",
            y_buf[si], cb_buf[si], cr_buf[si]
        );
        println!(
            "  SIMD:   ({}, {}, {})",
            simd_rgb[i],
            simd_rgb[i + 1],
            simd_rgb[i + 2]
        );
        println!(
            "  scalar: ({}, {}, {})",
            scalar_rgb[i],
            scalar_rgb[i + 1],
            scalar_rgb[i + 2]
        );
    }
}

/// Compare scanline reader RGB8 vs decode() streaming RGB on a STANDARD sRGB 4:4:4 JPEG.
/// If this also shows diffs, the bug is not specific to wide-gamut images.
#[test]
fn compare_scanline_vs_streaming_standard_444() {
    use imgref::ImgRefMut;

    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fuzz/corpus/seed/flower_444.jpg");
    let data = std::fs::read(&path).unwrap();

    println!("\n=== Comparing streaming vs scanline on flower_444.jpg ===");

    // Path 1: decode() — streaming
    let d = zenjpeg::decoder::Decoder::new();
    let result = d.decode(&data, Unstoppable).unwrap();
    let streaming_rgb = result.pixels_u8().unwrap();
    let w = result.width() as usize;
    let h = result.height() as usize;
    println!("Image: {w}x{h}");

    // Path 2: scanline reader → RGB8
    let mut reader = zenjpeg::decoder::Decoder::new()
        .scanline_reader(&data)
        .unwrap();
    let row_stride = w * 3;
    let mut scanline_rgb = vec![0u8; row_stride * h];
    let mut total_rows = 0;
    while total_rows < h {
        let remaining = h - total_rows;
        let batch = remaining.min(8);
        let out_slice = &mut scanline_rgb[total_rows * row_stride..];
        let img = ImgRefMut::new_stride(out_slice, w * 3, batch, row_stride);
        let rows = reader.read_rows_rgb8(img).unwrap();
        if rows == 0 {
            break;
        }
        total_rows += rows;
    }

    // Compare
    let mut max_diff = 0i32;
    let mut diff_count = 0usize;
    for i in 0..streaming_rgb.len() {
        let d = (streaming_rgb[i] as i32 - scanline_rgb[i] as i32).abs();
        if d > 0 {
            diff_count += 1;
        }
        max_diff = max_diff.max(d);
    }
    println!("streaming vs scanline: max={max_diff} diff_pixels={diff_count}");
    assert!(
        max_diff <= 1,
        "streaming vs scanline should match within ±1 for standard 4:4:4"
    );
}

/// Compare Jpegli vs Libjpeg IDCT on an Adobe RGB image.
#[test]
#[ignore = "requires corpus"]
fn compare_streaming_libjpeg_idct_adobe_rgb() {
    use imgref::ImgRefMut;

    let path = zenjpeg_bench_utils::corpus_builder_dir()
        .join("wide-gamut/adobe-rgb/flickr_841c1e16a9a5484a.jpg");
    let data = match std::fs::read(&path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Cannot read {}: {e}", path.display());
            return;
        }
    };

    println!("\n=== Streaming with Libjpeg IDCT on Adobe RGB image ===");

    // Default IDCT (Jpegli) streaming — disable ICC to compare raw decode
    let d = zenjpeg::decoder::Decoder::new().apply_icc(false);
    let result_jpegli = d.decode(&data, Unstoppable).unwrap();
    let jpegli_rgb = result_jpegli.pixels_u8().unwrap();
    let w = result_jpegli.width() as usize;
    let h = result_jpegli.height() as usize;

    // Libjpeg IDCT streaming — disable ICC to compare raw decode
    let d = zenjpeg::decoder::Decoder::new()
        .apply_icc(false)
        .idct_method(zenjpeg::decode::IdctMethod::Libjpeg);
    let result_lj = d.decode(&data, Unstoppable).unwrap();
    let lj_rgb = result_lj.pixels_u8().unwrap();

    // Libjpeg IDCT scanline reader
    let mut reader = zenjpeg::decoder::Decoder::new()
        .idct_method(zenjpeg::decode::IdctMethod::Libjpeg)
        .scanline_reader(&data)
        .unwrap();
    let row_stride = w * 3;
    let mut scanline_rgb = vec![0u8; row_stride * h];
    let mut total_rows = 0;
    while total_rows < h {
        let remaining = h - total_rows;
        let batch = remaining.min(8);
        let out_slice = &mut scanline_rgb[total_rows * row_stride..];
        let img = ImgRefMut::new_stride(out_slice, w * 3, batch, row_stride);
        let rows = reader.read_rows_rgb8(img).unwrap();
        if rows == 0 {
            break;
        }
        total_rows += rows;
    }

    // mozjpeg reference
    let (_, _, moz_rgb) = decode_mozjpeg_rgb(&data);

    // Compare streaming Jpegli vs Libjpeg
    let mut max_jpegli_lj = 0i32;
    for i in 0..jpegli_rgb.len() {
        max_jpegli_lj = max_jpegli_lj.max((jpegli_rgb[i] as i32 - lj_rgb[i] as i32).abs());
    }
    println!("streaming Jpegli IDCT vs Libjpeg IDCT: max={max_jpegli_lj}");

    // Compare streaming Libjpeg vs scanline Libjpeg
    let mut max_stream_scan_lj = 0i32;
    for i in 0..lj_rgb.len() {
        max_stream_scan_lj =
            max_stream_scan_lj.max((lj_rgb[i] as i32 - scanline_rgb[i] as i32).abs());
    }
    println!("streaming Libjpeg vs scanline Libjpeg: max={max_stream_scan_lj}");

    // Compare streaming Libjpeg vs mozjpeg
    let mut max_lj_moz = 0i32;
    for i in 0..lj_rgb.len() {
        max_lj_moz = max_lj_moz.max((lj_rgb[i] as i32 - moz_rgb[i] as i32).abs());
    }
    println!("streaming Libjpeg IDCT vs mozjpeg: max={max_lj_moz}");

    // Compare scanline Libjpeg vs mozjpeg
    let mut max_scan_lj_moz = 0i32;
    for i in 0..scanline_rgb.len() {
        max_scan_lj_moz = max_scan_lj_moz.max((scanline_rgb[i] as i32 - moz_rgb[i] as i32).abs());
    }
    println!("scanline Libjpeg IDCT vs mozjpeg: max={max_scan_lj_moz}");
}

/// Compare scanline reader RGB8 vs decode() streaming RGB to isolate the streaming path bug.
/// Both paths use the same ycbcr_planes_i16_to_rgb_u8 color conversion.
/// If scanline RGB8 matches mozjpeg but decode() doesn't, the streaming path's
/// IDCT/buffer management is wrong.
#[test]
#[ignore = "requires corpus"]
fn compare_scanline_rgb8_vs_streaming_decode() {
    use imgref::ImgRefMut;

    let path = zenjpeg_bench_utils::corpus_builder_dir()
        .join("wide-gamut/adobe-rgb/flickr_841c1e16a9a5484a.jpg");
    let data = match std::fs::read(&path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Cannot read {}: {e}", path.display());
            return;
        }
    };

    println!("\n=== Comparing scanline RGB8 vs streaming decode() ===");

    // Path 1: decode() — uses streaming path
    // Disable ICC to compare raw decode output (not ICC-corrected)
    let d = zenjpeg::decoder::Decoder::new().apply_icc(false);
    let result = d.decode(&data, Unstoppable).unwrap();
    let streaming_rgb = result.pixels_u8().unwrap();
    let w = result.width() as usize;
    let h = result.height() as usize;

    // Path 2: scanline reader → read_rows_rgb8
    let mut reader = zenjpeg::decoder::Decoder::new()
        .scanline_reader(&data)
        .unwrap();
    let sw = reader.width() as usize;
    let sh = reader.height() as usize;
    assert_eq!((w, h), (sw, sh));

    let row_stride = w * 3;
    let mut scanline_rgb = vec![0u8; row_stride * h];
    let mut total_rows = 0;
    while total_rows < h {
        let remaining = h - total_rows;
        let batch = remaining.min(8);
        let out_slice = &mut scanline_rgb[total_rows * row_stride..];
        let img = ImgRefMut::new_stride(out_slice, w * 3, batch, row_stride);
        let rows = reader.read_rows_rgb8(img).unwrap();
        if rows == 0 {
            break;
        }
        total_rows += rows;
    }

    // Path 3: mozjpeg
    let (_, _, moz_rgb) = decode_mozjpeg_rgb(&data);

    // Compare
    let mut max_streaming_scanline = 0i32;
    let mut max_streaming_moz = 0i32;
    let mut max_scanline_moz = 0i32;
    let mut worst_pos = (0usize, 0usize);
    let mut worst_ch = 0usize;

    for py in 0..h {
        for px in 0..w {
            let i = (py * w + px) * 3;
            for ch in 0..3 {
                let s = streaming_rgb[i + ch] as i32;
                let sl = scanline_rgb[i + ch] as i32;
                let m = moz_rgb[i + ch] as i32;

                let d_ss = (s - sl).abs();
                let d_sm = (s - m).abs();
                let d_slm = (sl - m).abs();

                if d_ss > max_streaming_scanline {
                    max_streaming_scanline = d_ss;
                    worst_pos = (px, py);
                    worst_ch = ch;
                }
                max_streaming_moz = max_streaming_moz.max(d_sm);
                max_scanline_moz = max_scanline_moz.max(d_slm);
            }
        }
    }

    println!("streaming decode() vs scanline RGB8: max={max_streaming_scanline}");
    println!("streaming decode() vs mozjpeg:       max={max_streaming_moz}");
    println!("scanline RGB8 vs mozjpeg:            max={max_scanline_moz}");

    if max_streaming_scanline > 2 {
        let (wpx, wpy) = worst_pos;
        let i = (wpy * w + wpx) * 3;
        println!("\nWorst at ({wpx},{wpy}) ch={worst_ch}:");
        println!(
            "  streaming: ({}, {}, {})",
            streaming_rgb[i],
            streaming_rgb[i + 1],
            streaming_rgb[i + 2]
        );
        println!(
            "  scanline:  ({}, {}, {})",
            scanline_rgb[i],
            scanline_rgb[i + 1],
            scanline_rgb[i + 2]
        );
        println!(
            "  mozjpeg:   ({}, {}, {})",
            moz_rgb[i],
            moz_rgb[i + 1],
            moz_rgb[i + 2]
        );

        // Show a 3x3 neighborhood around worst pixel
        println!("\n  3x3 neighborhood (streaming vs scanline):");
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let nx = wpx as i32 + dx;
                let ny = wpy as i32 + dy;
                if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                    let ni = (ny as usize * w + nx as usize) * 3;
                    print!(
                        "  ({},{}) s=({},{},{}) sl=({},{},{})",
                        nx,
                        ny,
                        streaming_rgb[ni],
                        streaming_rgb[ni + 1],
                        streaming_rgb[ni + 2],
                        scanline_rgb[ni],
                        scanline_rgb[ni + 1],
                        scanline_rgb[ni + 2],
                    );
                    let ch_diff: Vec<i32> = (0..3)
                        .map(|c| streaming_rgb[ni + c] as i32 - scanline_rgb[ni + c] as i32)
                        .collect();
                    println!(" diff=({},{},{})", ch_diff[0], ch_diff[1], ch_diff[2]);
                }
            }
        }
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
    let base = zenjpeg_bench_utils::jpegli_testdata_dir().join("jxl/flower");
    let test_cases: &[(&str, &str, i32)] = &[
        // (label, filename, max_allowed_diff)
        ("444_1x2", "flower.png.im_q85_444_1x2.jpg", 4),
        ("440", "flower.png.im_q85_440.jpg", 4),
        ("asymmetric", "flower.png.im_q85_asymmetric.jpg", 4),
        ("luma_subsample", "flower.png.im_q85_luma_subsample.jpg", 4),
        (
            "rgb_subsample_blue",
            "flower.png.im_q85_rgb_subsample_blue.jpg",
            4,
        ),
    ];

    let mut all_passed = true;
    for (label, filename, max_allowed) in test_cases {
        let path = base.join(filename).display().to_string();
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

    assert!(
        all_passed,
        "Some non-standard sampling modes exceeded max diff threshold"
    );
}

/// Test that decode() and scanline_reader produce consistent output for all standard
/// AND non-standard sampling modes (internal consistency, no mozjpeg reference needed).
#[test]
fn test_decode_scanline_consistency() {
    use imgref::ImgRefMut;

    let base = zenjpeg_bench_utils::jpegli_testdata_dir().join("jxl/flower");
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
        let path = base.join(filename);
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

    assert!(
        all_passed,
        "Some modes have inconsistent decode vs scanline output"
    );
}

/// Regression test: 444_1x2 scanline reader must match mozjpeg within IDCT rounding (max ≤ 3).
/// This was the original bug: scanline reader produced max=113 error for all-1x2 sampling
/// because StripProcessor classified it as S440 with undersized chroma buffers.
#[test]
fn regression_444_1x2_scanline_accuracy() {
    use imgref::ImgRefMut;

    let path =
        zenjpeg_bench_utils::jpegli_testdata_dir().join("jxl/flower/flower.png.im_q85_444_1x2.jpg");
    let data = match std::fs::read(&path) {
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
    let base = zenjpeg_bench_utils::jpegli_testdata_dir().join("jxl/flower");

    // 444_1x2: all components same sampling → chroma should be full resolution
    let path = base
        .join("flower.png.im_q85_444_1x2.jpg")
        .display()
        .to_string();
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

/// Diagnose the 3 corpus files with max_diff > 10 vs mozjpeg (default Jpegli IDCT).
/// Generates amplified diff PNGs and per-channel statistics.
///
/// The 3 files (from 754-file corpus comparison):
/// - source_jpegs/ab713625eeff48e1.jpg: max=22, mean=0.346
/// - wide-gamut/adobe-rgb/reddit_ac470c0702018bb7.jpg: max=17, mean=0.298
/// - source_jpegs/7ccea196894ff1ad.jpg: max=11, mean=0.320
#[test]
#[ignore = "requires corpus + output dir"]
fn diagnose_top3_outlier_diffs() {
    let corpus = zenjpeg_bench_utils::corpus_builder_dir();
    let out_dir = std::path::Path::new("/mnt/v/output/zenjpeg/diff-investigation");
    std::fs::create_dir_all(out_dir).unwrap();

    let files: &[(&str, &str)] = &[
        ("ab713625eeff48e1", "source_jpegs/ab713625eeff48e1.jpg"),
        (
            "reddit_ac470c07",
            "wide-gamut/adobe-rgb/reddit_ac470c0702018bb7.jpg",
        ),
        ("7ccea196894ff1ad", "source_jpegs/7ccea196894ff1ad.jpg"),
    ];

    for (label, rel_path) in files {
        let path = corpus.join(rel_path);
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP {label}: {e}");
                continue;
            }
        };

        println!("\n=== {label} ({rel_path}) ===");

        // Decode with mozjpeg (reference)
        let (mw, mh, moz_rgb) = decode_mozjpeg_rgb(&data);
        let w = mw as usize;
        let h = mh as usize;
        println!("  dimensions: {w}x{h}");

        // Decode with zenjpeg default (Jpegli IDCT, no ICC)
        let zen_default = zenjpeg::decoder::Decoder::new()
            .apply_icc(false)
            .decode(&data, Unstoppable)
            .unwrap();
        let zen_rgb = zen_default.pixels_u8().unwrap();

        // Decode with zenjpeg LibjpegCompat (Libjpeg IDCT, no ICC)
        let zen_compat = zenjpeg::decoder::Decoder::new()
            .apply_icc(false)
            .chroma_upsampling(zenjpeg::decoder::ChromaUpsampling::LibjpegCompat)
            .decode(&data, Unstoppable)
            .unwrap();
        let compat_rgb = zen_compat.pixels_u8().unwrap();

        // Per-channel stats: default vs mozjpeg
        let mut max_diff = [0i32; 3];
        let mut sum_diff = [0u64; 3];
        let mut diff_hist = [[0u32; 256]; 3]; // histogram of abs diffs per channel
        let mut worst_pos = [(0usize, 0usize); 3];

        for py in 0..h {
            for px in 0..w {
                let i = (py * w + px) * 3;
                for ch in 0..3 {
                    let d = (zen_rgb[i + ch] as i32 - moz_rgb[i + ch] as i32).abs();
                    if d > max_diff[ch] {
                        max_diff[ch] = d;
                        worst_pos[ch] = (px, py);
                    }
                    sum_diff[ch] += d as u64;
                    diff_hist[ch][d as usize] += 1;
                }
            }
        }

        let total = (w * h) as f64;
        let ch_names = ["R", "G", "B"];
        println!("\n  Default IDCT vs mozjpeg:");
        for ch in 0..3 {
            let (wx, wy) = worst_pos[ch];
            println!(
                "    {}: max={:>2} mean={:.4} worst@({wx},{wy})",
                ch_names[ch],
                max_diff[ch],
                sum_diff[ch] as f64 / total,
            );
        }

        // Per-channel stats: LibjpegCompat vs mozjpeg
        let mut compat_max = [0i32; 3];
        let mut compat_sum = [0u64; 3];
        for py in 0..h {
            for px in 0..w {
                let i = (py * w + px) * 3;
                for ch in 0..3 {
                    let d = (compat_rgb[i + ch] as i32 - moz_rgb[i + ch] as i32).abs();
                    compat_max[ch] = compat_max[ch].max(d);
                    compat_sum[ch] += d as u64;
                }
            }
        }
        println!("\n  Libjpeg IDCT vs mozjpeg:");
        for ch in 0..3 {
            println!(
                "    {}: max={:>2} mean={:.4}",
                ch_names[ch],
                compat_max[ch],
                compat_sum[ch] as f64 / total,
            );
        }

        // Diff histogram summary
        println!("\n  Diff histogram (default IDCT, combined channels):");
        for d in 0..=(*max_diff.iter().max().unwrap() as usize) {
            let count: u32 = diff_hist.iter().map(|h| h[d]).sum();
            let pct = count as f64 / (total * 3.0) * 100.0;
            if count > 0 {
                println!("    diff={d:>2}: {count:>10} ({pct:>6.2}%)");
            }
        }

        // Write 10x amplified diff PNG (default IDCT)
        let mut diff_img = vec![0u8; w * h * 3];
        for i in 0..w * h * 3 {
            let d = (zen_rgb[i] as i32 - moz_rgb[i] as i32).abs();
            diff_img[i] = (d * 10).min(255) as u8;
        }
        let diff_path = out_dir.join(format!("{label}_diff_10x.png"));
        write_rgb_png(&diff_path, &diff_img, w as u32, h as u32);
        println!("\n  Wrote: {}", diff_path.display());

        // Write side-by-side worst pixel region (32x32 crop around worst overall pixel)
        let overall_worst_ch = max_diff
            .iter()
            .enumerate()
            .max_by_key(|&(_, v)| *v)
            .unwrap()
            .0;
        let (cx, cy) = worst_pos[overall_worst_ch];
        let crop = 32usize;
        let x0 = cx.saturating_sub(crop / 2);
        let y0 = cy.saturating_sub(crop / 2);
        let x1 = (x0 + crop).min(w);
        let y1 = (y0 + crop).min(h);
        let cw = x1 - x0;
        let ch = y1 - y0;

        // 3 panels: mozjpeg | zen default | diff 20x
        let panel_w = cw * 3;
        let mut panel = vec![0u8; panel_w * ch * 3];
        for py in 0..ch {
            for px in 0..cw {
                let src_i = ((y0 + py) * w + (x0 + px)) * 3;
                let dst_base = (py * panel_w + px) * 3;
                // Panel 1: mozjpeg
                panel[dst_base] = moz_rgb[src_i];
                panel[dst_base + 1] = moz_rgb[src_i + 1];
                panel[dst_base + 2] = moz_rgb[src_i + 2];
                // Panel 2: zenjpeg
                let dst2 = (py * panel_w + cw + px) * 3;
                panel[dst2] = zen_rgb[src_i];
                panel[dst2 + 1] = zen_rgb[src_i + 1];
                panel[dst2 + 2] = zen_rgb[src_i + 2];
                // Panel 3: diff 20x
                let dst3 = (py * panel_w + cw * 2 + px) * 3;
                for c in 0..3 {
                    let d = (zen_rgb[src_i + c] as i32 - moz_rgb[src_i + c] as i32).abs();
                    panel[dst3 + c] = (d * 20).min(255) as u8;
                }
            }
        }
        let panel_path = out_dir.join(format!("{label}_worst_region.png"));
        write_rgb_png(&panel_path, &panel, panel_w as u32, ch as u32);
        println!("  Wrote: {}", panel_path.display());

        // Show actual pixel values at worst position
        let wi = (cy * w + cx) * 3;
        println!(
            "\n  Worst pixel ({cx},{cy}): mozjpeg=({},{},{}) zen=({},{},{}) diff=({},{},{})",
            moz_rgb[wi],
            moz_rgb[wi + 1],
            moz_rgb[wi + 2],
            zen_rgb[wi],
            zen_rgb[wi + 1],
            zen_rgb[wi + 2],
            zen_rgb[wi] as i32 - moz_rgb[wi] as i32,
            zen_rgb[wi + 1] as i32 - moz_rgb[wi + 1] as i32,
            zen_rgb[wi + 2] as i32 - moz_rgb[wi + 2] as i32,
        );
    }
}

/// Diagnose WHERE zen-vs-zune diffs occur on the 4 corpus outlier files.
/// Are they on MCU boundaries? Image edges? Interior?
#[test]
#[ignore = "requires corpus"]
fn diagnose_zen_vs_zune_outlier_locations() {
    let corpus = zenjpeg_bench_utils::corpus_builder_dir();

    // The 4 outlier files from corpus comparison (zen-vs-zune max > 10)
    let files = [
        "source_jpegs/ab713625eeff48e1.jpg", // max=22
    ];

    for rel_path in &files {
        let path = corpus.join(rel_path);
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP {rel_path}: {e}");
                continue;
            }
        };

        // Decode with zenjpeg (default Jpegli IDCT)
        let zen_img = zenjpeg::decoder::Decoder::new()
            .apply_icc(false)
            .decode(&data, Unstoppable)
            .unwrap();
        let w = zen_img.width() as usize;
        let h = zen_img.height() as usize;
        let zen_rgb = zen_img.pixels_u8().unwrap();

        // Decode with zenjpeg LibjpegCompat (Libjpeg i64 IDCT)
        let compat_img = zenjpeg::decoder::Decoder::new()
            .apply_icc(false)
            .chroma_upsampling(zenjpeg::decoder::ChromaUpsampling::LibjpegCompat)
            .decode(&data, Unstoppable)
            .unwrap();
        let compat_rgb = compat_img.pixels_u8().unwrap();

        // Decode with zune-jpeg
        let mut zdec =
            zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(&data));
        let zune_rgb = zdec.decode().unwrap();

        // Decode with mozjpeg
        let (_mw, _mh, moz_rgb) = decode_mozjpeg_rgb(&data);

        println!("\n=== {rel_path} ({w}x{h}) ===");
        println!(
            "MCU grid: {}x{}, padding: right={} bottom={}",
            (w + 15) / 16,
            (h + 15) / 16,
            if w % 16 == 0 { 0 } else { 16 - w % 16 },
            if h % 16 == 0 { 0 } else { 16 - h % 16 },
        );

        // Compare all pairs
        let pairs: &[(&str, &[u8], &str, &[u8])] = &[
            ("zen-default", zen_rgb, "zune", &zune_rgb),
            ("zen-default", zen_rgb, "mozjpeg", &moz_rgb),
            ("zen-compat", compat_rgb, "zune", &zune_rgb),
            ("zen-compat", compat_rgb, "mozjpeg", &moz_rgb),
            ("mozjpeg", &moz_rgb, "zune", &zune_rgb),
        ];

        for &(name_a, pix_a, name_b, pix_b) in pairs {
            if pix_a.len() != pix_b.len() {
                println!("  {name_a} vs {name_b}: size mismatch");
                continue;
            }

            let mut max_d = 0u8;
            let mut count_gt5 = 0usize;
            let mut count_gt2 = 0usize;
            let mut boundary_gt5 = 0usize; // on MCU row/col boundary ±1
            let mut edge_gt5 = 0usize; // in last MCU row or last MCU col of image
            let mut worst = Vec::new();

            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) * 3;
                    for ch in 0..3usize {
                        let d = (pix_a[i + ch] as i16 - pix_b[i + ch] as i16).unsigned_abs() as u8;
                        if d > max_d {
                            max_d = d;
                        }
                        if d > 2 {
                            count_gt2 += 1;
                        }
                        if d > 5 {
                            count_gt5 += 1;
                            let on_h_bnd = y % 16 <= 1 || y % 16 >= 14;
                            let on_v_bnd = x % 16 <= 1 || x % 16 >= 14;
                            if on_h_bnd || on_v_bnd {
                                boundary_gt5 += 1;
                            }
                            let on_bottom_edge = y >= h.saturating_sub(16);
                            let on_right_edge = x >= w.saturating_sub(16);
                            if on_bottom_edge || on_right_edge {
                                edge_gt5 += 1;
                            }
                            if worst.len() < 20
                                || d > worst
                                    .last()
                                    .map(|w: &(u8, usize, usize, usize)| w.0)
                                    .unwrap_or(0)
                            {
                                worst.push((d, x, y, ch));
                                worst.sort_by(|a, b| b.0.cmp(&a.0));
                                worst.truncate(20);
                            }
                        }
                    }
                }
            }

            println!("\n  {name_a} vs {name_b}: max={max_d} >2:{count_gt2} >5:{count_gt5}");
            if count_gt5 > 0 {
                println!(
                    "    Of {count_gt5} diffs>5: {boundary_gt5} on MCU boundary, {edge_gt5} on image edge"
                );
                let ch_names = ["R", "G", "B"];
                println!("    Top diffs:");
                for &(d, x, y, ch) in &worst {
                    let i = (y * w + x) * 3 + ch;
                    println!(
                        "      ({x:4},{y:4}) {}: a={:3} b={:3} diff={d:2} mcu=({},{}) mod16=({},{})",
                        ch_names[ch],
                        pix_a[i],
                        pix_b[i],
                        x / 16,
                        y / 16,
                        x % 16,
                        y % 16,
                    );
                }
            }
        }
    }
}

fn write_rgb_png(path: &std::path::Path, data: &[u8], w: u32, h: u32) {
    use std::io::BufWriter;
    let file = std::fs::File::create(path).unwrap();
    let bw = BufWriter::new(file);
    let mut encoder = png::Encoder::new(bw, w, h);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(data).unwrap();
}
