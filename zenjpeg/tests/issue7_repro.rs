//! Issue #7 reproduction: LibjpegCompat produces delta=49 vs mozjpeg on Canon 5D JPEG.
//!
//! Downloads a Canon EOS 5D Mark IV sRGB JPEG (800x537, baseline 4:2:0, DRI)
//! and compares zenjpeg LibjpegCompat decode against mozjpeg-sys (libjpeg-turbo FFI).
//!
//! Run: cargo test --release -p zenjpeg --test issue7_repro --features decoder -- --nocapture

use enough::Unstoppable;
use zenjpeg::decode::{ChromaUpsampling, Decoder};

/// Decode JPEG with mozjpeg-sys (libjpeg-turbo FFI) — the reference implementation.
fn decode_mozjpeg(jpeg: &[u8]) -> (u32, u32, Vec<u8>) {
    use mozjpeg_sys::*;
    use std::mem;
    unsafe {
        let mut err: jpeg_error_mgr = mem::zeroed();
        jpeg_std_error(&mut err);
        let mut cinfo: jpeg_decompress_struct = mem::zeroed();
        cinfo.common.err = &mut err;
        jpeg_create_decompress(&mut cinfo);
        jpeg_mem_src(&mut cinfo, jpeg.as_ptr(), jpeg.len() as _);
        jpeg_read_header(&mut cinfo, true as boolean);
        cinfo.out_color_space = J_COLOR_SPACE::JCS_RGB;
        jpeg_start_decompress(&mut cinfo);
        let w = cinfo.output_width;
        let h = cinfo.output_height;
        let stride = w as usize * cinfo.output_components as usize;
        let mut out = vec![0u8; h as usize * stride];
        while cinfo.output_scanline < h {
            let off = cinfo.output_scanline as usize * stride;
            let mut p = out[off..].as_mut_ptr();
            jpeg_read_scanlines(&mut cinfo, &mut p, 1);
        }
        jpeg_finish_decompress(&mut cinfo);
        jpeg_destroy_decompress(&mut cinfo);
        (w, h, out)
    }
}

/// Decode JPEG with zenjpeg in LibjpegCompat mode.
fn decode_zenjpeg_compat(jpeg: &[u8]) -> (u32, u32, Vec<u8>) {
    let result = Decoder::new()
        .apply_icc(false)
        .chroma_upsampling(ChromaUpsampling::Triangle)
        .decode(jpeg, Unstoppable)
        .expect("zenjpeg decode failed");
    let w = result.width();
    let h = result.height();
    let pixels = result.into_pixels_u8().expect("u8 pixels");
    (w, h, pixels)
}

/// Decode JPEG with zenjpeg using default (Triangle) upsampling.
fn decode_zenjpeg_default(jpeg: &[u8]) -> (u32, u32, Vec<u8>) {
    let result = Decoder::new()
        .apply_icc(false)
        .decode(jpeg, Unstoppable)
        .expect("zenjpeg decode failed");
    let w = result.width();
    let h = result.height();
    let pixels = result.into_pixels_u8().expect("u8 pixels");
    (w, h, pixels)
}

fn compare_pixels(name_a: &str, a: &[u8], name_b: &str, b: &[u8], w: u32, h: u32) {
    assert_eq!(a.len(), b.len(), "pixel buffer size mismatch");
    let npixels = (w * h) as usize;
    let mut max_r = 0i32;
    let mut max_g = 0i32;
    let mut max_b = 0i32;
    let mut max_any = 0i32;
    let mut differ_count = 0usize;
    let mut max_pos = (0, 0);

    for i in 0..npixels {
        let dr = (a[i * 3] as i32 - b[i * 3] as i32).abs();
        let dg = (a[i * 3 + 1] as i32 - b[i * 3 + 1] as i32).abs();
        let db = (a[i * 3 + 2] as i32 - b[i * 3 + 2] as i32).abs();
        let d = dr.max(dg).max(db);
        if d > max_any {
            max_any = d;
            max_pos = (i % w as usize, i / w as usize);
        }
        max_r = max_r.max(dr);
        max_g = max_g.max(dg);
        max_b = max_b.max(db);
        if d > 0 {
            differ_count += 1;
        }
    }

    let pct = differ_count as f64 / npixels as f64 * 100.0;
    println!("\n{name_a} vs {name_b}:");
    println!("  Max delta per channel: R={max_r}, G={max_g}, B={max_b}");
    println!(
        "  Max any-channel delta: {max_any} at ({}, {})",
        max_pos.0, max_pos.1
    );
    println!("  Differing pixels: {differ_count}/{npixels} ({pct:.2}%)");

    // Show histogram of differences
    let mut hist = [0u32; 256];
    for i in 0..npixels {
        let dr = (a[i * 3] as i32 - b[i * 3] as i32).abs();
        let dg = (a[i * 3 + 1] as i32 - b[i * 3 + 1] as i32).abs();
        let db = (a[i * 3 + 2] as i32 - b[i * 3 + 2] as i32).abs();
        let d = dr.max(dg).max(db) as usize;
        hist[d] += 1;
    }
    println!(
        "  Histogram: exact={}, delta1={}, delta2={}, delta3+={}",
        hist[0],
        hist[1],
        hist[2],
        hist[3..].iter().sum::<u32>()
    );
    for d in 3..256 {
        if hist[d] > 0 {
            println!("    delta={d}: {}", hist[d]);
        }
    }
}

#[test]
fn issue7_libjpeg_compat_delta() {
    let jpeg_path = "/tmp/issue7_test.jpg";
    let jpeg = match std::fs::read(jpeg_path) {
        Ok(data) => data,
        Err(_) => {
            eprintln!("Test image not found at {jpeg_path}, skipping.");
            eprintln!(
                "Download with: curl -sL -o /tmp/issue7_test.jpg 'https://s3-us-west-2.amazonaws.com/imageflow-resources/test_inputs/wide-gamut/srgb-reference/canon_eos_5d_mark_iv/wmc_81b268fc64ea796c.jpg'"
            );
            return;
        }
    };

    println!("Image size: {} bytes", jpeg.len());

    let (mw, mh, moz_pixels) = decode_mozjpeg(&jpeg);
    println!("mozjpeg decoded: {mw}x{mh}");

    let (zw, zh, zen_compat) = decode_zenjpeg_compat(&jpeg);
    println!("zenjpeg (LibjpegCompat) decoded: {zw}x{zh}");

    let (dw, dh, zen_default) = decode_zenjpeg_default(&jpeg);
    println!("zenjpeg (default/Triangle) decoded: {dw}x{dh}");

    assert_eq!((mw, mh), (zw, zh), "dimension mismatch");
    assert_eq!((mw, mh), (dw, dh), "dimension mismatch");

    compare_pixels(
        "zenjpeg-compat",
        &zen_compat,
        "mozjpeg",
        &moz_pixels,
        mw,
        mh,
    );
    compare_pixels(
        "zenjpeg-default",
        &zen_default,
        "mozjpeg",
        &moz_pixels,
        mw,
        mh,
    );
    compare_pixels(
        "zenjpeg-compat",
        &zen_compat,
        "zenjpeg-default",
        &zen_default,
        mw,
        mh,
    );

    // The actual assertion: LibjpegCompat should be within 1 of mozjpeg
    let npixels = (mw * mh) as usize;
    let mut max_compat_moz = 0i32;
    for i in 0..npixels * 3 {
        let d = (zen_compat[i] as i32 - moz_pixels[i] as i32).abs();
        max_compat_moz = max_compat_moz.max(d);
    }

    if max_compat_moz > 1 {
        eprintln!(
            "\nFAILURE: LibjpegCompat max delta vs mozjpeg = {max_compat_moz}, expected <= 1"
        );

        // Find where the worst differences are (by row)
        let w = mw as usize;
        let h = mh as usize;
        let mut worst_rows: Vec<(usize, i32)> = Vec::new();
        for y in 0..h {
            let mut row_max = 0i32;
            for x in 0..w {
                let idx = (y * w + x) * 3;
                for c in 0..3 {
                    let d = (zen_compat[idx + c] as i32 - moz_pixels[idx + c] as i32).abs();
                    row_max = row_max.max(d);
                }
            }
            if row_max > 2 {
                worst_rows.push((y, row_max));
            }
        }
        worst_rows.sort_by(|a, b| b.1.cmp(&a.1));
        println!("\nWorst rows (delta > 2):");
        for (y, d) in worst_rows.iter().take(20) {
            println!("  row {y}: max_delta={d}");
        }
    }

    assert!(
        max_compat_moz <= 1,
        "LibjpegCompat delta vs mozjpeg = {max_compat_moz}, expected <= 1"
    );
}

/// Test the scanline_reader path too — the issue mentions StripProcessor.
#[test]
fn issue7_scanline_reader_path() {
    let jpeg_path = "/tmp/issue7_test.jpg";
    let jpeg = match std::fs::read(jpeg_path) {
        Ok(data) => data,
        Err(_) => {
            eprintln!("Test image not found at {jpeg_path}, skipping.");
            return;
        }
    };

    let (mw, mh, moz_pixels) = decode_mozjpeg(&jpeg);

    // Decode via scanline_reader with LibjpegCompat
    let decoder = Decoder::new()
        .apply_icc(false)
        .chroma_upsampling(ChromaUpsampling::Triangle);
    let mut reader = decoder
        .scanline_reader(&jpeg)
        .expect("scanline_reader failed");
    let w = reader.width() as usize;
    let h = reader.height() as usize;
    let stride = w * 3;
    let mut pixels = vec![0u8; h * stride];
    let mut row = 0;
    while row < h {
        let buf =
            imgref::ImgRefMut::new_stride(&mut pixels[row * stride..], w * 3, h - row, stride);
        let rows_read = reader.read_rows_rgb8(buf).expect("read_rows_rgb8 failed");
        if rows_read == 0 {
            break;
        }
        row += rows_read;
    }

    assert_eq!((w as u32, h as u32), (mw, mh), "dimension mismatch");

    compare_pixels(
        "zen-scanline-compat",
        &pixels,
        "mozjpeg",
        &moz_pixels,
        mw,
        mh,
    );

    let mut max_delta = 0i32;
    for i in 0..pixels.len() {
        let d = (pixels[i] as i32 - moz_pixels[i] as i32).abs();
        max_delta = max_delta.max(d);
    }

    println!("\nScanline reader LibjpegCompat max delta vs mozjpeg: {max_delta}");
    assert!(
        max_delta <= 1,
        "Scanline reader LibjpegCompat delta vs mozjpeg = {max_delta}, expected <= 1"
    );
}

/// Test with apply_icc(true) — the default when CMS is enabled.
/// This was the likely original cause: ICC transform being applied
/// when both sides should be sRGB-to-sRGB (identity).
#[test]
fn issue7_with_icc_enabled() {
    let jpeg_path = "/tmp/issue7_test.jpg";
    let jpeg = match std::fs::read(jpeg_path) {
        Ok(data) => data,
        Err(_) => {
            eprintln!("Test image not found at {jpeg_path}, skipping.");
            return;
        }
    };

    let (mw, mh, moz_pixels) = decode_mozjpeg(&jpeg);

    // Decode with apply_icc(true) — default when cms feature is on
    let result = Decoder::new()
        .apply_icc(true)
        .chroma_upsampling(ChromaUpsampling::Triangle)
        .decode(&jpeg, Unstoppable)
        .expect("decode failed");
    let w = result.width();
    let h = result.height();
    let zen_pixels = result.into_pixels_u8().expect("u8 pixels");

    assert_eq!((w, h), (mw, mh), "dimension mismatch");

    compare_pixels(
        "zen-compat-icc",
        &zen_pixels,
        "mozjpeg",
        &moz_pixels,
        mw,
        mh,
    );

    let mut max_delta = 0i32;
    for i in 0..zen_pixels.len().min(moz_pixels.len()) {
        let d = (zen_pixels[i] as i32 - moz_pixels[i] as i32).abs();
        max_delta = max_delta.max(d);
    }

    println!("\nLibjpegCompat + apply_icc max delta vs mozjpeg: {max_delta}");
    // With ICC enabled and sRGB profile, should still be small
    // but if CMS is not compiled in, this is the same as apply_icc(false)
    if max_delta > 2 {
        eprintln!(
            "WARNING: apply_icc(true) produces delta={max_delta} — CMS transform may be the cause"
        );
    }
}

/// Test the coefficient/buffered decode path (force via output_target).
#[test]
fn issue7_coefficient_path() {
    use zenjpeg::decode::OutputTarget;

    let jpeg_path = "/tmp/issue7_test.jpg";
    let jpeg = match std::fs::read(jpeg_path) {
        Ok(data) => data,
        Err(_) => {
            eprintln!("Test image not found at {jpeg_path}, skipping.");
            return;
        }
    };

    let (mw, mh, moz_pixels) = decode_mozjpeg(&jpeg);

    // Force coefficient path by requesting f32 output
    let result = Decoder::new()
        .apply_icc(false)
        .chroma_upsampling(ChromaUpsampling::Triangle)
        .output_target(OutputTarget::SrgbF32)
        .decode(&jpeg, Unstoppable)
        .expect("decode failed");
    let w = result.width();
    let h = result.height();
    let f32_pixels = result.into_pixels_f32().expect("f32 pixels");
    // Convert f32 to u8 for comparison
    let zen_pixels: Vec<u8> = f32_pixels
        .iter()
        .map(|&v| (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8)
        .collect();

    assert_eq!((w, h), (mw, mh), "dimension mismatch");

    compare_pixels(
        "zen-compat-f32",
        &zen_pixels,
        "mozjpeg",
        &moz_pixels,
        mw,
        mh,
    );

    let mut max_delta = 0i32;
    for i in 0..zen_pixels.len().min(moz_pixels.len()) {
        let d = (zen_pixels[i] as i32 - moz_pixels[i] as i32).abs();
        max_delta = max_delta.max(d);
    }

    println!("\nCoefficient path (SrgbF32) LibjpegCompat max delta vs mozjpeg: {max_delta}");
    // f32 path uses different precision so expect slightly larger deltas
    if max_delta > 5 {
        eprintln!("WARNING: f32 coefficient path produces delta={max_delta} — may indicate bug");
    }
}

/// Test with dequant_bias(true) — jpegli's precision mode. This forces f32 IDCT
/// and applies Laplacian dequantization biases, producing the most divergent output.
#[test]
fn issue7_dequant_bias_path() {
    let jpeg_path = "/tmp/issue7_test.jpg";
    let jpeg = match std::fs::read(jpeg_path) {
        Ok(data) => data,
        Err(_) => {
            eprintln!("Test image not found at {jpeg_path}, skipping.");
            return;
        }
    };

    let (mw, mh, moz_pixels) = decode_mozjpeg(&jpeg);

    // Decode with dequant_bias — forces f32 IDCT
    let result = Decoder::new()
        .apply_icc(false)
        .dequant_bias(true)
        .chroma_upsampling(ChromaUpsampling::Triangle)
        .decode(&jpeg, Unstoppable)
        .expect("decode failed");
    let w = result.width();
    let h = result.height();
    let f32_pixels = result.into_pixels_f32().expect("f32 pixels");
    let zen_pixels: Vec<u8> = f32_pixels
        .iter()
        .map(|&v| (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8)
        .collect();

    assert_eq!((w, h), (mw, mh), "dimension mismatch");

    compare_pixels(
        "zen-bias-compat",
        &zen_pixels,
        "mozjpeg",
        &moz_pixels,
        mw,
        mh,
    );

    let mut max_delta = 0i32;
    for i in 0..zen_pixels.len().min(moz_pixels.len()) {
        let d = (zen_pixels[i] as i32 - moz_pixels[i] as i32).abs();
        max_delta = max_delta.max(d);
    }

    println!("\ndequant_bias + LibjpegCompat max delta vs mozjpeg: {max_delta}");
}

/// Test all decode paths produce matching output for this specific image.
#[test]
fn issue7_cross_path_consistency() {
    let jpeg_path = "/tmp/issue7_test.jpg";
    let jpeg = match std::fs::read(jpeg_path) {
        Ok(data) => data,
        Err(_) => {
            eprintln!("Test image not found at {jpeg_path}, skipping.");
            return;
        }
    };

    // Path 1: decode() - streaming
    let result1 = Decoder::new()
        .apply_icc(false)
        .chroma_upsampling(ChromaUpsampling::Triangle)
        .decode(&jpeg, Unstoppable)
        .expect("decode");
    let p1 = result1.into_pixels_u8().unwrap();

    // Path 2: scanline_reader() - streaming
    let decoder2 = Decoder::new()
        .apply_icc(false)
        .chroma_upsampling(ChromaUpsampling::Triangle);
    let mut reader = decoder2.scanline_reader(&jpeg).expect("scanline_reader");
    let w = reader.width() as usize;
    let h = reader.height() as usize;
    let stride = w * 3;
    let mut p2 = vec![0u8; h * stride];
    let mut row = 0;
    while row < h {
        let buf = imgref::ImgRefMut::new_stride(&mut p2[row * stride..], w * 3, h - row, stride);
        let rows_read = reader.read_rows_rgb8(buf).expect("read_rows");
        if rows_read == 0 {
            break;
        }
        row += rows_read;
    }

    // Path 1 and 2 should be byte-identical
    assert_eq!(p1.len(), p2.len(), "buffer size mismatch");
    let mut max_delta = 0i32;
    for i in 0..p1.len() {
        let d = (p1[i] as i32 - p2[i] as i32).abs();
        max_delta = max_delta.max(d);
    }
    println!("decode() vs scanline_reader() max delta: {max_delta}");
    assert_eq!(
        max_delta, 0,
        "decode() and scanline_reader() should be byte-identical for baseline 4:2:0"
    );
}
