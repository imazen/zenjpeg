#![cfg(feature = "ffi-tests")]
//! Visual regression tests: generate diff images comparing zenjpeg vs mozjpeg decoder output.
//!
//! Uses zensim-regress to produce side-by-side comparison montages (Expected | Actual | Diff)
//! saved to /mnt/v/output/zenjpeg/visual_diffs/. Each decode path is tested independently.
//!
//! Run: cargo test --release -p zenjpeg --test visual_diff_regression --features decoder -- --nocapture

use enough::Unstoppable;
use zenjpeg::decoder::Decoder;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use zensim_regress::diff_image::{create_comparison_montage_raw, generate_diff_image_raw};

const OUTPUT_DIR: &str = "/mnt/v/output/zenjpeg/visual_diffs";

/// Convert RGB (3 bytes/pixel) to RGBA (4 bytes/pixel) for zensim-regress.
fn rgb_to_rgba(rgb: &[u8]) -> Vec<u8> {
    let pixel_count = rgb.len() / 3;
    let mut rgba = Vec::with_capacity(pixel_count * 4);
    for i in 0..pixel_count {
        rgba.push(rgb[i * 3]);
        rgba.push(rgb[i * 3 + 1]);
        rgba.push(rgb[i * 3 + 2]);
        rgba.push(255);
    }
    rgba
}

/// Decode with mozjpeg-sys (libjpeg-turbo C).
/// `fancy`: true = triangle/fancy upsampling (default), false = box filter.
fn decode_mozjpeg_impl(data: &[u8], fancy: bool) -> (u32, u32, Vec<u8>) {
    use mozjpeg_sys::*;
    use std::mem;
    unsafe {
        let mut err: jpeg_error_mgr = mem::zeroed();
        jpeg_std_error(&mut err);
        let mut ci: jpeg_decompress_struct = mem::zeroed();
        ci.common.err = &mut err;
        jpeg_create_decompress(&mut ci);
        jpeg_mem_src(&mut ci, data.as_ptr(), data.len() as _);
        assert_eq!(
            jpeg_read_header(&mut ci, 1),
            1,
            "mozjpeg: read_header failed"
        );
        ci.out_color_space = J_COLOR_SPACE::JCS_RGB;
        ci.do_fancy_upsampling = if fancy { 1 } else { 0 };
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

/// Decode with mozjpeg fancy upsampling (default).
fn decode_mozjpeg(data: &[u8]) -> (u32, u32, Vec<u8>) {
    decode_mozjpeg_impl(data, true)
}

/// Decode with mozjpeg box filter (no fancy upsampling).
fn decode_mozjpeg_box(data: &[u8]) -> (u32, u32, Vec<u8>) {
    decode_mozjpeg_impl(data, false)
}

/// Decode with zenjpeg streaming (default path via Decoder::decode).
fn decode_zen_streaming(data: &[u8]) -> (u32, u32, Vec<u8>) {
    let decoder = Decoder::new();
    let img = decoder
        .decode(data, Unstoppable)
        .expect("zen streaming decode");
    (img.width, img.height, img.into_pixels_u8().unwrap())
}

/// Decode with zenjpeg scanline reader.
fn decode_zen_scanline(data: &[u8]) -> (u32, u32, Vec<u8>) {
    let decoder = Decoder::new();
    let mut reader = decoder.scanline_reader(data).expect("scanline_reader");
    let w = reader.width() as usize;
    let h = reader.height() as usize;
    let stride = w * 3;
    let mut pixels = vec![0u8; h * stride];
    let mut total_rows = 0;
    while !reader.is_finished() {
        let remaining = h - total_rows;
        let buf_start = total_rows * stride;
        let output = imgref::ImgRefMut::new(&mut pixels[buf_start..], stride, remaining);
        let rows = reader.read_rows_rgb8(output).expect("read_rows_rgb8");
        total_rows += rows;
    }
    assert_eq!(total_rows, h, "scanline: didn't read all rows");
    (w as u32, h as u32, pixels)
}

/// Decode with zenjpeg LibjpegCompat upsampling (should match mozjpeg closely).
fn decode_zen_libjpeg_compat(data: &[u8]) -> (u32, u32, Vec<u8>) {
    use zenjpeg::decode::ChromaUpsampling;
    let decoder = Decoder::new().chroma_upsampling(ChromaUpsampling::LibjpegCompat);
    let img = decoder
        .decode(data, Unstoppable)
        .expect("zen libjpeg_compat decode");
    (img.width, img.height, img.into_pixels_u8().unwrap())
}

/// Decode with zenjpeg box filter (NearestNeighbor).
fn decode_zen_box(data: &[u8]) -> (u32, u32, Vec<u8>) {
    use zenjpeg::decode::ChromaUpsampling;
    let decoder = Decoder::new().chroma_upsampling(ChromaUpsampling::NearestNeighbor);
    let img = decoder.decode(data, Unstoppable).expect("zen box decode");
    (img.width, img.height, img.into_pixels_u8().unwrap())
}

/// Encode a test pattern as 4:2:0 JPEG.
fn encode_420(pixels: &[u8], width: u32, height: u32, quality: f32) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter)
        .progressive(false)
        .allow_16bit_quant_tables(false);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("create encoder");
    enc.push_packed(pixels, Unstoppable).expect("push");
    enc.finish().expect("finish")
}

/// Encode a test pattern as 4:4:4 JPEG (no chroma subsampling).
fn encode_444(pixels: &[u8], width: u32, height: u32, quality: f32) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::None)
        .progressive(false)
        .allow_16bit_quant_tables(false);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("create encoder");
    enc.push_packed(pixels, Unstoppable).expect("push");
    enc.finish().expect("finish")
}

/// Encode a test pattern as progressive 4:2:0 JPEG.
fn encode_420_progressive(pixels: &[u8], width: u32, height: u32, quality: f32) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter)
        .progressive(true)
        .allow_16bit_quant_tables(false);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("create encoder");
    enc.push_packed(pixels, Unstoppable).expect("push");
    enc.finish().expect("finish")
}

/// Generate a high-contrast noise+patches test image. Alternates red/blue blocks
/// every 8 rows with horizontal green variation to stress chroma upsampling.
fn make_stress_image(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let block_y = y / 8;
            if block_y % 2 == 0 {
                data[idx] = 255;
                data[idx + 1] = 0;
                data[idx + 2] = 0;
            } else {
                data[idx] = 0;
                data[idx + 1] = 0;
                data[idx + 2] = 255;
            }
            if x % 4 < 2 {
                data[idx + 1] = ((x * 3 + y * 7) % 200) as u8;
            }
        }
    }
    data
}

/// Generate a photographic-like noise+patches image (more realistic DCT behavior).
fn make_noise_patches_image(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            // Base: smooth color gradient
            let r = ((x * 255) / width.max(1)) as u8;
            let g = ((y * 255) / height.max(1)) as u8;
            let b = (((x + y) * 128) / (width + height).max(1)) as u8;
            // Add pseudo-noise from hash mixing
            let hash = (x.wrapping_mul(2654435761) ^ y.wrapping_mul(2246822519)) as u32;
            let noise_r = (hash & 0x1F) as u8; // 0..31
            let noise_g = ((hash >> 5) & 0x1F) as u8;
            let noise_b = ((hash >> 10) & 0x1F) as u8;
            data[idx] = r.saturating_add(noise_r);
            data[idx + 1] = g.saturating_add(noise_g);
            data[idx + 2] = b.saturating_add(noise_b);
        }
    }
    data
}

/// Save a montage to disk. Creates the output dir if needed.
fn save_montage(montage: &image::RgbaImage, name: &str) {
    let dir = std::path::Path::new(OUTPUT_DIR);
    std::fs::create_dir_all(dir).expect("create output dir");
    let path = dir.join(format!("{name}.png"));
    montage.save(&path).expect("save montage");
    println!("  Saved: {}", path.display());
}

/// Compare two RGB pixel arrays, return (max_diff, mean_diff, boundary_max, interior_max).
fn analyze_diffs(a: &[u8], b: &[u8], width: usize, height: usize) -> (i32, f64, i32, i32) {
    let mcu_height = 16usize;
    let mut global_max = 0i32;
    let mut global_sum = 0i64;
    let mut boundary_max = 0i32;
    let mut interior_max = 0i32;
    let total_samples = (width * height * 3) as f64;

    for y in 0..height {
        let row_start = y * width * 3;
        let row_end = row_start + width * 3;
        let a_row = &a[row_start..row_end];
        let b_row = &b[row_start..row_end];

        let mut row_max = 0i32;
        for (av, bv) in a_row.iter().zip(b_row.iter()) {
            let d = (*av as i32 - *bv as i32).abs();
            row_max = row_max.max(d);
            global_sum += d as i64;
        }
        global_max = global_max.max(row_max);

        let in_mcu = y % mcu_height;
        let is_boundary = in_mcu == 0 || in_mcu == mcu_height - 1;
        if is_boundary {
            boundary_max = boundary_max.max(row_max);
        } else {
            interior_max = interior_max.max(row_max);
        }
    }

    let mean = global_sum as f64 / total_samples;
    (global_max, mean, boundary_max, interior_max)
}

/// Run a full visual diff comparison for a given test case.
/// Returns true if no stripe pattern detected.
///
/// Each zenjpeg decode path is compared against the matching mozjpeg mode:
/// - Fancy-upsampled paths (streaming, scanline, libjpeg_compat) → mozjpeg fancy
/// - Box filter path → mozjpeg box filter (do_fancy_upsampling=0)
fn run_visual_diff(
    label: &str,
    jpeg: &[u8],
    width: u32,
    height: u32,
    decode_paths: &[(&str, fn(&[u8]) -> (u32, u32, Vec<u8>))],
) -> bool {
    // Pre-decode both mozjpeg modes (lazy — only decode box if needed)
    let (moz_w, moz_h, moz_fancy_rgb) = decode_mozjpeg(jpeg);
    assert_eq!(moz_w, width, "{label}: mozjpeg width mismatch");
    assert_eq!(moz_h, height, "{label}: mozjpeg height mismatch");

    let moz_fancy_rgba = rgb_to_rgba(&moz_fancy_rgb);

    // Lazily decode mozjpeg box mode only if we have a box_filter path
    let has_box = decode_paths.iter().any(|(name, _)| *name == "box_filter");
    let (moz_box_rgb, moz_box_rgba) = if has_box {
        let (bw, bh, rgb) = decode_mozjpeg_box(jpeg);
        assert_eq!(bw, width, "{label}: mozjpeg-box width mismatch");
        assert_eq!(bh, height, "{label}: mozjpeg-box height mismatch");
        let rgba = rgb_to_rgba(&rgb);
        (rgb, rgba)
    } else {
        (Vec::new(), Vec::new())
    };

    let mut all_ok = true;

    for (path_name, decode_fn) in decode_paths {
        let (zen_w, zen_h, zen_rgb) = decode_fn(jpeg);
        assert_eq!(zen_w, width, "{label}/{path_name}: width mismatch");
        assert_eq!(zen_h, height, "{label}/{path_name}: height mismatch");

        // Compare against matching mozjpeg mode
        let is_box_filter = *path_name == "box_filter";
        let ref_rgb = if is_box_filter {
            &moz_box_rgb
        } else {
            &moz_fancy_rgb
        };
        let ref_rgba = if is_box_filter {
            &moz_box_rgba
        } else {
            &moz_fancy_rgba
        };

        let (max_diff, mean_diff, boundary_max, interior_max) =
            analyze_diffs(ref_rgb, &zen_rgb, width as usize, height as usize);

        let stripe_detected = boundary_max > interior_max + 3;
        let status = if stripe_detected { "STRIPE!" } else { "OK" };
        let ref_label = if is_box_filter {
            "vs moz-box"
        } else {
            "vs moz-fancy"
        };

        println!(
            "  {path_name:20} max={max_diff:3} mean={mean_diff:.3} boundary={boundary_max:3} interior={interior_max:3} [{status}] {ref_label}"
        );

        if stripe_detected {
            all_ok = false;
        }

        // Generate and save montage
        let zen_rgba = rgb_to_rgba(&zen_rgb);
        let amplification = 10u8; // amplify diffs 10x for visibility
        let montage =
            create_comparison_montage_raw(ref_rgba, &zen_rgba, width, height, amplification, 4);
        save_montage(&montage, &format!("{label}_{path_name}"));

        // Also save standalone diff image for easy inspection
        let diff_img = generate_diff_image_raw(ref_rgba, &zen_rgba, width, height, amplification);
        let diff_dir = std::path::Path::new(OUTPUT_DIR);
        diff_img
            .save(diff_dir.join(format!("{label}_{path_name}_diff.png")))
            .expect("save diff");
    }

    all_ok
}

// ============================================================================
// Test cases
// ============================================================================

/// Standard decode paths to test (all paths that produce RGB output).
fn standard_decode_paths() -> Vec<(&'static str, fn(&[u8]) -> (u32, u32, Vec<u8>))> {
    vec![
        ("streaming", decode_zen_streaming as fn(&[u8]) -> _),
        ("scanline", decode_zen_scanline),
        ("libjpeg_compat", decode_zen_libjpeg_compat),
        ("box_filter", decode_zen_box),
    ]
}

/// 4:2:0 high-contrast stress test at multiple sizes.
#[test]
fn test_visual_diff_420_stress() {
    let paths = standard_decode_paths();
    let sizes = [(64, 64), (128, 128), (96, 80), (255, 255), (512, 512)];

    println!("\n=== Visual Diff: 4:2:0 Stress Test (Q90) ===");
    for (w, h) in sizes {
        let label = format!("stress_420_{w}x{h}");
        println!("\n--- {label} ---");
        let pixels = make_stress_image(w, h);
        let jpeg = encode_420(&pixels, w as u32, h as u32, 90.0);
        let ok = run_visual_diff(&label, &jpeg, w as u32, h as u32, &paths);
        assert!(ok, "{label}: stripe pattern detected vs mozjpeg!");
    }
}

/// 4:2:0 noise+patches (photographic-like) at multiple sizes.
#[test]
fn test_visual_diff_420_noise_patches() {
    let paths = standard_decode_paths();
    let sizes = [(128, 128), (256, 256), (97, 63)];

    println!("\n=== Visual Diff: 4:2:0 Noise+Patches (Q85) ===");
    for (w, h) in sizes {
        let label = format!("noise_420_{w}x{h}");
        println!("\n--- {label} ---");
        let pixels = make_noise_patches_image(w, h);
        let jpeg = encode_420(&pixels, w as u32, h as u32, 85.0);
        let ok = run_visual_diff(&label, &jpeg, w as u32, h as u32, &paths);
        assert!(ok, "{label}: stripe pattern detected vs mozjpeg!");
    }
}

/// 4:4:4 (no chroma subsampling) — should match mozjpeg very closely.
#[test]
fn test_visual_diff_444() {
    let paths = standard_decode_paths();
    let sizes = [(128, 128), (256, 256)];

    println!("\n=== Visual Diff: 4:4:4 (Q90) ===");
    for (w, h) in sizes {
        let label = format!("noise_444_{w}x{h}");
        println!("\n--- {label} ---");
        let pixels = make_noise_patches_image(w, h);
        let jpeg = encode_444(&pixels, w as u32, h as u32, 90.0);
        let ok = run_visual_diff(&label, &jpeg, w as u32, h as u32, &paths);
        assert!(ok, "{label}: stripe pattern detected vs mozjpeg!");
    }
}

/// Progressive 4:2:0 — tests the coefficient-buffered decode path.
#[test]
fn test_visual_diff_progressive_420() {
    // Progressive uses coefficient path (not streaming), so only streaming+libjpeg_compat
    let paths: Vec<(&str, fn(&[u8]) -> (u32, u32, Vec<u8>))> = vec![
        ("streaming", decode_zen_streaming as fn(&[u8]) -> _),
        ("libjpeg_compat", decode_zen_libjpeg_compat),
        ("box_filter", decode_zen_box),
    ];

    println!("\n=== Visual Diff: Progressive 4:2:0 (Q85) ===");
    let sizes = [(128, 128), (256, 256)];
    for (w, h) in sizes {
        let label = format!("prog_420_{w}x{h}");
        println!("\n--- {label} ---");
        let pixels = make_stress_image(w, h);
        let jpeg = encode_420_progressive(&pixels, w as u32, h as u32, 85.0);
        let ok = run_visual_diff(&label, &jpeg, w as u32, h as u32, &paths);
        assert!(ok, "{label}: stripe pattern detected vs mozjpeg!");
    }
}

/// Low quality (Q50) — larger quant steps amplify any boundary artifacts.
#[test]
fn test_visual_diff_low_quality() {
    let paths = standard_decode_paths();

    println!("\n=== Visual Diff: Low Quality Q50 4:2:0 ===");
    let sizes = [(128, 128), (256, 256)];
    for (w, h) in sizes {
        let label = format!("lowq_420_{w}x{h}");
        println!("\n--- {label} ---");
        let pixels = make_stress_image(w, h);
        let jpeg = encode_420(&pixels, w as u32, h as u32, 50.0);
        let ok = run_visual_diff(&label, &jpeg, w as u32, h as u32, &paths);
        assert!(ok, "{label}: stripe pattern detected vs mozjpeg!");
    }
}

/// Test with mozjpeg-encoded JPEG (different quant tables, different encoder).
#[test]
fn test_visual_diff_mozjpeg_encoded() {
    let paths = standard_decode_paths();

    println!("\n=== Visual Diff: mozjpeg-encoded 4:2:0 (Q85) ===");
    let sizes = [(128, 128), (256, 256)];
    for (w, h) in sizes {
        let label = format!("mozenc_420_{w}x{h}");
        println!("\n--- {label} ---");
        let pixels = make_noise_patches_image(w, h);
        let jpeg = encode_with_mozjpeg(&pixels, w, h, 85);
        let ok = run_visual_diff(&label, &jpeg, w as u32, h as u32, &paths);
        assert!(ok, "{label}: stripe pattern detected vs mozjpeg!");
    }
}

/// Encode using mozjpeg (C encoder) for cross-encoder decode testing.
fn encode_with_mozjpeg(pixels: &[u8], width: usize, height: usize, quality: i32) -> Vec<u8> {
    use mozjpeg_sys::*;
    use std::mem;
    unsafe {
        let mut err: jpeg_error_mgr = mem::zeroed();
        jpeg_std_error(&mut err);
        let mut ci: jpeg_compress_struct = mem::zeroed();
        ci.common.err = &mut err;
        jpeg_create_compress(&mut ci);

        let mut buf: *mut u8 = std::ptr::null_mut();
        let mut buf_size: core::ffi::c_ulong = 0;
        jpeg_mem_dest(&mut ci, &mut buf, &mut buf_size);

        ci.image_width = width as u32;
        ci.image_height = height as u32;
        ci.input_components = 3;
        ci.in_color_space = J_COLOR_SPACE::JCS_RGB;
        jpeg_set_defaults(&mut ci);
        jpeg_set_quality(&mut ci, quality, 1);
        // Set 4:2:0
        (*ci.comp_info.offset(0)).h_samp_factor = 2;
        (*ci.comp_info.offset(0)).v_samp_factor = 2;
        (*ci.comp_info.offset(1)).h_samp_factor = 1;
        (*ci.comp_info.offset(1)).v_samp_factor = 1;
        (*ci.comp_info.offset(2)).h_samp_factor = 1;
        (*ci.comp_info.offset(2)).v_samp_factor = 1;

        jpeg_start_compress(&mut ci, 1);
        for y in 0..height {
            let row_ptr = pixels[y * width * 3..].as_ptr();
            jpeg_write_scanlines(&mut ci, &row_ptr, 1);
        }
        jpeg_finish_compress(&mut ci);

        let result = std::slice::from_raw_parts(buf, buf_size as usize).to_vec();
        // Don't free buf — short-lived test, OS reclaims
        jpeg_destroy_compress(&mut ci);
        result
    }
}

/// Real JPEG from corpus (if available).
#[test]
fn test_visual_diff_corpus_photo() {
    let corpus_files = ["/home/lilith/work/zen/zenjpeg/zenjpeg/fuzz/corpus/seed/flower_420.jpg"];
    let paths = standard_decode_paths();

    println!("\n=== Visual Diff: Corpus Photos ===");
    for path in corpus_files {
        let Ok(jpeg) = std::fs::read(path) else {
            println!("SKIP {path} (not found)");
            continue;
        };
        let name = std::path::Path::new(path)
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap();

        // Decode both mozjpeg modes
        let (w, h, moz_fancy_rgb) = decode_mozjpeg(&jpeg);
        let moz_fancy_rgba = rgb_to_rgba(&moz_fancy_rgb);
        let (_, _, moz_box_rgb) = decode_mozjpeg_box(&jpeg);
        let moz_box_rgba = rgb_to_rgba(&moz_box_rgb);

        println!("\n--- {name} ({w}x{h}) ---");
        for (path_name, decode_fn) in &paths {
            let (zen_w, zen_h, zen_rgb) = decode_fn(&jpeg);
            if zen_w != w || zen_h != h {
                println!("  {path_name}: size mismatch, skip");
                continue;
            }

            let is_box = *path_name == "box_filter";
            let ref_rgb = if is_box { &moz_box_rgb } else { &moz_fancy_rgb };
            let ref_rgba = if is_box {
                &moz_box_rgba
            } else {
                &moz_fancy_rgba
            };

            let (max_diff, mean_diff, boundary_max, interior_max) =
                analyze_diffs(ref_rgb, &zen_rgb, w as usize, h as usize);
            let stripe_detected = boundary_max > interior_max + 3;
            let status = if stripe_detected { "STRIPE!" } else { "OK" };
            let ref_label = if is_box { "vs moz-box" } else { "vs moz-fancy" };
            println!(
                "  {path_name:20} max={max_diff:3} mean={mean_diff:.3} boundary={boundary_max:3} interior={interior_max:3} [{status}] {ref_label}"
            );

            let zen_rgba = rgb_to_rgba(&zen_rgb);
            let montage = create_comparison_montage_raw(ref_rgba, &zen_rgba, w, h, 10, 4);
            save_montage(&montage, &format!("corpus_{name}_{path_name}"));

            assert!(
                !stripe_detected,
                "{name}/{path_name}: stripe pattern detected vs mozjpeg!"
            );
        }
    }
}

/// Verify that IdctMethod::Libjpeg with box filter matches mozjpeg box within max=1.
/// This confirms the remaining max=2-3 diff on the default box path is purely IDCT rounding.
#[test]
fn test_idct_method_libjpeg_matches_mozjpeg() {
    use zenjpeg::decode::{ChromaUpsampling, IdctMethod};

    let sizes = [(128, 128), (256, 256), (96, 80), (255, 255)];
    let qualities = [50.0, 85.0, 95.0];

    println!("\n=== IdctMethod::Libjpeg + Box vs mozjpeg Box ===");
    for (w, h) in sizes {
        for q in qualities {
            let pixels = make_stress_image(w, h);
            let jpeg = encode_420(&pixels, w as u32, h as u32, q);

            // Decode with zenjpeg: libjpeg IDCT + box filter
            let decoder = Decoder::new()
                .chroma_upsampling(ChromaUpsampling::NearestNeighbor)
                .idct_method(IdctMethod::Libjpeg);
            let img = decoder.decode(&jpeg, Unstoppable).expect("decode");
            let zen_rgb = img.into_pixels_u8().unwrap();

            // Decode with mozjpeg box filter
            let (_, _, moz_rgb) = decode_mozjpeg_box(&jpeg);

            let max_diff: i32 = zen_rgb
                .iter()
                .zip(moz_rgb.iter())
                .map(|(a, b)| (*a as i32 - *b as i32).abs())
                .max()
                .unwrap_or(0);

            println!("  {w}x{h} Q{q}: Libjpeg+box vs mozjpeg-box max_diff={max_diff}");
            assert!(
                max_diff <= 1,
                "{w}x{h} Q{q}: IdctMethod::Libjpeg + box should match mozjpeg within 1, got {max_diff}"
            );
        }
    }
}

/// Verify that LibjpegCompat (which auto-selects Libjpeg IDCT) matches mozjpeg fancy
/// within max=2. The remaining diff is from upsampler rounding, not IDCT.
#[test]
fn test_idct_method_libjpeg_compat_matches_mozjpeg() {
    use zenjpeg::decode::ChromaUpsampling;

    let sizes = [(128, 128), (256, 256), (96, 80)];

    println!("\n=== LibjpegCompat (auto Libjpeg IDCT) vs mozjpeg Fancy ===");
    for (w, h) in sizes {
        let pixels = make_stress_image(w, h);
        let jpeg = encode_420(&pixels, w as u32, h as u32, 85.0);

        // LibjpegCompat defaults to Libjpeg IDCT (via effective_idct_method)
        let decoder = Decoder::new().chroma_upsampling(ChromaUpsampling::LibjpegCompat);
        let img = decoder.decode(&jpeg, Unstoppable).expect("decode");
        let zen_rgb = img.into_pixels_u8().unwrap();

        let (_, _, moz_rgb) = decode_mozjpeg(&jpeg);

        let max_diff: i32 = zen_rgb
            .iter()
            .zip(moz_rgb.iter())
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap_or(0);

        println!("  {w}x{h}: LibjpegCompat vs mozjpeg-fancy max_diff={max_diff}");
        assert!(
            max_diff <= 2,
            "{w}x{h}: LibjpegCompat should match mozjpeg within 2, got {max_diff}"
        );
    }
}

/// Decode waterhouse.jpg from corpus and compare against mozjpeg to detect banding.
///
/// Banding manifests as systematic per-row or per-block-row diff spikes in smooth
/// gradient regions. This test analyzes:
/// - Per-row max/mean diffs (banding = periodic spikes every 8 or 16 rows)
/// - Block-row vs interior diff ratios (8px boundaries from IDCT)
/// - MCU-row boundary artifacts (16px boundaries from chroma upsampling)
/// - Channel-separated analysis (banding often affects chroma more than luma)
///
/// The waterhouse painting has large smooth sky/water gradients that are sensitive
/// to decoder precision differences.
#[test]
fn test_waterhouse_banding() {
    let corpus = zenjpeg_bench_utils::codec_corpus_dir();
    let Some(corpus) = corpus else {
        println!("SKIP: codec-corpus not found");
        return;
    };
    let path = corpus.join("imageflow/test_inputs/waterhouse.jpg");
    let jpeg = match std::fs::read(&path) {
        Ok(d) => d,
        Err(e) => {
            println!("SKIP: {}: {e}", path.display());
            return;
        }
    };

    let paths = standard_decode_paths();

    // Decode with mozjpeg (reference)
    let (w, h, moz_fancy_rgb) = decode_mozjpeg(&jpeg);
    let (_, _, moz_box_rgb) = decode_mozjpeg_box(&jpeg);
    let moz_fancy_rgba = rgb_to_rgba(&moz_fancy_rgb);
    let moz_box_rgba = rgb_to_rgba(&moz_box_rgb);

    println!("\n=== Waterhouse Banding Analysis ({w}x{h}) ===");

    for (path_name, decode_fn) in &paths {
        let (zen_w, zen_h, zen_rgb) = decode_fn(&jpeg);
        assert_eq!((zen_w, zen_h), (w, h), "{path_name}: size mismatch");

        let is_box = *path_name == "box_filter";
        let ref_rgb = if is_box { &moz_box_rgb } else { &moz_fancy_rgb };
        let ref_rgba = if is_box {
            &moz_box_rgba
        } else {
            &moz_fancy_rgba
        };
        let ref_label = if is_box { "vs moz-box" } else { "vs moz-fancy" };

        // Global stats
        let (max_diff, mean_diff, boundary_max, interior_max) =
            analyze_diffs(ref_rgb, &zen_rgb, w as usize, h as usize);

        // Per-row analysis for banding detection
        let width = w as usize;
        let height = h as usize;
        let stride = width * 3;
        let mut row_maxes = Vec::with_capacity(height);
        let mut row_means = Vec::with_capacity(height);

        for y in 0..height {
            let off = y * stride;
            let a = &ref_rgb[off..off + stride];
            let b = &zen_rgb[off..off + stride];
            let mut rmax = 0i32;
            let mut rsum = 0i64;
            for (av, bv) in a.iter().zip(b.iter()) {
                let d = (*av as i32 - *bv as i32).abs();
                rmax = rmax.max(d);
                rsum += d as i64;
            }
            row_maxes.push(rmax);
            row_means.push(rsum as f64 / stride as f64);
        }

        // Detect periodic banding: check if block-boundary rows (every 8px)
        // have systematically higher diffs than their neighbors
        let mut block_boundary_diffs = Vec::new(); // rows at y % 8 == 0
        let mut block_interior_diffs = Vec::new(); // all other rows
        let mut mcu_boundary_diffs = Vec::new(); // rows at y % 16 == {0, 15}

        for (y, &rm) in row_maxes.iter().enumerate() {
            if y % 8 == 0 || y % 8 == 7 {
                block_boundary_diffs.push(rm);
            } else {
                block_interior_diffs.push(rm);
            }
            if y % 16 == 0 || y % 16 == 15 {
                mcu_boundary_diffs.push(rm);
            }
        }

        let block_boundary_mean = if block_boundary_diffs.is_empty() {
            0.0
        } else {
            block_boundary_diffs.iter().sum::<i32>() as f64 / block_boundary_diffs.len() as f64
        };
        let block_interior_mean = if block_interior_diffs.is_empty() {
            0.0
        } else {
            block_interior_diffs.iter().sum::<i32>() as f64 / block_interior_diffs.len() as f64
        };
        let mcu_boundary_mean = if mcu_boundary_diffs.is_empty() {
            0.0
        } else {
            mcu_boundary_diffs.iter().sum::<i32>() as f64 / mcu_boundary_diffs.len() as f64
        };

        // Per-channel analysis (banding often hits one channel harder)
        let mut ch_max = [0i32; 3];
        let mut ch_sum = [0i64; 3];
        let npixels = (width * height) as f64;
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                for c in 0..3 {
                    let d = (ref_rgb[idx + c] as i32 - zen_rgb[idx + c] as i32).abs();
                    ch_max[c] = ch_max[c].max(d);
                    ch_sum[c] += d as i64;
                }
            }
        }
        let ch_mean: Vec<f64> = ch_sum.iter().map(|&s| s as f64 / npixels).collect();

        // Print top-10 worst rows for banding inspection
        let mut sorted_rows: Vec<(usize, i32, f64)> = row_maxes
            .iter()
            .zip(row_means.iter())
            .enumerate()
            .map(|(y, (&mx, &mn))| (y, mx, mn))
            .collect();
        sorted_rows.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.partial_cmp(&a.2).unwrap()));

        println!("\n--- {path_name} {ref_label} ---");
        println!(
            "  global: max={max_diff} mean={mean_diff:.4} mcu_boundary={boundary_max} interior={interior_max}"
        );
        println!(
            "  block boundaries (8px): mean_max={block_boundary_mean:.2}  interior: mean_max={block_interior_mean:.2}  ratio={:.2}",
            if block_interior_mean > 0.0 {
                block_boundary_mean / block_interior_mean
            } else {
                0.0
            }
        );
        println!("  MCU boundaries (16px): mean_max={mcu_boundary_mean:.2}");
        println!(
            "  per-channel max: R={} G={} B={}  mean: R={:.4} G={:.4} B={:.4}",
            ch_max[0], ch_max[1], ch_max[2], ch_mean[0], ch_mean[1], ch_mean[2]
        );
        println!("  worst 10 rows:");
        for &(y, mx, mn) in sorted_rows.iter().take(10) {
            let pos = if y % 16 == 0 || y % 16 == 15 {
                "MCU-bnd"
            } else if y % 8 == 0 || y % 8 == 7 {
                "blk-bnd"
            } else {
                "interior"
            };
            println!("    row {y:4}: max={mx:3} mean={mn:.3} [{pos}]");
        }

        // Save visual diff montage
        let zen_rgba = rgb_to_rgba(&zen_rgb);
        let montage = create_comparison_montage_raw(ref_rgba, &zen_rgba, w, h, 10, 4);
        save_montage(&montage, &format!("waterhouse_{path_name}"));

        let diff_img = generate_diff_image_raw(ref_rgba, &zen_rgba, w, h, 10);
        let diff_dir = std::path::Path::new(OUTPUT_DIR);
        diff_img
            .save(diff_dir.join(format!("waterhouse_{path_name}_diff.png")))
            .expect("save diff");

        // Assertions:
        // 1. No MCU-boundary stripe pattern
        let stripe_detected = boundary_max > interior_max + 3;
        assert!(
            !stripe_detected,
            "waterhouse/{path_name}: MCU stripe pattern detected (boundary={boundary_max} interior={interior_max})"
        );

        // 2. Block boundary diffs should not be systematically worse than interior.
        // A ratio > 1.5 suggests 8px-periodic banding from IDCT boundary effects.
        let block_ratio = if block_interior_mean > 0.5 {
            block_boundary_mean / block_interior_mean
        } else {
            1.0 // too-small diffs, no banding concern
        };
        assert!(
            block_ratio < 1.5,
            "waterhouse/{path_name}: block-boundary banding detected (ratio={block_ratio:.2}, boundary={block_boundary_mean:.2}, interior={block_interior_mean:.2})"
        );
    }
}

/// Verify streaming and scanline paths produce identical output (path consistency).
#[test]
fn test_decode_path_consistency() {
    let sizes = [(64, 64), (128, 128), (96, 80), (255, 255)];
    let qualities = [50.0, 85.0, 95.0];

    println!("\n=== Decode Path Consistency ===");
    for (w, h) in sizes {
        for q in qualities {
            let pixels = make_stress_image(w, h);
            let jpeg = encode_420(&pixels, w as u32, h as u32, q);

            let (_, _, stream_rgb) = decode_zen_streaming(&jpeg);
            let (_, _, scanline_rgb) = decode_zen_scanline(&jpeg);

            let max_diff: i32 = stream_rgb
                .iter()
                .zip(scanline_rgb.iter())
                .map(|(a, b)| (*a as i32 - *b as i32).abs())
                .max()
                .unwrap_or(0);

            let status = if max_diff == 0 { "identical" } else { "DIFF" };
            println!("  {w}x{h} Q{q}: stream vs scanline max_diff={max_diff} [{status}]");

            assert_eq!(
                max_diff, 0,
                "{w}x{h} Q{q}: streaming and scanline paths differ (max_diff={max_diff})"
            );
        }
    }
}
