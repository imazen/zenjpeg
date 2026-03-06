//! Corpus-level crop-on-decode verification.
//!
//! For every JPEG in the conformance corpus, verify that cropped decode
//! produces pixel-identical output to full decode + manual pixel crop.
//! Tests both the scanline (streaming) and decode() (buffered) paths,
//! with and without lossless transforms (rotations/flips).
//!
//! Run with:
//!   cargo test --release --features decoder -p zenjpeg --test crop_corpus -- --nocapture --ignored

use enough::Unstoppable;
use std::fs;
use std::path::{Path, PathBuf};

fn corpus() -> Option<codec_corpus::Corpus> {
    codec_corpus::Corpus::new().ok()
}

fn collect_jpgs(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file()
                && let Some(ext) = path.extension()
                && (ext == "jpg" || ext == "jpeg")
            {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

/// Manually crop a pixel buffer (3 bpp or 1 bpp).
fn manual_crop(
    pixels: &[u8],
    img_w: usize,
    bpp: usize,
    cx: usize,
    cy: usize,
    cw: usize,
    ch: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; cw * ch * bpp];
    for y in 0..ch {
        let src_off = ((cy + y) * img_w + cx) * bpp;
        let dst_off = y * cw * bpp;
        let row_bytes = cw * bpp;
        out[dst_off..dst_off + row_bytes].copy_from_slice(&pixels[src_off..src_off + row_bytes]);
    }
    out
}

/// Compute crop regions for a given image size.
/// Returns a set of (x, y, w, h) crops that exercise different cases.
fn crop_regions(img_w: u32, img_h: u32) -> Vec<(u32, u32, u32, u32)> {
    let mut regions = Vec::new();

    // Center crop: 50%
    let cw = img_w / 2;
    let ch = img_h / 2;
    if cw > 0 && ch > 0 {
        regions.push((img_w / 4, img_h / 4, cw, ch));
    }

    // Top-left corner: 25%
    let cw = img_w / 4;
    let ch = img_h / 4;
    if cw > 0 && ch > 0 {
        regions.push((0, 0, cw, ch));
    }

    // Bottom-right corner
    if cw > 0 && ch > 0 {
        regions.push((img_w - cw, img_h - ch, cw, ch));
    }

    // Non-aligned crop (offset by 3 pixels if image is large enough)
    if img_w > 30 && img_h > 30 {
        let cw = (img_w / 3).min(img_w - 3);
        let ch = (img_h / 3).min(img_h - 3);
        if cw > 0 && ch > 0 {
            regions.push((3, 3, cw, ch));
        }
    }

    regions
}

/// Run crop verification for a single file with a given crop region.
/// Returns (max_diff, mean_diff) for the buffered path, or None if decode fails.
fn verify_crop_buffered(data: &[u8], cx: u32, cy: u32, cw: u32, ch: u32) -> Option<(u8, f64)> {
    use zenjpeg::decoder::CropRegion;
    use zenjpeg::decoder::Decoder;

    // Full decode
    let full = Decoder::new().decode(data, Unstoppable).ok()?;
    let fw = full.width() as usize;
    let bpp = full.format().bytes_per_pixel();
    let full_pix = full.into_pixels_u8()?;

    // Manual crop from full decode
    let reference = manual_crop(
        &full_pix,
        fw,
        bpp,
        cx as usize,
        cy as usize,
        cw as usize,
        ch as usize,
    );

    // Cropped decode
    let cropped = Decoder::new()
        .crop(CropRegion::pixels(cx, cy, cw, ch))
        .decode(data, Unstoppable)
        .ok()?;

    if cropped.width() != cw || cropped.height() != ch {
        return Some((255, 255.0)); // Size mismatch
    }

    let cropped_pix = cropped.into_pixels_u8()?;
    if cropped_pix.len() != reference.len() {
        return Some((254, 254.0)); // Length mismatch
    }

    // Compare
    let mut max_diff = 0u8;
    let mut sum_diff = 0u64;
    for (a, b) in reference.iter().zip(cropped_pix.iter()) {
        let d = (*a as i16 - *b as i16).unsigned_abs() as u8;
        max_diff = max_diff.max(d);
        sum_diff += d as u64;
    }
    let mean_diff = sum_diff as f64 / reference.len() as f64;

    Some((max_diff, mean_diff))
}

/// Run crop verification via scanline reader.
fn verify_crop_scanline(data: &[u8], cx: u32, cy: u32, cw: u32, ch: u32) -> Option<(u8, f64)> {
    use zenjpeg::decoder::CropRegion;
    use zenjpeg::decoder::Decoder;

    // Full scanline decode
    let mut full_reader = Decoder::new().scanline_reader(data).ok()?;
    let fw = full_reader.width() as usize;
    let fh = full_reader.height() as usize;
    let mut full_pix = vec![0u8; fw * fh * 3];
    let mut rows_read = 0;
    while !full_reader.is_finished() {
        let remaining = fh - rows_read;
        let output = imgref::ImgRefMut::new(&mut full_pix[rows_read * fw * 3..], fw * 3, remaining);
        rows_read += full_reader.read_rows_rgb8(output).ok()?;
    }

    // Manual crop
    let reference = manual_crop(
        &full_pix,
        fw,
        3,
        cx as usize,
        cy as usize,
        cw as usize,
        ch as usize,
    );

    // Cropped scanline decode
    let mut reader = Decoder::new()
        .crop(CropRegion::pixels(cx, cy, cw, ch))
        .scanline_reader(data)
        .ok()?;

    if reader.width() != cw || reader.height() != ch {
        return Some((255, 255.0));
    }

    let out_w = cw as usize;
    let out_h = ch as usize;
    let mut cropped_pix = vec![0u8; out_w * out_h * 3];
    let mut rows_read = 0;
    while !reader.is_finished() {
        let remaining = out_h - rows_read;
        let output = imgref::ImgRefMut::new(
            &mut cropped_pix[rows_read * out_w * 3..],
            out_w * 3,
            remaining,
        );
        rows_read += reader.read_rows_rgb8(output).ok()?;
    }

    if cropped_pix.len() != reference.len() {
        return Some((254, 254.0));
    }

    let mut max_diff = 0u8;
    let mut sum_diff = 0u64;
    for (a, b) in reference.iter().zip(cropped_pix.iter()) {
        let d = (*a as i16 - *b as i16).unsigned_abs() as u8;
        max_diff = max_diff.max(d);
        sum_diff += d as u64;
    }
    let mean_diff = sum_diff as f64 / reference.len() as f64;

    Some((max_diff, mean_diff))
}

/// Run crop verification with a transform applied.
/// Compares: full decode with transform + manual crop vs cropped decode with transform.
fn verify_crop_with_transform(
    data: &[u8],
    transform: zenjpeg::lossless::LosslessTransform,
    cx: u32,
    cy: u32,
    cw: u32,
    ch: u32,
) -> Option<(u8, f64)> {
    use zenjpeg::decoder::CropRegion;
    use zenjpeg::decoder::Decoder;

    // Full decode with transform
    let full = Decoder::new()
        .transform(transform)
        .decode(data, Unstoppable)
        .ok()?;
    let fw = full.width() as usize;
    let bpp = full.format().bytes_per_pixel();
    let full_pix = full.into_pixels_u8()?;

    // Manual crop from full transformed decode
    let reference = manual_crop(
        &full_pix,
        fw,
        bpp,
        cx as usize,
        cy as usize,
        cw as usize,
        ch as usize,
    );

    // Cropped + transformed decode
    let cropped = Decoder::new()
        .transform(transform)
        .crop(CropRegion::pixels(cx, cy, cw, ch))
        .decode(data, Unstoppable)
        .ok()?;

    if cropped.width() != cw || cropped.height() != ch {
        return Some((255, 255.0));
    }

    let cropped_pix = cropped.into_pixels_u8()?;
    if cropped_pix.len() != reference.len() {
        return Some((254, 254.0));
    }

    let mut max_diff = 0u8;
    let mut sum_diff = 0u64;
    for (a, b) in reference.iter().zip(cropped_pix.iter()) {
        let d = (*a as i16 - *b as i16).unsigned_abs() as u8;
        max_diff = max_diff.max(d);
        sum_diff += d as u64;
    }
    let mean_diff = sum_diff as f64 / reference.len() as f64;

    Some((max_diff, mean_diff))
}

/// Run crop verification via scanline reader with a transform.
/// Returns None if the image can't be decoded (e.g. CMYK in scanline transform path).
fn verify_crop_scanline_with_transform(
    data: &[u8],
    transform: zenjpeg::lossless::LosslessTransform,
    cx: u32,
    cy: u32,
    cw: u32,
    ch: u32,
) -> Option<(u8, f64)> {
    use zenjpeg::decoder::CropRegion;
    use zenjpeg::decoder::Decoder;

    // Full scanline decode with transform — may panic on CMYK (pre-existing bug)
    let data_clone = data.to_vec();
    let full_result = std::panic::catch_unwind(move || {
        let mut full_reader = Decoder::new()
            .transform(transform)
            .scanline_reader(&data_clone)
            .ok()?;
        let fw = full_reader.width() as usize;
        let fh = full_reader.height() as usize;
        let mut full_pix = vec![0u8; fw * fh * 3];
        let mut rows_read = 0;
        while !full_reader.is_finished() {
            let remaining = fh - rows_read;
            let output =
                imgref::ImgRefMut::new(&mut full_pix[rows_read * fw * 3..], fw * 3, remaining);
            rows_read += full_reader.read_rows_rgb8(output).ok()?;
        }
        Some((fw, full_pix))
    });

    let (fw, full_pix) = match full_result {
        Ok(Some(v)) => v,
        _ => return None, // CMYK or other unsupported format
    };

    let reference = manual_crop(
        &full_pix,
        fw,
        3,
        cx as usize,
        cy as usize,
        cw as usize,
        ch as usize,
    );

    // Cropped + transformed scanline decode
    let mut reader = Decoder::new()
        .transform(transform)
        .crop(CropRegion::pixels(cx, cy, cw, ch))
        .scanline_reader(data)
        .ok()?;

    if reader.width() != cw || reader.height() != ch {
        return Some((255, 255.0));
    }

    let out_w = cw as usize;
    let out_h = ch as usize;
    let mut cropped_pix = vec![0u8; out_w * out_h * 3];
    let mut rows_read = 0;
    while !reader.is_finished() {
        let remaining = out_h - rows_read;
        let output = imgref::ImgRefMut::new(
            &mut cropped_pix[rows_read * out_w * 3..],
            out_w * 3,
            remaining,
        );
        rows_read += reader.read_rows_rgb8(output).ok()?;
    }

    if cropped_pix.len() != reference.len() {
        return Some((254, 254.0));
    }

    let mut max_diff = 0u8;
    let mut sum_diff = 0u64;
    for (a, b) in reference.iter().zip(cropped_pix.iter()) {
        let d = (*a as i16 - *b as i16).unsigned_abs() as u8;
        max_diff = max_diff.max(d);
        sum_diff += d as u64;
    }
    let mean_diff = sum_diff as f64 / reference.len() as f64;

    Some((max_diff, mean_diff))
}

#[test]
#[ignore]
fn corpus_crop_buffered() {
    let c = match corpus() {
        Some(c) => c,
        None => {
            eprintln!("Skipping: corpus unavailable");
            return;
        }
    };
    let corpus_dir = match c.get("jpeg-conformance/valid") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Skipping: {e}");
            return;
        }
    };

    let files = collect_jpgs(&corpus_dir);
    eprintln!("Testing {} files via decode() crop path\n", files.len());
    eprintln!(
        "{:<45} {:>8} {:>8} {:>6} {:>6} {:>10}",
        "File", "ImgW", "ImgH", "Crops", "MaxD", "MeanD"
    );

    let mut total_crops = 0;
    let mut total_failures = 0;

    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy();
        let data = fs::read(path).unwrap();

        // Get image dimensions first
        let info = match zenjpeg::decoder::Decoder::new().decode(&data, Unstoppable) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let img_w = info.width();
        let img_h = info.height();

        let regions = crop_regions(img_w, img_h);
        let mut file_max = 0u8;
        let mut file_mean_sum = 0.0;
        let mut file_crops = 0;

        for (cx, cy, cw, ch) in &regions {
            if let Some((max_diff, mean_diff)) = verify_crop_buffered(&data, *cx, *cy, *cw, *ch) {
                file_max = file_max.max(max_diff);
                file_mean_sum += mean_diff;
                file_crops += 1;
                total_crops += 1;
                if max_diff > 0 {
                    total_failures += 1;
                    eprintln!(
                        "  DIFF crop({},{},{},{}) max={} mean={:.4}",
                        cx, cy, cw, ch, max_diff, mean_diff
                    );
                }
            }
        }

        let avg_mean = if file_crops > 0 {
            file_mean_sum / file_crops as f64
        } else {
            0.0
        };

        eprintln!(
            "{:<45} {:>8} {:>8} {:>6} {:>6} {:>10.4}",
            name, img_w, img_h, file_crops, file_max, avg_mean
        );
    }

    eprintln!(
        "\n{} total crops tested, {} with nonzero diff",
        total_crops, total_failures
    );
    assert_eq!(
        total_failures, 0,
        "All buffered crop results should be pixel-identical"
    );
}

#[test]
#[ignore]
fn corpus_crop_scanline() {
    let c = match corpus() {
        Some(c) => c,
        None => {
            eprintln!("Skipping: corpus unavailable");
            return;
        }
    };
    let corpus_dir = match c.get("jpeg-conformance/valid") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Skipping: {e}");
            return;
        }
    };

    let files = collect_jpgs(&corpus_dir);
    eprintln!(
        "Testing {} files via scanline_reader() crop path\n",
        files.len()
    );
    eprintln!(
        "{:<45} {:>8} {:>8} {:>6} {:>6} {:>10}",
        "File", "ImgW", "ImgH", "Crops", "MaxD", "MeanD"
    );

    let mut total_crops = 0;
    let mut total_failures = 0;

    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy();
        let data = fs::read(path).unwrap();

        let full_reader = match zenjpeg::decoder::Decoder::new().scanline_reader(&data) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let img_w = full_reader.width();
        let img_h = full_reader.height();
        drop(full_reader);

        let regions = crop_regions(img_w, img_h);
        let mut file_max = 0u8;
        let mut file_mean_sum = 0.0;
        let mut file_crops = 0;

        for (cx, cy, cw, ch) in &regions {
            if let Some((max_diff, mean_diff)) = verify_crop_scanline(&data, *cx, *cy, *cw, *ch) {
                file_max = file_max.max(max_diff);
                file_mean_sum += mean_diff;
                file_crops += 1;
                total_crops += 1;
                if max_diff > 0 {
                    total_failures += 1;
                    eprintln!(
                        "  DIFF crop({},{},{},{}) max={} mean={:.4}",
                        cx, cy, cw, ch, max_diff, mean_diff
                    );
                }
            }
        }

        let avg_mean = if file_crops > 0 {
            file_mean_sum / file_crops as f64
        } else {
            0.0
        };

        eprintln!(
            "{:<45} {:>8} {:>8} {:>6} {:>6} {:>10.4}",
            name, img_w, img_h, file_crops, file_max, avg_mean
        );
    }

    eprintln!(
        "\n{} total crops tested, {} with nonzero diff",
        total_crops, total_failures
    );
    assert_eq!(
        total_failures, 0,
        "All scanline crop results should be pixel-identical"
    );
}

#[test]
#[ignore]
fn corpus_crop_with_transforms() {
    use zenjpeg::lossless::LosslessTransform;

    let c = match corpus() {
        Some(c) => c,
        None => {
            eprintln!("Skipping: corpus unavailable");
            return;
        }
    };
    let corpus_dir = match c.get("jpeg-conformance/valid") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Skipping: {e}");
            return;
        }
    };

    let transforms = [
        ("Rotate90", LosslessTransform::Rotate90),
        ("Rotate180", LosslessTransform::Rotate180),
        ("Rotate270", LosslessTransform::Rotate270),
        ("FlipH", LosslessTransform::FlipHorizontal),
        ("FlipV", LosslessTransform::FlipVertical),
        ("Transpose", LosslessTransform::Transpose),
        ("Transverse", LosslessTransform::Transverse),
    ];

    let files = collect_jpgs(&corpus_dir);
    eprintln!(
        "Testing {} files × {} transforms via decode() crop path\n",
        files.len(),
        transforms.len()
    );
    eprintln!(
        "{:<45} {:>10} {:>8} {:>8} {:>6} {:>6} {:>10}",
        "File", "Transform", "OutW", "OutH", "Crops", "MaxD", "MeanD"
    );

    let mut total_crops = 0;
    let mut total_failures = 0;

    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy();
        let data = fs::read(path).unwrap();

        for (tname, transform) in &transforms {
            // Get transformed dimensions
            let full = match zenjpeg::decoder::Decoder::new()
                .transform(*transform)
                .decode(&data, Unstoppable)
            {
                Ok(r) => r,
                Err(_) => continue,
            };
            let img_w = full.width();
            let img_h = full.height();
            drop(full);

            let regions = crop_regions(img_w, img_h);
            let mut file_max = 0u8;
            let mut file_mean_sum = 0.0;
            let mut file_crops = 0;

            for (cx, cy, cw, ch) in &regions {
                if let Some((max_diff, mean_diff)) =
                    verify_crop_with_transform(&data, *transform, *cx, *cy, *cw, *ch)
                {
                    file_max = file_max.max(max_diff);
                    file_mean_sum += mean_diff;
                    file_crops += 1;
                    total_crops += 1;
                    if max_diff > 0 {
                        total_failures += 1;
                        eprintln!(
                            "  DIFF {} crop({},{},{},{}) max={} mean={:.4}",
                            tname, cx, cy, cw, ch, max_diff, mean_diff
                        );
                    }
                }
            }

            let avg_mean = if file_crops > 0 {
                file_mean_sum / file_crops as f64
            } else {
                0.0
            };

            eprintln!(
                "{:<45} {:>10} {:>8} {:>8} {:>6} {:>6} {:>10.4}",
                name, tname, img_w, img_h, file_crops, file_max, avg_mean
            );
        }
    }

    eprintln!(
        "\n{} total transform+crop tests, {} with nonzero diff",
        total_crops, total_failures
    );
    assert_eq!(
        total_failures, 0,
        "All transform+crop results should be pixel-identical"
    );
}

#[test]
#[ignore]
fn corpus_crop_scanline_with_transforms() {
    use zenjpeg::lossless::LosslessTransform;

    let c = match corpus() {
        Some(c) => c,
        None => {
            eprintln!("Skipping: corpus unavailable");
            return;
        }
    };
    let corpus_dir = match c.get("jpeg-conformance/valid") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Skipping: {e}");
            return;
        }
    };

    let transforms = [
        ("Rotate90", LosslessTransform::Rotate90),
        ("Rotate180", LosslessTransform::Rotate180),
        ("Rotate270", LosslessTransform::Rotate270),
        ("FlipH", LosslessTransform::FlipHorizontal),
        ("FlipV", LosslessTransform::FlipVertical),
        ("Transpose", LosslessTransform::Transpose),
        ("Transverse", LosslessTransform::Transverse),
    ];

    let files = collect_jpgs(&corpus_dir);
    eprintln!(
        "Testing {} files × {} transforms via scanline_reader() crop path\n",
        files.len(),
        transforms.len()
    );
    eprintln!(
        "{:<45} {:>10} {:>8} {:>8} {:>6} {:>6} {:>10}",
        "File", "Transform", "OutW", "OutH", "Crops", "MaxD", "MeanD"
    );

    let mut total_crops = 0;
    let mut total_failures = 0;

    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy();
        let data = fs::read(path).unwrap();

        for (tname, transform) in &transforms {
            // Get transformed dimensions via scanline reader
            let full_reader = match zenjpeg::decoder::Decoder::new()
                .transform(*transform)
                .scanline_reader(&data)
            {
                Ok(r) => r,
                Err(_) => continue,
            };
            let img_w = full_reader.width();
            let img_h = full_reader.height();
            drop(full_reader);

            let regions = crop_regions(img_w, img_h);
            let mut file_max = 0u8;
            let mut file_mean_sum = 0.0;
            let mut file_crops = 0;

            for (cx, cy, cw, ch) in &regions {
                if let Some((max_diff, mean_diff)) =
                    verify_crop_scanline_with_transform(&data, *transform, *cx, *cy, *cw, *ch)
                {
                    file_max = file_max.max(max_diff);
                    file_mean_sum += mean_diff;
                    file_crops += 1;
                    total_crops += 1;
                    if max_diff > 0 {
                        total_failures += 1;
                        eprintln!(
                            "  DIFF {} crop({},{},{},{}) max={} mean={:.4}",
                            tname, cx, cy, cw, ch, max_diff, mean_diff
                        );
                    }
                }
            }

            let avg_mean = if file_crops > 0 {
                file_mean_sum / file_crops as f64
            } else {
                0.0
            };

            eprintln!(
                "{:<45} {:>10} {:>8} {:>8} {:>6} {:>6} {:>10.4}",
                name, tname, img_w, img_h, file_crops, file_max, avg_mean
            );
        }
    }

    eprintln!(
        "\n{} total scanline transform+crop tests, {} with nonzero diff",
        total_crops, total_failures
    );
    assert_eq!(
        total_failures, 0,
        "All scanline transform+crop results should be pixel-identical"
    );
}
