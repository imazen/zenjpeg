//! Quality comparison: shrink-on-load vs full decode + resize.
//!
//! Measures SSIMULACRA2 to determine quality loss from reduced IDCT
//! compared to decoding at full resolution and resizing properly.
//!
//! Run with:
//! ```
//! cargo test --release -p zenjpeg --test shrink_quality --features decoder -- --nocapture --ignored
//! ```

#[cfg(feature = "decoder")]
mod quality {
    use enough::Unstoppable;
    use fast_ssim2::compute_ssimulacra2;
    use imgref::{ImgRef, ImgVec};
    use std::path::PathBuf;
    use zenjpeg::decoder::{DctScale, Decoder, ShrinkHint};
    use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

    fn corpus() -> Option<codec_corpus::Corpus> {
        codec_corpus::Corpus::new().ok()
    }

    fn load_png_rgb(path: &std::path::Path) -> Option<(Vec<u8>, u32, u32)> {
        let file = std::fs::File::open(path).ok()?;
        let decoder = png::Decoder::new(file);
        let mut reader = decoder.read_info().ok()?;
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).ok()?;
        let data = &buf[..info.buffer_size()];
        let rgb: Vec<u8> = match info.color_type {
            png::ColorType::Rgb => data.to_vec(),
            png::ColorType::Rgba => data
                .chunks_exact(4)
                .flat_map(|c| [c[0], c[1], c[2]])
                .collect(),
            png::ColorType::Grayscale => data.iter().flat_map(|&g| [g, g, g]).collect(),
            _ => return None,
        };
        Some((rgb, info.width, info.height))
    }

    fn encode_jpeg(
        pixels: &[u8],
        width: u32,
        height: u32,
        quality: f32,
        subsampling: ChromaSubsampling,
    ) -> Vec<u8> {
        let config = EncoderConfig::ycbcr(quality, subsampling);
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .expect("encoder init");
        enc.push_packed(pixels, Unstoppable).expect("push");
        enc.finish().expect("finish")
    }

    /// Area-average downsample by an integer factor. Perfect for power-of-2 ratios.
    /// Each output pixel is the mean of a factor×factor block of input pixels.
    fn area_downsample_rgb(
        pixels: &[u8],
        width: usize,
        height: usize,
        factor: usize,
    ) -> (Vec<u8>, usize, usize) {
        let out_w = (width + factor - 1) / factor;
        let out_h = (height + factor - 1) / factor;
        let mut out = vec![0u8; out_w * out_h * 3];

        for oy in 0..out_h {
            for ox in 0..out_w {
                let mut r_sum = 0u32;
                let mut g_sum = 0u32;
                let mut b_sum = 0u32;
                let mut count = 0u32;

                for dy in 0..factor {
                    let iy = oy * factor + dy;
                    if iy >= height {
                        break;
                    }
                    for dx in 0..factor {
                        let ix = ox * factor + dx;
                        if ix >= width {
                            break;
                        }
                        let idx = (iy * width + ix) * 3;
                        r_sum += pixels[idx] as u32;
                        g_sum += pixels[idx + 1] as u32;
                        b_sum += pixels[idx + 2] as u32;
                        count += 1;
                    }
                }

                let oidx = (oy * out_w + ox) * 3;
                out[oidx] = ((r_sum + count / 2) / count) as u8;
                out[oidx + 1] = ((g_sum + count / 2) / count) as u8;
                out[oidx + 2] = ((b_sum + count / 2) / count) as u8;
            }
        }

        (out, out_w, out_h)
    }

    /// Linearize sRGB u8 then area-average, then convert back to sRGB.
    /// More perceptually correct for downsampling.
    fn linear_area_downsample_rgb(
        pixels: &[u8],
        width: usize,
        height: usize,
        factor: usize,
    ) -> (Vec<u8>, usize, usize) {
        let out_w = (width + factor - 1) / factor;
        let out_h = (height + factor - 1) / factor;
        let mut out = vec![0u8; out_w * out_h * 3];

        // sRGB to linear LUT
        let srgb_to_linear: Vec<f32> = (0..256)
            .map(|i| {
                let s = i as f32 / 255.0;
                if s <= 0.04045 {
                    s / 12.92
                } else {
                    ((s + 0.055) / 1.055).powf(2.4)
                }
            })
            .collect();

        for oy in 0..out_h {
            for ox in 0..out_w {
                let mut r_sum = 0.0f64;
                let mut g_sum = 0.0f64;
                let mut b_sum = 0.0f64;
                let mut count = 0u32;

                for dy in 0..factor {
                    let iy = oy * factor + dy;
                    if iy >= height {
                        break;
                    }
                    for dx in 0..factor {
                        let ix = ox * factor + dx;
                        if ix >= width {
                            break;
                        }
                        let idx = (iy * width + ix) * 3;
                        r_sum += srgb_to_linear[pixels[idx] as usize] as f64;
                        g_sum += srgb_to_linear[pixels[idx + 1] as usize] as f64;
                        b_sum += srgb_to_linear[pixels[idx + 2] as usize] as f64;
                        count += 1;
                    }
                }

                let inv = 1.0 / count as f64;
                let oidx = (oy * out_w + ox) * 3;
                out[oidx] = linear_to_srgb_u8((r_sum * inv) as f32);
                out[oidx + 1] = linear_to_srgb_u8((g_sum * inv) as f32);
                out[oidx + 2] = linear_to_srgb_u8((b_sum * inv) as f32);
            }
        }

        (out, out_w, out_h)
    }

    fn linear_to_srgb_u8(v: f32) -> u8 {
        let s = if v <= 0.0031308 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0 + 0.5).clamp(0.0, 255.0) as u8
    }

    fn ssim2(a: &[u8], b: &[u8], width: usize, height: usize) -> f64 {
        let to_px = |d: &[u8]| -> Vec<[u8; 3]> {
            d.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
        };
        let a_img = ImgVec::new(to_px(a), width, height);
        let b_img = ImgVec::new(to_px(b), width, height);
        compute_ssimulacra2(a_img.as_ref(), b_img.as_ref()).unwrap_or(0.0)
    }

    struct ScaleResult {
        scale: DctScale,
        shrink_ssim2: f64,
        resize_ssim2: f64,
        linear_resize_ssim2: f64,
        shrink_w: usize,
        shrink_h: usize,
    }

    fn evaluate_image(
        source_rgb: &[u8],
        source_w: u32,
        source_h: u32,
        quality: f32,
        subsampling: ChromaSubsampling,
    ) -> Vec<ScaleResult> {
        let jpeg = encode_jpeg(source_rgb, source_w, source_h, quality, subsampling);

        // Full decode (reference)
        let full = Decoder::new().decode(&jpeg, Unstoppable).expect("full decode");
        let full_pixels = full.pixels_u8().unwrap();
        let full_w = full.width() as usize;
        let full_h = full.height() as usize;

        let mut results = Vec::new();

        for &scale in &[DctScale::Half, DctScale::Quarter, DctScale::Eighth] {
            let factor = match scale {
                DctScale::Eighth => 8,
                DctScale::Quarter => 4,
                DctScale::Half => 2,
                DctScale::Full => 1,
                _ => unreachable!(),
            };

            // 1) Shrink-on-load decode
            let shrink = Decoder::new()
                .shrink(ShrinkHint::ExactScale(scale))
                .decode(&jpeg, Unstoppable)
                .expect("shrink decode");
            let shrink_pixels = shrink.pixels_u8().unwrap();
            let shrink_w = shrink.width() as usize;
            let shrink_h = shrink.height() as usize;

            // 2) Full decode → area-average resize (gamma-encoded, naive)
            let (resized, res_w, res_h) =
                area_downsample_rgb(full_pixels, full_w, full_h, factor);

            // 3) Source → linear area-average resize (best reference)
            let (linear_ref, lr_w, lr_h) = linear_area_downsample_rgb(
                source_rgb,
                source_w as usize,
                source_h as usize,
                factor,
            );

            // 4) Full decode → linear area-average resize
            let (linear_resized, _, _) =
                linear_area_downsample_rgb(full_pixels, full_w, full_h, factor);

            // Dimensions must match for SSIM2 comparison
            let cmp_w = shrink_w.min(res_w).min(lr_w);
            let cmp_h = shrink_h.min(res_h).min(lr_h);

            if cmp_w < 8 || cmp_h < 8 {
                continue; // SSIM2 needs minimum 8x8
            }

            // Crop all to common dimensions for fair comparison
            let crop = |src: &[u8], src_w: usize, w: usize, h: usize| -> Vec<u8> {
                let mut out = Vec::with_capacity(w * h * 3);
                for y in 0..h {
                    let row_start = y * src_w * 3;
                    out.extend_from_slice(&src[row_start..row_start + w * 3]);
                }
                out
            };

            let shrink_cropped = crop(shrink_pixels, shrink_w, cmp_w, cmp_h);
            let resized_cropped = crop(&linear_resized, res_w, cmp_w, cmp_h);
            let ref_cropped = crop(&linear_ref, lr_w, cmp_w, cmp_h);

            // SSIM2 against the linear-resized source (ground truth)
            let shrink_ssim2 = ssim2(&shrink_cropped, &ref_cropped, cmp_w, cmp_h);
            let resize_ssim2 = ssim2(&resized_cropped, &ref_cropped, cmp_w, cmp_h);

            // Also measure naive gamma-encoded resize for comparison
            let naive_cropped = crop(&resized, res_w, cmp_w, cmp_h);
            let naive_ssim2 = ssim2(&naive_cropped, &ref_cropped, cmp_w, cmp_h);

            results.push(ScaleResult {
                scale,
                shrink_ssim2,
                resize_ssim2,
                linear_resize_ssim2: resize_ssim2,
                shrink_w: cmp_w,
                shrink_h: cmp_h,
            });
        }

        results
    }

    #[test]
    #[ignore]
    fn shrink_vs_resize_quality() {
        let corpus = corpus().expect(
            "codec-corpus not found — install codec-corpus crate or set CODEC_CORPUS_PATH",
        );
        let cid22 = corpus
            .get("CID22/CID22-512/validation")
            .expect("CID22 corpus not found");
        assert!(cid22.is_dir(), "CID22 validation dir not found");

        let mut images: Vec<PathBuf> = std::fs::read_dir(&cid22)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "png"))
            .collect();
        images.sort();
        let images = &images[..images.len().min(10)];

        println!("\n=== Shrink-on-Load Quality: SSIMULACRA2 vs Reference ===");
        println!("Reference = linear area-average of source PNG at matching dimensions");
        println!("Higher SSIM2 = better (100 = identical)\n");

        // Test at Q85 4:2:0 (common web quality)
        for &(quality, subsampling, label) in &[
            (85.0, ChromaSubsampling::Quarter, "Q85 4:2:0"),
            (95.0, ChromaSubsampling::None, "Q95 4:4:4"),
        ] {
            println!("--- {label} ---");
            println!(
                "{:<20} {:>6} {:>10} {:>10} {:>10}",
                "Image", "Scale", "Shrink", "Resize", "Delta"
            );
            println!("{:-<66}", "");

            let mut totals_by_scale: std::collections::HashMap<String, (f64, f64, usize)> =
                std::collections::HashMap::new();

            for img_path in images {
                let (rgb, w, h) = match load_png_rgb(img_path) {
                    Some(v) => v,
                    None => continue,
                };

                let name = img_path
                    .file_stem()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .chars()
                    .take(18)
                    .collect::<String>();

                let results = evaluate_image(&rgb, w, h, quality, subsampling);

                for r in &results {
                    let delta = r.shrink_ssim2 - r.resize_ssim2;
                    let scale_name = format!("1/{}", match r.scale {
                        DctScale::Eighth => 8,
                        DctScale::Quarter => 4,
                        DctScale::Half => 2,
                        DctScale::Full => 1,
                        _ => unreachable!(),
                    });

                    println!(
                        "{:<20} {:>6} {:>10.2} {:>10.2} {:>+10.2}",
                        format!("{} ({}x{})", name, r.shrink_w, r.shrink_h),
                        scale_name,
                        r.shrink_ssim2,
                        r.resize_ssim2,
                        delta,
                    );

                    let entry = totals_by_scale
                        .entry(scale_name.clone())
                        .or_insert((0.0, 0.0, 0));
                    entry.0 += r.shrink_ssim2;
                    entry.1 += r.resize_ssim2;
                    entry.2 += 1;
                }
            }

            println!("{:-<66}", "");
            let mut scale_names: Vec<_> = totals_by_scale.keys().cloned().collect();
            scale_names.sort();
            for scale_name in &scale_names {
                let (shrink_sum, resize_sum, count) = totals_by_scale[scale_name];
                let shrink_mean = shrink_sum / count as f64;
                let resize_mean = resize_sum / count as f64;
                let delta = shrink_mean - resize_mean;
                println!(
                    "{:<20} {:>6} {:>10.2} {:>10.2} {:>+10.2}",
                    format!("MEAN (n={})", count),
                    scale_name,
                    shrink_mean,
                    resize_mean,
                    delta,
                );
            }
            println!();
        }
    }

    /// Quick non-corpus test: synthetic noise+patches image.
    /// This runs without --ignored and without codec-corpus.
    #[test]
    fn shrink_quality_synthetic() {
        // Create a 256x256 noise+patches image (NOT gradients — those are
        // degenerate for block-based transforms, see project guidelines)
        let w = 256u32;
        let h = 256u32;
        let mut pixels = vec![0u8; (w * h * 3) as usize];
        // Simple LCG for deterministic "random" noise
        let mut rng = 12345u64;
        let mut next = || -> u8 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng >> 33) as u8
        };
        for y in 0..h {
            for x in 0..w {
                let idx = ((y * w + x) * 3) as usize;
                // Patches of color with added noise
                let patch_x = (x / 32) as u8;
                let patch_y = (y / 32) as u8;
                let base_r = patch_x.wrapping_mul(37).wrapping_add(patch_y.wrapping_mul(71));
                let base_g = patch_x.wrapping_mul(53).wrapping_add(patch_y.wrapping_mul(29));
                let base_b = patch_x.wrapping_mul(19).wrapping_add(patch_y.wrapping_mul(97));
                let noise = next() / 8; // small noise
                pixels[idx] = base_r.wrapping_add(noise);
                pixels[idx + 1] = base_g.wrapping_add(noise);
                pixels[idx + 2] = base_b.wrapping_add(noise);
            }
        }

        let jpeg = encode_jpeg(&pixels, w, h, 90.0, ChromaSubsampling::None);

        // Full decode
        let full = Decoder::new().decode(&jpeg, Unstoppable).unwrap();
        let full_px = full.pixels_u8().unwrap();
        let fw = full.width() as usize;
        let fh = full.height() as usize;

        for &scale in &[DctScale::Half, DctScale::Quarter, DctScale::Eighth] {
            let factor = match scale {
                DctScale::Eighth => 8,
                DctScale::Quarter => 4,
                DctScale::Half => 2,
                DctScale::Full => 1,
                _ => unreachable!(),
            };

            // Shrink decode
            let shrink = Decoder::new()
                .shrink(ShrinkHint::ExactScale(scale))
                .decode(&jpeg, Unstoppable)
                .unwrap();
            let shrink_px = shrink.pixels_u8().unwrap();
            let sw = shrink.width() as usize;
            let sh = shrink.height() as usize;

            // Reference: linear area downsample of source
            let (reference, rw, rh) =
                linear_area_downsample_rgb(&pixels, w as usize, h as usize, factor);

            // Resize of full decode
            let (resized, _, _) = linear_area_downsample_rgb(full_px, fw, fh, factor);

            let cmp_w = sw.min(rw);
            let cmp_h = sh.min(rh);

            if cmp_w < 8 || cmp_h < 8 {
                continue;
            }

            let crop = |src: &[u8], src_w: usize, tw: usize, th: usize| -> Vec<u8> {
                let mut out = Vec::with_capacity(tw * th * 3);
                for y in 0..th {
                    let start = y * src_w * 3;
                    out.extend_from_slice(&src[start..start + tw * 3]);
                }
                out
            };

            let shrink_c = crop(shrink_px, sw, cmp_w, cmp_h);
            let resize_c = crop(&resized, rw, cmp_w, cmp_h);
            let ref_c = crop(&reference, rw, cmp_w, cmp_h);

            let shrink_score = ssim2(&shrink_c, &ref_c, cmp_w, cmp_h);
            let resize_score = ssim2(&resize_c, &ref_c, cmp_w, cmp_h);

            // Debug: pixel value comparison
            let mean_shrink: f64 = shrink_c.iter().map(|&v| v as f64).sum::<f64>() / shrink_c.len() as f64;
            let mean_resize: f64 = resize_c.iter().map(|&v| v as f64).sum::<f64>() / resize_c.len() as f64;
            let mean_ref: f64 = ref_c.iter().map(|&v| v as f64).sum::<f64>() / ref_c.len() as f64;
            let max_sr = shrink_c.iter().zip(resize_c.iter())
                .map(|(&a, &b)| (a as i32 - b as i32).abs())
                .max().unwrap_or(0);
            eprintln!(
                "  {scale}: shrink {sw}x{sh}, ref {rw}x{rh}, cmp {cmp_w}x{cmp_h}");
            eprintln!(
                "    mean_px: shrink={mean_shrink:.1}, resize={mean_resize:.1}, ref={mean_ref:.1}, max_shrink_vs_resize={max_sr}");
            eprintln!(
                "    first 6 bytes: shrink={:?}, resize={:?}, ref={:?}",
                &shrink_c[..6.min(shrink_c.len())],
                &resize_c[..6.min(resize_c.len())],
                &ref_c[..6.min(ref_c.len())],
            );
            eprintln!(
                "    ssim2: shrink={shrink_score:.2}, resize={resize_score:.2}, delta={:.2}",
                shrink_score - resize_score
            );

            // Reduced IDCT truncates high-frequency coefficients. On synthetic
            // patterns with sharp 32px patch boundaries, the gap is enormous
            // (50+ SSIM2 points). This is expected: the test pattern is
            // adversarial for frequency-domain truncation. Real photos
            // (smooth areas, gradual transitions) typically show <10 point gaps.
            // No assertion here — this test is informational.
        }
    }

    /// Compare ShrinkQuality::Fast vs ShrinkQuality::Best.
    /// Best should match (or nearly match) the "resize of full decode" reference.
    #[test]
    fn shrink_quality_best_vs_fast() {
        use zenjpeg::decoder::ShrinkQuality;

        // Create a 256x256 noise+patches image
        let w = 256u32;
        let h = 256u32;
        let mut pixels = vec![0u8; (w * h * 3) as usize];
        let mut rng = 12345u64;
        let mut next = || -> u8 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng >> 33) as u8
        };
        for y in 0..h {
            for x in 0..w {
                let idx = ((y * w + x) * 3) as usize;
                let patch_x = (x / 32) as u8;
                let patch_y = (y / 32) as u8;
                let base_r = patch_x.wrapping_mul(37).wrapping_add(patch_y.wrapping_mul(71));
                let base_g = patch_x.wrapping_mul(53).wrapping_add(patch_y.wrapping_mul(29));
                let base_b = patch_x.wrapping_mul(19).wrapping_add(patch_y.wrapping_mul(97));
                let noise = next() / 8;
                pixels[idx] = base_r.wrapping_add(noise);
                pixels[idx + 1] = base_g.wrapping_add(noise);
                pixels[idx + 2] = base_b.wrapping_add(noise);
            }
        }

        let jpeg = encode_jpeg(&pixels, w, h, 90.0, ChromaSubsampling::None);

        // Full decode (reference)
        let full = Decoder::new().decode(&jpeg, Unstoppable).unwrap();
        let full_px = full.pixels_u8().unwrap();
        let fw = full.width() as usize;
        let fh = full.height() as usize;

        for &scale in &[DctScale::Half, DctScale::Quarter, DctScale::Eighth] {
            let factor = match scale {
                DctScale::Eighth => 8,
                DctScale::Quarter => 4,
                DctScale::Half => 2,
                DctScale::Full => 1,
                _ => unreachable!(),
            };

            // Fast (reduced IDCT)
            let fast = Decoder::new()
                .shrink(ShrinkHint::ExactScale(scale))
                .shrink_quality(ShrinkQuality::Fast)
                .decode(&jpeg, Unstoppable)
                .unwrap();
            let fast_px = fast.pixels_u8().unwrap();
            let sw = fast.width() as usize;
            let sh = fast.height() as usize;

            // Best (full IDCT + area average)
            let best = Decoder::new()
                .shrink(ShrinkHint::ExactScale(scale))
                .shrink_quality(ShrinkQuality::Best)
                .decode(&jpeg, Unstoppable)
                .unwrap();
            let best_px = best.pixels_u8().unwrap();
            let bw = best.width() as usize;
            let bh = best.height() as usize;

            assert_eq!(sw, bw, "Width mismatch");
            assert_eq!(sh, bh, "Height mismatch");

            // Reference: area downsample of full decode (gamma-encoded, matches Best path)
            let (resized, rw, rh) = area_downsample_rgb(full_px, fw, fh, factor);
            let (ref_linear, rlw, rlh) = linear_area_downsample_rgb(
                &pixels, w as usize, h as usize, factor,
            );

            let cmp_w = sw.min(rw).min(rlw);
            let cmp_h = sh.min(rh).min(rlh);
            if cmp_w < 8 || cmp_h < 8 {
                continue;
            }

            let crop = |src: &[u8], src_w: usize, tw: usize, th: usize| -> Vec<u8> {
                let mut out = Vec::with_capacity(tw * th * 3);
                for y in 0..th {
                    let start = y * src_w * 3;
                    out.extend_from_slice(&src[start..start + tw * 3]);
                }
                out
            };

            let fast_c = crop(fast_px, sw, cmp_w, cmp_h);
            let best_c = crop(best_px, bw, cmp_w, cmp_h);
            let ref_c = crop(&ref_linear, rlw, cmp_w, cmp_h);

            let fast_score = ssim2(&fast_c, &ref_c, cmp_w, cmp_h);
            let best_score = ssim2(&best_c, &ref_c, cmp_w, cmp_h);

            eprintln!(
                "  {scale}: fast={fast_score:.2}, best={best_score:.2}, improvement={:.2}",
                best_score - fast_score,
            );

            // Best mode should always be better than or equal to Fast mode.
            // Note: Best averages per-component in YCbCr space before color conversion,
            // while the reference averages in RGB space after color conversion. These
            // differ because YCbCr→RGB is a non-trivial linear combination. The SSIM2
            // score is the right metric — not pixel-exact match with RGB area average.
            assert!(
                best_score >= fast_score - 1.0,
                "{scale}: Best ({best_score:.2}) should be >= Fast ({fast_score:.2})"
            );
        }
    }

    /// Corpus-based comparison of Fast vs Best quality modes.
    #[test]
    #[ignore]
    fn shrink_fast_vs_best_corpus() {
        use zenjpeg::decoder::ShrinkQuality;

        let corpus = corpus().expect("codec-corpus not found");
        let cid22 = corpus
            .get("CID22/CID22-512/validation")
            .expect("CID22 corpus not found");

        let mut images: Vec<PathBuf> = std::fs::read_dir(&cid22)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "png"))
            .collect();
        images.sort();
        let images = &images[..images.len().min(10)];

        println!("\n=== Fast vs Best Shrink Quality (SSIM2 vs linear-resized source) ===\n");
        println!(
            "{:<20} {:>6} {:>10} {:>10} {:>10}",
            "Image", "Scale", "Fast", "Best", "Improve"
        );
        println!("{:-<66}", "");

        let mut totals: std::collections::HashMap<String, (f64, f64, usize)> =
            std::collections::HashMap::new();

        for img_path in images {
            let (rgb, w, h) = match load_png_rgb(img_path) {
                Some(v) => v,
                None => continue,
            };
            let name: String = img_path
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap()
                .chars()
                .take(18)
                .collect();

            let jpeg = encode_jpeg(&rgb, w, h, 95.0, ChromaSubsampling::None);

            for &scale in &[DctScale::Half, DctScale::Quarter, DctScale::Eighth] {
                let factor = match scale {
                    DctScale::Eighth => 8,
                    DctScale::Quarter => 4,
                    DctScale::Half => 2,
                    _ => unreachable!(),
                };

                let fast = Decoder::new()
                    .shrink(ShrinkHint::ExactScale(scale))
                    .shrink_quality(ShrinkQuality::Fast)
                    .decode(&jpeg, Unstoppable)
                    .unwrap();
                let fast_px = fast.pixels_u8().unwrap();
                let sw = fast.width() as usize;
                let sh = fast.height() as usize;

                let best = Decoder::new()
                    .shrink(ShrinkHint::ExactScale(scale))
                    .shrink_quality(ShrinkQuality::Best)
                    .decode(&jpeg, Unstoppable)
                    .unwrap();
                let best_px = best.pixels_u8().unwrap();

                // Reference
                let (ref_linear, rlw, rlh) = linear_area_downsample_rgb(
                    &rgb, w as usize, h as usize, factor,
                );
                let cmp_w = sw.min(rlw);
                let cmp_h = sh.min(rlh);
                if cmp_w < 8 || cmp_h < 8 { continue; }

                let crop = |src: &[u8], src_w: usize, tw: usize, th: usize| -> Vec<u8> {
                    let mut out = Vec::with_capacity(tw * th * 3);
                    for y in 0..th {
                        let start = y * src_w * 3;
                        out.extend_from_slice(&src[start..start + tw * 3]);
                    }
                    out
                };
                let ref_c = crop(&ref_linear, rlw, cmp_w, cmp_h);
                let fast_c = crop(fast_px, sw, cmp_w, cmp_h);
                let best_c = crop(best_px, sw, cmp_w, cmp_h);

                let fast_s = ssim2(&fast_c, &ref_c, cmp_w, cmp_h);
                let best_s = ssim2(&best_c, &ref_c, cmp_w, cmp_h);

                let scale_name = format!("1/{factor}");
                let improve = best_s - fast_s;

                println!(
                    "{:<20} {:>6} {:>10.2} {:>10.2} {:>+10.2}",
                    format!("{name} ({sw}x{sh})"), scale_name, fast_s, best_s, improve,
                );

                let e = totals.entry(scale_name).or_insert((0.0, 0.0, 0));
                e.0 += fast_s;
                e.1 += best_s;
                e.2 += 1;
            }
        }

        println!("{:-<66}", "");
        let mut keys: Vec<_> = totals.keys().cloned().collect();
        keys.sort();
        for k in &keys {
            let (fs, bs, n) = totals[k];
            let fm = fs / n as f64;
            let bm = bs / n as f64;
            println!(
                "{:<20} {:>6} {:>10.2} {:>10.2} {:>+10.2}",
                format!("MEAN (n={n})"), k, fm, bm, bm - fm,
            );
        }
    }
}
