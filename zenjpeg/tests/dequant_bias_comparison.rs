//! Compare decoder quality: zenjpeg default vs dequant_bias vs C++ jpegli vs zune-jpeg.
//!
//! Measures SSIMULACRA2 of each decoder's output against the original source image
//! to determine which decoder produces the highest fidelity reconstruction.
//! Higher SSIMULACRA2 = better quality. 100 = identical.
//!
//! Run with:
//! ```
//! cargo test --release -p zenjpeg --test dequant_bias_comparison --features decoder -- --nocapture --ignored
//! ```

#[cfg(feature = "decoder")]
mod comparison {
    use enough::Unstoppable;
    use fast_ssim2::compute_ssimulacra2;
    use imgref::ImgVec;
    use std::path::PathBuf;
    use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

    fn corpus_dir() -> Option<PathBuf> {
        let p = PathBuf::from(
            std::env::var("CODEC_CORPUS")
                .unwrap_or_else(|_| "/home/lilith/work/codec-eval/codec-corpus".into()),
        );
        if p.is_dir() {
            Some(p)
        } else {
            None
        }
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
        progressive: bool,
    ) -> Vec<u8> {
        let config =
            EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter).progressive(progressive);
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .expect("encoder init");
        enc.push_packed(pixels, Unstoppable).expect("push");
        enc.finish().expect("finish")
    }

    fn decode_zenjpeg(data: &[u8], bias: bool) -> Vec<u8> {
        let decoder = zenjpeg::decoder::Decoder::new().dequant_bias(bias);
        let img = decoder.decode(data, Unstoppable).expect("zenjpeg decode");
        img.data
    }

    fn decode_zune(data: &[u8]) -> Vec<u8> {
        use zune_jpeg::zune_core::bytestream::ZCursor;
        let cursor = ZCursor::new(data);
        let mut decoder = zune_jpeg::JpegDecoder::new(cursor);
        decoder.decode().expect("zune decode")
    }

    /// Decode using C++ jpegli via FFI (libjpeg-compatible API).
    unsafe fn decode_cjpegli(data: &[u8]) -> Vec<u8> {
        use jpegli_internals_sys::*;
        use std::mem::MaybeUninit;

        let mut err: MaybeUninit<jpeg_error_mgr> = MaybeUninit::zeroed();
        jpeg_std_error(err.as_mut_ptr());
        let mut err = err.assume_init();

        let mut cinfo: MaybeUninit<jpeg_decompress_struct> = MaybeUninit::zeroed();
        let cinfo_ptr = cinfo.as_mut_ptr();
        (*cinfo_ptr).err = &mut err;
        jpeg_CreateDecompress(
            cinfo_ptr,
            JPEG_LIB_VERSION as i32,
            std::mem::size_of::<jpeg_decompress_struct>(),
        );
        let cinfo_ptr = cinfo.as_mut_ptr();

        jpeg_mem_src(cinfo_ptr, data.as_ptr(), data.len() as _);
        jpeg_read_header(cinfo_ptr, 1);
        (*cinfo_ptr).out_color_space = JCS_EXT_RGB as u32;
        jpeg_start_decompress(cinfo_ptr);

        let width = (*cinfo_ptr).output_width as usize;
        let height = (*cinfo_ptr).output_height as usize;
        let components = (*cinfo_ptr).output_components as usize;
        let row_stride = width * components;
        let mut output = vec![0u8; height * row_stride];

        let mut row_ptrs = [std::ptr::null_mut::<u8>(); 8];
        while ((*cinfo_ptr).output_scanline as usize) < height {
            let start = (*cinfo_ptr).output_scanline as usize;
            let remaining = height - start;
            let count = remaining.min(8);
            for i in 0..count {
                row_ptrs[i] = output[(start + i) * row_stride..].as_mut_ptr();
            }
            jpeg_read_scanlines(cinfo_ptr, row_ptrs.as_mut_ptr(), count as u32);
        }

        jpeg_finish_decompress(cinfo_ptr);
        jpeg_destroy_decompress(cinfo_ptr);
        output
    }

    fn compute_ssim2(a: &[u8], b: &[u8], width: usize, height: usize) -> f64 {
        let to_pixels =
            |d: &[u8]| -> Vec<[u8; 3]> { d.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect() };
        let a_img = ImgVec::new(to_pixels(a), width, height);
        let b_img = ImgVec::new(to_pixels(b), width, height);
        compute_ssimulacra2(a_img.as_ref(), b_img.as_ref()).unwrap_or(0.0)
    }

    fn max_pixel_diff(a: &[u8], b: &[u8]) -> u8 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| x.abs_diff(y))
            .max()
            .unwrap_or(0)
    }

    fn mean_abs_diff(a: &[u8], b: &[u8]) -> f64 {
        let sum: u64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| x.abs_diff(y) as u64)
            .sum();
        sum as f64 / a.len() as f64
    }

    #[test]
    #[ignore]
    fn compare_decoder_quality() {
        let corpus = corpus_dir().expect("codec-corpus not found");
        let cid22 = corpus.join("CID22/CID22-512/validation");
        assert!(cid22.is_dir(), "CID22 validation dir not found");

        let mut images: Vec<PathBuf> = std::fs::read_dir(&cid22)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "png"))
            .collect();
        images.sort();
        let images = &images[..images.len().min(10)];

        let qualities = [50.0f32, 75.0, 85.0, 95.0];

        println!();
        println!("=== Decoder Quality Comparison (SSIMULACRA2 vs Original, higher = better) ===");
        println!();

        // Accumulators per decoder per quality
        // 0=zenjpeg, 1=zenjpeg+bias, 2=cjpegli, 3=zune
        let mut totals: Vec<[f64; 4]> = vec![[0.0; 4]; qualities.len()];
        let mut counts: Vec<usize> = vec![0; qualities.len()];

        for (qi, &quality) in qualities.iter().enumerate() {
            println!("--- Quality {} ---", quality);
            println!(
                "{:<20} {:>10} {:>10} {:>10} {:>10}  {:>8} {:>8}",
                "Image", "zenjpeg", "zen+bias", "cjpegli", "zune-jpeg", "max_diff", "mean_diff"
            );
            println!(
                "{:<20} {:>10} {:>10} {:>10} {:>10}  {:>8} {:>8}",
                "", "(SSIM2)", "(SSIM2)", "(SSIM2)", "(SSIM2)", "bias-cpp", "bias-cpp"
            );

            for img_path in images.iter() {
                let (rgb, width, height) = match load_png_rgb(img_path) {
                    Some(v) => v,
                    None => continue,
                };

                let jpeg = encode_jpeg(&rgb, width, height, quality, false);
                let w = width as usize;
                let h = height as usize;

                let pixels_zen = decode_zenjpeg(&jpeg, false);
                let pixels_bias = decode_zenjpeg(&jpeg, true);
                let pixels_cpp = unsafe { decode_cjpegli(&jpeg) };
                let pixels_zune = decode_zune(&jpeg);

                let scores = [
                    compute_ssim2(&rgb, &pixels_zen, w, h),
                    compute_ssim2(&rgb, &pixels_bias, w, h),
                    compute_ssim2(&rgb, &pixels_cpp, w, h),
                    compute_ssim2(&rgb, &pixels_zune, w, h),
                ];

                let max_diff_bias_cpp = max_pixel_diff(&pixels_bias, &pixels_cpp);
                let mean_diff_bias_cpp = mean_abs_diff(&pixels_bias, &pixels_cpp);

                let name = img_path.file_stem().unwrap().to_str().unwrap();
                println!(
                    "{:<20} {:>10.4} {:>10.4} {:>10.4} {:>10.4}  {:>8} {:>8.4}",
                    name,
                    scores[0],
                    scores[1],
                    scores[2],
                    scores[3],
                    max_diff_bias_cpp,
                    mean_diff_bias_cpp
                );

                for (i, &s) in scores.iter().enumerate() {
                    totals[qi][i] += s;
                }
                counts[qi] += 1;
            }

            if counts[qi] > 0 {
                let n = counts[qi] as f64;
                println!(
                    "{:<20} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                    "MEAN",
                    totals[qi][0] / n,
                    totals[qi][1] / n,
                    totals[qi][2] / n,
                    totals[qi][3] / n,
                );
            }
            println!();
        }

        // Summary
        println!("=== Summary: Mean SSIMULACRA2 vs Original (higher = better) ===");
        println!(
            "{:>8} {:>10} {:>10} {:>10} {:>10}  {:>12} {:>12}",
            "Quality", "zenjpeg", "zen+bias", "cjpegli", "zune-jpeg", "bias-zen", "bias-cpp"
        );
        for (qi, &quality) in qualities.iter().enumerate() {
            if counts[qi] == 0 {
                continue;
            }
            let n = counts[qi] as f64;
            let means: Vec<f64> = (0..4).map(|i| totals[qi][i] / n).collect();
            // Absolute SSIM2 point improvement of bias over default
            let bias_vs_zen = means[1] - means[0];
            // Absolute SSIM2 point difference of bias vs cpp
            let bias_vs_cpp = means[1] - means[2];
            println!(
                "{:>8.0} {:>10.4} {:>10.4} {:>10.4} {:>10.4}  {:>+12.4} {:>+12.4}",
                quality, means[0], means[1], means[2], means[3], bias_vs_zen, bias_vs_cpp
            );
        }

        println!();
        println!("Positive bias-zen = bias is HIGHER SSIM2 (better) than default zenjpeg");
        println!("Positive bias-cpp = bias is HIGHER SSIM2 (better) than C++ jpegli");
        println!("Units: SSIMULACRA2 points (100 = identical to original)");
    }

    /// Frymire quality sweep: single photographic image (1118x1105), fine quality steps.
    #[test]
    #[ignore]
    fn compare_decoder_quality_frymire() {
        let corpus = corpus_dir().expect("codec-corpus not found");
        let frymire = corpus.join("imageflow/test_inputs/frymire.png");
        assert!(frymire.exists(), "frymire.png not found at {:?}", frymire);

        let (rgb, width, height) = load_png_rgb(&frymire).expect("failed to load frymire.png");
        let w = width as usize;
        let h = height as usize;

        let qualities: Vec<f32> = vec![
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 87.0, 90.0, 92.0,
            95.0, 97.0, 99.0,
        ];

        println!();
        println!(
            "=== Frymire ({}x{}) Decoder Quality Sweep (SSIMULACRA2 vs Original) ===",
            width, height
        );
        println!();
        println!(
            "{:>5} {:>8} {:>10} {:>10} {:>10} {:>10}  {:>9} {:>9} {:>6}",
            "Q",
            "bytes",
            "zenjpeg",
            "zen+bias",
            "cjpegli",
            "zune-jpeg",
            "bias-zen",
            "bias-cpp",
            "maxdif"
        );

        for &quality in &qualities {
            let jpeg = encode_jpeg(&rgb, width, height, quality, false);
            let size = jpeg.len();

            let pixels_zen = decode_zenjpeg(&jpeg, false);
            let pixels_bias = decode_zenjpeg(&jpeg, true);
            let pixels_cpp = unsafe { decode_cjpegli(&jpeg) };
            let pixels_zune = decode_zune(&jpeg);

            let s_zen = compute_ssim2(&rgb, &pixels_zen, w, h);
            let s_bias = compute_ssim2(&rgb, &pixels_bias, w, h);
            let s_cpp = compute_ssim2(&rgb, &pixels_cpp, w, h);
            let s_zune = compute_ssim2(&rgb, &pixels_zune, w, h);

            let max_diff = max_pixel_diff(&pixels_bias, &pixels_cpp);

            println!(
                "{:>5.0} {:>8} {:>10.4} {:>10.4} {:>10.4} {:>10.4}  {:>+9.4} {:>+9.4} {:>6}",
                quality,
                size,
                s_zen,
                s_bias,
                s_cpp,
                s_zune,
                s_bias - s_zen,
                s_bias - s_cpp,
                max_diff
            );
        }

        println!();
        println!("bias-zen: SSIM2 gain of dequant_bias over default (positive = better)");
        println!(
            "bias-cpp: SSIM2 gap vs C++ jpegli (positive = bias better, negative = C++ better)"
        );
        println!("maxdif:   max pixel diff between zen+bias and cjpegli");
    }

    #[test]
    #[ignore]
    fn compare_decoder_pairwise() {
        let corpus = corpus_dir().expect("codec-corpus not found");
        let cid22 = corpus.join("CID22/CID22-512/validation");
        assert!(cid22.is_dir(), "CID22 validation dir not found");

        let mut images: Vec<PathBuf> = std::fs::read_dir(&cid22)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "png"))
            .collect();
        images.sort();
        let images = &images[..images.len().min(6)];

        let quality = 85.0f32;

        println!();
        println!(
            "=== Pairwise Decoder SSIMULACRA2 (between decoders, Q{}) ===",
            quality
        );
        println!("Higher SSIM2 = more similar. 100 = identical.");
        println!();

        let names = ["zenjpeg", "zen+bias", "cjpegli", "zune-jpeg"];
        let mut pair_totals = [[0.0f64; 4]; 4];
        let mut count = 0usize;

        for img_path in images.iter() {
            let (rgb, width, height) = match load_png_rgb(img_path) {
                Some(v) => v,
                None => continue,
            };

            let jpeg = encode_jpeg(&rgb, width, height, quality, false);
            let w = width as usize;
            let h = height as usize;

            let outputs: Vec<Vec<u8>> = vec![
                decode_zenjpeg(&jpeg, false),
                decode_zenjpeg(&jpeg, true),
                unsafe { decode_cjpegli(&jpeg) },
                decode_zune(&jpeg),
            ];

            let name = img_path.file_stem().unwrap().to_str().unwrap();
            println!("Image: {}", name);
            println!(
                "{:>12} {:>10} {:>10} {:>10} {:>10}",
                "", names[0], names[1], names[2], names[3]
            );
            for i in 0..4 {
                print!("{:>12}", names[i]);
                for j in 0..4 {
                    if i == j {
                        print!("{:>10}", "-");
                    } else if j > i {
                        let d = compute_ssim2(&outputs[i], &outputs[j], w, h);
                        pair_totals[i][j] += d;
                        print!("{:>10.2}", d);
                    } else {
                        print!("{:>10}", "");
                    }
                }
                println!();
            }
            count += 1;
            println!();
        }

        if count > 0 {
            let n = count as f64;
            println!(
                "=== Mean Pairwise SSIMULACRA2 (Q{}, {} images) ===",
                quality, count
            );
            println!(
                "{:>12} {:>10} {:>10} {:>10} {:>10}",
                "", names[0], names[1], names[2], names[3]
            );
            for i in 0..4 {
                print!("{:>12}", names[i]);
                for j in 0..4 {
                    if i == j {
                        print!("{:>10}", "-");
                    } else if j > i {
                        print!("{:>10.2}", pair_totals[i][j] / n);
                    } else {
                        print!("{:>10}", "");
                    }
                }
                println!();
            }
        }
    }
}
