use enough::Unstoppable;
use zenjpeg::decoder::{DctScale, Decoder, ShrinkHint};
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn encode_jpeg(pixels: &[u8], w: u32, h: u32, quality: f32, sub: ChromaSubsampling) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(quality, sub);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(pixels, Unstoppable).unwrap();
    enc.finish().unwrap()
}

fn encode_grayscale(pixels: &[u8], w: u32, h: u32, quality: f32) -> Vec<u8> {
    let config = EncoderConfig::grayscale(quality);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Gray8Srgb)
        .unwrap();
    enc.push_packed(pixels, Unstoppable).unwrap();
    enc.finish().unwrap()
}

/// Test grayscale to eliminate color conversion from the equation.
#[test]
fn shrink_grayscale_quality() {
    for w in [64u32, 72, 80, 96, 128] {
        let h = 16u32;
        let mut pixels = vec![0u8; (w * h) as usize];
        let mut rng = 42u64;
        let mut next = || -> u8 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 33) as u8
        };
        for y in 0..h {
            for x in 0..w {
                let bx = (x / 8) as u8;
                let by = (y / 8) as u8;
                pixels[(y * w + x) as usize] = bx
                    .wrapping_mul(37)
                    .wrapping_add(by.wrapping_mul(71))
                    .wrapping_add(next() / 32);
            }
        }

        let jpeg = encode_grayscale(&pixels, w, h, 95.0);

        // Full decode
        let full = Decoder::new().decode(&jpeg, Unstoppable).unwrap();
        let fp = full.pixels_u8().unwrap();
        let fw = full.width() as usize;
        // Gray decoded as RGB: 3 bytes per pixel, all channels same
        let fp_gray: Vec<u8> = fp.chunks_exact(3).map(|c| c[0]).collect();

        // Half decode
        let half = Decoder::new()
            .shrink(ShrinkHint::ExactScale(DctScale::Half))
            .decode(&jpeg, Unstoppable)
            .unwrap();
        let hp = half.pixels_u8().unwrap();
        let hw = half.width() as usize;
        let hh = half.height() as usize;
        // Gray decoded as RGB
        let hp_gray: Vec<u8> = hp.chunks_exact(3).map(|c| c[0]).collect();

        let mut max_diff = 0i32;
        let mut first_bad = None;
        for y in 0..hh {
            for x in 0..hw {
                let mut sum = 0u32;
                let mut cnt = 0u32;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let fy = y * 2 + dy;
                        let fx = x * 2 + dx;
                        if fy < full.height() as usize && fx < fw {
                            sum += fp_gray[fy * fw + fx] as u32;
                            cnt += 1;
                        }
                    }
                }
                let avg = (sum + cnt / 2) / cnt;
                let sv = hp_gray[y * hw + x] as i32;
                let diff = (sv - avg as i32).abs();
                if diff > 20 && first_bad.is_none() {
                    first_bad = Some((x, y, sv, avg));
                }
                max_diff = max_diff.max(diff);
            }
        }
        let bad_str = first_bad
            .map(|(x, y, sv, avg)| format!(", first_bad=({x},{y}) sv={sv} avg={avg}"))
            .unwrap_or_default();
        eprintln!(
            "  gray w={w} ({} MCUs) → {hw}x{hh}: max_diff={max_diff}{bad_str}",
            w / 8
        );
    }
}

/// Test with RGB 4:4:4, print the actual Y strip values for diagnosis.
#[test]
fn shrink_rgb_strip_diagnosis() {
    let w = 72u32;
    let h = 8u32;
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    let mut rng = 42u64;
    let mut next = || -> u8 {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng >> 33) as u8
    };
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 3) as usize;
            let bx = (x / 8) as u8;
            pixels[idx] = bx.wrapping_mul(37).wrapping_add(next() / 32);
            pixels[idx + 1] = bx
                .wrapping_mul(53)
                .wrapping_add(50)
                .wrapping_add(next() / 32);
            pixels[idx + 2] = bx
                .wrapping_mul(19)
                .wrapping_add(100)
                .wrapping_add(next() / 32);
        }
    }

    let jpeg = encode_jpeg(&pixels, w, h, 95.0, ChromaSubsampling::None);

    // Full decode
    let full = Decoder::new().decode(&jpeg, Unstoppable).unwrap();
    let fp = full.pixels_u8().unwrap();
    let fw = full.width() as usize;

    // Half decode via scanline reader (to see raw strip values)
    let mut reader = Decoder::new()
        .shrink(ShrinkHint::ExactScale(DctScale::Half))
        .scanline_reader(&jpeg)
        .unwrap();
    let hw = reader.width() as usize;
    let hh = reader.height() as usize;
    let mut hp = vec![0u8; hw * hh * 3];
    let mut rows_read = 0;
    while rows_read < hh {
        let out = imgref::ImgRefMut::new(&mut hp[rows_read * hw * 3..], hw * 3, hh - rows_read);
        let count = reader.read_rows_rgb8(out).unwrap();
        if count == 0 {
            break;
        }
        rows_read += count;
    }

    eprintln!("Full: {fw}x{}, Half: {hw}x{hh}", full.height());

    // Print first row of shrink output at block boundaries
    eprintln!("Half first row RGB at block boundaries:");
    for block in 0..(hw / 4 + 1).min(10) {
        let x = block * 4;
        if x >= hw {
            break;
        }
        let i = x * 3;
        eprintln!(
            "  block {block} (x={x}): ({},{},{})",
            hp[i],
            hp[i + 1],
            hp[i + 2]
        );
    }

    // Full area-averaged first row at same positions
    eprintln!("Full area-avg first row at block boundaries:");
    for block in 0..(hw / 4 + 1).min(10) {
        let x = block * 4;
        if x >= hw {
            break;
        }
        let fx = x * 2;
        let mut r = 0u32;
        let mut g = 0u32;
        let mut b = 0u32;
        for dy in 0..2 {
            for dx in 0..2 {
                let idx = (dy * fw + fx + dx) * 3;
                r += fp[idx] as u32;
                g += fp[idx + 1] as u32;
                b += fp[idx + 2] as u32;
            }
        }
        eprintln!(
            "  block {block} (x={x}): ({},{},{})",
            (r + 2) / 4,
            (g + 2) / 4,
            (b + 2) / 4
        );
    }

    // Compute max diff per block
    eprintln!("\nPer-block max diff (shrink vs area-avg of full):");
    for block in 0..(hw / 4).min(10) {
        let mut max_diff = 0i32;
        for by in 0..4.min(hh) {
            for bx in 0..4 {
                let x = block * 4 + bx;
                let y = by;
                if x >= hw || y >= hh {
                    continue;
                }
                for c in 0..3 {
                    let mut sum = 0u32;
                    let mut cnt = 0u32;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let fy = y * 2 + dy;
                            let fx = x * 2 + dx;
                            if fy < full.height() as usize && fx < fw {
                                sum += fp[(fy * fw + fx) * 3 + c] as u32;
                                cnt += 1;
                            }
                        }
                    }
                    let avg = (sum + cnt / 2) / cnt;
                    let sv = hp[(y * hw + x) * 3 + c] as i32;
                    let diff = (sv - avg as i32).abs();
                    max_diff = max_diff.max(diff);
                }
            }
        }
        eprintln!("  block {block}: max_diff={max_diff}");
    }
}
