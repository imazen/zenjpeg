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

#[test]
fn shrink_256x256_solid() {
    let w = 256u32;
    let h = 256u32;
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    // Solid red
    for i in (0..pixels.len()).step_by(3) {
        pixels[i] = 200;
        pixels[i + 1] = 50;
        pixels[i + 2] = 50;
    }
    let jpeg = encode_jpeg(&pixels, w, h, 90.0, ChromaSubsampling::None);
    let half = Decoder::new()
        .shrink(ShrinkHint::ExactScale(DctScale::Half))
        .decode(&jpeg, Unstoppable)
        .unwrap();
    let hp = half.pixels_u8().unwrap();
    let hw = half.width() as usize;
    let hh = half.height() as usize;
    eprintln!("Solid half: {}x{}", hw, hh);

    // All pixels should be very close to (200, 50, 50)
    let mut max_diff = 0i32;
    let mut bad = None;
    for y in 0..hh {
        for x in 0..hw {
            let i = (y * hw + x) * 3;
            for c in 0..3 {
                let expected = [200i32, 50, 50][c];
                let actual = hp[i + c] as i32;
                let diff = (actual - expected).abs();
                if diff > max_diff {
                    max_diff = diff;
                    if diff > 3 {
                        bad = Some((x, y, c, actual, expected));
                    }
                }
            }
        }
    }
    eprintln!("Solid: max_diff={max_diff}");
    if let Some((x, y, c, a, e)) = bad {
        eprintln!("First bad: ({x},{y}) c={c} actual={a} expected={e}");
    }
}

#[test]
fn shrink_256x256_multirow_check() {
    let w = 256u32;
    let h = 256u32;
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    let mut rng = 12345u64;
    let mut next = || -> u8 {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng >> 33) as u8
    };
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 3) as usize;
            let patch_x = (x / 32) as u8;
            let patch_y = (y / 32) as u8;
            pixels[idx] = patch_x
                .wrapping_mul(37)
                .wrapping_add(patch_y.wrapping_mul(71));
            pixels[idx + 1] = patch_x
                .wrapping_mul(53)
                .wrapping_add(patch_y.wrapping_mul(29));
            pixels[idx + 2] = patch_x
                .wrapping_mul(19)
                .wrapping_add(patch_y.wrapping_mul(97));
            let noise = next() / 8;
            pixels[idx] = pixels[idx].wrapping_add(noise);
            pixels[idx + 1] = pixels[idx + 1].wrapping_add(noise);
            pixels[idx + 2] = pixels[idx + 2].wrapping_add(noise);
        }
    }

    let jpeg = encode_jpeg(&pixels, w, h, 90.0, ChromaSubsampling::None);

    // Full decode
    let full = Decoder::new().decode(&jpeg, Unstoppable).unwrap();
    let fp = full.pixels_u8().unwrap();
    let fw = full.width() as usize;
    let fh = full.height() as usize;

    // Half decode
    let half = Decoder::new()
        .shrink(ShrinkHint::ExactScale(DctScale::Half))
        .decode(&jpeg, Unstoppable)
        .unwrap();
    let hp = half.pixels_u8().unwrap();
    let hw = half.width() as usize;
    let hh = half.height() as usize;

    eprintln!("Full: {}x{}, Half: {}x{}", fw, fh, hw, hh);

    // Check each MCU row strip (4 rows at half scale)
    let mcu_height = 4;
    let num_mcu_rows = (hh + mcu_height - 1) / mcu_height;

    for mcu_row in 0..num_mcu_rows {
        let mut max_diff = 0i32;
        let mut worst = None;

        for row_in_strip in 0..mcu_height {
            let y = mcu_row * mcu_height + row_in_strip;
            if y >= hh {
                break;
            }

            for x in 0..hw {
                for c in 0..3 {
                    // Area average of 2x2 from full
                    let mut sum = 0u32;
                    let mut cnt = 0u32;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let fy = y * 2 + dy;
                            let fx = x * 2 + dx;
                            if fy < fh && fx < fw {
                                sum += fp[(fy * fw + fx) * 3 + c] as u32;
                                cnt += 1;
                            }
                        }
                    }
                    let avg = (sum + cnt / 2) / cnt;
                    let sv = hp[(y * hw + x) * 3 + c] as i32;
                    let diff = (sv - avg as i32).abs();
                    if diff > max_diff {
                        max_diff = diff;
                        worst = Some((x, y, c, sv, avg as i32));
                    }
                }
            }
        }

        if max_diff > 10 {
            let (x, y, c, sv, avg) = worst.unwrap();
            eprintln!("  MCU row {mcu_row}: max_diff={max_diff} worst=({x},{y}) c={c} shrink={sv} avg={avg}");
        }
    }
}
