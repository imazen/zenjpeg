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

/// Test: full-scale via scanline reader vs standard decode
/// This tells us if the scanline path itself has issues independent of shrink.
#[test]
fn scanline_full_vs_standard() {
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

    // Standard decode (parser path)
    let standard = Decoder::new().decode(&jpeg, Unstoppable).unwrap();
    let sp = standard.pixels_u8().unwrap();
    eprintln!("Standard: {}x{}", standard.width(), standard.height());

    // Scanline decode at full scale (via shrink hint with Full)
    let mut reader = Decoder::new()
        .shrink(ShrinkHint::ExactScale(DctScale::Full))
        .scanline_reader(&jpeg)
        .unwrap();
    let hw = reader.width() as usize;
    let hh = reader.height() as usize;
    let mut scanline_pixels = vec![0u8; hw * hh * 3];
    let mut rows_read = 0;
    while rows_read < hh {
        let out = imgref::ImgRefMut::new(
            &mut scanline_pixels[rows_read * hw * 3..],
            hw * 3,
            hh - rows_read,
        );
        let count = reader.read_rows_rgb8(out).unwrap();
        if count == 0 {
            break;
        }
        rows_read += count;
    }
    eprintln!("Scanline Full: {}x{}", hw, hh);

    // Compare first row pixel by pixel
    let mut max_diff = 0i32;
    for x in 0..hw {
        for c in 0..3 {
            let sv = scanline_pixels[x * 3 + c] as i32;
            let stv = sp[x * 3 + c] as i32;
            let diff = (sv - stv).abs();
            if diff > max_diff {
                max_diff = diff;
                if diff > 2 {
                    eprintln!("  DIFF at x={x} c={c}: scanline={sv} standard={stv} diff={diff}");
                }
            }
        }
    }
    eprintln!("Full-scale scanline vs standard: max_diff={max_diff}");

    // Now half scale
    let half = Decoder::new()
        .shrink(ShrinkHint::ExactScale(DctScale::Half))
        .decode(&jpeg, Unstoppable)
        .unwrap();
    let hp = half.pixels_u8().unwrap();
    let halfw = half.width() as usize;
    let halfh = half.height() as usize;
    eprintln!("Half: {}x{}", halfw, halfh);

    // Area-average the SCANLINE full-scale output for comparison with half
    let mut max_diff_half = 0i32;
    for y in 0..halfh {
        for x in 0..halfw {
            for c in 0..3 {
                let mut sum = 0u32;
                let mut cnt = 0u32;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let fy = y * 2 + dy;
                        let fx = x * 2 + dx;
                        if fy < hh && fx < hw {
                            sum += scanline_pixels[(fy * hw + fx) * 3 + c] as u32;
                            cnt += 1;
                        }
                    }
                }
                let avg = (sum + cnt / 2) / cnt;
                let sv = hp[(y * halfw + x) * 3 + c] as i32;
                let diff = (sv - avg as i32).abs();
                if diff > max_diff_half {
                    max_diff_half = diff;
                    if diff > 10 {
                        eprintln!(
                            "  HALF DIFF at ({x},{y}) c={c}: shrink={sv} avg={avg} diff={diff}"
                        );
                    }
                }
            }
        }
    }
    eprintln!("Half vs area-avg-of-scanline-full: max_diff={max_diff_half}");
}
