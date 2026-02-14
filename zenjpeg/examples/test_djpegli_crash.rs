//! Temporary: test which encoder settings trigger the djpegli crash.

fn main() {
    use enough::Unstoppable;
    use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

    let size = 256u32;
    let mut data = vec![0u8; (size * size * 3) as usize];
    for y in 0..size as usize {
        for x in 0..size as usize {
            let idx = (y * size as usize + x) * 3;
            let bx = (x / 8) as u32;
            let by = (y / 8) as u32;
            let block_hash = bx
                .wrapping_mul(2654435761)
                .wrapping_add(by.wrapping_mul(40503));
            let block_type = block_hash % 4;
            let px = x as u32;
            let py = y as u32;
            let mut h = px
                .wrapping_mul(374761393)
                .wrapping_add(py.wrapping_mul(668265263));
            h = (h ^ (h >> 13)).wrapping_mul(1274126177);
            let n = (h >> 24) as u8;
            match block_type {
                0 => {
                    let b = ((bx.wrapping_mul(17) ^ by.wrapping_mul(31)) & 0xFF) as u8;
                    data[idx] = b.wrapping_add(n >> 2);
                    data[idx + 1] = b.wrapping_add(n >> 1);
                    data[idx + 2] = b.wrapping_add(n >> 3);
                }
                1 => {
                    data[idx] = ((x * 255) / size as usize) as u8;
                    data[idx + 1] = ((y * 255) / size as usize) as u8;
                    data[idx + 2] = n >> 2;
                }
                2 => {
                    let e = if (x % 8 < 4) ^ (y % 8 < 4) {
                        200u8
                    } else {
                        55u8
                    };
                    data[idx] = e;
                    data[idx + 1] = e.wrapping_add(n >> 4);
                    data[idx + 2] = 255 - e;
                }
                _ => {
                    data[idx] = n;
                    data[idx + 1] = n.wrapping_mul(3);
                    data[idx + 2] = n.wrapping_mul(7);
                }
            }
        }
    }

    // Also generate smooth gradient data
    let mut smooth = vec![0u8; (size * size * 3) as usize];
    for y in 0..size as usize {
        for x in 0..size as usize {
            let idx = (y * size as usize + x) * 3;
            smooth[idx] = ((x * 255) / size as usize) as u8;
            smooth[idx + 1] = ((y * 255) / size as usize) as u8;
            smooth[idx + 2] = 128;
        }
    }

    // Random noise (pure noise, no patches)
    let mut pure_noise = vec![0u8; (size * size * 3) as usize];
    for y in 0..size as usize {
        for x in 0..size as usize {
            let idx = (y * size as usize + x) * 3;
            let px = x as u32;
            let py = y as u32;
            let mut h = px
                .wrapping_mul(374761393)
                .wrapping_add(py.wrapping_mul(668265263));
            h = (h ^ (h >> 13)).wrapping_mul(1274126177);
            let n = (h >> 24) as u8;
            pure_noise[idx] = n;
            pure_noise[idx + 1] = n.wrapping_mul(3);
            pure_noise[idx + 2] = n.wrapping_mul(7);
        }
    }

    let config = || {
        EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
            .progressive(true)
            .restart_mcu_rows(0)
    };

    let variants: Vec<(&str, EncoderConfig, &[u8], u32)> = vec![
        ("noise_256", config(), &data, 256),
        ("noise_16", config(), &data, 16),
        ("smooth_256", config(), &smooth, 256),
        ("smooth_16", config(), &smooth, 16),
        ("pure_noise_256", config(), &pure_noise, 256),
        ("pure_noise_16", config(), &pure_noise, 16),
        // Actual baseline (non-progressive)
        (
            "noise_baseline",
            EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
                .progressive(false)
                .restart_mcu_rows(0),
            &data,
            256,
        ),
    ];

    for (name, config, pixels, sz) in &variants {
        let (w, h) = (*sz, *sz);
        let pix = &pixels[..(w * h * 3) as usize];

        let mut enc = config
            .clone()
            .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(pix, Unstoppable).unwrap();
        let jpeg = enc.finish().unwrap();
        let path = format!("/tmp/djpegli_test_{}.jpg", name);
        std::fs::write(&path, &jpeg).unwrap();
        eprintln!("{}: {} bytes -> {}", name, jpeg.len(), path);
    }
}
