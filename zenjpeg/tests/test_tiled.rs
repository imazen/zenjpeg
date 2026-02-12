use enough::Unstoppable;
use zenjpeg::decoder::{DctScale, Decoder, ShrinkHint};
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn encode_jpeg(pixels: &[u8], w: u32, h: u32, quality: f32, sub: ChromaSubsampling) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(quality, sub);
    let mut enc = config.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).unwrap();
    enc.push_packed(pixels, Unstoppable).unwrap();
    enc.finish().unwrap()
}

/// Create a tiled image where every 8x8 block has identical content.
/// After JPEG encoding, all blocks should have identical coefficients.
/// If the half-scale output differs between blocks, the issue is in
/// buffer layout/offset, not in the IDCT math.
#[test]
fn shrink_tiled_uniform_blocks() {
    let tile: [u8; 8 * 8 * 3] = {
        let mut t = [0u8; 192];
        for y in 0..8 {
            for x in 0..8 {
                let idx = (y * 8 + x) * 3;
                t[idx] = (x * 30 + y * 10) as u8;
                t[idx + 1] = (x * 10 + y * 30 + 50) as u8;
                t[idx + 2] = (200u8).wrapping_sub((x * 15 + y * 15) as u8);
            }
        }
        t
    };

    // 80 wide = 10 MCU columns, exceeds the 8-MCU threshold
    let w = 80u32;
    let h = 8u32;
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let tx = (x % 8) as usize;
            let ty = (y % 8) as usize;
            let src = (ty * 8 + tx) * 3;
            let dst = ((y * w + x) * 3) as usize;
            pixels[dst] = tile[src];
            pixels[dst + 1] = tile[src + 1];
            pixels[dst + 2] = tile[src + 2];
        }
    }

    let jpeg = encode_jpeg(&pixels, w, h, 95.0, ChromaSubsampling::None);

    // Half-scale decode
    let half = Decoder::new()
        .shrink(ShrinkHint::ExactScale(DctScale::Half))
        .decode(&jpeg, Unstoppable).unwrap();
    let hp = half.pixels_u8().unwrap();
    let hw = half.width() as usize;
    let hh = half.height() as usize;
    eprintln!("Half: {}x{}", hw, hh);

    // Block size is 4 for half scale
    let bs = 4;
    let num_blocks = hw / bs;

    // Print first row of each block
    eprintln!("First pixel of each 4-pixel block (row 0):");
    for b in 0..num_blocks {
        let x = b * bs;
        let i = x * 3;
        eprintln!("  block {b} (x={x}): R={} G={} B={}", hp[i], hp[i+1], hp[i+2]);
    }

    // Compare all blocks against block 0 — they should all be identical
    // (tiled input means identical DCT coefficients per block)
    let mut max_diff = 0i32;
    let mut first_bad_block = None;
    for b in 1..num_blocks {
        for row in 0..hh.min(bs) {
            for col in 0..bs {
                for c in 0..3 {
                    let x0 = col;
                    let x1 = b * bs + col;
                    let v0 = hp[(row * hw + x0) * 3 + c] as i32;
                    let v1 = hp[(row * hw + x1) * 3 + c] as i32;
                    let diff = (v0 - v1).abs();
                    if diff > max_diff {
                        max_diff = diff;
                        if diff > 2 && first_bad_block.is_none() {
                            first_bad_block = Some((b, row, col, c, v0, v1, diff));
                        }
                    }
                }
            }
        }
    }

    eprintln!("Max diff between blocks: {max_diff}");
    if let Some((b, row, col, c, v0, v1, diff)) = first_bad_block {
        eprintln!("First bad: block {b} ({row},{col}) c={c}: block0={v0} block{b}={v1} diff={diff}");
    }

    // With tiled input and uniform blocks, max diff should be very small (just JPEG compression noise)
    assert!(max_diff <= 5, "Tiled blocks should produce identical output, got max_diff={max_diff}");
}
