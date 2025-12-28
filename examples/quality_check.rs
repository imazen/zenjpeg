fn main() {
    let width = 64usize;
    let height = 64usize;

    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 4) % 256) as u8;
            let g = ((y * 4) % 256) as u8;
            let b = (((x + y) * 2) % 256) as u8;
            rgb.push(r);
            rgb.push(g);
            rgb.push(b);
        }
    }

    println!("Direct mozjpeg-oxide (default settings):");
    for q in [30u8, 50, 70, 85, 95] {
        let jpeg = mozjpeg_oxide::Encoder::new()
            .quality(q)
            .encode_rgb(&rgb, width as u32, height as u32)
            .unwrap();
        println!("  Q{}: {} bytes", q, jpeg.len());
    }

    println!("\nDirect mozjpeg-oxide (matching zenjpeg settings):");
    for q in [30u8, 50, 70, 85, 95] {
        // Match zenjpeg's settings: 4:4:4, Huffman opt, deringing based on image
        let use_deringing = zenjpeg::analysis::should_use_deringing(&rgb, width, height);
        let jpeg = mozjpeg_oxide::Encoder::new()
            .quality(q)
            .subsampling(mozjpeg_oxide::Subsampling::S444)
            .optimize_huffman(true)
            .overshoot_deringing(use_deringing)
            .encode_rgb(&rgb, width as u32, height as u32)
            .unwrap();
        println!("  Q{}: {} bytes (deringing={})", q, jpeg.len(), use_deringing);
    }

    println!("\nVia zenjpeg delegation:");
    for q in [30u8, 50, 70, 85, 95] {
        let jpeg = zenjpeg::Encoder::new()
            .quality(zenjpeg::Quality::Standard(q))
            .encode_rgb(&rgb, width, height)
            .unwrap();
        println!("  Q{}: {} bytes", q, jpeg.len());
    }
}
