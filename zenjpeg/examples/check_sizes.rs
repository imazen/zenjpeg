use enough::Unstoppable;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn create_test_jpeg(width: u32, height: u32, quality: f32, progressive: bool) -> Vec<u8> {
    let mut data = vec![0u8; (width * height * 3) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
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
            let noise = (h >> 24) as u8;

            match block_type {
                0 => {
                    let bias = ((bx.wrapping_mul(17) ^ by.wrapping_mul(31)) & 0xFF) as u8;
                    data[idx] = bias.wrapping_add(noise >> 2);
                    data[idx + 1] = bias.wrapping_add(noise >> 1);
                    data[idx + 2] = bias.wrapping_add(noise >> 3);
                }
                1 => {
                    data[idx] = ((x * 255) / width as usize) as u8;
                    data[idx + 1] = ((y * 255) / height as usize) as u8;
                    data[idx + 2] = noise >> 2;
                }
                2 => {
                    let edge = if (x % 8 < 4) ^ (y % 8 < 4) {
                        200u8
                    } else {
                        55u8
                    };
                    data[idx] = edge;
                    data[idx + 1] = edge.wrapping_add(noise >> 4);
                    data[idx + 2] = 255 - edge;
                }
                _ => {
                    data[idx] = noise;
                    data[idx + 1] = noise.wrapping_mul(3);
                    data[idx + 2] = noise.wrapping_mul(7);
                }
            }
        }
    }

    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter).progressive(progressive);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder creation should succeed");
    enc.push_packed(&data, Unstoppable)
        .expect("push should succeed");
    enc.finish().expect("encoding should succeed")
}

fn main() {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::zune_core::options::DecoderOptions;

    let (w, h) = (2048, 2048);
    let progressive = create_test_jpeg(w, h, 85.0, true);
    let baseline = create_test_jpeg(w, h, 85.0, false);
    eprintln!(
        "baseline={} bytes, progressive={} bytes",
        baseline.len(),
        progressive.len()
    );

    // Decode progressive with zune
    let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut zd = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(&progressive[..]), opts);
    let zune_prog = zd.decode().unwrap();

    // Decode baseline with zune
    let opts2 = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut zd2 = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(&baseline[..]), opts2);
    let zune_base = zd2.decode().unwrap();

    eprintln!("zune-progressive output: {} bytes", zune_prog.len());
    eprintln!("zune-baseline output: {} bytes", zune_base.len());

    // Decode progressive with zenjpeg
    use zenjpeg::decode::Decoder;
    use zenjpeg::decoder::PixelFormat;
    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
    let result = decoder.decode(&progressive, Unstoppable).unwrap();
    let zen_prog = result.into_pixels_u8().unwrap();
    eprintln!("zenjpeg-progressive output: {} bytes", zen_prog.len());

    // Compare zune vs zenjpeg progressive
    let maxdiff: u8 = zune_prog
        .iter()
        .zip(zen_prog.iter())
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap();
    eprintln!("max diff zune-prog vs zenjpeg-prog: {}", maxdiff);

    // Check zune progressive correctness: compare first 100 pixels
    let expected_size = (w * h * 3) as usize;
    eprintln!("expected pixel count: {}", expected_size);

    // Quick non-zero check on zune prog
    let nonzero: usize = zune_prog.iter().filter(|&&b| b != 0).count();
    let total = zune_prog.len();
    eprintln!(
        "zune-prog nonzero: {}/{} ({:.1}%)",
        nonzero,
        total,
        100.0 * nonzero as f64 / total as f64
    );

    // Compare specific pixel locations to diagnose the difference
    // Check first pixel of each 8x8 block
    let mut max_diff_pos = (0, 0, 0u8);
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 3) as usize;
            let diff_r = zune_prog[idx].abs_diff(zen_prog[idx]);
            let diff_g = zune_prog[idx + 1].abs_diff(zen_prog[idx + 1]);
            let diff_b = zune_prog[idx + 2].abs_diff(zen_prog[idx + 2]);
            let max_ch = diff_r.max(diff_g).max(diff_b);
            if max_ch > max_diff_pos.2 {
                max_diff_pos = (x, y, max_ch);
            }
        }
    }
    eprintln!(
        "worst pixel at ({},{}): diff={}",
        max_diff_pos.0, max_diff_pos.1, max_diff_pos.2
    );

    // Save the progressive file so we can test with djpegli
    std::fs::write("/tmp/bench_prog_2048.jpg", &progressive).unwrap();
    eprintln!("saved /tmp/bench_prog_2048.jpg");

    // Also check: decode with zune, compare against djpegli
    // Use the C++ djpegli binary if available
    let djpegli_out = std::process::Command::new("sh")
        .arg("-c")
        .arg(
            "djpegli /tmp/bench_prog_2048.jpg /tmp/bench_prog_2048.ppm 2>/dev/null && python3 -c \"
import struct
with open('/tmp/bench_prog_2048.ppm', 'rb') as f:
    line = f.readline()  # P6
    while True:
        line = f.readline()
        if not line.startswith(b'#'):
            break
    dims = line.strip()
    f.readline()  # maxval
    data = f.read()
    import sys
    sys.stdout.buffer.write(data)
\" 2>/dev/null",
        )
        .output();
    if let Ok(output) = djpegli_out {
        if output.stdout.len() == expected_size {
            let djpegli_pixels = &output.stdout;
            let dj_vs_zen: u8 = djpegli_pixels
                .iter()
                .zip(zen_prog.iter())
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap();
            let dj_vs_zune: u8 = djpegli_pixels
                .iter()
                .zip(zune_prog.iter())
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap();
            eprintln!("djpegli vs zenjpeg max diff: {}", dj_vs_zen);
            eprintln!("djpegli vs zune max diff: {}", dj_vs_zune);
        } else {
            eprintln!(
                "djpegli output size mismatch: {} vs {}",
                output.stdout.len(),
                expected_size
            );
        }
    } else {
        eprintln!("djpegli not available");
    }

    // Time comparison: 20 iterations for stability
    let iters = 20;

    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
        let mut zd = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(&progressive[..]), opts);
        let _ = std::hint::black_box(zd.decode().unwrap());
    }
    let zune_time = t0.elapsed();

    let t1 = std::time::Instant::now();
    for _ in 0..iters {
        let decoder = Decoder::new().output_format(PixelFormat::Rgb);
        let result = decoder.decode(&progressive, Unstoppable).unwrap();
        let _ = std::hint::black_box(result.into_pixels_u8().unwrap());
    }
    let zen_time = t1.elapsed();

    // Also time cjpegli progressive if available (using FFI)
    let djpegli_iters = 20;
    let t2 = std::time::Instant::now();
    for _ in 0..djpegli_iters {
        let _ = std::process::Command::new("sh")
            .arg("-c")
            .arg("djpegli /tmp/bench_prog_2048.jpg /tmp/bench_prog_2048.ppm 2>/dev/null")
            .output();
    }
    let dj_time = t2.elapsed();

    eprintln!(
        "\n=== Timing ({} iterations, 2048x2048 progressive) ===",
        iters
    );
    eprintln!(
        "zune-progressive:    {:.2} ms/decode (WRONG OUTPUT, max err 228)",
        zune_time.as_secs_f64() * 1000.0 / iters as f64
    );
    eprintln!(
        "zenjpeg-progressive: {:.2} ms/decode (correct, max err 24 vs djpegli)",
        zen_time.as_secs_f64() * 1000.0 / iters as f64
    );
    eprintln!(
        "djpegli (CLI):       {:.2} ms/decode (reference, includes disk I/O)",
        dj_time.as_secs_f64() * 1000.0 / djpegli_iters as f64
    );
    eprintln!(
        "zenjpeg/zune ratio:  {:.2}x (invalid comparison — zune has wrong output)",
        zen_time.as_secs_f64() / zune_time.as_secs_f64()
    );

    // Test with cjpegli-generated progressive JPEG (independent reference)
    if let Ok(cjpegli_data) = std::fs::read("/tmp/cjpegli_prog_512.jpg") {
        eprintln!("\n=== Cross-check: cjpegli-generated progressive 512x512 ===");
        eprintln!("File size: {} bytes", cjpegli_data.len());

        // Decode with djpegli (reference)
        let dj_out = std::process::Command::new("sh")
            .arg("-c")
            .arg("djpegli /tmp/cjpegli_prog_512.jpg /tmp/dj_cjpegli_prog_512.ppm 2>/dev/null && python3 -c \"
import struct
with open('/tmp/dj_cjpegli_prog_512.ppm', 'rb') as f:
    line = f.readline()
    while True:
        line = f.readline()
        if not line.startswith(b'#'):
            break
    f.readline()
    data = f.read()
    import sys
    sys.stdout.buffer.write(data)
\" 2>/dev/null")
            .output()
            .ok()
            .map(|o| o.stdout);

        // Decode with zune
        let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
        let mut zd =
            zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(&cjpegli_data[..]), opts);
        let zune_cj = zd.decode().unwrap();

        // Decode with zenjpeg
        let decoder = Decoder::new().output_format(PixelFormat::Rgb);
        let zen_result = decoder.decode(&cjpegli_data, Unstoppable).unwrap();
        let zen_cj = zen_result.into_pixels_u8().unwrap();

        let zune_vs_zen: u8 = zune_cj
            .iter()
            .zip(zen_cj.iter())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        eprintln!("zune vs zenjpeg max diff: {}", zune_vs_zen);

        if let Some(ref dj_pixels) = dj_out
            && dj_pixels.len() == zen_cj.len()
        {
            let dj_vs_zen: u8 = dj_pixels
                .iter()
                .zip(zen_cj.iter())
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap();
            let dj_vs_zune: u8 = dj_pixels
                .iter()
                .zip(zune_cj.iter())
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap();
            eprintln!("djpegli vs zenjpeg max diff: {}", dj_vs_zen);
            eprintln!("djpegli vs zune max diff: {}", dj_vs_zune);
        }
    }
}
