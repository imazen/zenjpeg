fn main() {
    use enough::Unstoppable;
    use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};
    use zenjpeg_bench_utils::{bytes_to_rgb, ImageData, QualityMetrics};

    let img = ImageData::from_path(std::path::Path::new(
        "/home/lilith/work/codec-eval/codec-corpus/gb82/baby-lossless.png",
    ))
    .unwrap();
    let reference = bytes_to_rgb(&img.pixels, img.width, img.height);
    eprintln!(
        "Image: {}x{}, pixels={}",
        img.width,
        img.height,
        img.pixels.len()
    );

    // Encode with zenjpeg Q90 (no auto_optimize)
    let config = EncoderConfig::ycbcr(90.0f32, ChromaSubsampling::Quarter);
    let mut e = config
        .encode_from_bytes(img.width as u32, img.height as u32, PixelLayout::Rgb8Srgb)
        .unwrap();
    e.push_packed(&img.pixels, Unstoppable).unwrap();
    let zen_jpeg = e.finish().unwrap();
    eprintln!("Encoded: {} bytes", zen_jpeg.len());

    // Save for external testing
    std::fs::write("/tmp/zen_diag_q90.jpg", &zen_jpeg).unwrap();

    // === Decode with 3 different decoders ===

    // 1. zune-jpeg
    {
        use zune_jpeg::zune_core::bytestream::ZCursor;
        use zune_jpeg::zune_core::colorspace::ColorSpace;
        use zune_jpeg::zune_core::options::DecoderOptions;
        let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
        let mut dec = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(&zen_jpeg), opts);
        let pixels = dec.decode().unwrap();
        let info = dec.info().unwrap();
        let mid =
            info.width as usize * (info.height as usize / 2) * 3 + (info.width as usize / 2) * 3;
        eprint!("[zune-jpeg] mid pixels: ");
        for &b in &pixels[mid..mid + 30] {
            eprint!("{:3} ", b);
        }
        eprintln!(" ({}x{})", info.width, info.height);
    }

    // 2. djpeg (libjpeg-turbo)
    {
        let djpeg = std::process::Command::new("djpeg")
            .arg("-ppm")
            .arg("/tmp/zen_diag_q90.jpg")
            .output()
            .unwrap();
        if djpeg.status.success() {
            let ppm = &djpeg.stdout;
            // Parse PPM header to get width/height
            let header = String::from_utf8_lossy(&ppm[..100]);
            let parts: Vec<&str> = header.split_whitespace().collect();
            let w: usize = parts[1].parse().unwrap();
            let h: usize = parts[2].parse().unwrap();
            // Skip header (3 newlines for P6)
            let mut pos = 0;
            let mut newlines = 0;
            while pos < ppm.len() && newlines < 3 {
                if ppm[pos] == b'\n' {
                    newlines += 1;
                }
                pos += 1;
            }
            let mid = w * (h / 2) * 3 + (w / 2) * 3;
            eprint!("[djpeg] mid pixels: ");
            for &b in &ppm[pos + mid..pos + mid + 30] {
                eprint!("{:3} ", b);
            }
            eprintln!(" ({}x{})", w, h);
        } else {
            eprintln!("[djpeg] FAILED: {}", String::from_utf8_lossy(&djpeg.stderr));
        }
    }

    // 3. zenjpeg's own decoder
    {
        let decoded = zenjpeg::decoder::Decoder::new()
            .decode(&zen_jpeg, Unstoppable)
            .unwrap();
        let pixels = decoded.pixels_u8().unwrap();
        let w = decoded.width() as usize;
        let h = decoded.height() as usize;
        let mid = w * (h / 2) * 3 + (w / 2) * 3;
        eprint!("[zenjpeg dec] mid pixels: ");
        for &b in &pixels[mid..mid + 30] {
            eprint!("{:3} ", b);
        }
        eprintln!(" ({}x{})", w, h);
    }

    // === Compare with original ===
    let mid = img.width * (img.height / 2) * 3 + (img.width / 2) * 3;
    eprint!("[original] mid pixels: ");
    for &b in &img.pixels[mid..mid + 30] {
        eprint!("{:3} ", b);
    }
    eprintln!();

    // === Also try encoding with cjpegli CLI for comparison ===
    {
        let status = std::process::Command::new("cjpegli")
            .arg("/home/lilith/work/codec-eval/codec-corpus/gb82/baby-lossless.png")
            .arg("/tmp/zen_diag_cjpegli_q90.jpg")
            .arg("-q")
            .arg("90")
            .status()
            .unwrap();
        if status.success() {
            let cpp_jpeg = std::fs::read("/tmp/zen_diag_cjpegli_q90.jpg").unwrap();
            // Decode with zune-jpeg
            use zune_jpeg::zune_core::bytestream::ZCursor;
            use zune_jpeg::zune_core::colorspace::ColorSpace;
            use zune_jpeg::zune_core::options::DecoderOptions;
            let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
            let mut dec = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(&cpp_jpeg), opts);
            let pixels = dec.decode().unwrap();
            let info = dec.info().unwrap();
            let mid = info.width as usize * (info.height as usize / 2) * 3
                + (info.width as usize / 2) * 3;
            eprint!("[cjpegli zune] mid pixels: ");
            for &b in &pixels[mid..mid + 30] {
                eprint!("{:3} ", b);
            }
            eprintln!(
                " ({}x{}, {} bytes)",
                info.width,
                info.height,
                cpp_jpeg.len()
            );
        }
    }

    // === Metrics comparison ===
    // Decode zen with zune for metrics (same pipeline as bench-utils)
    let zen_dec = zenjpeg_bench_utils::decode_jpeg_to_rgb(&zen_jpeg).unwrap();
    let zen_ba = QualityMetrics::butteraugli(reference.as_ref(), zen_dec.as_ref());
    let zen_ss2 = QualityMetrics::ssimulacra2(reference.as_ref(), zen_dec.as_ref());
    let zen_rms = QualityMetrics::rms(reference.as_ref(), zen_dec.as_ref());
    eprintln!(
        "[zune→metrics] BA={:.4}, SS2={:.2}, RMS={:.2}",
        zen_ba, zen_ss2, zen_rms
    );

    // Decode zen with zenjpeg decoder for metrics
    let zen_own_dec = zenjpeg::decoder::Decoder::new()
        .decode(&zen_jpeg, Unstoppable)
        .unwrap();
    let zen_own_rgb = bytes_to_rgb(
        zen_own_dec.pixels_u8().unwrap(),
        zen_own_dec.width() as usize,
        zen_own_dec.height() as usize,
    );
    let zen_own_ba = QualityMetrics::butteraugli(reference.as_ref(), zen_own_rgb.as_ref());
    let zen_own_ss2 = QualityMetrics::ssimulacra2(reference.as_ref(), zen_own_rgb.as_ref());
    let zen_own_rms = QualityMetrics::rms(reference.as_ref(), zen_own_rgb.as_ref());
    eprintln!(
        "[zenjpeg-dec→metrics] BA={:.4}, SS2={:.2}, RMS={:.2}",
        zen_own_ba, zen_own_ss2, zen_own_rms
    );
}
