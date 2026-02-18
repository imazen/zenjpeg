use enough::Unstoppable;

fn main() {
    let data = std::fs::read("/tmp/cjpegli_prog_64.jpg").expect("read");

    // Decode with zenjpeg
    use zenjpeg::decode::Decoder;
    use zenjpeg::decoder::PixelFormat;
    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
    let result = decoder.decode(&data, Unstoppable).expect("decode");
    let zen = result.into_pixels_u8().expect("pixels");

    // Decode with zune-jpeg
    let mut zune_dec = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(&data));
    let zune = zune_dec.decode().expect("zune decode");

    // Read djpegli
    let dj_ppm = std::fs::read("/tmp/dj_64.ppm").expect("dj");
    let mut offset = 0;
    while dj_ppm[offset] != b'\n' {
        offset += 1;
    }
    offset += 1;
    while dj_ppm[offset] == b'#' {
        while dj_ppm[offset] != b'\n' {
            offset += 1;
        }
        offset += 1;
    }
    while dj_ppm[offset] != b'\n' {
        offset += 1;
    }
    offset += 1;
    while dj_ppm[offset] != b'\n' {
        offset += 1;
    }
    offset += 1;
    let dj = &dj_ppm[offset..];

    // Compare all pairs
    fn compare(a: &[u8], b: &[u8], name: &str) {
        let mut max_diff = 0u8;
        let mut sum = 0u64;
        let mut worst_idx = 0usize;
        for i in 0..a.len().min(b.len()) {
            let d = a[i].abs_diff(b[i]);
            sum += d as u64;
            if d > max_diff {
                max_diff = d;
                worst_idx = i;
            }
        }
        let pixel = worst_idx / 3;
        let channel = worst_idx % 3;
        let px = pixel % 64;
        let py = pixel / 64;
        eprintln!(
            "{}: max_diff={:3}, mean={:.2} (worst at pixel ({},{}) ch{} a={} b={})",
            name,
            max_diff,
            sum as f64 / a.len() as f64,
            px,
            py,
            channel,
            a[worst_idx],
            b[worst_idx]
        );
    }

    eprintln!(
        "zen: {} bytes, zune: {} bytes, dj: {} bytes",
        zen.len(),
        zune.len(),
        dj.len()
    );
    compare(&zen, &dj, "zenjpeg vs djpegli");
    compare(&zune, &dj, "zune    vs djpegli");
    compare(&zen, &zune, "zenjpeg vs zune   ");

    // Show first 4 pixels from each
    eprintln!("\nFirst 4 pixels:");
    for i in 0..4 {
        let idx = i * 3;
        eprintln!(
            "  px{}: zen=({},{},{}) zune=({},{},{}) dj=({},{},{})",
            i,
            zen[idx],
            zen[idx + 1],
            zen[idx + 2],
            zune[idx],
            zune[idx + 1],
            zune[idx + 2],
            dj[idx],
            dj[idx + 1],
            dj[idx + 2]
        );
    }
}
