//! Try cjpegli with explicit distance `-d 15.267` vs `-q 10` — same thing?
//! Also try a couple of neighbors to see size sensitivity.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let png_path = manifest.join("tests/images/frymire.png");
    let img = image::open(&png_path).unwrap().to_rgb8();
    let (w, h) = img.dimensions();
    let rgb = img.into_raw();

    let ppm = "/tmp/q10_ddist.ppm";
    let mut f = fs::File::create(ppm).unwrap();
    writeln!(f, "P6\n{w} {h}\n255").unwrap();
    f.write_all(&rgb).unwrap();
    drop(f);
    let cjpegli = manifest.parent().unwrap().join("internal/jpegli-cpp/build/tools/cjpegli");

    // Reference Rust encode at Q=10
    let config = EncoderConfig::ycbcr(10.0, ChromaSubsampling::Quarter).progressive(false);
    let mut e = config.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).unwrap();
    e.push_packed(&rgb, enough::Unstoppable).unwrap();
    let rust = e.finish().unwrap();
    println!("zen Q=10.0 baseline: {} bytes", rust.len());

    let cases = [
        ("-q 10 -p 0",   vec!["-q", "10", "-p", "0"]),
        ("-d 15.267 -p 0", vec!["-d", "15.267", "-p", "0"]),
        ("-d 14.0 -p 0", vec!["-d", "14.0", "-p", "0"]),
        ("-d 15.0 -p 0", vec!["-d", "15.0", "-p", "0"]),
        ("-d 16.0 -p 0", vec!["-d", "16.0", "-p", "0"]),
        ("-d 17.0 -p 0", vec!["-d", "17.0", "-p", "0"]),
    ];
    let out = "/tmp/q10_cpp_sweep.jpg";
    for (label, args) in &cases {
        let mut all: Vec<String> = vec![ppm.into(), out.into()];
        all.extend(args.iter().map(|s| s.to_string()));
        all.extend(["--chroma_subsampling".into(), "420".into()]);
        let _ = Command::new(&cjpegli).args(&all)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status().unwrap();
        let n = fs::metadata(out).unwrap().len();
        println!("  cjpegli {label:<22} → {n} bytes");
    }

    // And try at a few zen distance-equivalents by Quality approx
    // (cjpegli -q 10 → distance 15.267 per its log).
    let ds: [f32; 6] = [10.0, 5.0, 3.0, 1.0, 15.0, 16.0];
    for d in ds {
        // construct distance-like Quality
        let q = zenjpeg::encoder::Quality::ApproxButteraugli(d);
        let config = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter).progressive(false);
        let mut e = config.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).unwrap();
        e.push_packed(&rgb, enough::Unstoppable).unwrap();
        let z = e.finish().unwrap();
        println!("  zen distance≈{d} baseline: {} bytes", z.len());
    }
}
