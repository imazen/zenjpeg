//! Measure deringing's impact at each quality level.

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

    let ppm = "/tmp/q10_sweep.ppm";
    let mut f = fs::File::create(ppm).unwrap();
    writeln!(f, "P6\n{w} {h}\n255").unwrap();
    f.write_all(&rgb).unwrap();
    drop(f);
    let cjpegli = manifest.parent().unwrap().join("internal/jpegli-cpp/build/tools/cjpegli");
    let out = "/tmp/q10_sweep_cpp.jpg";

    println!("{:<5} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Q", "zen_def", "zen_nd", "cpp", "def-cpp", "nd-cpp", "def-nd");

    for q in [5u8, 10, 20, 30, 50, 70, 85, 95] {
        // zen default
        let c = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter).progressive(false);
        let mut e = c.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).unwrap();
        e.push_packed(&rgb, enough::Unstoppable).unwrap();
        let def = e.finish().unwrap();

        // zen no deringing
        let c = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
            .progressive(false)
            .deringing(false);
        let mut e = c.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).unwrap();
        e.push_packed(&rgb, enough::Unstoppable).unwrap();
        let nd = e.finish().unwrap();

        // cjpegli
        Command::new(&cjpegli)
            .args([ppm, out, "-q", &q.to_string(), "--chroma_subsampling", "420", "-p", "0"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status().unwrap();
        let cpp_n = fs::metadata(out).unwrap().len() as i64;

        println!("{:<5} {:>10} {:>10} {:>10} {:>+10} {:>+10} {:>+10}",
            q, def.len(), nd.len(), cpp_n,
            def.len() as i64 - cpp_n, nd.len() as i64 - cpp_n, def.len() as i64 - nd.len() as i64);
    }
}
