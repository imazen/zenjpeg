//! Dump zenjpeg and cjpegli Q10 baseline 4:2:0 JPEGs side-by-side and break down
//! where the 2.3KB size gap lives.
//!
//! Run: `cargo run --release -p zenjpeg --features __test-utils --example q10_investigate_dump`
//!
//! Writes:
//!   /tmp/q10_zen.jpg        — zenjpeg encode
//!   /tmp/q10_cpp.jpg        — cjpegli encode (via CLI)
//!   /tmp/q10_zen.ppm        — temp PPM
//!
//! Prints per-marker byte breakdown.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let png_path = manifest.join("tests/images/frymire.png");

    // Load PNG via `image` crate (already a dev-dep in the workspace).
    let img = image::open(&png_path).expect("load frymire.png").to_rgb8();
    let (w, h) = img.dimensions();
    let rgb = img.into_raw();
    eprintln!("Loaded frymire.png: {w}x{h}, rgb {} bytes", rgb.len());

    // --- Rust encode ---------------------------------------------------------
    let config = EncoderConfig::ycbcr(10.0, ChromaSubsampling::Quarter).progressive(false);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encoder");
    enc.push_packed(&rgb, enough::Unstoppable).expect("push");
    let rust_jpeg = enc.finish().expect("finish");
    fs::write("/tmp/q10_zen.jpg", &rust_jpeg).unwrap();
    eprintln!("zenjpeg Q10 4:2:0 baseline: {} bytes", rust_jpeg.len());

    // --- C++ cjpegli CLI encode ---------------------------------------------
    let ppm = "/tmp/q10_zen.ppm";
    {
        let mut f = fs::File::create(ppm).unwrap();
        writeln!(f, "P6").unwrap();
        writeln!(f, "{w} {h}").unwrap();
        writeln!(f, "255").unwrap();
        f.write_all(&rgb).unwrap();
    }

    let cjpegli = manifest
        .parent()
        .unwrap()
        .join("internal/jpegli-cpp/build/tools/cjpegli");
    let out = "/tmp/q10_cpp.jpg";
    let status = Command::new(&cjpegli)
        .args([
            ppm,
            out,
            "-q",
            "10",
            "--chroma_subsampling",
            "420",
            "-p",
            "0",
        ])
        .status()
        .expect("cjpegli");
    assert!(status.success(), "cjpegli failed");
    let cpp_jpeg = fs::read(out).unwrap();
    eprintln!("cjpegli  Q10 4:2:0 baseline: {} bytes", cpp_jpeg.len());

    let delta = rust_jpeg.len() as i64 - cpp_jpeg.len() as i64;
    eprintln!("Δ = {delta:+} bytes ({:.3}%)", delta as f64 * 100.0 / cpp_jpeg.len() as f64);

    println!("\n=== zen breakdown ===");
    breakdown("zen", &rust_jpeg);
    println!("\n=== cpp breakdown ===");
    breakdown("cpp", &cpp_jpeg);

    println!("\n=== side by side ===");
    let z = markers(&rust_jpeg);
    let c = markers(&cpp_jpeg);
    println!("{:<20} {:>10} {:>10} {:>10}", "section", "zen", "cpp", "Δ");
    for (k, zv) in &z {
        let cv = c.iter().find(|(n, _)| n == k).map(|(_, v)| *v).unwrap_or(0);
        println!("{:<20} {:>10} {:>10} {:>+10}", k, zv, cv, *zv as i64 - cv as i64);
    }
    // any section only in cpp
    for (k, cv) in &c {
        if !z.iter().any(|(n, _)| n == k) {
            println!("{:<20} {:>10} {:>10} {:>+10}", k, 0, cv, -(*cv as i64));
        }
    }
}

fn breakdown(tag: &str, data: &[u8]) {
    let m = markers(data);
    for (k, v) in &m {
        println!("  {tag} {:<16} {} bytes", k, v);
    }
}

/// Walk marker segments + entropy-coded data. Returns (name, byte_count) pairs.
fn markers(d: &[u8]) -> Vec<(String, usize)> {
    let mut out: Vec<(String, usize)> = Vec::new();
    let mut add = |name: &str, n: usize| {
        if let Some(entry) = out.iter_mut().find(|(k, _)| k == name) {
            entry.1 += n;
        } else {
            out.push((name.to_string(), n));
        }
    };

    let mut i = 0usize;
    if d.len() < 2 || d[0] != 0xFF || d[1] != 0xD8 {
        panic!("no SOI");
    }
    add("SOI", 2);
    i += 2;

    while i < d.len() {
        if d[i] != 0xFF {
            panic!("expected marker at {i:#x}, got {:#x}", d[i]);
        }
        // skip fill bytes
        let mut j = i;
        while j < d.len() && d[j] == 0xFF {
            j += 1;
        }
        if j >= d.len() {
            break;
        }
        let marker = d[j];
        let header = j - i + 1; // 0xFF... + marker byte

        // Markers with no length: SOI(D8), EOI(D9), RSTn(D0..D7), TEM(01)
        let standalone = matches!(
            marker,
            0xD0..=0xD9 | 0x01
        );
        if marker == 0xD9 {
            add("EOI", header);
            i = j + 1;
            break;
        }
        if standalone {
            // shouldn't normally happen outside scan data
            add(&format!("marker_{:02X}", marker), header);
            i = j + 1;
            continue;
        }

        // read 2-byte length
        let length = u16::from_be_bytes([d[j + 1], d[j + 2]]) as usize;
        let seg_end = j + 1 + length; // length includes the length bytes

        let name = match marker {
            0xC0 => "SOF0",
            0xC2 => "SOF2",
            0xC4 => "DHT",
            0xDA => "SOS_header",
            0xDB => "DQT",
            0xDD => "DRI",
            0xE0..=0xEF => "APPn",
            0xFE => "COM",
            _ => "other",
        };
        let seg_bytes = header + length; // 0xFF + marker + length bytes
        add(name, seg_bytes);

        if marker == 0xDA {
            // Entropy-coded scan data follows until next non-RST marker.
            let mut k = seg_end;
            let scan_start = k;
            while k < d.len() {
                if d[k] == 0xFF {
                    // could be stuffed byte 0xFF 0x00, or RST, or next real marker
                    if k + 1 < d.len() {
                        let next = d[k + 1];
                        if next == 0x00 {
                            k += 2;
                            continue;
                        }
                        if (0xD0..=0xD7).contains(&next) {
                            k += 2;
                            continue;
                        }
                        // other non-zero byte: real marker, end of scan
                        break;
                    }
                    break;
                }
                k += 1;
            }
            let scan_len = k - scan_start;
            add("scan_data", scan_len);
            i = k;
        } else {
            i = seg_end;
        }
    }

    out
}
