//! Deeper analysis of the Q10 baseline scan_data gap.
//!
//! - Count RST markers and stuffed 0xFF bytes in each file.
//! - Dequantize and compare quant tables (should be identical).
//! - Try disabling trellis / adaptive quant in zenjpeg to see if the gap is
//!   coding efficiency vs quant-decision.
//!
//! Run: `cargo run --release -p zenjpeg --features __test-utils --example q10_investigate_scan`

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let png_path = manifest.join("tests/images/frymire.png");
    let img = image::open(&png_path).expect("load").to_rgb8();
    let (w, h) = img.dimensions();
    let rgb = img.into_raw();

    // Reference (cpp) -----------------------------------------------
    let ppm = "/tmp/q10_scan.ppm";
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
    let cpp_path = "/tmp/q10_cpp_baseline.jpg";
    Command::new(&cjpegli)
        .args([ppm, cpp_path, "-q", "10", "--chroma_subsampling", "420", "-p", "0"])
        .status()
        .unwrap();
    let cpp = fs::read(cpp_path).unwrap();

    // Zen default -----------------------------------------------
    let rust_default = encode_rust(&rgb, w, h, None);
    fs::write("/tmp/q10_zen_default.jpg", &rust_default).unwrap();

    // Zen: disable adaptive quant
    let rust_no_aq = encode_rust(&rgb, w, h, Some(Opts { adaptive: false, trellis: true, dri: true }));
    fs::write("/tmp/q10_zen_no_aq.jpg", &rust_no_aq).unwrap();

    // Zen: no trellis
    let rust_no_trellis = encode_rust(&rgb, w, h, Some(Opts { adaptive: true, trellis: false, dri: true }));
    fs::write("/tmp/q10_zen_no_trellis.jpg", &rust_no_trellis).unwrap();

    // Zen: no DRI
    let rust_no_dri = encode_rust(&rgb, w, h, Some(Opts { adaptive: true, trellis: true, dri: false }));
    fs::write("/tmp/q10_zen_no_dri.jpg", &rust_no_dri).unwrap();

    // Zen: no AQ + no trellis
    let rust_bare = encode_rust(
        &rgb,
        w,
        h,
        Some(Opts {
            adaptive: false,
            trellis: false,
            dri: false,
        }),
    );
    fs::write("/tmp/q10_zen_bare.jpg", &rust_bare).unwrap();

    // Zen: no deringing
    let cfg = EncoderConfig::ycbcr(10.0, ChromaSubsampling::Quarter)
        .progressive(false)
        .deringing(false);
    let mut e = cfg
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("enc");
    e.push_packed(&rgb, enough::Unstoppable).expect("push");
    let rust_no_dering = e.finish().expect("finish");
    fs::write("/tmp/q10_zen_no_dering.jpg", &rust_no_dering).unwrap();

    // Zen: no deringing AND no DRI
    let cfg = EncoderConfig::ycbcr(10.0, ChromaSubsampling::Quarter)
        .progressive(false)
        .deringing(false)
        .restart_mcu_rows(0);
    let mut e = cfg
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("enc");
    e.push_packed(&rgb, enough::Unstoppable).expect("push");
    let rust_no_dering_no_dri = e.finish().expect("finish");
    fs::write("/tmp/q10_zen_no_dering_no_dri.jpg", &rust_no_dering_no_dri).unwrap();

    println!("{:<30} {:>8}  {:>10}", "config", "bytes", "Δ vs cpp");
    println!("{:-<60}", "");
    let baseline = cpp.len() as i64;
    let print = |n: &str, v: &[u8]| {
        let d = v.len() as i64 - baseline;
        println!("{:<30} {:>8}  {:>+10}", n, v.len(), d);
    };
    print("cjpegli -q 10 420 baseline", &cpp);
    print("zen default", &rust_default);
    print("zen no_aq (adaptive=false)", &rust_no_aq);
    print("zen no_trellis", &rust_no_trellis);
    print("zen no_dri", &rust_no_dri);
    print("zen bare (no_aq, no_trellis, no_dri)", &rust_bare);
    print("zen no_dering", &rust_no_dering);
    print("zen no_dering + no_dri", &rust_no_dering_no_dri);

    println!("\nRST marker + stuffed-byte stats:");
    stats("cpp", &cpp);
    stats("zen default", &rust_default);
    stats("zen no_dri", &rust_no_dri);
    stats("zen bare", &rust_bare);

    // Print quant tables
    println!("\nDQT table contents:");
    print_dqt("cpp", &cpp);
    print_dqt("zen", &rust_default);
}

struct Opts {
    adaptive: bool,
    trellis: bool,
    dri: bool,
}

fn encode_rust(rgb: &[u8], w: u32, h: u32, opts: Option<Opts>) -> Vec<u8> {
    let mut config = EncoderConfig::ycbcr(10.0, ChromaSubsampling::Quarter).progressive(false);
    if let Some(o) = opts {
        config = config.aq_enabled(o.adaptive);
        if !o.dri {
            config = config.restart_mcu_rows(0);
        }
        // trellis flag ignored; default is already off (None). Kept for interface symmetry.
        let _ = o.trellis;
    }
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("enc");
    enc.push_packed(rgb, enough::Unstoppable).expect("push");
    enc.finish().expect("finish")
}

fn stats(name: &str, d: &[u8]) {
    // Find first SOS then count restart markers + stuffed bytes in scan.
    let mut i = 0;
    while i + 1 < d.len() {
        if d[i] == 0xFF && d[i + 1] == 0xDA {
            let len = u16::from_be_bytes([d[i + 2], d[i + 3]]) as usize;
            let scan_start = i + 2 + len;
            // Walk scan data until non-stuffed non-RST marker.
            let mut j = scan_start;
            let mut stuffed = 0usize;
            let mut rst = 0usize;
            let mut raw = 0usize;
            while j < d.len() {
                if d[j] == 0xFF {
                    if j + 1 >= d.len() {
                        break;
                    }
                    let nx = d[j + 1];
                    if nx == 0x00 {
                        stuffed += 1;
                        raw += 1; // the 0xFF is a real byte
                        j += 2;
                        continue;
                    }
                    if (0xD0..=0xD7).contains(&nx) {
                        rst += 1;
                        j += 2;
                        continue;
                    }
                    break;
                }
                raw += 1;
                j += 1;
            }
            println!(
                "  {name:<25} raw_bytes={raw} stuffed_ff={stuffed} rst_markers={rst} total_scan={}",
                j - scan_start
            );
            return;
        }
        i += 1;
    }
}

fn print_dqt(name: &str, d: &[u8]) {
    let mut i = 0;
    while i + 1 < d.len() {
        if d[i] == 0xFF && d[i + 1] == 0xDB {
            let len = u16::from_be_bytes([d[i + 2], d[i + 3]]) as usize;
            let end = i + 2 + len;
            let mut p = i + 4;
            while p < end {
                let pq_tq = d[p];
                let pq = pq_tq >> 4;
                let tq = pq_tq & 0x0F;
                p += 1;
                let elem_bytes = if pq == 0 { 1 } else { 2 };
                let mut vals = Vec::with_capacity(64);
                for _ in 0..64 {
                    let v = if elem_bytes == 1 {
                        d[p] as u16
                    } else {
                        u16::from_be_bytes([d[p], d[p + 1]])
                    };
                    p += elem_bytes;
                    vals.push(v);
                }
                println!(
                    "  {name} DQT table {tq} precision {} values: {:?}",
                    if pq == 0 { 8 } else { 16 },
                    vals
                );
            }
            return;
        }
        i += 1;
    }
}
