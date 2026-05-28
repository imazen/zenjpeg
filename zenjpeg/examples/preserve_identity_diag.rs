//! Diagnostic: emit_preserved with IDENTITY quant scale should
//! produce a JPEG whose decoded pixels match a fresh decode of the
//! source. If they don't, our emit is broken — and the divergence
//! tells us where.

use enough::Unstoppable;
use zenjpeg::decode::DecodeConfig;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality};
use zenjpeg::types::Subsampling;

use zenjpeg::recompress::expert::{EmitConfig, QuantScale, emit_preserved};

fn make_test_image(w: u32, h: u32) -> Vec<u8> {
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let s = (x.wrapping_mul(2654435761) ^ y).wrapping_mul(2246822519);
            rgb.push(((x * 7 + y * 3) % 240 + (s & 0x0F)) as u8);
            rgb.push(((x * 5 + y * 11) % 220 + ((s >> 4) & 0x1F)) as u8);
            rgb.push(((x * 13 + y * 2) % 200 + ((s >> 9) & 0x3F)) as u8);
        }
    }
    rgb
}

fn diff_pixels(a: &[u8], b: &[u8]) -> (usize, u16, u32) {
    let mut n_diff = 0usize;
    let mut max_diff = 0u16;
    let mut total_diff = 0u32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as i32 - *y as i32).unsigned_abs() as u16;
        if d > 0 {
            n_diff += 1;
            if d > max_diff {
                max_diff = d;
            }
            total_diff += d as u32;
        }
    }
    (n_diff, max_diff, total_diff)
}

fn run_case(w: u32, h: u32, q: f32, chroma: ChromaSubsampling) {
    println!("\n== {}x{} q={} chroma={:?} ==", w, h, q, chroma);
    let rgb = make_test_image(w, h);
    let cfg = EncoderConfig::ycbcr(Quality::ApproxJpegli(q), chroma).progressive(false);
    let mut enc = cfg
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encode setup");
    enc.push_packed(&rgb, Unstoppable).expect("encode push");
    let source = enc.finish().expect("encode finish");
    println!("source bytes: {}", source.len());

    let coeffs = DecodeConfig::new()
        .decode_coefficients(&source, Unstoppable)
        .expect("decode coefficients");

    // Identity emit
    let emit_cfg = EmitConfig::uniform_scale(QuantScale::IDENTITY);
    let subs = match chroma {
        ChromaSubsampling::None => Subsampling::S444,
        ChromaSubsampling::HalfHorizontal => Subsampling::S422,
        ChromaSubsampling::Quarter => Subsampling::S420,
        ChromaSubsampling::HalfVertical => Subsampling::S440,
        _ => Subsampling::S420,
    };
    let emitted = match emit_preserved(&coeffs, subs, &emit_cfg) {
        Ok(b) => b,
        Err(e) => {
            println!("emit failed: {:?}", e);
            return;
        }
    };
    println!("emitted bytes: {}", emitted.len());

    // Decode both and diff
    let dec_source = DecodeConfig::new()
        .decode(&source, Unstoppable)
        .expect("decode source");
    let dec_emit = match DecodeConfig::new().decode(&emitted, Unstoppable) {
        Ok(d) => d,
        Err(e) => {
            println!("emitted didn't decode: {:?}", e);
            // Write to file for forensic
            std::fs::write("/tmp/preserve_emit_bad.jpg", &emitted).ok();
            std::fs::write("/tmp/preserve_emit_source.jpg", &source).ok();
            return;
        }
    };

    let sp = dec_source.pixels_u8().expect("source u8");
    let ep = dec_emit.pixels_u8().expect("emitted u8");
    if dec_source.width != dec_emit.width || dec_source.height != dec_emit.height {
        println!(
            "DIM MISMATCH: source {}x{} vs emitted {}x{}",
            dec_source.width, dec_source.height, dec_emit.width, dec_emit.height
        );
        return;
    }
    let (n, max_d, total) = diff_pixels(sp, ep);
    println!(
        "pixel diff: {} pixels differ (max={}, total={}, mean={:.3})",
        n,
        max_d,
        total,
        if n > 0 { total as f32 / n as f32 } else { 0.0 }
    );

    if n > 0 && n < 32 {
        for (i, (x, y)) in sp.iter().zip(ep.iter()).enumerate() {
            if x != y {
                let pixel = i / 3;
                let comp = i % 3;
                let py = pixel as u32 / dec_source.width;
                let px = pixel as u32 % dec_source.width;
                println!(
                    "  ({},{})[{}]: src={} emit={} diff={}",
                    px,
                    py,
                    ["R", "G", "B"][comp],
                    x,
                    y,
                    *x as i32 - *y as i32
                );
            }
        }
    } else if n > 0 {
        // Bin differences by pixel-x to see column pattern
        let mut by_x: std::collections::BTreeMap<u32, (usize, u32)> =
            std::collections::BTreeMap::new();
        for (i, (x, y)) in sp.iter().zip(ep.iter()).enumerate() {
            if x != y {
                let pixel = i / 3;
                let px = pixel as u32 % dec_source.width;
                let entry = by_x.entry(px).or_insert((0, 0));
                entry.0 += 1;
                entry.1 += (*x as i32 - *y as i32).unsigned_abs();
            }
        }
        let cols: Vec<_> = by_x.iter().collect();
        if !cols.is_empty() {
            println!("  differs-by-x-col histogram (first/last 5):");
            for &(x, (cnt, tot)) in cols.iter().take(5) {
                println!("    x={:3}: cnt={} sum_diff={}", x, cnt, tot);
            }
            println!("    ...");
            for &(x, (cnt, tot)) in cols.iter().rev().take(5).collect::<Vec<_>>().iter().rev() {
                println!("    x={:3}: cnt={} sum_diff={}", x, cnt, tot);
            }
        }
        // By pixel-y
        let mut by_y: std::collections::BTreeMap<u32, (usize, u32)> =
            std::collections::BTreeMap::new();
        for (i, (x, y)) in sp.iter().zip(ep.iter()).enumerate() {
            if x != y {
                let pixel = i / 3;
                let py = pixel as u32 / dec_source.width;
                let entry = by_y.entry(py).or_insert((0, 0));
                entry.0 += 1;
                entry.1 += (*x as i32 - *y as i32).unsigned_abs();
            }
        }
        let rows: Vec<_> = by_y.iter().collect();
        if !rows.is_empty() {
            println!("  differs-by-y-row histogram (first/last 5):");
            for &(y, (cnt, tot)) in rows.iter().take(5) {
                println!("    y={:3}: cnt={} sum_diff={}", y, cnt, tot);
            }
            println!("    ...");
            for &(y, (cnt, tot)) in rows.iter().rev().take(5).collect::<Vec<_>>().iter().rev() {
                println!("    y={:3}: cnt={} sum_diff={}", y, cnt, tot);
            }
        }
    }
}

fn dump_smallest_case() {
    // 8×8 image — exactly 1 block, simplest possible case
    let w = 8u32;
    let h = 8u32;
    let mut rgb = vec![128u8; (w * h * 3) as usize];
    // Set a known pattern: top-left red gradient
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            rgb[i] = (x * 30) as u8;
            rgb[i + 1] = (y * 30) as u8;
            rgb[i + 2] = 200;
        }
    }
    let cfg = EncoderConfig::ycbcr(Quality::ApproxJpegli(95.0), ChromaSubsampling::None);
    let mut enc = cfg
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encode setup");
    enc.push_packed(&rgb, Unstoppable).expect("encode push");
    let source = enc.finish().expect("encode finish");
    std::fs::write("/tmp/preserve_diag_source.jpg", &source).expect("write");
    println!(
        "source bytes: {} → /tmp/preserve_diag_source.jpg",
        source.len()
    );

    let coeffs = DecodeConfig::new()
        .decode_coefficients(&source, Unstoppable)
        .expect("decode coefficients");

    println!("DecodedCoefficients:");
    println!("  width: {}", coeffs.width);
    println!("  height: {}", coeffs.height);
    println!("  components: {}", coeffs.components.len());
    for (i, c) in coeffs.components.iter().enumerate() {
        println!(
            "    [{}] id={} blocks={}x{} h={} v={} q_idx={} coeffs_len={}",
            i,
            c.id,
            c.blocks_wide,
            c.blocks_high,
            c.h_samp,
            c.v_samp,
            c.quant_table_idx,
            c.coeffs.len()
        );
        println!(
            "        first block: {:?}",
            &c.coeffs[..64.min(c.coeffs.len())]
        );
    }
    for (i, t) in coeffs.quant_tables.iter().enumerate() {
        match t {
            Some(qt) => println!("  quant_table[{}]: {:?}", i, qt),
            None => println!("  quant_table[{}]: None", i),
        }
    }

    let emit_cfg = EmitConfig::uniform_scale(QuantScale::IDENTITY);
    let emitted = emit_preserved(&coeffs, Subsampling::S444, &emit_cfg).expect("emit");
    std::fs::write("/tmp/preserve_diag_emit.jpg", &emitted).expect("write");
    println!(
        "\nemitted bytes: {} → /tmp/preserve_diag_emit.jpg",
        emitted.len()
    );

    let dec_source = DecodeConfig::new()
        .decode(&source, Unstoppable)
        .expect("decode source");
    let dec_emit = DecodeConfig::new()
        .decode(&emitted, Unstoppable)
        .expect("decode emit");

    let sp = dec_source.pixels_u8().unwrap();
    let ep = dec_emit.pixels_u8().unwrap();
    println!("\nDecoded source pixels: {:?}", sp);
    println!("Decoded emit   pixels: {:?}", ep);
}

fn dump_decoded_coefficients(w: u32, h: u32, q: f32, chroma: ChromaSubsampling) {
    println!("\n==== {}x{} q={} chroma={:?} ====", w, h, q, chroma);
    let rgb = make_test_image(w, h);
    let cfg = EncoderConfig::ycbcr(Quality::ApproxJpegli(q), chroma).progressive(false);
    let mut enc = cfg.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).unwrap();
    enc.push_packed(&rgb, Unstoppable).unwrap();
    let source = enc.finish().unwrap();

    let coeffs = DecodeConfig::new()
        .decode_coefficients(&source, Unstoppable)
        .unwrap();
    println!(
        "DecodedCoefficients dims: {}x{}",
        coeffs.width, coeffs.height
    );
    for (i, c) in coeffs.components.iter().enumerate() {
        println!(
            "  comp[{}] id={} blocks={}x{} h_samp={} v_samp={} q_idx={}",
            i, c.id, c.blocks_wide, c.blocks_high, c.h_samp, c.v_samp, c.quant_table_idx
        );
    }
}

fn main() {
    println!("==== component shape diagnostics ====");
    dump_decoded_coefficients(64, 64, 90.0, ChromaSubsampling::Quarter); // MCU-aligned
    dump_decoded_coefficients(72, 56, 75.0, ChromaSubsampling::Quarter); // partial
    dump_decoded_coefficients(67, 53, 75.0, ChromaSubsampling::Quarter); // partial
    dump_decoded_coefficients(80, 64, 75.0, ChromaSubsampling::Quarter); // MCU-aligned

    println!("\n==== identity emit tests ====");
    run_case(64, 64, 90.0, ChromaSubsampling::None);
    run_case(64, 64, 90.0, ChromaSubsampling::Quarter);
    run_case(128, 128, 75.0, ChromaSubsampling::Quarter);
    run_case(128, 128, 50.0, ChromaSubsampling::Quarter);
    run_case(128, 128, 30.0, ChromaSubsampling::Quarter);
    run_case(64, 64, 90.0, ChromaSubsampling::HalfHorizontal);
    run_case(64, 64, 90.0, ChromaSubsampling::HalfVertical);
    // Block-aligned heights with non-MCU-aligned widths.
    run_case(72, 56, 75.0, ChromaSubsampling::Quarter); // both even, MCU-aligned (5*16=80? no 72/16=4.5 → 5 MCUs)
    run_case(80, 64, 75.0, ChromaSubsampling::Quarter); // both MCU-aligned
    // Odd dimensions force partial-MCU edge handling.
    run_case(67, 53, 75.0, ChromaSubsampling::Quarter);
    run_case(127, 89, 50.0, ChromaSubsampling::Quarter);
}
