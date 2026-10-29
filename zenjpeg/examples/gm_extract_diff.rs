//! Diff the two Ultra HDR gain-map extraction + decode paths for one JPEG:
//! ultrahdr-rs (what the v2 hdr-corpus-convert tool consumed) vs zenjpeg
//! extras (what the zencodec adapter consumes). Prints slice equality and
//! decoded-map equality so corpus byte-diffs can be attributed.
use enough::Unstoppable;
use zenjpeg::ultrahdr::UltraHdrExtras;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: gm_extract_diff <file.jpg>");
    let data = std::fs::read(&path).unwrap();

    let dec = ultrahdr_rs::Decoder::new(&data).unwrap();
    let gm_a = dec.gainmap_jpeg().expect("ultrahdr-rs: no gain map");

    let cfg = zenjpeg::decoder::DecodeConfig::new().preserve_all_metadata();
    let mut res = cfg.decode(&data, Unstoppable).unwrap();
    let extras = res.take_extras().expect("zenjpeg: no extras");
    let gm_b = extras.gainmap().expect("zenjpeg: no gain map");

    let first_diff = gm_a.iter().zip(gm_b).position(|(a, b)| a != b);
    println!(
        "extracted slice: A(ultrahdr-rs)={}B B(zenjpeg)={}B equal={} first_diff={:?}",
        gm_a.len(),
        gm_b.len(),
        gm_a == gm_b,
        first_diff
    );

    // v2 corpus-tool decode: explicit Gray, stored orientation.
    let gray = zenjpeg::decoder::Decoder::new()
        .output_format(zenjpeg::decoder::PixelFormat::Gray)
        .auto_orient(false)
        .decode(gm_a, Unstoppable)
        .unwrap();
    let gray_px = gray.pixels_u8().unwrap();

    // Adapter decode: extras.decode_gainmap().
    let gm = extras
        .decode_gainmap()
        .expect("no gain map")
        .expect("decode failed");
    println!(
        "decoded: v2 Gray {}x{} 1ch vs adapter {}x{} {}ch",
        gray.width(),
        gray.height(),
        gm.width,
        gm.height,
        gm.channels
    );
    if gm.channels == 1 && (gm.width, gm.height) == (gray.width(), gray.height()) {
        let n = gray_px.iter().zip(&gm.data).filter(|(a, b)| a != b).count();
        let maxd = gray_px
            .iter()
            .zip(&gm.data)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        println!(
            "map bytes differing: {n} of {} (max abs {maxd})",
            gray_px.len()
        );
    }
}
