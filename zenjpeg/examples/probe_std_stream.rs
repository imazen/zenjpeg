//! Emit baseline/444/std frymire encodes and verify they decode.
use enough::Unstoppable;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn load_frymire() -> (Vec<u8>, u32, u32) {
    let png_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/images/frymire.png");
    let decoder = png::Decoder::new(std::io::BufReader::new(
        std::fs::File::open(png_path).unwrap(),
    ));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    buf.truncate(info.buffer_size());
    let rgb = match info.color_type {
        png::ColorType::Rgb => buf,
        png::ColorType::Rgba => buf
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect(),
        other => panic!("{other:?}"),
    };
    (rgb, info.width, info.height)
}

fn main() {
    let out_dir = std::env::args().nth(1).unwrap();
    let (px, w, h) = load_frymire();
    for q in [10u8, 50, 90] {
        let mut enc = EncoderConfig::ycbcr(q as f32, ChromaSubsampling::None)
            .progressive(false)
            .optimize_huffman(false)
            .restart_mcu_rows(0)
            .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(&px, Unstoppable).unwrap();
        let jpeg = enc.finish().unwrap();
        std::fs::write(format!("{out_dir}/frymire-std-q{q}.jpg"), &jpeg).unwrap();
        // Also decode with our own decoder
        match zenjpeg::decoder::Decoder::new().decode(&jpeg, Unstoppable) {
            Ok(r) => println!("q{q}: zen decode OK {}x{}", r.width, r.height),
            Err(e) => println!("q{q}: zen decode ERR {e}"),
        }
    }
}
