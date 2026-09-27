//! Measure fixed-Huffman 4:4:4 sizes on tests/images/1.png for relocking the
//! parity size tables after the table-completion fix.
use enough::Unstoppable;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn main() {
    let png_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/images/1.png");
    let decoder = png::Decoder::new(std::io::BufReader::new(
        std::fs::File::open(png_path).unwrap(),
    ));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    buf.truncate(info.buffer_size());
    let rgb: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => buf,
        png::ColorType::Rgba => buf
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect(),
        other => panic!("{other:?}"),
    };
    let (w, h) = (info.width, info.height);
    for q in [
        5u8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
    ] {
        let mut enc = EncoderConfig::ycbcr(q as f32, ChromaSubsampling::None)
            .progressive(false)
            .optimize_huffman(false)
            .restart_mcu_rows(0)
            .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(&rgb, Unstoppable).unwrap();
        let jpeg = enc.finish().unwrap();
        zenjpeg::decoder::Decoder::new()
            .decode(&jpeg, Unstoppable)
            .expect("must decode");
        // parity_reference_locked variant: default restart markers
        let mut enc2 = EncoderConfig::ycbcr(q as f32, ChromaSubsampling::None)
            .progressive(false)
            .optimize_huffman(false)
            .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc2.push_packed(&rgb, Unstoppable).unwrap();
        let jpeg2 = enc2.finish().unwrap();
        zenjpeg::decoder::Decoder::new()
            .decode(&jpeg2, Unstoppable)
            .expect("must decode");
        println!("{q} {} {}", jpeg.len(), jpeg2.len());
    }
}
