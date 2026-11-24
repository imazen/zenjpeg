//! Resident-input XYB decode driver for perf and zenbench.
//! prepare <P6.ppm> <directory> writes Q85 XYB/YCbCr fixtures using zenjpeg.
//! run <jpeg> <iterations> profiles repeated fresh-output, one-thread decodes.
//! dump <jpeg> <prefix> writes RGB8 and little-endian sRGB f32 for exact A/B.
//! bench <directory> interleaves all six fixtures with zenbench.
use enough::Unstoppable;
use std::{hint::black_box, path::Path};
use zenjpeg::{
    decoder::{Decoder, OutputTarget},
    encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, XybSubsampling},
};

const NAMES: [&str; 6] = [
    "xyb-bquarter-baseline",
    "xyb-bquarter-progressive",
    "xyb-full-baseline",
    "xyb-full-progressive",
    "ycbcr-420-baseline",
    "ycbcr-444-baseline",
];

fn main() {
    let args: Vec<_> = std::env::args().collect();
    match args[1].as_str() {
        "prepare" => {
            let ppm = std::fs::read(&args[2]).unwrap();
            let header_end = ppm.windows(5).position(|w| w == b"\n255\n").unwrap() + 5;
            let header = std::str::from_utf8(&ppm[..header_end]).unwrap();
            let mut words = header.split_whitespace();
            assert_eq!(words.next(), Some("P6"));
            let width: u32 = words.next().unwrap().parse().unwrap();
            let height: u32 = words.next().unwrap().parse().unwrap();
            let rgb = &ppm[header_end..];
            assert_eq!(rgb.len(), width as usize * height as usize * 3);
            std::fs::create_dir_all(&args[3]).unwrap();
            for (i, name) in NAMES.iter().enumerate() {
                let config = match i {
                    0..=1 => EncoderConfig::xyb(85.0, XybSubsampling::BQuarter),
                    2..=3 => EncoderConfig::xyb(85.0, XybSubsampling::Full),
                    4 => EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter),
                    _ => EncoderConfig::ycbcr(85.0, ChromaSubsampling::None),
                }
                .progressive(i == 1 || i == 3);
                let mut enc = config
                    .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
                    .unwrap();
                enc.push_packed(rgb, Unstoppable).unwrap();
                std::fs::write(
                    Path::new(&args[3]).join(format!("{name}.jpg")),
                    enc.finish().unwrap(),
                )
                .unwrap();
            }
        }
        "run" => {
            let jpeg = std::fs::read(&args[2]).unwrap();
            let iterations: usize = args[3].parse().unwrap();
            let decoder = Decoder::new().num_threads(1);
            for _ in 0..iterations {
                black_box(decoder.decode(black_box(&jpeg), Unstoppable).unwrap());
            }
        }
        "dump" => {
            let jpeg = std::fs::read(&args[2]).unwrap();
            let decoded = Decoder::new()
                .num_threads(1)
                .decode(&jpeg, Unstoppable)
                .unwrap();
            std::fs::write(format!("{}.rgb8", args[3]), decoded.pixels_u8().unwrap()).unwrap();
            let decoded = Decoder::new()
                .num_threads(1)
                .output_target(OutputTarget::SrgbF32)
                .decode(&jpeg, Unstoppable)
                .unwrap();
            use std::io::Write;
            let mut file = std::io::BufWriter::new(
                std::fs::File::create(format!("{}.rgbf32", args[3])).unwrap(),
            );
            for &v in decoded.pixels_f32().unwrap() {
                file.write_all(&v.to_le_bytes()).unwrap();
            }
        }
        "bench" => {
            let mut suite = zenbench::Suite::new();
            let files: Vec<_> = NAMES
                .iter()
                .map(|name| std::fs::read(Path::new(&args[2]).join(format!("{name}.jpg"))).unwrap())
                .collect();
            suite.group("xyb-decode-q85", |g| {
                g.throughput(zenbench::Throughput::Elements(1024 * 1024));
                for (name, jpeg) in NAMES.iter().zip(files) {
                    g.bench(*name, move |b| {
                        b.iter(|| {
                            black_box(
                                Decoder::new()
                                    .num_threads(1)
                                    .decode(black_box(&jpeg), Unstoppable)
                                    .unwrap(),
                            )
                        })
                    });
                }
            });
            let result = zenbench::engine_new(suite).run();
            zenbench::postprocess_result(&result);
        }
        _ => panic!("expected prepare, run or bench"),
    }
}
