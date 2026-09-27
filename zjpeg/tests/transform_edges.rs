use std::process::Command;
use zenjpeg::decoder::DecodeConfig;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, Exif, Orientation, PixelLayout};

#[test]
fn cli_requires_trim_and_leaves_output_untouched_on_rejection() {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("transform-edges-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("source.jpg");
    let output = dir.join("output.jpg");
    let pixels: Vec<u8> = (0..32 * 24 * 3)
        .map(|i| ((i * 73 + i / 17) % 256) as u8)
        .collect();
    let mut enc = EncoderConfig::ycbcr(90, ChromaSubsampling::Quarter)
        .request()
        .exif(Exif::build().orientation(Orientation::Rotate90))
        .encode_from_bytes(32, 24, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&pixels, enough::Unstoppable).unwrap();
    let jpeg = enc.finish().unwrap();
    std::fs::write(&input, &jpeg).unwrap();
    for args in [
        vec!["transform", "--auto-orient"],
        vec!["transform", "--rotate", "90"],
        vec!["process", "--orient", "auto"],
        vec!["process", "--rotate", "90"],
    ] {
        std::fs::write(&output, b"existing output").unwrap();
        let invoke = |trim: bool| {
            let mut command = Command::new(env!("CARGO_BIN_EXE_zjpeg"));
            command
                .args(&args)
                .arg(&input)
                .arg("--output")
                .arg(&output)
                .arg("--force");
            if trim {
                command.arg("--trim");
            }
            command.output().unwrap()
        };
        let rejected = invoke(false);
        assert!(!rejected.status.success(), "{args:?}");
        assert!(String::from_utf8_lossy(&rejected.stderr).contains("MCU"));
        assert_eq!(std::fs::read(&output).unwrap(), b"existing output");
        assert_eq!(std::fs::read(&input).unwrap(), jpeg);
        let trimmed = invoke(true);
        assert!(
            trimmed.status.success(),
            "{}",
            String::from_utf8_lossy(&trimmed.stderr)
        );
        let bytes = std::fs::read(&output).unwrap();
        let info = DecodeConfig::new().read_info(&bytes).unwrap();
        assert_eq!((info.dimensions.width, info.dimensions.height), (16, 32));
    }
    std::fs::remove_file(input).unwrap();
    std::fs::remove_file(output).unwrap();
    std::fs::remove_dir(dir).unwrap();
}
