//! The zencodec exits for JPEG frames above eight bits: nothing is narrowed
//! unless the caller asked for fewer bits. `native_precision` holds the native
//! reader equal to libjpeg-turbo; `oracle_*` here holds this path to the same
//! oracle directly, through the widening.
#![cfg(all(feature = "zencodec", feature = "extended-jpeg"))]
use std::borrow::Cow;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use zencodec::decode::{
    Decode, DecodeJob, DecodeOutput, DecodeRowSink, DecoderConfig, SinkError, StreamingDecode,
};
use zenjpeg::{CmykHandling, JpegDecodeJob, JpegDecoderConfig};
use zenpixels::{ChannelType, PixelDescriptor, PixelFormat, PixelSliceMut};

const SEEDS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/fuzz/seeds/native/");

fn seed(name: &str) -> Vec<u8> {
    std::fs::read(format!("{SEEDS}{name}.jpg")).unwrap()
}

fn corpus_12_bit() -> Vec<u8> {
    let corpus = codec_corpus::Corpus::new().expect("codec-corpus init");
    let dir = corpus.get("mozjpeg").expect("codec-corpus mozjpeg");
    std::fs::read(dir.join("testorig12.jpg")).expect("testorig12.jpg")
}

fn decode(data: &[u8], preferred: &[PixelDescriptor]) -> DecodeOutput {
    JpegDecoderConfig::new()
        .job()
        .decoder(Cow::Borrowed(data), preferred)
        .unwrap()
        .decode()
        .unwrap()
}

/// The frame's codes from the native API, whose equality with libjpeg-turbo is
/// `native_precision`'s job.
fn native_codes(data: &[u8], gray: bool, rgba: bool) -> Vec<u16> {
    use zenjpeg::decoder::{Decoder, OutputTarget, PixelFormat as Native};
    Decoder::new()
        .output_target(OutputTarget::Native16)
        .output_format(if gray {
            Native::Gray
        } else if rgba {
            Native::Rgba
        } else {
            Native::Rgb
        })
        .decode(data, enough::Unstoppable)
        .unwrap()
        .pixels_u16()
        .unwrap()
        .to_vec()
}

fn nearest_16(code: u16, bits: u8) -> u16 {
    let max = (1u32 << bits) - 1;
    ((u32::from(code) * 65535 + max / 2) / max) as u16
}

fn words(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_ne_bytes(*b))
        .collect()
}

fn output_words(output: &DecodeOutput) -> Vec<u16> {
    let pixels = output.pixels();
    (0..pixels.rows())
        .flat_map(|y| words(pixels.row(y)))
        .collect()
}

#[derive(Default)]
struct Sink {
    descriptor: Option<PixelDescriptor>,
    size: (u32, u32),
    stride: usize,
    buffer: Vec<u16>,
    finished: bool,
}

impl DecodeRowSink for Sink {
    fn begin(
        &mut self,
        width: u32,
        height: u32,
        descriptor: PixelDescriptor,
    ) -> Result<(), SinkError> {
        // Padded rows, so a write past a row's active samples is visible.
        self.size = (width, height);
        self.descriptor = Some(descriptor);
        self.stride = width as usize * descriptor.channels() + 5;
        self.buffer = vec![0xABCD; self.stride * height as usize];
        Ok(())
    }

    fn provide_next_buffer(
        &mut self,
        y: u32,
        height: u32,
        width: u32,
        descriptor: PixelDescriptor,
    ) -> Result<PixelSliceMut<'_>, SinkError> {
        // The stride must be a whole number of pixels for the view.
        let pixel = descriptor.channels();
        let stride = self.stride.div_ceil(pixel) * pixel;
        if stride != self.stride {
            self.stride = stride;
            self.buffer = vec![0xABCD; stride * self.size.1 as usize];
        }
        let start = y as usize * stride;
        let end = start + height as usize * stride;
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut self.buffer[start..end]);
        Ok(PixelSliceMut::new(bytes, width, height, stride * 2, descriptor).unwrap())
    }

    fn finish(&mut self) -> Result<(), SinkError> {
        self.finished = true;
        Ok(())
    }
}

/// One-shot, push and streaming exits for one preference list agree.
fn check_paths(data: &[u8], preferred: &[PixelDescriptor]) -> (PixelDescriptor, Vec<u16>) {
    check_job_paths(data, preferred, || JpegDecoderConfig::new().job())
}

fn check_job_paths(
    data: &[u8],
    preferred: &[PixelDescriptor],
    job: impl Fn() -> JpegDecodeJob,
) -> (PixelDescriptor, Vec<u16>) {
    let one_shot = job()
        .decoder(Cow::Borrowed(data), preferred)
        .unwrap()
        .decode()
        .unwrap();
    let descriptor = one_shot.pixels().descriptor();
    assert_eq!(descriptor.channel_type(), ChannelType::U16);
    let samples = output_words(&one_shot);

    let mut sink = Sink::default();
    let info = job()
        .push_decoder(Cow::Borrowed(data), &mut sink, preferred)
        .unwrap();
    assert!(sink.finished);
    assert_eq!(info.native_format, descriptor);
    let active = sink.size.0 as usize * descriptor.channels();
    let mut pushed = Vec::new();
    for row in sink.buffer.chunks_exact(sink.stride) {
        pushed.extend_from_slice(&row[..active]);
        assert!(
            row[active..].iter().all(|&v| v == 0xABCD),
            "push wrote into row padding"
        );
    }
    assert_eq!(pushed, samples, "push vs one-shot");

    let mut streaming = job()
        .streaming_decoder(Cow::Borrowed(data), preferred)
        .unwrap();
    let mut streamed = Vec::new();
    while let Some((y, rows)) = streaming.next_batch().unwrap() {
        assert_eq!(rows.descriptor(), descriptor);
        assert_eq!(y as usize, streamed.len() / active);
        assert_eq!(
            rows.exact_depth(),
            one_shot.pixels().exact_depth(),
            "streaming depth evidence"
        );
        for r in 0..rows.rows() {
            streamed.extend(words(rows.row(r)));
        }
    }
    assert_eq!(streamed, samples, "streaming vs one-shot");
    (descriptor, samples)
}

#[test]
fn corpus_twelve_bit_keeps_its_depth_by_default() {
    let data = corpus_12_bit();
    let (descriptor, samples) = check_paths(&data, &[]);
    assert_eq!(descriptor, PixelDescriptor::RGB16_SRGB);
    let codes = native_codes(&data, false, false);
    let expected: Vec<u16> = codes.iter().map(|&c| nearest_16(c, 12)).collect();
    assert_eq!(samples, expected);

    let output = decode(&data, &[]);
    assert_eq!(output.info().source_color.bit_depth, Some(12));
    let exact = output.pixels().exact_depth().unwrap();
    assert!(exact.color_exact_at(16));
    // Every code returns through the descriptor's exact 12-bit grid.
    for (&code, &sample) in codes.iter().zip(&samples) {
        assert_eq!(((u32::from(sample) * 4095 + 32767) / 65535) as u16, code);
    }
    let distinct: std::collections::BTreeSet<u16> = samples.iter().copied().collect();
    assert!(distinct.len() > 256, "{} distinct samples", distinct.len());
}

#[test]
fn output_info_and_advertised_exits_name_the_depth() {
    let data = corpus_12_bit();
    let job = JpegDecoderConfig::new().job();
    assert_eq!(
        job.output_info(&data).unwrap().native_format,
        PixelDescriptor::RGB16_SRGB
    );
    let advertised = JpegDecoderConfig::supported_descriptors();
    for descriptor in [
        PixelDescriptor::RGB16_SRGB,
        PixelDescriptor::RGBA16_SRGB,
        PixelDescriptor::GRAY16_SRGB,
    ] {
        assert!(advertised.contains(&descriptor), "{descriptor:?}");
    }
    assert!(JpegDecoderConfig::capabilities().native_16bit());
}

#[test]
fn every_seed_above_eight_bits_keeps_its_depth() {
    // Gray and color, DCT and lossless, 12 and 16 bits; progressive included.
    for (name, bits, gray) in [
        ("01-p12-c1-sof30-mode0-s1", 12, true),
        ("02-p12-c1-sof30-mode1-s1", 12, true),
        ("03-p12-c1-sof30-mode2-s1", 12, true),
        ("04-p12-c1-sof30-mode3-s1", 12, true),
        ("11-p12-c3-sof30-mode0-s1", 12, false),
        ("12-p12-c3-sof30-mode1-s1", 12, false),
        ("13-p12-c3-sof30-mode2-s1", 12, false),
        ("14-p12-c3-sof30-mode3-s1", 12, false),
        ("32-p12-c3-sof31-mode0-s1", 12, false),
        ("45-p12-c3-sof30-mode0-s3", 12, false),
    ] {
        let data = seed(name);
        let (descriptor, samples) = check_paths(&data, &[]);
        assert_eq!(
            descriptor,
            if gray {
                PixelDescriptor::GRAY16_SRGB
            } else {
                PixelDescriptor::RGB16_SRGB
            },
            "{name}"
        );
        let expected: Vec<u16> = native_codes(&data, gray, false)
            .iter()
            .map(|&c| nearest_16(c, bits))
            .collect();
        assert_eq!(samples, expected, "{name}");
    }
}

#[test]
fn crop_hint_takes_the_same_samples_on_every_path() {
    let data = corpus_12_bit();
    let full = output_words(&decode(&data, &[]));
    let (x, y, w, h) = (16usize, 16usize, 64usize, 32usize);
    let (_, cropped) = check_job_paths(&data, &[], || {
        JpegDecoderConfig::new()
            .job()
            .with_crop_hint(x as u32, y as u32, w as u32, h as u32)
    });
    let stride = 227 * 3;
    let expected: Vec<u16> = (y..y + h)
        .flat_map(|row| full[row * stride + x * 3..row * stride + (x + w) * 3].to_vec())
        .collect();
    assert_eq!(cropped, expected);
}

#[test]
fn orientation_applies_to_u16_exits_on_every_path() {
    // Little-endian TIFF with Orientation 6, in an APP1 ahead of the frame.
    let mut app1 =
        b"Exif\0\0II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0\0\0\0\0".to_vec();
    let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xE1];
    jpeg.extend_from_slice(&((app1.len() + 2) as u16).to_be_bytes());
    jpeg.append(&mut app1);
    let plain = corpus_12_bit();
    jpeg.extend_from_slice(&plain[2..]);

    let upright = output_words(&decode(&plain, &[]));
    let (_, turned) = check_job_paths(&jpeg, &[], || {
        JpegDecoderConfig::new()
            .job()
            .with_orientation(zencodec::OrientationHint::Correct)
    });
    // Rotate 90 degrees clockwise: (x, y) lands at (h - 1 - y, x).
    let (w, h) = (227usize, 149usize);
    let mut expected = vec![0u16; upright.len()];
    for y in 0..h {
        for x in 0..w {
            let to = (x * h + (h - 1 - y)) * 3;
            let from = (y * w + x) * 3;
            expected[to..to + 3].copy_from_slice(&upright[from..from + 3]);
        }
    }
    assert_eq!(turned, expected);
}

#[test]
fn rgba_exit_adds_opaque_alpha_without_touching_color() {
    let data = corpus_12_bit();
    let (descriptor, rgba) = check_paths(&data, &[PixelDescriptor::RGBA16_SRGB]);
    assert_eq!(descriptor, PixelDescriptor::RGBA16_SRGB);
    let (_, rgb) = check_paths(&data, &[PixelDescriptor::RGB16_SRGB]);
    for (px, expected) in rgba.as_chunks::<4>().0.iter().zip(rgb.as_chunks::<3>().0) {
        assert_eq!(&px[..3], expected);
        assert_eq!(px[3], 65535);
    }
}

#[test]
fn a_list_offering_more_than_u8_does_not_narrow() {
    let data = corpus_12_bit();
    for preferred in [
        vec![PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGB16_SRGB],
        JpegDecoderConfig::supported_descriptors().to_vec(),
    ] {
        let (descriptor, samples) = check_paths(&data, &preferred);
        assert_eq!(descriptor, PixelDescriptor::RGB16_SRGB);
        assert_eq!(samples, output_words(&decode(&data, &[])));
    }
    // A later float offer holds the samples too, ahead of U8.
    let output = decode(
        &data,
        &[
            PixelDescriptor::RGB8_SRGB,
            PixelDescriptor::RGBF32_LINEAR,
            PixelDescriptor::RGB16_SRGB,
        ],
    );
    assert_eq!(
        output.pixels().descriptor().channel_type(),
        ChannelType::F32
    );
    // Eight-bit frames keep plain first-supported ranking.
    let eight = seed("10-p8-c3-sof30-mode0-s1");
    let output = decode(
        &eight,
        &[PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGB16_SRGB],
    );
    assert_eq!(output.pixels().descriptor(), PixelDescriptor::RGB8_SRGB);
}

#[test]
fn an_explicit_u8_request_is_honored_on_every_path() {
    let data = corpus_12_bit();
    let preferred = [PixelDescriptor::RGB8_SRGB];
    let one_shot = decode(&data, &preferred);
    assert_eq!(one_shot.pixels().descriptor(), PixelDescriptor::RGB8_SRGB);
    assert_eq!(one_shot.pixels().exact_depth(), None);
    let bytes: Vec<u8> = (0..one_shot.pixels().rows())
        .flat_map(|y| one_shot.pixels().row(y).to_vec())
        .collect();
    let expected: Vec<u8> = native_codes(&data, false, false)
        .iter()
        .map(|&c| ((u32::from(c) * 255 + 2047) / 4095) as u8)
        .collect();
    assert_eq!(bytes, expected);

    let mut streaming = JpegDecoderConfig::new()
        .job()
        .streaming_decoder(Cow::Borrowed(&data), &preferred)
        .unwrap();
    let mut streamed = Vec::new();
    while let Some((_, rows)) = streaming.next_batch().unwrap() {
        for r in 0..rows.rows() {
            streamed.extend_from_slice(rows.row(r));
        }
    }
    assert_eq!(streamed, bytes);
}

#[test]
fn eight_bit_frames_widen_exactly_when_asked_and_stay_u8_otherwise() {
    for (name, gray) in [
        ("00-p8-c1-sof30-mode0-s1", true),
        ("10-p8-c3-sof30-mode0-s1", false),
        ("46-p8-c1-sof31-mode0-s1", true),
    ] {
        let data = seed(name);
        let plain = decode(&data, &[]);
        assert_eq!(
            plain.pixels().descriptor().channel_type(),
            ChannelType::U8,
            "{name}"
        );
        let preferred = [if gray {
            PixelDescriptor::GRAY16_SRGB
        } else {
            PixelDescriptor::RGB16_SRGB
        }];
        let (descriptor, samples) = check_paths(&data, &preferred);
        assert_eq!(descriptor, preferred[0], "{name}");
        let bytes: Vec<u8> = (0..plain.pixels().rows())
            .flat_map(|y| plain.pixels().row(y).to_vec())
            .collect();
        let expected: Vec<u16> = bytes.iter().map(|&b| u16::from(b) * 257).collect();
        assert_eq!(samples, expected, "{name}");
    }
}

#[test]
fn lossless_below_eight_bits_widens_from_its_codes() {
    let data = seed("30-p2-c1-sof31-mode0-s1");
    let plain = decode(&data, &[]);
    assert_eq!(plain.pixels().descriptor().channel_type(), ChannelType::U8);
    let (_, samples) = check_paths(&data, &[PixelDescriptor::GRAY16_SRGB]);
    let expected: Vec<u16> = native_codes(&data, true, false)
        .iter()
        .map(|&c| nearest_16(c, 2))
        .collect();
    assert_eq!(samples, expected);
}

#[test]
fn deep_raw_cmyk_leaves_as_f32_unless_cmyk8_is_named() {
    for (name, bits) in [
        ("16-p12-c4-sof30-mode0-s1", 12u32),
        ("33-p16-c4-sof31-mode0-s1", 16),
    ] {
        let data = seed(name);
        let output = decode(&data, &[]);
        assert_eq!(
            output.pixels().descriptor().pixel_format(),
            PixelFormat::CmykF32Inverted,
            "{name}"
        );
        let codes = zenjpeg::decoder::Decoder::new()
            .output_target(zenjpeg::decoder::OutputTarget::Native16)
            .output_format(zenjpeg::decoder::PixelFormat::Cmyk)
            .decode(&data, enough::Unstoppable)
            .unwrap()
            .pixels_u16()
            .unwrap()
            .to_vec();
        let pixels = output.pixels();
        let floats: Vec<f32> = (0..pixels.rows())
            .flat_map(|y| {
                pixels
                    .row(y)
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_ne_bytes(*b))
                    .collect::<Vec<_>>()
            })
            .collect();
        let max = ((1u64 << bits) - 1) as f32;
        let expected: Vec<f32> = codes.iter().map(|&c| f32::from(c) / max).collect();
        assert_eq!(floats, expected, "{name}");

        let narrowed = decode(&data, &[PixelDescriptor::CMYK8]);
        assert_eq!(
            narrowed.pixels().descriptor(),
            PixelDescriptor::CMYK8,
            "{name}"
        );
    }
}

#[test]
fn deep_cmyk_converted_to_rgb_keeps_its_depth() {
    let data = seed("16-p12-c4-sof30-mode0-s1");
    let output = JpegDecoderConfig::new()
        .cmyk_handling(CmykHandling::BadRgb)
        .job()
        .decoder(Cow::Borrowed(&data), &[])
        .unwrap()
        .decode()
        .unwrap();
    assert_eq!(output.pixels().descriptor(), PixelDescriptor::RGB16_SRGB);
}

// ---------------------------------------------------------------------------
// libjpeg-turbo, run explicitly with ZENJPEG_TURBO_BIN (scripts/build-native-oracle.sh).
// ---------------------------------------------------------------------------

fn djpeg(args: &[&str], input: &[u8]) -> (usize, Vec<u16>) {
    let bin = PathBuf::from(
        std::env::var_os("ZENJPEG_TURBO_BIN")
            .expect("set ZENJPEG_TURBO_BIN to pinned libjpeg-turbo tools directory"),
    );
    let mut child = Command::new(bin.join("djpeg-static"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("launch oracle");
    child.stdin.take().unwrap().write_all(input).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    // P5/P6, width, height, max, one whitespace byte, then big-endian samples.
    let pnm = output.stdout;
    let mut pos = 0;
    let mut fields = Vec::new();
    while fields.len() < 4 {
        let start = pos;
        while !pnm[pos].is_ascii_whitespace() {
            pos += 1;
        }
        fields.push(String::from_utf8(pnm[start..pos].to_vec()).unwrap());
        pos += 1;
    }
    let channels = if fields[0] == "P5" { 1 } else { 3 };
    let samples = pnm[pos..]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_be_bytes(*b))
        .collect();
    (channels, samples)
}

#[test]
#[ignore = "requires explicitly configured libjpeg-turbo 3.2.0 djpeg"]
fn oracle_zencodec_u16_exit_matches_djpeg_codes() {
    let mut files = vec![("testorig12.jpg".to_owned(), corpus_12_bit())];
    for name in [
        "01-p12-c1-sof30-mode0-s1",
        "11-p12-c3-sof30-mode0-s1",
        "12-p12-c3-sof30-mode1-s1",
        "32-p12-c3-sof31-mode0-s1",
        "45-p12-c3-sof30-mode0-s3",
    ] {
        files.push((name.to_owned(), seed(name)));
    }
    for (name, data) in files {
        let (channels, expected) = djpeg(&[], &data);
        let output = decode(&data, &[]);
        assert_eq!(output.pixels().descriptor().channels(), channels, "{name}");
        let samples = output_words(&output);
        assert_eq!(samples.len(), expected.len(), "{name}");
        for (i, (&sample, &code)) in samples.iter().zip(&expected).enumerate() {
            assert_eq!(sample, nearest_16(code, 12), "{name}: sample {i}");
        }
    }
}
