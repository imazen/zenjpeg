//! Shared deterministic source and explicit decoder-oracle helpers.
use zenjpeg::decoder::{ChromaUpsampling, Decoder, IdctMethod, OutputTarget, PixelFormat};
use zenjpeg::encoder::{EncoderConfig, PixelLayout as Layout, Unstoppable};

pub fn pixels(w: usize, h: usize, layout: Layout, pattern: usize) -> Vec<u8> {
    let mut v = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let mut rgb = match pattern {
                0 => [81; 3],
                1 => [47, 139, 203],
                2 => {
                    if (x + y) % 2 == 0 {
                        [24, 180, 91]
                    } else {
                        [230, 57, 164]
                    }
                }
                _ => [
                    (x * 43 + y * 71) as u8,
                    (x * 79 + y * 19 + 37) as u8,
                    if x == w - 1 || y == h - 1 { 251 } else { 32 },
                ],
            };
            if matches!(
                layout,
                Layout::Bgr8Srgb | Layout::Bgrx8Srgb | Layout::Bgra8Srgb
            ) {
                rgb.swap(0, 2);
            }
            for c in 0..layout.channels() {
                let value = if c < 3 { rgb[c] } else { 17 };
                match layout {
                    Layout::Rgb16Linear
                    | Layout::Rgbx16Linear
                    | Layout::Rgba16Linear
                    | Layout::Gray16Linear => {
                        v.extend_from_slice(&(value as u16 * 257).to_ne_bytes())
                    }
                    Layout::RgbF32Linear
                    | Layout::RgbxF32Linear
                    | Layout::RgbaF32Linear
                    | Layout::GrayF32Linear => {
                        v.extend_from_slice(&(value as f32 / 255.).to_ne_bytes())
                    }
                    _ => v.push(value),
                }
            }
        }
    }
    v
}

pub fn streamed(
    cfg: &EncoderConfig,
    src: &[u8],
    w: usize,
    h: usize,
    layout: Layout,
    extra: usize,
    parts: &[usize],
) -> Vec<u8> {
    let row = w * layout.bytes_per_pixel();
    let stride = row + extra;
    let mut padded = vec![0xE7; stride * h];
    for y in 0..h {
        padded[y * stride..y * stride + row].copy_from_slice(&src[y * row..(y + 1) * row]);
    }
    let mut enc = cfg.encode_from_bytes(w as u32, h as u32, layout).unwrap();
    let mut y = 0;
    let mut i = 0;
    while y < h {
        let n = parts[i % parts.len()].min(h - y);
        enc.push(&padded[y * stride..], n, stride, Unstoppable)
            .unwrap();
        y += n;
        i += 1;
    }
    enc.finish().unwrap()
}

pub fn plane(w: usize, h: usize, extra: usize, channel: usize, hs: usize, vs: usize) -> Vec<f32> {
    let mut v = vec![1234.; (w + extra) * h];
    for y in 0..h {
        for x in 0..w {
            v[y * (w + extra) + x] = match channel {
                0 => ((x * 43 + y * 71) % 193) as f32 - 90.,
                1 => (((x / hs) * 19 + (y / vs) * 31) % 151) as f32 - 71.,
                _ => (((x / hs) * 37 + (y / vs) * 11) % 173) as f32 - 83.,
            };
        }
    }
    v
}

pub fn compare_turbo(jpeg: &[u8], w: usize, h: usize, label: &str) {
    use std::{
        io::Write,
        path::PathBuf,
        process::{Command, Stdio},
    };
    let bin = PathBuf::from(
        std::env::var_os("ZENJPEG_TURBO_BIN")
            .expect("set ZENJPEG_TURBO_BIN to pinned libjpeg-turbo tools directory"),
    );
    let mut child = Command::new(bin.join("djpeg-static"))
        .args(["-nosmooth", "-rgb"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("launch pinned djpeg");
    let mut stdin = child.stdin.take().unwrap();
    let out = std::thread::scope(|scope| {
        let writer = scope.spawn(move || stdin.write_all(jpeg).unwrap());
        let out = child.wait_with_output().unwrap();
        writer.join().unwrap();
        out
    });
    assert!(
        out.status.success(),
        "{label}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let header = format!("P6\n{w} {h}\n255\n");
    assert!(
        out.stdout.starts_with(header.as_bytes()),
        "{label}: PPM header"
    );
    let expected = &out.stdout[header.len()..];
    let native = Decoder::new()
        .idct_method(IdctMethod::Libjpeg)
        .output_target(OutputTarget::Srgb8)
        .output_format(PixelFormat::Rgb)
        .chroma_upsampling(ChromaUpsampling::NearestNeighbor)
        .decode(jpeg, Unstoppable)
        .unwrap();
    assert_eq!(native.pixels_u8().unwrap(), expected, "{label}");
}
