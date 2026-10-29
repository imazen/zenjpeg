#![cfg(feature = "__diagnostics")]

use zenjpeg::decoder::Decoder;
use zenjpeg::encoder::ProgressiveScanMode;
use zenjpeg::encoder::{AqCoupling, TrellisConfig};
use zenjpeg::encoder::{
    ChromaSubsampling, EncoderConfig, PixelLayout, Unstoppable, XybSubsampling,
};

fn image(w: usize, h: usize, flat_chroma: bool) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let n = ((x * 37 + y * 71 + x * y * 13) % 160) as u8;
            let color = if flat_chroma {
                [n, n, n]
            } else {
                [
                    n,
                    n.wrapping_add((x * 11) as u8),
                    n.wrapping_add((y * 17) as u8),
                ]
            };
            v.extend_from_slice(&color);
        }
    }
    v
}

fn check(config: EncoderConfig, width: usize, height: usize, pixels: &[u8]) {
    let mut plain = config
        .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
        .unwrap();
    plain.push_packed(pixels, Unstoppable).unwrap();
    let expected = plain.finish().unwrap();
    let mut encoder = config
        .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
        .unwrap();
    let capture = encoder.capture_diagnostics().unwrap();
    assert!(encoder.capture_diagnostics().is_err());
    for row in pixels.chunks_exact(width * 3) {
        encoder.push(row, 1, width * 3, Unstoppable).unwrap();
    }
    let capture = capture
        .try_take()
        .expect_err("the encoder still owns its capture");
    let bytes = encoder.finish().unwrap();
    assert_eq!(bytes, expected, "observing must not alter the JPEG");
    let diagnostics = capture.try_take().unwrap();
    if diagnostics.color_path == "YCbCr" {
        let mut independent = jpeg_decoder::Decoder::new(std::io::Cursor::new(&bytes));
        independent.decode().expect("independent JPEG decode");
        let info = independent.info().unwrap();
        assert_eq!((info.width as usize, info.height as usize), (width, height));
    }
    let decoded = Decoder::new()
        .decode_coefficients(&bytes, Unstoppable)
        .unwrap();
    assert_eq!(diagnostics.components.len(), decoded.components.len());
    for (index, (component, reference)) in diagnostics
        .components
        .iter()
        .zip(&decoded.components)
        .enumerate()
    {
        assert_eq!(component.component_id, reference.id);
        assert_eq!(
            diagnostics.sampling_factors[index],
            (reference.h_samp, reference.v_samp)
        );
        let (w, h) = component.block_grid;
        assert_eq!(component.blocks.len(), (w * h) as usize);
        for y in 0..h as usize {
            for x in 0..w as usize {
                let block = &component.blocks[y * w as usize + x];
                assert_eq!(
                    &block.coef_levels[..],
                    reference.block_at(x, y),
                    "component {} block {x},{y}",
                    component.component_id
                );
                assert!(block.coef_pre_quant.iter().all(|v| v.is_finite()));
                if reference.block_at(x, y).iter().any(|v| *v != 0) {
                    assert!(
                        block.coef_pre_quant.iter().any(|v| *v != 0.0),
                        "uncaptured nonzero block {x},{y}"
                    );
                }
                assert!(block.aq_strength.is_finite());
            }
        }
    }
}

#[test]
fn capture_matches_final_jpeg_and_is_byte_neutral() {
    let (w, h) = (33, 19);
    for flat in [false, true] {
        let pixels = image(w, h, flat);
        for sub in [
            ChromaSubsampling::None,
            ChromaSubsampling::HalfHorizontal,
            ChromaSubsampling::Quarter,
            ChromaSubsampling::HalfVertical,
        ] {
            for progressive in [false, true] {
                let mode = if progressive {
                    ProgressiveScanMode::Progressive
                } else {
                    ProgressiveScanMode::Baseline
                };
                check(
                    EncoderConfig::ycbcr(83.0, sub).scan_mode(mode),
                    w,
                    h,
                    &pixels,
                );
            }
        }
        check(
            EncoderConfig::ycbcr(83.0, ChromaSubsampling::Quarter)
                .scan_mode(ProgressiveScanMode::Baseline)
                .optimize_huffman(false),
            w,
            h,
            &pixels,
        );
        check(EncoderConfig::grayscale(83.0), w, h, &pixels);
        check(EncoderConfig::rgb(83.0), w, h, &pixels);
        for sub in [XybSubsampling::Full, XybSubsampling::BQuarter] {
            check(EncoderConfig::xyb(83.0, sub), w, h, &pixels);
        }
        let mut trellis = TrellisConfig::new();
        trellis.dc_enabled = true;
        trellis.aq_coupling = AqCoupling {
            scale: 2.0,
            ..AqCoupling::OFF
        };
        check(
            EncoderConfig::ycbcr(83.0, ChromaSubsampling::Quarter).trellis(trellis),
            w,
            h,
            &pixels,
        );
    }
}

#[test]
fn capture_rejects_late_installation() {
    let mut encoder = EncoderConfig::ycbcr(80.0, ChromaSubsampling::Quarter)
        .encode_from_bytes(8, 8, PixelLayout::Rgb8Srgb)
        .unwrap();
    encoder.push(&[64; 24], 1, 24, Unstoppable).unwrap();
    assert!(encoder.capture_diagnostics().is_err());
}

#[cfg(feature = "boundary-rd")]
#[test]
fn capture_matches_boundary_refinement() {
    use zenjpeg::encoder::{BoundaryRd, BoundaryRdConfig};
    let (w, h) = (33, 19);
    check(
        EncoderConfig::ycbcr(80.0, ChromaSubsampling::Quarter)
            .boundary_rd(BoundaryRd::On(BoundaryRdConfig::default())),
        w,
        h,
        &image(w, h, false),
    );
}

#[test]
fn typed_rgb_encoder_exposes_the_same_capture() {
    let config = EncoderConfig::ycbcr(80.0, ChromaSubsampling::None);
    let pixels = vec![rgb::RGB8::new(80, 100, 120); 16 * 16];
    let mut encoder = config.encode_from_rgb::<rgb::RGB8>(16, 16).unwrap();
    let capture = encoder.capture_diagnostics().unwrap();
    encoder.push(&pixels, 16, 16, Unstoppable).unwrap();
    let bytes = encoder.finish().unwrap();
    let diag = capture.try_take().unwrap();
    let decoded = Decoder::new()
        .decode_coefficients(&bytes, Unstoppable)
        .unwrap();
    assert_eq!(
        diag.components[0].blocks[0].coef_levels.as_slice(),
        decoded.components[0].block(0)
    );
}
