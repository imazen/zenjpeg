//! Regression coverage for #204: pixel removal requires explicit consent.
use enough::Unstoppable;
use zenjpeg::decode::DecodeConfig;
use zenjpeg::encode::EncoderSegments;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use zenjpeg::lossless::{
    EdgeHandling, LosslessTransform, TransformConfig, apply_exif_orientation, transform,
};

fn jpeg(w: u32, h: u32, sampling: Option<ChromaSubsampling>, orientation: u8) -> Vec<u8> {
    let mut exif = b"Exif\0\0II\x2a\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0".to_vec();
    exif.extend_from_slice(&[orientation, 0, 0, 0, 0, 0, 0, 0]);
    let (config, layout, channels) = match sampling {
        Some(s) => (EncoderConfig::ycbcr(90, s), PixelLayout::Rgb8Srgb, 3),
        None => (EncoderConfig::grayscale(90), PixelLayout::Gray8Srgb, 1),
    };
    // Textured luma with flat chroma exercises DC-only chroma too.
    let pixels: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            let v = ((i * 73 + i / w * 41) % 256) as u8;
            std::iter::repeat_n(v, channels)
        })
        .collect();
    let mut enc = config
        .with_segments(EncoderSegments::new().set_exif(exif))
        .encode_from_bytes(w, h, layout)
        .unwrap();
    enc.push_packed(&pixels, Unstoppable).unwrap();
    enc.finish().unwrap()
}

#[test]
fn default_orientation_preserves_all_pixels_or_rejects() {
    use ChromaSubsampling::*;
    for sampling in [
        Option::None,
        Some(ChromaSubsampling::None),
        Some(Quarter),
        Some(HalfHorizontal),
        Some(HalfVertical),
    ] {
        let (mw, mh) = sampling.map_or((8, 8), |s| {
            (8 * u32::from(s.h_factor()), 8 * u32::from(s.v_factor()))
        });
        for (w, h) in [(32, 32), (31, 32), (32, 31), (31, 31), (7, 7)] {
            for orientation in 1..=8 {
                let input = jpeg(w, h, sampling, orientation);
                let transform_kind = LosslessTransform::from_exif_orientation(orientation).unwrap();
                // JPEG perfect-transform rules, expressed independently by EXIF value.
                let needs_width = matches!(orientation, 2 | 3 | 7 | 8);
                let needs_height = matches!(orientation, 3 | 4 | 6 | 7);
                let rejects = (needs_width && w % mw != 0) || (needs_height && h % mh != 0);
                for output in [
                    apply_exif_orientation(&input, Unstoppable),
                    transform(
                        &input,
                        &TransformConfig {
                            transform: transform_kind,
                            ..Default::default()
                        },
                        Unstoppable,
                    ),
                ] {
                    assert_eq!(
                        output.is_err(),
                        rejects,
                        "{sampling:?} {w}x{h} EXIF {orientation}"
                    );
                    if let Ok(output) = output {
                        let decoded = DecodeConfig::new().decode(&output, Unstoppable).unwrap();
                        let expected = if transform_kind.swaps_dimensions() {
                            (h, w)
                        } else {
                            (w, h)
                        };
                        assert_eq!((decoded.width(), decoded.height()), expected);
                    }
                }
                if !rejects {
                    let upright = apply_exif_orientation(&input, Unstoppable).unwrap();
                    assert_eq!(
                        apply_exif_orientation(&upright, Unstoppable).unwrap(),
                        upright
                    );
                }
            }
        }
    }
}

#[test]
fn explicit_trim_cannot_emit_an_empty_image() {
    let input = jpeg(7, 7, Some(ChromaSubsampling::Quarter), 6);
    assert!(
        transform(
            &input,
            &TransformConfig {
                transform: LosslessTransform::Rotate90,
                edge_handling: EdgeHandling::TrimPartialBlocks,
            },
            Unstoppable
        )
        .is_err()
    );
}

#[cfg(feature = "layout")]
#[test]
fn layout_defaults_to_rejecting_pixel_loss() {
    let input = jpeg(32, 24, Some(ChromaSubsampling::Quarter), 6);
    let cfg = zenjpeg::layout::LayoutConfig::new(90_f32);
    assert!(
        cfg.request(&input)
            .rotate_90()
            .execute(&Unstoppable)
            .is_err()
    );
    assert!(
        cfg.request(&input)
            .rotate_90()
            .optimize_for_decode()
            .execute(&Unstoppable)
            .is_err()
    );
}
