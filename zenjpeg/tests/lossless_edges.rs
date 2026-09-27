//! Regression coverage for #204: pixel removal requires explicit consent.
use enough::Unstoppable;
use zenjpeg::decode::DecodeConfig;
use zenjpeg::encode::EncoderSegments;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use zenjpeg::lossless::{
    EdgeHandling, LosslessTransform, TransformConfig, apply_exif_orientation, transform,
};

fn jpeg(w: u32, h: u32, sampling: Option<ChromaSubsampling>, orientation: u8) -> Vec<u8> {
    let mut exif = b"II\x2a\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0".to_vec();
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
                let info = DecodeConfig::new()
                    .auto_orient(false)
                    .read_info(&input)
                    .unwrap();
                assert_eq!(
                    zenjpeg::lossless::parse_exif_orientation(info.exif.as_deref().unwrap()),
                    Some(orientation)
                );
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
                        let decoded = DecodeConfig::new()
                            .auto_orient(false)
                            .decode(&output, Unstoppable)
                            .unwrap();
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

#[test]
fn explicit_orientation_trim_matches_the_retained_pixels() {
    use zenjpeg::lossless::apply_exif_orientation_with_edge_handling;
    let input = jpeg(32, 23, None, 6);
    assert!(apply_exif_orientation(&input, Unstoppable).is_err());
    let output = apply_exif_orientation_with_edge_handling(
        &input,
        EdgeHandling::TrimPartialBlocks,
        Unstoppable,
    )
    .unwrap();
    let dec = DecodeConfig::new().auto_orient(false);
    let original = dec.decode(&input, Unstoppable).unwrap();
    let upright = dec.decode(&output, Unstoppable).unwrap();
    assert_eq!((upright.width(), upright.height()), (16, 32));
    let channels = original.pixels_u8().unwrap().len() / (32 * 23);
    for sy in 0..16 {
        for sx in 0..32 {
            for c in 0..channels {
                let source = original.pixels_u8().unwrap()[(sy * 32 + sx) * channels + c];
                let dest = upright.pixels_u8().unwrap()[(sx * 16 + 15 - sy) * channels + c];
                assert!(source.abs_diff(dest) <= 2);
            }
        }
    }
    assert_eq!(
        zenjpeg::lossless::parse_exif_orientation(upright.extras().unwrap().exif().unwrap()),
        Some(1)
    );
    assert_eq!(
        apply_exif_orientation(&output, Unstoppable).unwrap(),
        output
    );
}

#[test]
fn rejected_orientation_obeys_cancellation() {
    struct Cancel;
    impl enough::Stop for Cancel {
        fn check(&self) -> Result<(), enough::StopReason> {
            Err(enough::StopReason::Cancelled)
        }
    }
    let input = jpeg(32, 24, Some(ChromaSubsampling::Quarter), 6);
    let err = apply_exif_orientation(&input, Cancel).unwrap_err();
    assert!(matches!(
        err.kind(),
        zenjpeg::decoder::ErrorKind::Cancelled(_)
    ));
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
