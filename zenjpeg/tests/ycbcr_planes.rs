//! `DecodedYCbCr::planes` and `YCbCrPlanes::from_planes` borrow the decoder's
//! `Vec`s as zenpixels plane views, without copying.

use enough::Unstoppable;
use zenjpeg::decoder::Decoder;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, YCbCrPlanes};
use zenpixels::plane::{AnyPlane, FloatEncoding, PlaneEncoding, PlaneRole, PlaneSpec};
use zenpixels::sample::ChromaSubsampling as Subsampling;

#[test]
fn decoded_ycbcr_planes_borrow_without_copying() {
    let (w, h) = (40u32, 24u32);
    let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 251) as u8).collect();
    let mut enc = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encoder");
    enc.push_packed(&rgb, Unstoppable).expect("push");
    let jpeg = enc.finish().expect("finish");

    let decoded = Decoder::new()
        .decode_to_ycbcr_f32(&jpeg, Unstoppable)
        .expect("decode");
    let (layout, planes) = decoded.planes().expect("planes");

    let luma = FloatEncoding::new(255.0, -128.0).unwrap();
    let chroma = FloatEncoding::new(255.0, 0.0).unwrap();
    assert_eq!(
        layout.planes(),
        &[
            PlaneSpec::new(PlaneRole::Luma, PlaneEncoding::Float(luma)),
            PlaneSpec::new(PlaneRole::Cb, PlaneEncoding::Float(chroma)),
            PlaneSpec::new(PlaneRole::Cr, PlaneEncoding::Float(chroma)),
        ]
    );
    assert_eq!(layout.subsampling(), Subsampling::Yuv444);

    let vecs = [&decoded.y, &decoded.cb, &decoded.cr];
    let mut refs = Vec::new();
    for (plane, v) in planes.iter().zip(vecs) {
        let AnyPlane::F32(p) = plane else {
            panic!("expected an f32 plane");
        };
        assert_eq!((p.width(), p.height()), (w as usize, h as usize));
        for (row_idx, row) in p.rows().enumerate() {
            assert!(core::ptr::eq(
                row.as_ptr(),
                v[row_idx * w as usize..].as_ptr()
            ));
        }
        refs.push(*p);
    }

    let enc_planes = YCbCrPlanes::from_planes(refs[0], refs[1], refs[2]);
    assert!(core::ptr::eq(enc_planes.y.as_ptr(), decoded.y.as_ptr()));
    assert!(core::ptr::eq(enc_planes.cb.as_ptr(), decoded.cb.as_ptr()));
    assert!(core::ptr::eq(enc_planes.cr.as_ptr(), decoded.cr.as_ptr()));
    assert_eq!(
        (
            enc_planes.y_stride,
            enc_planes.cb_stride,
            enc_planes.cr_stride
        ),
        (w as usize, w as usize, w as usize)
    );
}
