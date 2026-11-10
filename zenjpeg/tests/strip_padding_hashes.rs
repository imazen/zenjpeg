//! Versioned decode identity, supplemental to the independent padding oracles.
//! Run the explicit regeneration recipe with pinned turbo, review the text diff.
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use zenjpeg::decoder::{ChromaUpsampling, Decoder, IdctMethod, OutputTarget, PixelFormat};
use zenjpeg::encoder::{
    ChromaSubsampling as Sub, DownsamplingMethod as Method, EncoderConfig, PixelLayout as Layout,
    Unstoppable, XybSubsampling, YCbCrPlanes,
};
#[path = "support/strip_padding_inputs.rs"]
mod inputs;
use inputs::{compare_turbo, pixels, plane, streamed};

const SUBS: [Sub; 4] = [
    Sub::None,
    Sub::HalfHorizontal,
    Sub::Quarter,
    Sub::HalfVertical,
];
const SCHEMA: &str = "zenjpeg-strip-padding-decode-v1";
const DECODER: &str = "scalar; Libjpeg IDCT; nearest chroma; Srgb8; RGB; no ICC transform";

struct Group {
    hash: Sha256,
    cases: usize,
    turbo: bool,
}
impl Group {
    fn new(key: &str, turbo: bool) -> Self {
        let mut hash = Sha256::new();
        frame(&mut hash, SCHEMA.as_bytes());
        frame(&mut hash, DECODER.as_bytes());
        frame(&mut hash, key.as_bytes());
        Self {
            hash,
            cases: 0,
            turbo,
        }
    }
    fn add(&mut self, label: &str, jpeg: &[u8], w: usize, h: usize, independent: bool) {
        let decoded = Decoder::new()
            .idct_method(IdctMethod::Libjpeg)
            .chroma_upsampling(ChromaUpsampling::NearestNeighbor)
            .output_target(OutputTarget::Srgb8)
            .output_format(PixelFormat::Rgb)
            .decode(jpeg, Unstoppable)
            .unwrap();
        let rgb = decoded.pixels_u8().unwrap();
        assert_eq!(rgb.len(), w * h * 3);
        if independent && self.turbo {
            compare_turbo(jpeg, w, h, label);
        }
        frame(&mut self.hash, label.as_bytes());
        self.hash.update((w as u64).to_le_bytes());
        self.hash.update((h as u64).to_le_bytes());
        frame(&mut self.hash, rgb);
        self.cases += 1;
    }
    fn finish(self) -> serde_json::Value {
        serde_json::json!({"cases": self.cases, "rgb_sha256": format!("{:x}", self.hash.finalize()),
            "independent_turbo": self.turbo})
    }
}
fn frame(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}
fn geometries() -> Vec<(usize, usize)> {
    // Every mod-16 residue in both axes, then combined corners after 1/2 strips.
    let mut v: Vec<_> = (1..=16).map(|r| (r, (r * 7) % 16 + 1)).collect();
    v.extend([
        (1, 33),
        (33, 1),
        (7, 17),
        (17, 32),
        (31, 33),
        (32, 8),
        (33, 49),
        (129, 50),
    ]);
    v
}
fn byte_group(
    key: &str,
    cfg: EncoderConfig,
    layout: Layout,
    turbo: bool,
    independent: bool,
) -> serde_json::Value {
    let mut group = Group::new(key, turbo);
    let mut geometries = geometries();
    if key.starts_with("automatic-parallel/") {
        geometries.extend([(257, 513), (513, 519)]);
    }
    for (i, (w, h)) in geometries.into_iter().enumerate() {
        for pattern in 0..4 {
            let src = pixels(w, h, layout, pattern);
            let jpeg = streamed(&cfg, &src, w, h, layout, 13, &[1, 7, 19, 2]);
            group.add(
                &format!("geometry{i}/pattern{pattern}"),
                &jpeg,
                w,
                h,
                independent,
            );
        }
    }
    group.finish()
}
fn manifest(independent: bool) -> serde_json::Value {
    // A separate test executable confines this process-global test dispatch state.
    let _lock = archmage::testing::lock_token_testing();
    archmage::dangerously_disable_tokens_except_wasm(true).unwrap();
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            archmage::dangerously_disable_tokens_except_wasm(false).unwrap();
        }
    }
    let _restore = Restore;
    let mut groups = BTreeMap::new();
    // Defaults (AQ/deringing enabled), one canonical quality per producer group.
    for sub in SUBS {
        for method in [Method::Box, Method::GammaAware, Method::GammaAwareIterative] {
            for (q, progressive) in [(5, false), (40, true), (100, false)] {
                let key = format!("ycbcr/{sub:?}/{method:?}/q{q}/progressive{progressive}");
                let cfg = EncoderConfig::ycbcr(q, sub)
                    .downsampling_method(method)
                    .progressive(progressive);
                groups.insert(
                    key.clone(),
                    byte_group(&key, cfg, Layout::Rgb8Srgb, true, independent),
                );
            }
        }
    }
    for layout in [
        Layout::Rgb8Srgb,
        Layout::Rgba8Srgb,
        Layout::Rgbx8Srgb,
        Layout::Bgr8Srgb,
        Layout::Bgra8Srgb,
        Layout::Bgrx8Srgb,
        Layout::Rgb16Linear,
        Layout::Rgba16Linear,
        Layout::Rgbx16Linear,
        Layout::RgbF32Linear,
        Layout::RgbaF32Linear,
        Layout::RgbxF32Linear,
        Layout::Gray8Srgb,
        Layout::Gray16Linear,
        Layout::GrayF32Linear,
    ] {
        for method in [Method::Box, Method::GammaAware, Method::GammaAwareIterative] {
            let key = format!("format/{layout:?}/420/{method:?}/q40");
            groups.insert(
                key.clone(),
                byte_group(
                    &key,
                    EncoderConfig::ycbcr(40, Sub::Quarter).downsampling_method(method),
                    layout,
                    true,
                    independent,
                ),
            );
        }
    }
    for sub in [XybSubsampling::Full, XybSubsampling::BQuarter] {
        for layout in [Layout::Rgb16Linear, Layout::RgbF32Linear] {
            let key = format!("xyb-deep/{sub:?}/{layout:?}/q40");
            groups.insert(
                key.clone(),
                byte_group(
                    &key,
                    EncoderConfig::xyb(40, sub),
                    layout,
                    false,
                    independent,
                ),
            );
        }
    }
    for (name, cfg, turbo) in [
        ("rgb", EncoderConfig::rgb(40), true),
        ("gray", EncoderConfig::grayscale(40), true),
        (
            "xyb-full",
            EncoderConfig::xyb(40, XybSubsampling::Full),
            false,
        ),
        (
            "xyb-bquarter",
            EncoderConfig::xyb(40, XybSubsampling::BQuarter),
            false,
        ),
    ] {
        for progressive in [false, true] {
            let key = format!("color/{name}/q40/progressive{progressive}");
            groups.insert(
                key.clone(),
                byte_group(
                    &key,
                    cfg.clone().progressive(progressive),
                    Layout::Rgb8Srgb,
                    turbo,
                    independent,
                ),
            );
        }
    }
    for sub in SUBS {
        for sampled in [false, true] {
            let key = format!("planar/{sub:?}/presampled{sampled}/q40");
            let mut group = Group::new(&key, true);
            let hs = if sampled { sub.h_factor() as usize } else { 1 };
            let vs = if sampled { sub.v_factor() as usize } else { 1 };
            for (i, (w, h)) in geometries().into_iter().enumerate() {
                let cw = w.div_ceil(hs);
                let ch = h.div_ceil(vs);
                let y = plane(w, h, 3, 0, 1, 1);
                let cb = plane(
                    cw,
                    ch,
                    3,
                    1,
                    if sampled { 1 } else { sub.h_factor() as usize },
                    if sampled { 1 } else { sub.v_factor() as usize },
                );
                let cr = plane(
                    cw,
                    ch,
                    3,
                    2,
                    if sampled { 1 } else { sub.h_factor() as usize },
                    if sampled { 1 } else { sub.v_factor() as usize },
                );
                let cfg = EncoderConfig::ycbcr(40, sub);
                let mut enc = cfg.encode_from_ycbcr_planar(w as u32, h as u32).unwrap();
                let strip = sub.v_factor() as usize * 8;
                let mut row = 0;
                while row < h {
                    let rows = strip.min(h - row);
                    let planes = YCbCrPlanes {
                        y: &y[row * (w + 3)..],
                        y_stride: w + 3,
                        cb: &cb[(row / vs) * (cw + 3)..],
                        cb_stride: cw + 3,
                        cr: &cr[(row / vs) * (cw + 3)..],
                        cr_stride: cw + 3,
                    };
                    if sampled {
                        enc.push_subsampled(&planes, rows, Unstoppable).unwrap();
                    } else {
                        enc.push(&planes, rows, Unstoppable).unwrap();
                    }
                    row += rows;
                }
                group.add(
                    &format!("geometry{i}"),
                    &enc.finish().unwrap(),
                    w,
                    h,
                    independent,
                );
            }
            groups.insert(key, group.finish());
        }
    }
    #[cfg(feature = "parallel")]
    {
        use zenjpeg::encoder::ParallelEncoding;
        for sub in SUBS {
            let key = format!("fused-parallel/{sub:?}/q40");
            let mut group = Group::new(&key, true);
            for (w, h) in [(257, 513), (511, 519), (512, 528), (513, 545)] {
                for pattern in 0..4 {
                    let src = pixels(w, h, Layout::Rgb8Srgb, pattern);
                    let jpeg = EncoderConfig::ycbcr(40, sub)
                        .progressive(false)
                        .encode_bytes_parallel(&src, w as u32, h as u32)
                        .unwrap();
                    group.add(
                        &format!("{w}x{h}/pattern{pattern}"),
                        &jpeg,
                        w,
                        h,
                        independent,
                    );
                }
            }
            groups.insert(key, group.finish());
            let key = format!("automatic-parallel/{sub:?}/q40");
            let cfg = EncoderConfig::ycbcr(40, sub)
                .progressive(false)
                .parallel(ParallelEncoding::Auto);
            groups.insert(
                key.clone(),
                byte_group(&key, cfg, Layout::Rgb8Srgb, true, independent),
            );
        }
    }
    serde_json::json!({"schema":SCHEMA, "decoder":DECODER,
        "framing":"length-prefixed UTF-8 schema, decoder, group; repeated length-prefixed label, u64le width/height, length-prefixed row-major RGB8",
        "source":"strip_padding_inputs pixels/plane v1; geometry order in strip_padding_hashes.rs; stride sentinel E7; AQ/deringing defaults",
        "groups":groups})
}
fn expected() -> serde_json::Value {
    serde_json::from_str(include_str!("support/strip_padding_decode_v1.json")).unwrap()
}
#[test]
fn versioned_padding_decode_hashes() {
    let actual = manifest(false);
    let expected = expected();
    #[cfg(not(feature = "parallel"))]
    let expected = {
        let mut expected = expected;
        expected["groups"]
            .as_object_mut()
            .unwrap()
            .retain(|k, _| !k.contains("parallel"));
        expected
    };
    check(&actual, &expected);
}
#[test]
#[ignore = "explicit pinned turbo verification; run just strip-padding-hash-oracle"]
fn independent_padding_decode_hashes() {
    check(&manifest(true), &expected());
}
#[test]
#[ignore = "explicit regeneration writes a review candidate outside the repository"]
fn regenerate_padding_decode_hashes() {
    let path =
        std::env::var_os("ZENJPEG_PADDING_HASH_CANDIDATE").expect("set candidate output path");
    let actual = manifest(true);
    std::fs::write(path, serde_json::to_string_pretty(&actual).unwrap() + "\n").unwrap();
}

fn check(actual: &serde_json::Value, expected: &serde_json::Value) {
    let a = actual["groups"].as_object().unwrap();
    let e = expected["groups"].as_object().unwrap();
    assert_eq!(a.keys().collect::<Vec<_>>(), e.keys().collect::<Vec<_>>());
    for (key, value) in a {
        assert_eq!(
            value, &e[key],
            "{key}: decode identity changed; review independent oracles before updating goldens"
        );
    }
    for key in ["schema", "decoder", "framing", "source"] {
        assert_eq!(actual[key], expected[key]);
    }
    eprintln!(
        "decode goldens: {} groups, {} decodes",
        a.len(),
        a.values()
            .map(|v| v["cases"].as_u64().unwrap())
            .sum::<u64>()
    );
}

#[path = "support/progressive_decode_hashes.rs"]
mod progressive_decode_hashes;

#[test]
#[ignore = "explicit pinned turbo validation; run just progressive-decode-oracle"]
fn independent_progressive_decode_hashes() {
    let jpeg = progressive_decode_hashes::encode_noise();
    // Validate pixels independently before checking the versioned identity.
    compare_turbo(
        &jpeg,
        progressive_decode_hashes::WIDTH as usize,
        progressive_decode_hashes::HEIGHT as usize,
        "progressive AC-refinement noise",
    );
    progressive_decode_hashes::check_noise_decode(&jpeg);
    let jpeg = progressive_decode_hashes::encode_gradient();
    compare_turbo(
        &jpeg,
        progressive_decode_hashes::GRADIENT_SIZE as usize,
        progressive_decode_hashes::GRADIENT_SIZE as usize,
        "progressive grayscale gradient",
    );
    progressive_decode_hashes::check_gradient_decode(&jpeg);
}
