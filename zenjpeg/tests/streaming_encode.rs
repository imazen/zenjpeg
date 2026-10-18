#![cfg(feature = "zencodec")]
//! Streaming encode output through zencodec's `Encoder::begin` /
//! `pending_output`: fixed-table sequential JPEG streams its header and
//! MCU rows as rows arrive, byte-identical to the one-shot encode;
//! optimized-table and progressive configurations report `AT_FINISH`.
//! Explicit settings survive the effort preset.

use zencodec::encode::{EncodeJob, Encoder, EncoderConfig, OutputTiming};
use zencodec::{CategorizedError, ErrorCategory, Metadata, MetadataPolicy, ResourceLimits};
use zencodec_testkit as tk;
use zenjpeg::encoder::ChromaSubsampling;
use zenjpeg::{JpegDecoderConfig as D, JpegEncoderConfig as E};
use zenpixels::{PixelDescriptor, PixelSlice};

/// Fixed general-purpose tables, sequential scan: the streaming configuration.
fn fixed(sub: ChromaSubsampling) -> E {
    E::ycbcr(85.0, sub)
        .with_progressive(false)
        .with_huffman(false)
}

/// Deterministic noisy RGB8 (or gray) pixels, so every MCU codes bytes.
fn pixels(w: u32, h: u32, ch: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(w as usize * h as usize * ch);
    let mut x = 0x9e37_79b9u32;
    for i in 0..(w as usize * h as usize * ch) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        // Smooth gradient plus noise, like a photo.
        v.push(((i % 251) as u32 / 2 + (x & 63)) as u8);
    }
    v
}

fn desc(ch: usize) -> PixelDescriptor {
    match ch {
        1 => PixelDescriptor::GRAY8_SRGB,
        3 => PixelDescriptor::RGB8_SRGB,
        _ => PixelDescriptor::RGBA8_SRGB,
    }
}

fn meta() -> Metadata {
    let xmp = br#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF/></x:xmpmeta>"#;
    // Minimal TIFF header: little-endian, IFD at 8, no entries.
    let exif = [b'I', b'I', 42, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    Metadata::none().with_exif(&exif[..]).with_xmp(&xmp[..])
}

fn one_shot(cfg: &E, px: &[u8], w: u32, h: u32, ch: usize, m: &Metadata) -> Vec<u8> {
    let s = PixelSlice::new(px, w, h, w as usize * ch, desc(ch)).unwrap();
    cfg.clone()
        .job()
        .with_metadata_policy(m.clone(), MetadataPolicy::PreserveExact)
        .encoder()
        .unwrap()
        .encode(s)
        .unwrap()
        .into_vec()
}

/// `begin`, push in strips of `strip` rows, drain after each. Returns the
/// timing, the file, and the bytes pending after `begin` and after each
/// pushed row count.
fn streamed(
    cfg: &E,
    px: &[u8],
    (w, h, ch): (u32, u32, usize),
    strip: u32,
    m: &Metadata,
) -> (OutputTiming, Vec<u8>, Vec<(u32, usize)>) {
    let stride = w as usize * ch;
    let mut enc = cfg
        .clone()
        .job()
        .with_metadata_policy(m.clone(), MetadataPolicy::PreserveExact)
        .encoder()
        .unwrap();
    let timing = enc.begin(w, h, desc(ch)).unwrap();
    let mut out = Vec::new();
    let mut seen = vec![(0, enc.pending_output().len())];
    let take = |enc: &mut zenjpeg::JpegEncoder, out: &mut Vec<u8>| {
        let n = enc.pending_output().len();
        out.extend_from_slice(enc.pending_output());
        enc.consume_output(n);
    };
    take(&mut enc, &mut out);
    let mut y = 0;
    while y < h {
        let n = strip.min(h - y);
        let s = PixelSlice::new(
            &px[y as usize * stride..(y + n) as usize * stride],
            w,
            n,
            stride,
            desc(ch),
        )
        .unwrap();
        enc.push_rows(s).unwrap();
        y += n;
        seen.push((y, out.len() + enc.pending_output().len()));
        take(&mut enc, &mut out);
    }
    out.extend_from_slice(enc.finish().unwrap().data());
    (timing, out, seen)
}

#[test]
fn fixed_tables_stream_byte_identical_to_one_shot() {
    let cases = [
        (ChromaSubsampling::Quarter, 3usize, 31u32),
        (ChromaSubsampling::None, 3, 15),
        (ChromaSubsampling::Quarter, 4, 31),
        // Gray input keeps the configuration's 16-row strips.
        (ChromaSubsampling::Quarter, 1, 31),
        (ChromaSubsampling::None, 1, 15),
    ];
    for (sub, ch, lag) in cases {
        let cfg = fixed(sub);
        for (w, h) in [(1u32, 1u32), (37, 21), (64, 48), (203, 131)] {
            let px = pixels(w, h, ch);
            for m in [Metadata::none(), meta()] {
                let want = one_shot(&cfg, &px, w, h, ch, &m);
                for strip in [1u32, 7, 16, h] {
                    let (timing, got, _) = streamed(&cfg, &px, (w, h, ch), strip, &m);
                    // An image no taller than the lag would be held whole:
                    // that is AT_FINISH, not a stream.
                    let want_timing = if h > lag {
                        OutputTiming::streaming(lag)
                    } else {
                        OutputTiming::AT_FINISH
                    };
                    assert_eq!(timing, want_timing, "{sub:?} ch{ch} {w}x{h}");
                    assert_eq!(
                        got, want,
                        "{sub:?} ch{ch} {w}x{h} strip {strip}: streamed bytes differ"
                    );
                }
            }
        }
    }
}

#[test]
fn metadata_is_in_the_header_pending_at_begin() {
    let cfg = fixed(ChromaSubsampling::Quarter);
    let mut enc = cfg
        .job()
        .with_metadata_policy(meta(), MetadataPolicy::PreserveExact)
        .encoder()
        .unwrap();
    assert!(enc.begin(64, 48, desc(3)).unwrap().streams());
    let header = enc.pending_output();
    let has = |needle: &[u8]| header.windows(needle.len()).any(|w| w == needle);
    assert_eq!(&header[..2], &[0xFF, 0xD8], "SOI");
    assert!(has(b"Exif\0\0"), "EXIF APP1 in the header");
    assert!(
        has(b"http://ns.adobe.com/xap/1.0/\0"),
        "XMP APP1 in the header"
    );
    assert!(has(&[0xFF, 0xDA]), "header runs through SOS");
}

#[test]
fn reported_lag_bounds_the_rows_held_back() {
    for (sub, ch) in [
        (ChromaSubsampling::Quarter, 3usize),
        (ChromaSubsampling::None, 3),
        (ChromaSubsampling::Quarter, 1),
    ] {
        let (w, h) = (96u32, 120u32);
        let px = pixels(w, h, ch);
        let (timing, _, seen) = streamed(&fixed(sub), &px, (w, h, ch), 1, &Metadata::none());
        let lag = timing.lag_rows().unwrap();
        let header = seen[0].1;
        let first = seen.iter().find(|&&(_, n)| n > header).unwrap().0;
        eprintln!("{sub:?} ch{ch}: lag {lag}, first scan bytes after {first} rows");
        // Rows are held back by at most `lag`: once `lag + 1` rows are in,
        // the first MCU row's bytes are out.
        assert!(
            first <= lag + 1,
            "{sub:?} ch{ch}: first scan bytes after {first} rows, lag {lag}"
        );
        // And the lag is not slack: no scan bytes before one strip is full.
        let strip = if matches!(sub, ChromaSubsampling::Quarter) {
            16
        } else {
            8
        };
        assert!(first >= strip, "{sub:?} ch{ch}: bytes after {first} rows");
        // Output grows with every further strip.
        let mut prev = 0;
        for &(rows, n) in &seen {
            if rows > lag && rows % strip == 0 && rows < h {
                assert!(n > prev, "{sub:?} ch{ch}: no new bytes by row {rows}");
                prev = n;
            }
        }
    }
}

#[test]
fn optimized_and_progressive_report_at_finish() {
    let (w, h) = (64u32, 48u32);
    let px = pixels(w, h, 3);
    for cfg in [
        E::new(),                         // effort 1: progressive, optimized
        E::new().with_progressive(false), // sequential, optimized
        E::new().with_huffman(false),     // fixed tables, but progressive
        E::new().with_generic_effort(2),  // progressive search
    ] {
        let m = meta();
        let (timing, got, seen) = streamed(&cfg, &px, (w, h, 3), 16, &m);
        assert_eq!(timing, OutputTiming::AT_FINISH, "{cfg:?}");
        assert!(
            seen.iter().all(|&(_, n)| n == 0),
            "nothing pending: {cfg:?}"
        );
        assert_eq!(got, one_shot(&cfg, &px, w, h, 3, &m), "{cfg:?}");
    }
}

/// The SOF marker type: 0xC0 baseline, 0xC1 extended sequential, 0xC2 progressive.
fn sof(jpeg: &[u8]) -> u8 {
    jpeg.windows(2)
        .find(|w| w[0] == 0xFF && (0xC0..=0xC2).contains(&w[1]))
        .unwrap()[1]
}

#[test]
fn explicit_settings_survive_the_effort_preset() {
    let (w, h) = (64u32, 48u32);
    let px = pixels(w, h, 3);
    let m = Metadata::none();
    // The default effort chooses progressive; an explicit choice wins.
    assert_eq!(sof(&one_shot(&E::new(), &px, w, h, 3, &m)), 0xC2);
    let seq = one_shot(&E::new().with_progressive(false), &px, w, h, 3, &m);
    assert_eq!(sof(&seq), 0xC0);
    // Fixed tables are a different, larger table set than optimized ones.
    let fixed = one_shot(&fixed(ChromaSubsampling::Quarter), &px, w, h, 3, &m);
    assert_ne!(fixed, seq);
    // A named preset is not replaced by the default effort's preset.
    let preset = E::from_preset("jpegli_baseline", 85.0).unwrap();
    assert_eq!(sof(&one_shot(&preset, &px, w, h, 3, &m)), 0xC0);
    // A later effort request replaces the named preset.
    let effort = E::from_preset("jpegli_baseline", 85.0)
        .unwrap()
        .with_generic_effort(1);
    assert_eq!(sof(&one_shot(&effort, &px, w, h, 3, &m)), 0xC2);
}

#[test]
fn canvas_size_push_rows_keeps_the_configuration() {
    // Rows pushed with a known canvas used to be re-encoded as baseline
    // with fixed tables whatever the configuration said.
    let (w, h) = (64u32, 48u32);
    let px = pixels(w, h, 3);
    let cfg = E::new();
    let mut enc = cfg.clone().job().with_canvas_size(w, h).encoder().unwrap();
    for y in (0..h).step_by(16) {
        let s = PixelSlice::new(
            &px[y as usize * w as usize * 3..][..16 * w as usize * 3],
            w,
            16,
            w as usize * 3,
            desc(3),
        )
        .unwrap();
        enc.push_rows(s).unwrap();
    }
    let got = enc.finish().unwrap().into_vec();
    assert_eq!(got, one_shot(&cfg, &px, w, h, 3, &Metadata::none()));
    assert_eq!(sof(&got), 0xC2);
}

#[test]
fn row_protocol_errors() {
    let (w, h) = (32u32, 32u32);
    let px = pixels(w, h, 3);
    let cfg = fixed(ChromaSubsampling::Quarter);
    let all = || PixelSlice::new(&px, w, h, w as usize * 3, desc(3)).unwrap();
    // More rows than declared.
    let mut enc = cfg.clone().job().encoder().unwrap();
    enc.begin(w, h - 8, desc(3)).unwrap();
    assert!(enc.push_rows(all()).is_err());
    // Another pixel format than declared.
    let mut enc = cfg.clone().job().encoder().unwrap();
    enc.begin(w, h, desc(4)).unwrap();
    assert!(enc.push_rows(all()).is_err());
    // begin after rows.
    let mut enc = cfg.clone().job().with_canvas_size(w, h).encoder().unwrap();
    enc.push_rows(PixelSlice::new(&px, w, 8, w as usize * 3, desc(3)).unwrap())
        .unwrap();
    assert!(enc.begin(w, h, desc(3)).is_err());
    // Fewer rows than declared.
    let mut enc = cfg.clone().job().encoder().unwrap();
    enc.begin(w, h, desc(3)).unwrap();
    enc.push_rows(PixelSlice::new(&px, w, 8, w as usize * 3, desc(3)).unwrap())
        .unwrap();
    assert!(enc.finish().is_err());
}

#[test]
fn output_limit_counts_streamed_bytes() {
    let (w, h) = (203u32, 131u32);
    let px = pixels(w, h, 3);
    let cfg = fixed(ChromaSubsampling::Quarter);
    let total = one_shot(&cfg, &px, w, h, 3, &Metadata::none()).len() as u64;
    let limited = |max: u64| {
        let mut enc = cfg
            .clone()
            .job()
            .with_limits(ResourceLimits::none().with_max_output(max))
            .encoder()
            .unwrap();
        enc.begin(w, h, desc(3))?;
        let stride = w as usize * 3;
        let mut y = 0;
        while y < h {
            let n = 16.min(h - y);
            let s = PixelSlice::new(
                &px[y as usize * stride..(y + n) as usize * stride],
                w,
                n,
                stride,
                desc(3),
            )
            .unwrap();
            enc.push_rows(s)?;
            let k = enc.pending_output().len();
            enc.consume_output(k);
            y += n;
        }
        enc.finish().map(|_| ())
    };
    assert!(limited(total).is_ok(), "exactly the file size fits");
    assert!(limited(total - 1).is_err(), "one byte less does not");
}

#[test]
fn testkit_streaming_output_honesty() {
    let img = tk::TestImage::rgb8_gradient(97, 70);
    tk::check_streaming_output_honesty(fixed(ChromaSubsampling::Quarter), D::new(), &img).unwrap();
    tk::check_streaming_output_honesty(fixed(ChromaSubsampling::None), D::new(), &img).unwrap();
    tk::check_streaming_output_honesty(E::new(), D::new(), &img).unwrap();
}

#[test]
fn an_mpf_gain_map_reports_at_finish() {
    // The MPF index records the primary image's length, which is known
    // only at the end, so a fixed-table encode with a gain map cannot stream.
    let (w, h) = (64u32, 48u32);
    let px = pixels(w, h, 3);
    let gain_map = one_shot(
        &fixed(ChromaSubsampling::Quarter),
        &pixels(16, 16, 1),
        16,
        16,
        1,
        &Metadata::none(),
    );
    let mut cfg = fixed(ChromaSubsampling::Quarter);
    *cfg.inner_mut() = cfg.inner().clone().add_gainmap(gain_map.clone());
    let m = Metadata::none();
    let (timing, got, seen) = streamed(&cfg, &px, (w, h, 3), 16, &m);
    assert_eq!(timing, OutputTiming::AT_FINISH);
    assert!(seen.iter().all(|&(_, n)| n == 0), "nothing pending");
    assert_eq!(got, one_shot(&cfg, &px, w, h, 3, &m));
    assert!(
        got.ends_with(&gain_map),
        "the gain map follows the primary image"
    );
}

#[test]
fn strided_crops_push_without_a_trailing_stride() {
    // A crop of a wider image: rows are `stride` apart and the slice ends at
    // the last row's pixels. Every path must take it, streamed or not, and
    // give the bytes of the same pixels packed.
    let (full_w, w, h) = (300u32, 64u32, 48u32);
    let wide = pixels(full_w, h, 3);
    let stride = full_w as usize * 3;
    let packed: Vec<u8> = (0..h as usize)
        .flat_map(|y| wide[y * stride..y * stride + w as usize * 3].to_vec())
        .collect();
    let cfg = fixed(ChromaSubsampling::Quarter);
    let want = one_shot(&cfg, &packed, w, h, 3, &Metadata::none());
    let crop = |y: u32, n: u32| {
        let start = y as usize * stride;
        let end = start + (n as usize - 1) * stride + w as usize * 3;
        PixelSlice::new(&wide[start..end], w, n, stride, desc(3)).unwrap()
    };
    let mut enc = cfg.clone().job().encoder().unwrap();
    enc.begin(w, h, desc(3)).unwrap();
    let mut got = Vec::new();
    for y in (0..h).step_by(16) {
        enc.push_rows(crop(y, 16)).unwrap();
        let n = enc.pending_output().len();
        got.extend_from_slice(enc.pending_output());
        enc.consume_output(n);
    }
    got.extend_from_slice(enc.finish().unwrap().data());
    assert_eq!(got, want, "streamed crop");
    let whole = cfg.job().encoder().unwrap().encode(crop(0, h)).unwrap();
    assert_eq!(whole.data(), &want[..], "one-shot crop");
}

#[test]
fn a_failed_begin_leaves_the_encoder_as_it_was() {
    let (w, h) = (64u32, 48u32);
    let px = pixels(w, h, 3);
    let cfg = fixed(ChromaSubsampling::Quarter);
    let mut enc = cfg.clone().job().encoder().unwrap();
    // Gray+alpha is not an encode layout: refused, nothing declared.
    assert!(enc.begin(w, h, PixelDescriptor::GRAYA8_SRGB).is_err());
    assert!(enc.pending_output().is_empty());
    // The same encoder then declares and streams normally.
    assert!(enc.begin(w, h, desc(3)).unwrap().streams());
    let mut got = enc.pending_output().to_vec();
    let n = got.len();
    enc.consume_output(n);
    for y in (0..h).step_by(16) {
        let s = PixelSlice::new(&px[y as usize * 192..][..16 * 192], w, 16, 192, desc(3)).unwrap();
        enc.push_rows(s).unwrap();
        let n = enc.pending_output().len();
        got.extend_from_slice(enc.pending_output());
        enc.consume_output(n);
    }
    got.extend_from_slice(enc.finish().unwrap().data());
    assert_eq!(got, one_shot(&cfg, &px, w, h, 3, &Metadata::none()));
}

#[test]
fn one_shot_paths_are_refused_after_begin() {
    let px = pixels(32, 32, 3);
    let cfg = fixed(ChromaSubsampling::Quarter);
    let mut enc = cfg.clone().job().encoder().unwrap();
    enc.begin(32, 32, desc(3)).unwrap();
    let s = PixelSlice::new(&px, 32, 32, 96, desc(3)).unwrap();
    assert!(enc.encode(s).is_err());
    let mut enc = cfg.job().encoder().unwrap();
    enc.begin(32, 32, desc(3)).unwrap();
    let e = enc.begin(32, 32, desc(3)).unwrap_err();
    assert!(e.to_string().contains("twice"), "{e}");
}

#[test]
fn prefer_streaming_picks_fixed_tables_unless_chosen_explicitly() {
    let (w, h) = (64u32, 48u32);
    let begin = |cfg: E| cfg.job().encoder().unwrap().begin(w, h, desc(3)).unwrap();
    assert_eq!(begin(E::new()), OutputTiming::AT_FINISH);
    assert_eq!(E::new().prefers_streaming(), Some(false));
    let prefer = E::new().with_prefer_streaming(true);
    assert_eq!(prefer.prefers_streaming(), Some(true));
    assert_eq!(begin(prefer.clone()), OutputTiming::streaming(31));
    // The same bytes as asking for fixed tables and a sequential scan.
    let px = pixels(w, h, 3);
    let explicit = E::new().with_progressive(false).with_huffman(false);
    assert_eq!(
        one_shot(&prefer, &px, w, h, 3, &Metadata::none()),
        one_shot(&explicit, &px, w, h, 3, &Metadata::none())
    );
    // An explicit choice wins.
    assert_eq!(
        begin(E::new().with_progressive(true).with_prefer_streaming(true)),
        OutputTiming::AT_FINISH
    );
    assert_eq!(
        begin(E::new().with_huffman(true).with_prefer_streaming(true)),
        OutputTiming::AT_FINISH
    );
}

#[test]
fn native_mutable_configuration_outranks_effort_and_streaming_preference() {
    use zenjpeg::encoder::{EncoderConfig as NativeConfig, PixelLayout};
    let (w, h) = (67, 93);
    let px = pixels(w, h, 3);
    for progressive in [false, true] {
        let native = NativeConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
            .progressive(progressive)
            .huffman(progressive);
        let mut encoder = native
            .request()
            .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
            .unwrap();
        encoder
            .push(&px, h as usize, w as usize * 3, zencodec::Unstoppable)
            .unwrap();
        let want = encoder.finish().unwrap();
        for prefer in [false, true] {
            let mut adapter = E::new()
                .with_generic_effort(2)
                .with_prefer_streaming(prefer);
            *adapter.inner_mut() = native.clone();
            assert_eq!(one_shot(&adapter, &px, w, h, 3, &Metadata::none()), want);
            let (timing, got, _) = streamed(&adapter, &px, (w, h, 3), 7, &Metadata::none());
            assert_eq!(got, want);
            assert_eq!(timing.streams(), !progressive);
        }
    }
}

/// A push cut short by the job's stop token leaves fewer rows than declared;
/// `finish` reports the cancellation, not that row count.
#[test]
fn finish_after_a_cancelled_push_reports_the_cancellation() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    struct Trip(Arc<AtomicBool>);
    impl enough::Stop for Trip {
        fn check(&self) -> Result<(), enough::StopReason> {
            if self.0.load(Ordering::Relaxed) {
                Err(enough::StopReason::Cancelled)
            } else {
                Ok(())
            }
        }
    }
    let (w, h) = (64u32, 64u32);
    let px = pixels(w, h, 3);
    let stride = w as usize * 3;
    let tripped = Arc::new(AtomicBool::new(false));
    let mut enc = fixed(ChromaSubsampling::Quarter)
        .job()
        .with_stop(zencodec::StopToken::new(Trip(tripped.clone())))
        .encoder()
        .unwrap();
    assert!(enc.begin(w, h, desc(3)).unwrap().streams());
    let strip = |y: u32| {
        PixelSlice::new(
            &px[y as usize * stride..][..16 * stride],
            w,
            16,
            stride,
            desc(3),
        )
        .unwrap()
    };
    enc.push_rows(strip(0)).unwrap();
    tripped.store(true, Ordering::Relaxed);
    let pushed = enc.push_rows(strip(16)).unwrap_err();
    assert_eq!(
        pushed.category(),
        ErrorCategory::Stopped(enough::StopReason::Cancelled)
    );
    let finished = enc.finish().unwrap_err();
    assert_eq!(
        finished.category(),
        ErrorCategory::Stopped(enough::StopReason::Cancelled),
        "{finished}"
    );
}
