//! APP10 `ZENJ` encoder parameter record (zenjpeg#189): emitted by default,
//! read back by `detect::probe`, verified against the file's DQT tables,
//! chained across generations, and absent when disabled.

use enough::Unstoppable;
use zenjpeg::detect::record::{
    ColorFront, DialScale, EncodeFlags, SubsamplingCode, TableSource, dqt_crc32, read_record,
};
use zenjpeg::detect::{EncodeRecord, probe};
use zenjpeg::encode::{
    ChromaSubsampling, EncoderConfig, OptimizationPreset, PixelLayout, Quality, QuantTableConfig,
};

fn gradient_rgb(w: u32, h: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|y| {
            (0..w).flat_map(move |x| {
                [
                    ((x * 255) / w) as u8,
                    ((y * 255) / h) as u8,
                    (((x ^ y) * 7) & 255) as u8,
                ]
            })
        })
        .collect()
}

fn encode(config: &EncoderConfig, w: u32, h: u32) -> Vec<u8> {
    let rgb = gradient_rgb(w, h);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("builder");
    enc.push_packed(&rgb, Unstoppable).expect("push");
    enc.finish().expect("finish")
}

fn count_app10(jpeg: &[u8]) -> usize {
    let mut n = 0;
    let mut pos = 2;
    while pos + 4 <= jpeg.len() && jpeg[pos] == 0xFF && jpeg[pos + 1] != 0xDA {
        if jpeg[pos + 1] == 0xEA {
            n += 1;
        }
        let len = ((jpeg[pos + 2] as usize) << 8) | jpeg[pos + 3] as usize;
        pos += 2 + len;
    }
    n
}

#[test]
fn default_encode_carries_a_verified_record() {
    let cfg = EncoderConfig::ycbcr(Quality::ApproxMozjpeg(35), ChromaSubsampling::Quarter)
        .optimization(OptimizationPreset::HybridProgressive)
        .pre_blur(0.4);
    let jpeg = encode(&cfg, 64, 48);
    assert_eq!(count_app10(&jpeg), 1);

    let p = probe(&jpeg).expect("probe");
    let rec = p.record.as_ref().expect("record present");
    assert!(rec.verified, "newest entry must match the file's DQT");
    assert_eq!(rec.generations(), 1);
    let e = rec.current().expect("current entry");
    assert_eq!(e.quality_scale, DialScale::Mozjpeg);
    assert_eq!(e.quality, 35.0);
    assert_eq!(e.table_source, TableSource::Jpegli);
    assert_eq!(e.color_front, ColorFront::YCbCr);
    assert_eq!(e.subsampling, SubsamplingCode::Quarter);
    assert!(e.flags.has(EncodeFlags::AQ));
    assert!(e.flags.has(EncodeFlags::TRELLIS_AC));
    assert!(e.flags.has(EncodeFlags::DERINGING));
    assert!(e.flags.has(EncodeFlags::PROGRESSIVE));
    assert_eq!(e.pre_blur, 0.375, "sigma is stored at 1/16 resolution");
    assert_eq!(Some(e.dqt_crc32), dqt_crc32(&jpeg));
    assert_ne!(e.version, (0, 0, 0));
}

#[test]
fn robidoux_tables_are_named_where_the_fingerprint_says_unknown() {
    let cfg = EncoderConfig::ycbcr(55.0, ChromaSubsampling::None)
        .optimization(OptimizationPreset::MozjpegBaseline);
    let jpeg = encode(&cfg, 64, 64);
    let p = probe(&jpeg).expect("probe");
    let e = p
        .record
        .as_ref()
        .and_then(EncodeRecord::current)
        .expect("entry");
    assert_eq!(e.table_source, TableSource::MozjpegRobidoux);
    assert!(!e.flags.has(EncodeFlags::AQ));
    assert!(!e.flags.has(EncodeFlags::PROGRESSIVE));
    assert_eq!(e.quality_scale, DialScale::Jpegli);
    assert_eq!(e.quality, 55.0);
    // `p.encoder` / `p.quality` stay the fingerprint estimate (today `Unknown`
    // for these tables); the record is the ground truth beside it.
}

#[test]
fn disabled_means_no_marker_and_no_record() {
    let cfg = EncoderConfig::ycbcr(80.0, ChromaSubsampling::Quarter).encode_record(false);
    let jpeg = encode(&cfg, 32, 32);
    assert_eq!(count_app10(&jpeg), 0);
    assert!(read_record(&jpeg).is_none());
    assert!(probe(&jpeg).expect("probe").record.is_none());
}

#[test]
fn history_chains_and_a_rewritten_table_unverifies_it() {
    let gen1 = encode(
        &EncoderConfig::ycbcr(60.0, ChromaSubsampling::Quarter),
        48,
        48,
    );
    let prior = read_record(&gen1).expect("gen1 record").entries;
    let gen2 = encode(
        &EncoderConfig::ycbcr(90.0, ChromaSubsampling::None).record_history(prior.clone()),
        48,
        48,
    );
    let rec = read_record(&gen2).expect("gen2 record");
    assert!(rec.verified);
    assert_eq!(rec.generations(), 2);
    assert_eq!(
        rec.entries[0], prior[0],
        "the prior generation rides along unchanged"
    );
    assert_eq!(rec.entries[1].quality, 90.0);
    assert_ne!(rec.entries[0].dqt_crc32, rec.entries[1].dqt_crc32);

    // Tamper with one DQT value: the record no longer describes these tables.
    let mut tampered = gen2.clone();
    let dqt = tampered
        .windows(2)
        .position(|w| w == [0xFF, 0xDB])
        .expect("DQT");
    tampered[dqt + 5] ^= 0x01;
    let rec = read_record(&tampered).expect("record still present");
    assert!(!rec.verified);
    assert!(rec.current().is_none());
    assert_eq!(rec.generations(), 2, "prior context is kept");
}

#[test]
fn xyb_and_grayscale_describe_their_front_end() {
    let xyb = encode(&EncoderConfig::xyb(70.0, Default::default()), 32, 32);
    let e = read_record(&xyb).expect("xyb record");
    assert!(e.verified);
    assert_eq!(e.entries[0].color_front, ColorFront::Xyb);
    let gray = encode(&EncoderConfig::grayscale(70.0), 32, 32);
    let e = read_record(&gray).expect("gray record");
    assert_eq!(e.entries[0].color_front, ColorFront::Grayscale);
    assert_eq!(e.entries[0].subsampling, SubsamplingCode::NotApplicable);
    assert!(matches!(
        EncoderConfig::ycbcr(70.0, ChromaSubsampling::Quarter).get_quant_table_config(),
        QuantTableConfig::Jpegli { .. }
    ));
}

#[cfg(feature = "recompress")]
#[test]
fn recompress_appends_a_generation_to_the_chain() {
    use zenjpeg::recompress::{RecompressOptions, RecompressResult, recompress};
    // A detailed source so the router has something to requantize.
    let (w, h) = (256, 256);
    let rgb: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            let (x, y) = (i % w, i / w);
            [
                ((x * 7 + y * 3) % 251) as u8,
                ((x ^ y) % 241) as u8,
                ((x * y) % 239) as u8,
            ]
        })
        .collect();
    let mut enc = EncoderConfig::ycbcr(92.0, ChromaSubsampling::Quarter)
        .encode_from_bytes(w as u32, h as u32, PixelLayout::Rgb8Srgb)
        .expect("builder");
    enc.push_packed(&rgb, Unstoppable).expect("push");
    let src = enc.finish().expect("finish");
    let src_rec = read_record(&src).expect("source record");
    assert!(src_rec.verified);

    match recompress(&src, &RecompressOptions::new(60.0)).expect("recompress") {
        RecompressResult::Recompressed {
            bytes, strategy, ..
        } => {
            let rec = read_record(&bytes).unwrap_or_else(|| panic!("record after {strategy:?}"));
            assert!(
                rec.verified,
                "{strategy:?}: newest entry must match the output tables"
            );
            assert_eq!(rec.generations(), 2, "{strategy:?}");
            assert_eq!(rec.entries[0], src_rec.entries[0]);
            assert_eq!(
                rec.entries[1].encoder,
                zenjpeg::detect::record::EncoderId::Zenjpeg
            );
            assert_ne!(rec.entries[1].dqt_crc32, rec.entries[0].dqt_crc32);
        }
        other => panic!("expected a recompressed result, got {other:?}"),
    }
}
