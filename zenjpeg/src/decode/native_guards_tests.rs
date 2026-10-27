//! Internal admission boundaries that callers cannot reach with a valid reader.
use super::*;
use alloc::borrow::Cow;
const GRAY8: &[u8] = include_bytes!("../../fuzz/seeds/native/00-p8-c1-sof30-mode0-s1.jpg");
const GRAY12: &[u8] = include_bytes!("../../fuzz/seeds/native/01-p12-c1-sof30-mode0-s1.jpg");
const TWO12: &[u8] = include_bytes!("../../fuzz/seeds/native/06-p12-c2-sof30-mode0-s1.jpg");
const THREE12: &[u8] = include_bytes!("../../fuzz/seeds/native/11-p12-c3-sof30-mode0-s1.jpg");
const LOSSLESS: &[u8] = include_bytes!("../../fuzz/seeds/native/32-p12-c3-sof31-mode0-s1.jpg");
fn parser(data: &[u8]) -> JpegParser<'_> {
    let mut parser = JpegParser::new(data, 1_000_000, None).unwrap();
    parser.read_header().unwrap();
    parser
}
#[test]
fn native_internal_admission_boundaries() {
    let mut p = parser(GRAY12);
    assert!(p.decode(&Unstoppable).is_err()); // Ordinary coefficient mode must opt in.
    p.color_model = color_model::ColorModel::Xyb;
    assert!(Decoder::new().validate_native_options(&p).is_err());
    let mut p = parser(LOSSLESS);
    p.native_precision = true;
    assert!(p.decode(&Unstoppable).is_err()); // SOF3 has no DCT coefficients.
    let result = Decoder::new().decode(GRAY8, Unstoppable).unwrap();
    assert!(ScanlineReader::from_native_result(Cow::Borrowed(GRAY8), result).is_err());
}
#[test]
fn native_row_sources_refuse_reads_after_height() {
    let mut p = parser(TWO12);
    let mut rows = p
        .component_rows(ChromaUpsampling::Triangle, &Unstoppable)
        .unwrap();
    for _ in 0..p.height {
        rows.next_row(TWO12, &Unstoppable).unwrap();
    }
    assert!(rows.next_row(TWO12, &Unstoppable).is_err());
    let mut p = parser(LOSSLESS);
    let mut rows = p.predictive_reader(&Unstoppable).unwrap();
    for _ in 0..p.height {
        rows.next_row(LOSSLESS, &Unstoppable).unwrap();
    }
    assert!(rows.next_row(LOSSLESS, &Unstoppable).is_err());
}
#[test]
fn component_source_rechecks_scan_membership() {
    let mut data = THREE12.to_vec();
    let sos = crate::container::marker::iter(&data)
        .find(|s| matches!(s.kind, crate::container::marker::MarkerKind::Sos))
        .unwrap()
        .offset;
    data[sos + 3] = 10;
    data[sos + 4] = 2;
    data.drain(sos + 9..sos + 11);
    let mut p = parser(&data);
    // A stale caller classification must not make an incomplete scan interleaved.
    p.interleaved = true;
    assert!(
        p.component_rows(ChromaUpsampling::Triangle, &Unstoppable)
            .is_err()
    );
}
#[test]
fn gray_coefficients_preserve_eight_bit_precision() {
    let coefficients = Decoder::new()
        .decode_coefficients(GRAY8, Unstoppable)
        .unwrap();
    let mut reader = ScanlineReader::from_coefficients(
        Cow::Borrowed(GRAY8),
        coefficients,
        ChromaUpsampling::Triangle,
        IdctMethod::Libjpeg,
        OutputTarget::Native16,
        DeblockMode::Off,
        zencodec::AllocPreference::CodecDefault,
        color_model::ColorModel::Gray,
        8,
    )
    .unwrap();
    let mut row = [0u16; 9];
    assert_eq!(
        reader
            .read_rows_gray16(imgref::ImgRefMut::new(&mut row, 9, 1))
            .unwrap(),
        1
    );
    assert!(row.iter().all(|v| *v <= 255));
}

#[test]
fn predictive_source_rejects_unresolved_height() {
    let mut p = parser(LOSSLESS);
    p.height = 0;
    assert!(p.predictive_reader(&Unstoppable).is_err());
}

#[test]
fn predictive_sampling_and_late_markers() {
    for vertical in [false, true] {
        let mut p = parser(LOSSLESS);
        if vertical {
            p.components[0].v_samp_factor = 2;
        } else {
            p.components[0].h_samp_factor = 2;
        }
        assert!(p.predictive_reader(&Unstoppable).is_err());
    }
    // A legal empty ancillary segment after the scan still has to be walked.
    for marker in [0xe1, 0xfe, 0xdb] {
        let mut data = LOSSLESS.to_vec();
        let end = data.len() - 2;
        data.splice(end..end, [0xff, marker, 0, 2]);
        let mut p = parser(&data);
        assert!(p.predictive_reader(&Unstoppable).is_ok());
        let mut p = parser(&data);
        p.collect_native_metadata(&Unstoppable).unwrap();
    }
}

#[test]
fn sequential_scan_length_is_part_of_admission() {
    let mut data = THREE12.to_vec();
    let sos = crate::container::marker::iter(&data)
        .find(|s| matches!(s.kind, crate::container::marker::MarkerKind::Sos))
        .unwrap()
        .offset;
    data[sos + 3] += 1;
    let mut p = parser(&data);
    assert!(
        p.component_rows(ChromaUpsampling::Triangle, &Unstoppable)
            .is_err()
    );
}

#[test]
fn predictive_requires_every_component_and_one_frame() {
    let mut p = parser(LOSSLESS);
    p.num_components += 1;
    assert!(p.predictive_reader(&Unstoppable).is_err());
    let mut data = LOSSLESS.to_vec();
    let frame = crate::container::marker::iter(&data)
        .find(|s| matches!(s.kind, crate::container::marker::MarkerKind::Sof(_)))
        .unwrap();
    let duplicate = data[frame.offset..frame.offset + frame.length].to_vec();
    let end = data.len() - 2;
    data.splice(end..end, duplicate);
    assert!(parser(&data).predictive_reader(&Unstoppable).is_err());
}

#[test]
fn native_collected_parts_normalize_without_truncation() {
    let config = Decoder::new().output_target(OutputTarget::Native16);
    let decoded = config.decode(GRAY12, Unstoppable).unwrap();
    let expected: Vec<_> = decoded
        .pixels_u16()
        .unwrap()
        .iter()
        .map(|&v| v as f32 / 4095.0)
        .collect();
    let (bytes, floats, _, _, _, _) = decoded.into_parts();
    assert!(bytes.is_none());
    assert_eq!(floats.unwrap(), expected);
}

#[test]
fn native_output_size_checks_overflow_before_allocation() {
    let mut p = parser(GRAY12);
    p.width = u32::MAX;
    p.height = u32::MAX;
    assert!(
        Decoder::new()
            .decode_native(GRAY12, &mut p, &Unstoppable, None, false)
            .is_err()
    );
}
