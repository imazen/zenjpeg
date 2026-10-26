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
