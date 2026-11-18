//! Exact baseline entropy controls for opt-in reconstruction metadata.
use enough::Unstoppable;
use zenjpeg::decoder::{Decoder, Strictness};

fn marker(out: &mut Vec<u8>, id: u8, payload: &[u8]) {
    out.extend([255, id]);
    out.extend(u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
    out.extend(payload);
}
fn flush(out: &mut Vec<u8>, bits: &mut Vec<u8>, padding: u8) {
    for chunk in bits.chunks(8) {
        let mut byte = 0;
        for i in 0..8 {
            byte = (byte << 1) | chunk.get(i).copied().unwrap_or(padding);
        }
        out.push(byte);
        if byte == 255 {
            out.push(0);
        }
    }
    bits.clear();
}
fn source(
    components: u8,
    zrl_at: Option<usize>,
    nonzero_after_zrl: bool,
    restart: bool,
    padding: u8,
) -> Vec<u8> {
    let mut out = vec![255, 0xd8];
    let mut quant = vec![0];
    quant.extend([1; 64]);
    marker(&mut out, 0xdb, &quant);
    let mut sof = vec![8, 0, 8, 0, 16, components];
    for c in 1..=components {
        sof.extend([c, 0x11, 0]);
    }
    marker(&mut out, 0xc0, &sof);
    let mut dht = vec![0, 1];
    dht.extend([0; 15]);
    dht.push(0);
    dht.extend([0x10, 1, 1, 1]);
    dht.extend([0; 13]);
    dht.extend([0, 0xf0, 1]);
    marker(&mut out, 0xc4, &dht);
    if restart {
        marker(&mut out, 0xdd, &[0, 1]);
    }
    let mut sos = vec![components];
    for c in 1..=components {
        sos.extend([c, 0]);
    }
    sos.extend([0, 63, 0]);
    marker(&mut out, 0xda, &sos);
    let mut bits = Vec::new();
    for block in 0..usize::from(components) * 2 {
        if restart && block == usize::from(components) {
            flush(&mut out, &mut bits, padding);
            out.extend([255, 0xd0]);
        }
        bits.push(0); // DC category zero.
        if zrl_at == Some(block) {
            bits.extend([1, 0]);
            if nonzero_after_zrl {
                bits.extend([1, 1, 0, 1]);
            }
        }
        bits.push(0); // EOB.
    }
    flush(&mut out, &mut bits, padding);
    out.extend([255, 0xd9]);
    out
}

#[test]
fn baseline_metadata_tracks_only_trailing_zrls_in_scan_order() {
    for components in [1, 3] {
        for restart in [false, true] {
            for padding in [0, 1] {
                for zrl_at in [None, Some(0), Some(usize::from(components) * 2 - 1)] {
                    for nonzero in [false, true] {
                        let input = source(components, zrl_at, nonzero, restart, padding);
                        let ordinary = Decoder::new()
                            .decode_coefficients(&input, Unstoppable)
                            .unwrap();
                        let (coefficients, metadata) = Decoder::new()
                            .decode_coefficients_with_jbrd_metadata(&input, Unstoppable)
                            .unwrap();
                        assert_eq!(coefficients.width, ordinary.width);
                        assert_eq!(coefficients.height, ordinary.height);
                        assert_eq!(coefficients.components.len(), ordinary.components.len());
                        for (a, b) in coefficients.components.iter().zip(&ordinary.components) {
                            assert_eq!(a.coeffs, b.coeffs);
                        }
                        assert_eq!(
                            metadata.scans.len(),
                            1,
                            "baseline scan record, components={components}, restart={restart}"
                        );
                        let scan = &metadata.scans[0];
                        assert_eq!((scan.ss, scan.se, scan.ah, scan.al), (0, 63, 0, 0));
                        assert!(scan.reset_points.is_empty());
                        let expected = if nonzero {
                            Vec::new()
                        } else {
                            zrl_at
                                .map(|index| vec![(u32::try_from(index).unwrap(), 1)])
                                .unwrap_or_default()
                        };
                        assert_eq!(
                            scan.extra_zero_runs, expected,
                            "components={components} restart={restart} padding={padding} zrl={zrl_at:?} nonzero={nonzero}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn native_nonzero_long_code_and_edge_scan_order_matches_metadata() {
    let rows = [
        (
            "slow11-nonzero-before",
            include_str!("testdata/jpeg_scan_v012_slow11-nonzero-before.jpg.hex"),
            Some((1, 2)),
        ),
        (
            "slow11-zrl-before-nonzero",
            include_str!("testdata/jpeg_scan_v012_slow11-zrl-before-nonzero.jpg.hex"),
            None,
        ),
        (
            "nonzero-before-zrl",
            include_str!("testdata/jpeg_scan_v012_nonzero-before-zrl.jpg.hex"),
            Some((0, 1)),
        ),
        (
            "zrl-before-nonzero",
            include_str!("testdata/jpeg_scan_v012_zrl-before-nonzero.jpg.hex"),
            None,
        ),
        (
            "three-zrl",
            include_str!("testdata/jpeg_scan_v012_three-zrl.jpg.hex"),
            Some((0, 3)),
        ),
        (
            "zrl-finishes-block",
            include_str!("testdata/jpeg_scan_v012_zrl-finishes-block.jpg.hex"),
            Some((0, 3)),
        ),
        (
            "long-codes-nonzero-before",
            include_str!("testdata/jpeg_scan_v012_long-codes-nonzero-before.jpg.hex"),
            Some((1, 2)),
        ),
        (
            "long-codes-zrl-before-nonzero",
            include_str!("testdata/jpeg_scan_v012_long-codes-zrl-before-nonzero.jpg.hex"),
            None,
        ),
        (
            "gray-absolute22",
            include_str!("testdata/jpeg_scan_v012_gray-absolute22.jpg.hex"),
            Some((1, 1)),
        ),
        (
            "interleaved-last",
            include_str!("testdata/jpeg_scan_v012_interleaved-last.jpg.hex"),
            Some((5, 2)),
        ),
        (
            "interleaved-first-long",
            include_str!("testdata/jpeg_scan_v012_interleaved-first-long.jpg.hex"),
            Some((0, 1)),
        ),
        (
            "edge-padding-luma",
            include_str!("testdata/jpeg_scan_v012_edge-padding-luma.jpg.hex"),
            Some((7, 1)),
        ),
        (
            "edge-visible-chroma-long",
            include_str!("testdata/jpeg_scan_v012_edge-visible-chroma-long.jpg.hex"),
            Some((11, 2)),
        ),
    ];
    for (label, text, expected) in rows {
        let compact: String = text.chars().filter(|c| !c.is_whitespace()).collect();
        let input: Vec<u8> = compact
            .as_bytes()
            .chunks_exact(2)
            .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
            .collect();
        for strictness in [
            Strictness::Strict,
            Strictness::Balanced,
            Strictness::Lenient,
            Strictness::Permissive,
        ] {
            let original = Decoder::new()
                .strictness(strictness)
                .decode_coefficients(&input, Unstoppable)
                .unwrap();
            let (decoded, metadata) = Decoder::new()
                .strictness(strictness)
                .decode_coefficients_with_jbrd_metadata(&input, Unstoppable)
                .unwrap();
            assert_eq!(decoded.components.len(), original.components.len());
            for (a, b) in decoded.components.iter().zip(&original.components) {
                assert_eq!(a.coeffs, b.coeffs, "{label} coefficient identity");
            }
            assert_eq!(metadata.scans.len(), 1, "{label}");
            assert!(metadata.scans[0].reset_points.is_empty());
            assert_eq!(
                metadata.scans[0].extra_zero_runs,
                expected.into_iter().collect::<Vec<_>>(),
                "{label} global scan-block index"
            );
        }
    }
}

#[test]
fn truncated_baseline_metadata_retains_ordinary_error_policy() {
    for text in [
        include_str!("testdata/jpeg_scan_v012_slow11-nonzero-before.jpg.hex"),
        include_str!("testdata/jpeg_scan_v012_edge-padding-luma.jpg.hex"),
        include_str!("testdata/jpeg_scan_v012_edge-visible-chroma-long.jpg.hex"),
    ] {
        let compact: String = text.chars().filter(|c| !c.is_whitespace()).collect();
        let input: Vec<u8> = compact
            .as_bytes()
            .chunks_exact(2)
            .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
            .collect();
        let sos = input.windows(2).position(|v| v == [255, 0xda]).unwrap();
        let start = sos + 2 + usize::from(u16::from_be_bytes([input[sos + 2], input[sos + 3]]));
        let mut strict_errors = 0;
        for end in start..input.len() - 2 {
            let mut truncated = input[..end].to_vec();
            truncated.extend([255, 0xd9]);
            for strictness in [
                Strictness::Strict,
                Strictness::Balanced,
                Strictness::Lenient,
                Strictness::Permissive,
            ] {
                let original = Decoder::new()
                    .strictness(strictness)
                    .decode_coefficients(&truncated, Unstoppable);
                let tracked = Decoder::new()
                    .strictness(strictness)
                    .decode_coefficients_with_jbrd_metadata(&truncated, Unstoppable);
                assert_eq!(
                    original.is_ok(),
                    tracked.is_ok(),
                    "{strictness:?}: truncation {end}"
                );
                match (original, tracked) {
                    (Ok(original), Ok((tracked, _))) => {
                        assert_eq!(original.components.len(), tracked.components.len());
                        for (a, b) in original.components.iter().zip(&tracked.components) {
                            assert_eq!(a.coeffs, b.coeffs, "{strictness:?}: truncation {end}");
                        }
                    }
                    (Err(_), Err(_)) if strictness == Strictness::Strict => strict_errors += 1,
                    (Err(_), Err(_)) => {}
                    _ => unreachable!(),
                }
            }
        }
        assert!(strict_errors > 0);
    }
}
