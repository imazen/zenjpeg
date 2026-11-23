//! Bounded fixed-fixture encode/decode bench; no corpus discovery or skips.
#![forbid(unsafe_code)]
#[path = "../examples/support/leaf_dispatch.rs"]
mod workload;
use zenbench::prelude::*;

fn bench(suite: &mut Suite) {
    for (name, bytes) in workload::decode_cases() {
        let expected = workload::decode(bytes);
        assert!(!expected.is_empty());
        suite.group(format!("leaf/decode/{name}"), |g| {
            g.throughput(Throughput::Bytes(bytes.len() as u64));
            g.bench("zenjpeg", move |b| {
                b.iter(|| black_box(workload::decode(black_box(bytes))))
            });
        });
    }
    for case in workload::encode_cases() {
        let encoded = case.encode();
        let mut oracle = jpeg_decoder::Decoder::new(std::io::Cursor::new(&encoded));
        assert!(!oracle.decode().unwrap().is_empty());
        assert!(!workload::decode(&encoded).is_empty());
        eprintln!(
            "{}: {}x{}, {} bytes, independently decoded",
            case.name,
            case.width,
            case.height,
            encoded.len()
        );
        suite.group(format!("leaf/encode/{}", case.name), |g| {
            g.throughput(Throughput::Elements(
                u64::from(case.width) * u64::from(case.height),
            ));
            g.bench("zenjpeg", move |b| b.iter(|| black_box(case.encode())));
        });
    }
}
zenbench::main!(bench);
