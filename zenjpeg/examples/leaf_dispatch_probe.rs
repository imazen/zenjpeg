//! Linked size/parity consumer. Writes bounded encoded/raster witnesses to argv[1].
#![forbid(unsafe_code)]
#[path = "support/leaf_dispatch.rs"]
mod workload;

fn main() {
    #[cfg(not(target_os = "unknown"))]
    let out = {
        let path = std::path::PathBuf::from(
            std::env::args_os()
                .nth(1)
                .expect("output directory required"),
        );
        std::fs::create_dir_all(&path).unwrap();
        path
    };
    for (name, bytes) in workload::decode_cases() {
        let raster = workload::decode(bytes);
        #[cfg(not(target_os = "unknown"))]
        std::fs::write(out.join(format!("decode-{name}.rgb")), &raster).unwrap();
        std::hint::black_box((name, raster));
    }
    for case in workload::encode_cases() {
        let encoded = case.encode();
        let raster = workload::decode(&encoded);
        let mut oracle = jpeg_decoder::Decoder::new(std::io::Cursor::new(&encoded));
        let independent = oracle.decode().unwrap();
        #[cfg(not(target_os = "unknown"))]
        {
            std::fs::write(out.join(format!("encode-{}.jpg", case.name)), &encoded).unwrap();
            std::fs::write(out.join(format!("encode-{}.rgb", case.name)), &raster).unwrap();
            std::fs::write(out.join(format!("oracle-{}.rgb", case.name)), &independent).unwrap();
        }
        std::hint::black_box((encoded, raster, independent));
    }
}
