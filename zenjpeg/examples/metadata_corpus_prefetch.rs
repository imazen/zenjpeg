//! Populate test datasets serially before parallel or cross-platform tests.
//! Run with CODEC_CORPUS_CACHE set to the same absolute path used by tests.
#[path = "../tests/support/compact_icc.rs"]
#[allow(dead_code)]
mod compact_icc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let corpus = codec_corpus::Corpus::new()?;
    for dataset in [
        "jpeg-conformance",
        "ultrahdr-conformance",
        "imageflow/test_inputs",
        "mozjpeg",
        "zune/test-images/jpeg",
        "zune/fuzz-corpus/jpeg",
        "image-rs/tests/images/jpg",
        "image-rs/tests/images/jpg/progressive",
    ] {
        let path = corpus.get(dataset)?;
        println!("{dataset}: {}", path.display());
    }
    println!("compact ICC profiles: {}", compact_icc::root().display());
    Ok(())
}
