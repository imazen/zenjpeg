//! Populate test datasets serially before parallel or cross-platform tests.
//! Run with CODEC_CORPUS_CACHE set to the same absolute path used by tests.
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
    ] {
        let path = corpus.get(dataset)?;
        println!("{dataset}: {}", path.display());
    }
    Ok(())
}
