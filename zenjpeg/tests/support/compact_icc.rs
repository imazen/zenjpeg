// CC0 profiles from Compact-ICC-Profiles, pinned by immutable source revision.
const REVISION: &str = "bdd84663061bc4ae95ca70decff54f581e27f702";

pub fn root() -> std::path::PathBuf {
    codec_corpus::Corpus::new()
        .unwrap()
        .with_max_age(std::time::Duration::MAX)
        .zip_url(
            &format!("compact-icc-{REVISION}"),
            &format!("https://github.com/saucecontrol/Compact-ICC-Profiles/archive/{REVISION}.zip"),
        )
        .expect("pinned ICC corpus must be available")
}

pub fn profile(name: &str) -> Vec<u8> {
    std::fs::read(root().join("profiles").join(name)).expect("pinned ICC fixture must exist")
}
