//! Dump estimate_all for a given (encoder, subsampling, source, target).
//! Usage: dump_estimates <source_zensim> <target>
use zenjpeg::recompress::expert::{CalibrationLookup, EncoderClass};
use zenjpeg::types::Subsampling;

fn main() {
    let src: f32 = std::env::args().nth(1).unwrap().parse().unwrap();
    let tgt: f32 = std::env::args().nth(2).unwrap().parse().unwrap();
    let est =
        CalibrationLookup::SEED.estimate_all(EncoderClass::Mozjpeg, Subsampling::S420, src, tgt);
    println!("Mozjpeg S420 source={src} target={tgt}");
    for c in &est {
        println!(
            "  {:?}: projected={:.2} ratio={:.4} dial={:?}",
            c.kind,
            c.estimate.projected_zensim_a,
            c.estimate.projected_size_ratio,
            c.estimate.dial_zensim_a
        );
    }
}
