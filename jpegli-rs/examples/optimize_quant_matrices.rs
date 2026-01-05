//! Simulated annealing optimizer for jpegli quantization matrices.
//!
//! **NOTE**: This example is currently broken - it uses `Ssim2Reference` which
//! doesn't exist in fast-ssim2. Needs refactoring to use `compute_frame_ssimulacra2`.
//!
//! Optimizes base quantization matrices to maximize SSIMULACRA2 quality
//! at a given file size (or minimize file size at a given quality).
//!
//! The search space includes:
//! - Base YCbCr matrix (192 values: Y[64], Cb[64], Cr[64])
//! - Global scale factor
//! - Frequency exponents (64 values)
//!
//! Usage:
//!   cargo run --release --example optimize_quant_matrices -- <corpus_dir> [options]
//!
//! Options:
//!   --quality <N>        Target quality level (default: 85)
//!   --iterations <N>     SA iterations (default: 10000)
//!   --output <file>      Output file for best matrices (JSON)
//!   --resume <file>      Resume from checkpoint
//!   --seed <N>           Random seed for reproducibility

fn main() {
    eprintln!("ERROR: This example is currently broken.");
    eprintln!();
    eprintln!("It uses `fast_ssim2::Ssim2Reference` which doesn't exist in fast-ssim2.");
    eprintln!("The example needs refactoring to use `compute_frame_ssimulacra2` directly.");
    eprintln!();
    eprintln!("TODO: Cache the reference image preprocessing manually, or accept");
    eprintln!("      the performance hit of recomputing per evaluation.");
    std::process::exit(1);
}
