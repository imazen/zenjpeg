//! # zenjpeg - High-Quality JPEG Encoder
//!
//! zenjpeg combines the best techniques from mozjpeg and jpegli to achieve
//! Pareto-optimal compression across both low and high quality settings.
//!
//! ## Key Features
//!
//! - **Adaptive quality selection**: Automatically chooses the best encoding
//!   strategy based on target quality
//! - **Trellis quantization** (from mozjpeg): Rate-distortion optimized
//!   coefficient selection, especially effective at low quality
//! - **Adaptive quantization** (from jpegli): Content-aware bit allocation
//!   based on perceptual importance
//! - **Perceptual optimization**: Uses Butteraugli/SSIMULACRA2 for quality
//!   assessment and tuning
//!
//! ## Usage
//!
//! ```rust,ignore
//! use zenjpeg::{Encoder, Quality};
//!
//! let encoder = Encoder::new()
//!     .quality(Quality::Perceptual(85.0))
//!     .optimize_for_web(true);
//!
//! let jpeg_data = encoder.encode_rgb(&pixels, width, height)?;
//! ```
//!
//! ## Quality Modes
//!
//! - `Quality::Low(q)`: Uses mozjpeg-style trellis for best low-bitrate results
//! - `Quality::High(q)`: Uses jpegli-style adaptive quantization for high quality
//! - `Quality::Perceptual(target)`: Automatically selects strategy to hit target
//!   perceptual quality (SSIMULACRA2 score)

// Core modules
pub mod analysis;
mod consts;
mod consts_moz;
mod error;
mod types;

// Encoding pipeline
mod color;
mod dct;
mod encode;
mod entropy;
mod huffman;
mod quant;

// Advanced features
pub mod adaptive_quant;
mod progressive;
mod trellis;

// Strategy selection
mod strategy;

// Forked jpegli encoder for experimental improvements
pub mod jpegli;

// Public API
pub use encode::Encoder;
pub use error::Error;
pub use progressive::ScanInfo;
pub use types::{ColorSpace, EncodingStrategy, OptimizeFor, PixelFormat, Quality, ScanScript, Subsampling};

/// Result type for zenjpeg operations
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_encode() {
        // Placeholder test - will be implemented with actual encoder
    }
}

#[cfg(test)]
mod debug_tests {
    use super::*;
    
    #[test]
    fn debug_quality_encoding() {
        use quant::QuantTableSet;
        
        // Check quant table values at different quality levels
        for q in [30, 50, 70, 95] {
            let tables = QuantTableSet::standard(q as u8);
            println!("Q{}: luma[0]={}, luma[1]={}, luma[63]={}", 
                q, tables.luma.values[0], tables.luma.values[1], tables.luma.values[63]);
        }
    }
}
