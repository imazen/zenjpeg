//! UltraHDR support for jpegli.
//!
//! This module provides integration with [`ultrahdr_core`] for HDR gain map
//! encoding and decoding. UltraHDR images contain an SDR base JPEG plus a
//! secondary gain map JPEG that allows reconstruction of HDR content on
//! capable displays while remaining compatible with SDR viewers.
//!
//! # Overview
//!
//! - **Encoding**: Tonemap HDR → compute gain map → encode base+gainmap → assemble
//! - **Decoding**: Decode base → extract gain map → apply boost → reconstruct HDR
//!
//! # Example: Encode UltraHDR from HDR source
//!
//! ```rust,ignore
//! use jpegli::ultrahdr::{
//!     encode_ultrahdr, GainMapConfig, ToneMapConfig, UhdrRawImage, Unstoppable,
//! };
//! use jpegli::encoder::{EncoderConfig, ChromaSubsampling};
//!
//! // Load HDR image (e.g., from OpenEXR, HDR photo, etc.)
//! let hdr = UhdrRawImage::from_data(
//!     width, height, PixelFormat::Rgba32F, gamut, transfer, data,
//! )?;
//!
//! // Encode with default settings
//! let jpeg = encode_ultrahdr(
//!     &hdr,
//!     &GainMapConfig::default(),
//!     &ToneMapConfig::default(),
//!     &EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter),
//!     75.0, // gain map quality
//!     Unstoppable,
//! )?;
//! ```
//!
//! # Example: Decode and reconstruct HDR
//!
//! ```rust,ignore
//! use jpegli::decoder::Decoder;
//! use jpegli::ultrahdr::{reconstruct_hdr, HdrOutputFormat, UltraHdrExtras, Unstoppable};
//!
//! let decoded = Decoder::new().decode(&jpeg_data)?;
//!
//! // Check if it's UltraHDR
//! if let Some(extras) = decoded.extras() {
//!     if extras.is_ultrahdr() {
//!         let hdr = reconstruct_hdr(
//!             decoded.pixels(),
//!             decoded.width(),
//!             decoded.height(),
//!             extras,
//!             4.0, // display boost (1.0=SDR, 4.0=typical HDR)
//!             HdrOutputFormat::LinearFloat,
//!             Unstoppable,
//!         )?;
//!     }
//! }
//! ```

mod decode;
mod encode;

// Re-export the main workflow functions
pub use decode::{reconstruct_hdr, reencode_ultrahdr, tonemapper_from_ultrahdr, UltraHdrExtras};
pub use encode::{encode_ultrahdr, encode_ultrahdr_with_tonemapper};

// Re-export core types from ultrahdr-core (aliased to avoid collisions)
pub use ultrahdr_core::{
    // Tonemapping
    color::tonemap::{AdaptiveTonemapper, FitConfig, FitMode, FitStats, ToneMapConfig},
    // Gainmap functions
    gainmap::{apply_gainmap, compute_gainmap, GainMapConfig, HdrOutputFormat},
    // Metadata
    metadata::xmp::{generate_xmp, parse_xmp},
    // Color types (aliased to avoid collision with jpegli types)
    ColorGamut as UhdrColorGamut,
    ColorTransfer as UhdrColorTransfer,
    // Gain map types
    GainMap,
    GainMapMetadata,
    PixelFormat as UhdrPixelFormat,
    RawImage as UhdrRawImage,
    // Cancellation
    Unstoppable,
};

// Re-export the Stop trait from enough (same one used by jpegli)
pub use enough::Stop;
