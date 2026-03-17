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
//! use zenjpeg::ultrahdr::{
//!     encode_ultrahdr, GainMapConfig, ToneMapConfig, UhdrRawImage, Unstoppable,
//! };
//! use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling};
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
//! # Example: Decode and reconstruct HDR (streaming)
//!
//! ```rust,ignore
//! use zenjpeg::ultrahdr::{create_hdr_reconstructor, UltraHdrExtras};
//!
//! // Use UltraHdrReader for streaming decode
//! let mut reader = UltraHdrReader::new(&jpeg_data, config)?;
//!
//! // Or with separate decoder + reconstructor:
//! // Note: streaming APIs now work with linear f32 RGB. The caller must convert
//! // sRGB u8 to linear f32 before feeding rows to the reconstructor.
//! let reconstructor = create_hdr_reconstructor(
//!     width, height, &extras, 4.0,
//! )?;
//!
//! // Process rows (input: linear f32 RGB, output: linear f32 RGBA)
//! let hdr_batch = reconstructor.process_rows(&sdr_linear_f32, batch_height as u32)?;
//! ```

mod decode;
mod encode;

// Re-export the main workflow functions
pub use decode::{UltraHdrExtras, create_hdr_reconstructor, tonemapper_from_ultrahdr};
pub use encode::{
    create_gainmap_computer, encode_ultrahdr, encode_ultrahdr_with_tonemapper, encode_with_gainmap,
};

// Re-export core types from ultrahdr-core (aliased to avoid collisions)
pub use ultrahdr_core::{
    // Color types (aliased to avoid collision with jpegli types)
    ColorGamut as UhdrColorGamut,
    ColorTransfer as UhdrColorTransfer,
    // Gain map types
    GainMap,
    GainMapMetadata,
    PixelFormat as UhdrPixelFormat,
    RawImage as UhdrRawImage,
    // Fraction types (used by ISO 21496-1 binary format)
    UnsignedFraction,
    // Cancellation
    Unstoppable,
    // Tonemapping
    color::tonemap::{AdaptiveTonemapper, FitConfig, FitMode, FitStats, ToneMapConfig},
    // Gainmap functions (full-image)
    gainmap::{GainMapConfig, HdrOutputFormat, apply_gainmap, compute_gainmap},
    // Streaming APIs (low-memory processing, linear f32 I/O)
    // - RowDecoder/RowEncoder: full gainmap in memory, row-based SDR/HDR
    // - StreamDecoder/StreamEncoder: dual streaming for parallel decode
    gainmap::{RowDecoder, RowEncoder, StreamDecoder, StreamEncoder},
    // Metadata
    metadata::iso21496::{parse_iso21496, serialize_iso21496},
    metadata::xmp::{generate_xmp, parse_xmp},
};

// Re-export the Stop trait from enough (same one used by jpegli)
pub use enough::Stop;

// Re-export streaming UltraHDR reader types from decode module
pub use crate::decode::{GainMapMemory, UltraHdrMode, UltraHdrReader, UltraHdrReaderConfig};
