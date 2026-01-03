//! Type-safe pixel format handling.
//!
//! This module provides the [`Pixel`] trait for compile-time type-safe pixel formats,
//! along with convenient type aliases for common formats.
//!
//! # Supported Formats
//!
//! | Type | Channels | Alpha | Bytes | Description |
//! |------|----------|-------|-------|-------------|
//! | [`RGB8`] | 3 | No | 3 | 8-bit RGB |
//! | [`RGBA8`] | 4 | Yes | 4 | 8-bit RGBA (alpha ignored on encode) |
//! | [`Gray8`] | 1 | No | 1 | 8-bit grayscale |
//! | [`RGB16`] | 3 | No | 6 | 16-bit RGB (high precision input) |
//! | [`RGBA16`] | 4 | Yes | 8 | 16-bit RGBA (alpha ignored on encode) |
//! | [`Gray16`] | 1 | No | 2 | 16-bit grayscale |
//!
//! # Example
//!
//! ```rust
//! use jpegli::{RGB8, RGBA8, Gray8};
//!
//! // All pixel types implement bytemuck::Pod for zero-copy operations
//! let pixels: &[RGB8] = &[
//!     RGB8::new(255, 0, 0),   // Red
//!     RGB8::new(0, 255, 0),   // Green
//!     RGB8::new(0, 0, 255),   // Blue
//! ];
//!
//! // Convert to bytes without copying
//! let bytes: &[u8] = bytemuck::cast_slice(pixels);
//! assert_eq!(bytes.len(), 9); // 3 pixels * 3 bytes
//! ```

use bytemuck::{Pod, Zeroable};

// Re-export rgb types with convenient names
pub use rgb::alt::Gray as GrayRaw;
pub use rgb::{RGB, RGBA};

/// 8-bit RGB pixel (3 bytes).
pub type RGB8 = RGB<u8>;

/// 8-bit RGBA pixel (4 bytes). Alpha is ignored during JPEG encoding.
pub type RGBA8 = RGBA<u8>;

/// 8-bit grayscale pixel (1 byte).
pub type Gray8 = GrayRaw<u8>;

/// 16-bit RGB pixel (6 bytes). Higher precision input for quality-critical workflows.
pub type RGB16 = RGB<u16>;

/// 16-bit RGBA pixel (8 bytes). Alpha is ignored during JPEG encoding.
pub type RGBA16 = RGBA<u16>;

/// 16-bit grayscale pixel (2 bytes).
pub type Gray16 = GrayRaw<u16>;

// Sealed trait pattern to prevent external implementations
mod private {
    pub trait Sealed {}
}

impl private::Sealed for RGB8 {}
impl private::Sealed for RGBA8 {}
impl private::Sealed for Gray8 {}
impl private::Sealed for RGB16 {}
impl private::Sealed for RGBA16 {}
impl private::Sealed for Gray16 {}

/// Marker trait for pixel types that can be encoded/decoded by jpegli.
///
/// This trait is sealed and cannot be implemented outside this crate.
/// All supported pixel types implement [`bytemuck::Pod`] for zero-copy operations.
///
/// # Supported Types
///
/// - [`RGB8`], [`RGBA8`], [`Gray8`] - 8-bit formats
/// - [`RGB16`], [`RGBA16`], [`Gray16`] - 16-bit formats (higher precision)
///
/// # Note on RGBA/RGBA16
///
/// Alpha channels are **ignored** during JPEG encoding since JPEG doesn't support
/// transparency. On decode, alpha is set to fully opaque (255 for u8, 65535 for u16).
pub trait Pixel: Pod + Zeroable + Sized + private::Sealed + 'static {
    /// Number of bytes per pixel.
    const BYTES: usize;

    /// Number of color channels (including alpha if present).
    const CHANNELS: usize;

    /// Whether this format has an alpha channel.
    ///
    /// Alpha is ignored on encode and set to opaque on decode.
    const HAS_ALPHA: bool;

    /// Bit depth per channel (8 or 16).
    const BIT_DEPTH: u8;
}

impl Pixel for RGB8 {
    const BYTES: usize = 3;
    const CHANNELS: usize = 3;
    const HAS_ALPHA: bool = false;
    const BIT_DEPTH: u8 = 8;
}

impl Pixel for RGBA8 {
    const BYTES: usize = 4;
    const CHANNELS: usize = 4;
    const HAS_ALPHA: bool = true;
    const BIT_DEPTH: u8 = 8;
}

impl Pixel for Gray8 {
    const BYTES: usize = 1;
    const CHANNELS: usize = 1;
    const HAS_ALPHA: bool = false;
    const BIT_DEPTH: u8 = 8;
}

impl Pixel for RGB16 {
    const BYTES: usize = 6;
    const CHANNELS: usize = 3;
    const HAS_ALPHA: bool = false;
    const BIT_DEPTH: u8 = 16;
}

impl Pixel for RGBA16 {
    const BYTES: usize = 8;
    const CHANNELS: usize = 4;
    const HAS_ALPHA: bool = true;
    const BIT_DEPTH: u8 = 16;
}

impl Pixel for Gray16 {
    const BYTES: usize = 2;
    const CHANNELS: usize = 1;
    const HAS_ALPHA: bool = false;
    const BIT_DEPTH: u8 = 16;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_sizes() {
        assert_eq!(std::mem::size_of::<RGB8>(), RGB8::BYTES);
        assert_eq!(std::mem::size_of::<RGBA8>(), RGBA8::BYTES);
        assert_eq!(std::mem::size_of::<Gray8>(), Gray8::BYTES);
        assert_eq!(std::mem::size_of::<RGB16>(), RGB16::BYTES);
        assert_eq!(std::mem::size_of::<RGBA16>(), RGBA16::BYTES);
        assert_eq!(std::mem::size_of::<Gray16>(), Gray16::BYTES);
    }

    #[test]
    fn test_bytemuck_cast() {
        let pixels: Vec<RGB8> = vec![
            RGB8::new(255, 0, 0),
            RGB8::new(0, 255, 0),
            RGB8::new(0, 0, 255),
        ];

        // Zero-copy cast to bytes
        let bytes: &[u8] = bytemuck::cast_slice(&pixels);
        assert_eq!(bytes, &[255, 0, 0, 0, 255, 0, 0, 0, 255]);

        // Zero-copy cast back
        let back: &[RGB8] = bytemuck::cast_slice(bytes);
        assert_eq!(back, &pixels);
    }

    #[test]
    fn test_16bit_bytemuck_cast() {
        let pixels: Vec<RGB16> = vec![RGB16::new(65535, 0, 0), RGB16::new(0, 65535, 0)];

        let bytes: &[u8] = bytemuck::cast_slice(&pixels);
        assert_eq!(bytes.len(), 12); // 2 pixels * 6 bytes

        let back: &[RGB16] = bytemuck::cast_slice(bytes);
        assert_eq!(back, &pixels);
    }

    #[test]
    fn test_gray_pixel() {
        let pixels: Vec<Gray8> = vec![Gray8::new(0), Gray8::new(128), Gray8::new(255)];

        let bytes: &[u8] = bytemuck::cast_slice(&pixels);
        assert_eq!(bytes, &[0, 128, 255]);
    }
}
