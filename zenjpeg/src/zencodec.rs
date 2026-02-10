//! zencodec-types trait implementations for zenjpeg.
//!
//! Provides [`JpegEncoding`] and [`JpegDecoding`] types that implement the
//! [`Encoding`] / [`Decoding`] traits from zencodec-types, wrapping the native
//! zenjpeg API.
//!
//! The native API remains untouched — this is a thin adapter layer.

extern crate alloc;
use alloc::vec::Vec;

use imgref::{ImgRef, ImgVec};
use rgb::{Gray, Rgb, Rgba};
use zencodec_types::{
    DecodeOutput, Decoding, DecodingJob, EncodeOutput, Encoding, EncodingJob, ImageFormat,
    ImageInfo, ImageMetadata, PixelData, Stop,
};

use crate::encode::encoder_config::EncoderConfig;
use crate::encode::encoder_types::{ChromaSubsampling, Quality};
use crate::encode::exif::Exif;
use crate::error::Error;

// ── Encoding ────────────────────────────────────────────────────────────────

/// JPEG encoder configuration implementing [`Encoding`].
///
/// Wraps [`EncoderConfig`] with limit fields for the trait interface.
/// Defaults to YCbCr 4:2:0 at quality 85.
///
/// # Examples
///
/// ```rust
/// use zencodec_types::Encoding;
/// use zenjpeg::JpegEncoding;
///
/// let enc = JpegEncoding::new()
///     .with_quality(90.0)
///     .with_progressive(true);
/// ```
#[derive(Clone, Debug)]
pub struct JpegEncoding {
    inner: EncoderConfig,
    limit_pixels: Option<u64>,
    limit_memory: Option<u64>,
    limit_output: Option<u64>,
}

impl JpegEncoding {
    /// Create a default YCbCr 4:2:0 config at quality 85.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter),
            limit_pixels: None,
            limit_memory: None,
            limit_output: None,
        }
    }

    /// Create a YCbCr config with quality and subsampling.
    #[must_use]
    pub fn ycbcr(quality: f32, subsampling: ChromaSubsampling) -> Self {
        Self {
            inner: EncoderConfig::ycbcr(quality, subsampling),
            limit_pixels: None,
            limit_memory: None,
            limit_output: None,
        }
    }

    /// Create a grayscale config with quality.
    #[must_use]
    pub fn grayscale(quality: f32) -> Self {
        Self {
            inner: EncoderConfig::grayscale(quality),
            limit_pixels: None,
            limit_memory: None,
            limit_output: None,
        }
    }

    /// Enable progressive JPEG encoding.
    #[must_use]
    pub fn with_progressive(mut self, enable: bool) -> Self {
        self.inner = self.inner.progressive(enable);
        self
    }

    /// Enable SharpYUV chroma downsampling (better edges, slower).
    #[must_use]
    pub fn with_sharp_yuv(mut self, enable: bool) -> Self {
        self.inner = self.inner.sharp_yuv(enable);
        self
    }

    /// Set chroma subsampling mode.
    #[must_use]
    pub fn with_subsampling(self, subsampling: ChromaSubsampling) -> Self {
        let quality = self.inner.quality;
        Self {
            inner: EncoderConfig::ycbcr(quality, subsampling),
            ..self
        }
    }

    /// Access the underlying [`EncoderConfig`].
    #[must_use]
    pub fn inner(&self) -> &EncoderConfig {
        &self.inner
    }

    /// Mutable access to the underlying [`EncoderConfig`].
    pub fn inner_mut(&mut self) -> &mut EncoderConfig {
        &mut self.inner
    }
}

impl Default for JpegEncoding {
    fn default() -> Self {
        Self::new()
    }
}

impl Encoding for JpegEncoding {
    type Error = Error;
    type Job<'a> = JpegEncodeJob<'a>;

    fn with_quality(mut self, quality: f32) -> Self {
        self.inner = self.inner.quality(Quality::ApproxJpegli(quality));
        self
    }

    fn with_effort(self, _effort: u32) -> Self {
        // JPEG doesn't have a separate effort parameter.
        // Progressive mode is the main speed/quality tradeoff but it's
        // controlled via with_progressive().
        self
    }

    fn with_lossless(self, _lossless: bool) -> Self {
        // JPEG is inherently lossy; ignore.
        self
    }

    fn with_alpha_quality(self, _quality: f32) -> Self {
        // JPEG doesn't support alpha; ignore.
        self
    }

    fn with_limit_pixels(mut self, max: u64) -> Self {
        self.limit_pixels = Some(max);
        self
    }

    fn with_limit_memory(mut self, bytes: u64) -> Self {
        self.limit_memory = Some(bytes);
        self
    }

    fn with_limit_output(mut self, bytes: u64) -> Self {
        self.limit_output = Some(bytes);
        self
    }

    fn job(&self) -> JpegEncodeJob<'_> {
        JpegEncodeJob {
            config: self,
            stop: None,
            icc: None,
            exif: None,
            xmp: None,
            limit_pixels: None,
            limit_memory: None,
        }
    }
}

// ── Encode job ──────────────────────────────────────────────────────────────

/// Per-operation JPEG encode job.
///
/// Created by [`JpegEncoding::job()`]. Borrows temporary data (stop token,
/// metadata) and is consumed by terminal encode methods.
pub struct JpegEncodeJob<'a> {
    config: &'a JpegEncoding,
    stop: Option<&'a dyn Stop>,
    icc: Option<&'a [u8]>,
    exif: Option<&'a [u8]>,
    xmp: Option<&'a [u8]>,
    limit_pixels: Option<u64>,
    limit_memory: Option<u64>,
}

impl<'a> JpegEncodeJob<'a> {
    /// Encode using the native request API. Common path for all pixel types.
    fn do_encode(self, pixels: &[Rgb<u8>], w: u32, h: u32) -> Result<EncodeOutput, Error> {
        let mut req = self.config.inner.request();

        if let Some(icc) = self.icc {
            req = req.icc_profile(icc);
        }
        if let Some(exif) = self.exif {
            req = req.exif(Exif::raw(exif));
        }
        if let Some(xmp) = self.xmp {
            req = req.xmp(xmp);
        }
        if let Some(stop) = self.stop {
            req = req.stop(stop);
        }

        let data = req.encode(pixels, w, h)?;
        Ok(EncodeOutput::new(data, ImageFormat::Jpeg))
    }
}

impl<'a> EncodingJob<'a> for JpegEncodeJob<'a> {
    type Error = Error;

    fn with_stop(mut self, stop: &'a dyn Stop) -> Self {
        self.stop = Some(stop);
        self
    }

    fn with_metadata(mut self, meta: &'a ImageMetadata<'a>) -> Self {
        if let Some(icc) = meta.icc_profile {
            self.icc = Some(icc);
        }
        if let Some(exif) = meta.exif {
            self.exif = Some(exif);
        }
        if let Some(xmp) = meta.xmp {
            self.xmp = Some(xmp);
        }
        self
    }

    fn with_icc(mut self, icc: &'a [u8]) -> Self {
        self.icc = Some(icc);
        self
    }

    fn with_exif(mut self, exif: &'a [u8]) -> Self {
        self.exif = Some(exif);
        self
    }

    fn with_xmp(mut self, xmp: &'a [u8]) -> Self {
        self.xmp = Some(xmp);
        self
    }

    fn with_limit_pixels(mut self, max: u64) -> Self {
        self.limit_pixels = Some(max);
        self
    }

    fn with_limit_memory(mut self, bytes: u64) -> Self {
        self.limit_memory = Some(bytes);
        self
    }

    fn encode_rgb8(self, img: ImgRef<'_, Rgb<u8>>) -> Result<EncodeOutput, Self::Error> {
        let (buf, w, h) = img.to_contiguous_buf();
        self.do_encode(&buf, w as u32, h as u32)
    }

    fn encode_rgba8(self, img: ImgRef<'_, Rgba<u8>>) -> Result<EncodeOutput, Self::Error> {
        // JPEG doesn't support alpha — strip it.
        let (buf, w, h) = img.to_contiguous_buf();
        let rgb: Vec<Rgb<u8>> = buf
            .iter()
            .map(|p: &Rgba<u8>| Rgb {
                r: p.r,
                g: p.g,
                b: p.b,
            })
            .collect();
        self.do_encode(&rgb, w as u32, h as u32)
    }

    fn encode_gray8(self, img: ImgRef<'_, Gray<u8>>) -> Result<EncodeOutput, Self::Error> {
        // Expand gray to RGB (JPEG grayscale uses its own color mode but
        // the EncoderConfig is already set — if it's YCbCr, RGB with R=G=B
        // produces the same result as grayscale subsampling).
        let (buf, w, h) = img.to_contiguous_buf();
        let rgb: Vec<Rgb<u8>> = buf
            .iter()
            .map(|p: &Gray<u8>| {
                let v = p.value();
                Rgb { r: v, g: v, b: v }
            })
            .collect();
        self.do_encode(&rgb, w as u32, h as u32)
    }
}

// ── Decoding ────────────────────────────────────────────────────────────────

/// JPEG decoder configuration implementing [`Decoding`].
///
/// Wraps [`crate::decode::DecodeConfig`] with the trait interface.
///
/// # Examples
///
/// ```rust,ignore
/// use zencodec_types::Decoding;
/// use zenjpeg::JpegDecoding;
///
/// let dec = JpegDecoding::new()
///     .with_limit_pixels(100_000_000);
/// let output = dec.decode(&jpeg_bytes)?;
/// ```
#[derive(Clone, Debug)]
pub struct JpegDecoding {
    #[cfg(feature = "decoder")]
    inner: crate::decode::DecodeConfig,
    limit_file_size: Option<u64>,
    // When the decoder feature is disabled, we still need fields for limits
    // so that Decoding trait methods work (they just won't decode).
    #[cfg(not(feature = "decoder"))]
    max_pixels: Option<u64>,
    #[cfg(not(feature = "decoder"))]
    max_memory: Option<u64>,
}

impl JpegDecoding {
    /// Create a default decoder config.
    #[must_use]
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "decoder")]
            inner: crate::decode::DecodeConfig::new(),
            limit_file_size: None,
            #[cfg(not(feature = "decoder"))]
            max_pixels: None,
            #[cfg(not(feature = "decoder"))]
            max_memory: None,
        }
    }
}

impl Default for JpegDecoding {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoding for JpegDecoding {
    type Error = Error;
    type Job<'a> = JpegDecodeJob<'a>;

    fn with_limit_pixels(mut self, max: u64) -> Self {
        #[cfg(feature = "decoder")]
        {
            self.inner = self.inner.max_pixels(max);
        }
        #[cfg(not(feature = "decoder"))]
        {
            self.max_pixels = Some(max);
        }
        self
    }

    fn with_limit_memory(mut self, bytes: u64) -> Self {
        #[cfg(feature = "decoder")]
        {
            self.inner = self.inner.max_memory(bytes);
        }
        #[cfg(not(feature = "decoder"))]
        {
            self.max_memory = Some(bytes);
        }
        self
    }

    fn with_limit_dimensions(mut self, _width: u32, _height: u32) -> Self {
        // zenjpeg's decoder doesn't have per-dimension limits, only max_pixels.
        // Use max_pixels = width * height as an approximation.
        let max = _width as u64 * _height as u64;
        #[cfg(feature = "decoder")]
        {
            self.inner = self.inner.max_pixels(max);
        }
        #[cfg(not(feature = "decoder"))]
        {
            self.max_pixels = Some(max);
        }
        self
    }

    fn with_limit_file_size(mut self, bytes: u64) -> Self {
        self.limit_file_size = Some(bytes);
        self
    }

    fn job(&self) -> JpegDecodeJob<'_> {
        JpegDecodeJob {
            config: self,
            stop: None,
            limit_pixels: None,
            limit_memory: None,
        }
    }

    fn probe(&self, data: &[u8]) -> Result<ImageInfo, Self::Error> {
        #[cfg(feature = "decoder")]
        {
            let info = self.inner.read_info(data)?;
            Ok(to_image_info(&info))
        }
        #[cfg(not(feature = "decoder"))]
        {
            let _ = data;
            Err(Error::unsupported_feature("decoder feature required for probing"))
        }
    }
}

// ── Decode job ──────────────────────────────────────────────────────────────

/// Per-operation JPEG decode job.
///
/// Created by [`JpegDecoding::job()`]. Borrows a stop token and is consumed
/// by terminal decode methods.
pub struct JpegDecodeJob<'a> {
    config: &'a JpegDecoding,
    stop: Option<&'a dyn Stop>,
    limit_pixels: Option<u64>,
    limit_memory: Option<u64>,
}

impl<'a> DecodingJob<'a> for JpegDecodeJob<'a> {
    type Error = Error;

    fn with_stop(mut self, stop: &'a dyn Stop) -> Self {
        self.stop = Some(stop);
        self
    }

    fn with_limit_pixels(mut self, max: u64) -> Self {
        self.limit_pixels = Some(max);
        self
    }

    fn with_limit_memory(mut self, bytes: u64) -> Self {
        self.limit_memory = Some(bytes);
        self
    }

    fn decode(self, data: &[u8]) -> Result<DecodeOutput, Self::Error> {
        #[cfg(feature = "decoder")]
        {
            use crate::types::PixelFormat;

            // Build decoder config with overrides
            let mut cfg = self.config.inner.clone();

            if let Some(max) = self.limit_pixels {
                cfg = cfg.max_pixels(max);
            }
            if let Some(bytes) = self.limit_memory {
                cfg = cfg.max_memory(bytes);
            }

            // Ensure metadata preservation is enabled
            cfg = cfg.preserve_all();

            let stop = self.stop.unwrap_or(&enough::Unstoppable);
            let result = cfg.decode(data, stop)?;

            let w = result.width();
            let h = result.height();
            let format = result.format();

            // Extract metadata from extras before consuming pixels
            let mut info = ImageInfo::new(w, h, ImageFormat::Jpeg);
            if let Some(extras) = result.extras() {
                if let Some(icc) = extras.icc_profile() {
                    info = info.with_icc_profile(icc.to_vec());
                }
                if let Some(exif) = extras.exif() {
                    info = info.with_exif(exif.to_vec());
                }
                if let Some(xmp) = extras.xmp() {
                    info = info.with_xmp(xmp.as_bytes().to_vec());
                }
            }

            // Now consume pixels
            let pixel_data = match format {
                PixelFormat::Gray => {
                    let pixels_u8 = result.into_pixels_u8().unwrap_or_default();
                    let gray: Vec<Gray<u8>> = pixels_u8.iter().map(|&v| Gray::new(v)).collect();
                    PixelData::Gray8(ImgVec::new(gray, w as usize, h as usize))
                }
                PixelFormat::Rgb => {
                    let pixels_u8 = result.into_pixels_u8().unwrap_or_default();
                    let rgb = bytes_to_rgb(&pixels_u8);
                    PixelData::Rgb8(ImgVec::new(rgb, w as usize, h as usize))
                }
                _ => {
                    // For other formats, best effort — treat as RGB bytes
                    let pixels_u8 = result.into_pixels_u8().unwrap_or_default();
                    let rgb = bytes_to_rgb(&pixels_u8);
                    PixelData::Rgb8(ImgVec::new(rgb, w as usize, h as usize))
                }
            };

            Ok(DecodeOutput::new(pixel_data, info))
        }

        #[cfg(not(feature = "decoder"))]
        {
            let _ = data;
            Err(Error::unsupported_feature("decoder feature required"))
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Convert JpegInfo to zencodec_types ImageInfo.
#[cfg(feature = "decoder")]
fn to_image_info(info: &crate::decode::JpegInfo) -> ImageInfo {
    let mut img_info = ImageInfo::new(
        info.dimensions.width,
        info.dimensions.height,
        ImageFormat::Jpeg,
    );

    if let Some(ref icc) = info.icc_profile {
        img_info = img_info.with_icc_profile(icc.clone());
    }
    if let Some(ref exif) = info.exif {
        img_info = img_info.with_exif(exif.clone());
    }
    if let Some(ref xmp) = info.xmp {
        img_info = img_info.with_xmp(xmp.as_bytes().to_vec());
    }

    img_info
}

/// Convert raw bytes (3 bytes per pixel) to Vec<Rgb<u8>>.
fn bytes_to_rgb(bytes: &[u8]) -> Vec<Rgb<u8>> {
    bytes
        .chunks_exact(3)
        .map(|c| Rgb {
            r: c[0],
            g: c[1],
            b: c[2],
        })
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use imgref::Img;
    use zencodec_types::Encoding;

    #[test]
    fn encoding_default_roundtrip() {
        let enc = JpegEncoding::new().with_quality(80.0);
        let pixels = vec![Rgb { r: 128, g: 64, b: 32 }; 64];
        let img = Img::new(pixels, 8, 8);
        let output = enc.encode_rgb8(img.as_ref()).unwrap();
        assert!(!output.bytes().is_empty());
        assert_eq!(output.format(), ImageFormat::Jpeg);
        // Verify it starts with JPEG SOI marker
        assert_eq!(&output.bytes()[0..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn encoding_with_metadata() {
        let enc = JpegEncoding::new().with_quality(85.0);
        let pixels = vec![Rgb { r: 255, g: 0, b: 0 }; 16];
        let img = Img::new(pixels, 4, 4);

        let icc = b"fake icc profile data";
        let output = enc
            .job()
            .with_icc(icc)
            .encode_rgb8(img.as_ref())
            .unwrap();
        assert!(!output.bytes().is_empty());
    }

    #[test]
    fn encoding_gray8() {
        let enc = JpegEncoding::new().with_quality(90.0);
        let pixels = vec![Gray::new(128u8); 64];
        let img = Img::new(pixels, 8, 8);
        let output = enc.encode_gray8(img.as_ref()).unwrap();
        assert!(!output.bytes().is_empty());
        assert_eq!(output.format(), ImageFormat::Jpeg);
    }

    #[test]
    fn encoding_rgba8_strips_alpha() {
        let enc = JpegEncoding::new().with_quality(85.0);
        let pixels = vec![
            Rgba {
                r: 100,
                g: 150,
                b: 200,
                a: 128,
            };
            64
        ];
        let img = Img::new(pixels, 8, 8);
        let output = enc.encode_rgba8(img.as_ref()).unwrap();
        assert!(!output.bytes().is_empty());
    }

    #[cfg(feature = "decoder")]
    #[test]
    fn decode_roundtrip() {
        use zencodec_types::Decoding;

        // Encode
        let enc = JpegEncoding::new().with_quality(95.0);
        let pixels = vec![Rgb { r: 200, g: 100, b: 50 }; 64];
        let img = Img::new(pixels, 8, 8);
        let encoded = enc.encode_rgb8(img.as_ref()).unwrap();

        // Decode
        let dec = JpegDecoding::new();
        let output = dec.decode(encoded.bytes()).unwrap();
        assert_eq!(output.info().width, 8);
        assert_eq!(output.info().height, 8);
        assert_eq!(output.info().format, ImageFormat::Jpeg);
    }

    #[cfg(feature = "decoder")]
    #[test]
    fn probe_info() {
        use zencodec_types::Decoding;

        let enc = JpegEncoding::new().with_quality(85.0);
        let pixels = vec![Rgb { r: 0, g: 0, b: 0 }; 100];
        let img = Img::new(pixels, 10, 10);
        let encoded = enc.encode_rgb8(img.as_ref()).unwrap();

        let dec = JpegDecoding::new();
        let info = dec.probe(encoded.bytes()).unwrap();
        assert_eq!(info.width, 10);
        assert_eq!(info.height, 10);
        assert_eq!(info.format, ImageFormat::Jpeg);
    }
}
