//! Native integer output collected from the same row reader exposed publicly.
use super::*;
use crate::foundation::alloc::try_alloc_vec;

impl DecodeConfig {
    pub(super) fn decode_native(
        &self,
        data: &[u8],
        parser: &mut JpegParser<'_>,
        stop: &dyn Stop,
        forced_exif: bool,
    ) -> Result<DecodeResult> {
        if stop.should_stop() {
            return Err(Error::cancelled());
        }
        self.validate_native_options(parser)?;
        let precision = parser.precision;
        let format = self.output_format.unwrap_or(PixelFormat::Rgb);
        let layout = crate::color::planes::Layout8::from_pixel_format(format).ok_or_else(|| {
            Error::unsupported_feature("native samples require RGB, BGR, RGBA, BGRA or gray output")
        })?;
        let channels = layout.bytes_per_pixel();
        let mut reader = self.scanline_reader(data)?;
        let (width, height) = (reader.width(), reader.height());
        let count = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(channels))
            .ok_or_else(|| Error::size_overflow("native output"))?;
        parser.charge_memory(
            count
                .checked_mul(if self.output_target.is_f32() {
                    4
                } else if self.output_target == OutputTarget::Native16 {
                    2
                } else {
                    1
                })
                .ok_or_else(|| Error::size_overflow("native output"))?,
            "native output",
        )?;
        let mut samples: Vec<u16> = try_alloc_vec(width as usize * channels, "native row")?;
        let mut output = if self.output_target == OutputTarget::Native16 {
            OwnedDecodedPixels::U16(try_alloc_vec(count, "native output")?)
        } else if self.output_target.is_f32() {
            OwnedDecodedPixels::F32(try_alloc_vec(count, "native output")?)
        } else {
            OwnedDecodedPixels::U8(try_alloc_vec(count, "native output")?)
        };
        let mut row: Vec<rgb::RGB<u16>> = try_alloc_vec(width as usize, "native RGB row")?;
        let max = ((1u32 << precision) - 1) as u16;
        for y in 0..height as usize {
            if stop.should_stop() {
                return Err(Error::cancelled());
            }
            let dest = &mut samples[..];
            if format == PixelFormat::Gray {
                reader.read_rows_gray16(imgref::ImgRefMut::new(dest, width as usize, 1))?;
            } else {
                reader.read_rows_rgb16(imgref::ImgRefMut::new(&mut row, width as usize, 1))?;
                for (dst, p) in dest.chunks_exact_mut(channels).zip(&row) {
                    let rgb = if matches!(
                        format,
                        PixelFormat::Bgr | PixelFormat::Bgra | PixelFormat::Bgrx
                    ) {
                        [p.b, p.g, p.r]
                    } else {
                        [p.r, p.g, p.b]
                    };
                    dst[..3].copy_from_slice(&rgb);
                    if channels == 4 {
                        dst[3] = max;
                    }
                }
            }
            let start = y * samples.len();
            let end = start + samples.len();
            match &mut output {
                OwnedDecodedPixels::U16(pixels) => pixels[start..end].copy_from_slice(&samples),
                OwnedDecodedPixels::U8(pixels) => {
                    for (dst, &src) in pixels[start..end].iter_mut().zip(&samples) {
                        *dst = ((src as u32 * 255 + max as u32 / 2) / max as u32) as u8;
                    }
                }
                OwnedDecodedPixels::F32(pixels) => {
                    for (dst, &src) in pixels[start..end].iter_mut().zip(&samples) {
                        let value = src as f32 / max as f32;
                        *dst = if self.output_target.is_linear() {
                            linear_srgb::default::srgb_to_linear(value)
                        } else {
                            value
                        };
                    }
                }
            }
        }
        // Metadata uses the shared marker framing, including inter/post-scan APPs.
        parser.collect_native_metadata(stop)?;
        let extras = finalize_extras(parser.take_extras(), data, forced_exif);
        let warnings = parser.take_warnings();
        let mut result = match output {
            OwnedDecodedPixels::U16(pixels) => {
                DecodeResult::new_u16(width, height, format, precision, pixels, extras, warnings)
            }
            OwnedDecodedPixels::F32(pixels) => DecodeResult::new_f32(
                width,
                height,
                format,
                self.output_target,
                pixels,
                extras,
                warnings,
            ),
            OwnedDecodedPixels::U8(pixels) => DecodeResult::new_u8(
                width,
                height,
                format,
                self.output_target,
                pixels,
                extras,
                warnings,
            ),
        };
        result.set_sample_precision(precision);
        Ok(result)
    }
}
