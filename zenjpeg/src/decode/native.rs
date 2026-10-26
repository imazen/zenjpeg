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
        stop.check()?;
        self.validate_native_options(parser)?;
        let precision = parser.precision;
        let format = self.output_format.unwrap_or(PixelFormat::Rgb);
        let row_format = match format {
            PixelFormat::GrayF32 if self.output_target.is_f32() => PixelFormat::Gray,
            PixelFormat::RgbaF32 if self.output_target.is_f32() => PixelFormat::Rgba,
            other => other,
        };
        let layout =
            crate::color::planes::Layout8::from_pixel_format(if row_format == PixelFormat::Cmyk {
                PixelFormat::Rgba
            } else {
                row_format
            })
            .ok_or_else(|| {
                Error::unsupported_feature(
                    "native samples require RGB, BGR, RGBA, BGRA or gray output",
                )
            })?;
        let channels = layout.bytes_per_pixel();
        // Reserve the output budget before constructing a coefficient-backed
        // reader, so the two stages cannot each spend the whole memory limit.
        // Crops conservatively reserve the uncropped output, as transforms do.
        let output_bytes = (parser.width as usize)
            .checked_mul(parser.height as usize)
            .and_then(|n| n.checked_mul(channels))
            .and_then(|n| {
                n.checked_mul(if self.output_target.is_f32() {
                    4
                } else if self.output_target == OutputTarget::Native16 {
                    2
                } else {
                    1
                })
            })
            .ok_or_else(|| Error::size_overflow("native output"))?;
        parser.charge_memory(output_bytes, "native output")?;
        let mut row_config = self.clone();
        if self.max_memory != 0 && self.max_memory != u64::MAX {
            row_config.max_memory = (self.max_memory - output_bytes as u64).max(1);
        }
        let mut reader = if parser.mode == JpegMode::Lossless {
            let predictive = parser.predictive_reader(stop)?;
            let mut reader = ScanlineReader::from_predictive(
                alloc::borrow::Cow::Borrowed(data),
                parser.width,
                parser.height,
                parser.num_components,
                precision,
                predictive,
                parser.color_model,
                core::array::from_fn(|i| parser.components[i].id),
            );
            self.apply_crop(&mut reader, parser.width, parser.height, 1)?;
            reader
        } else {
            row_config.scanline_reader(data)?
        };
        let (width, height) = (reader.width(), reader.height());
        let count = width as usize * height as usize * channels;
        let mut samples: Vec<u16> = try_alloc_vec(width as usize * channels, "native row")?;
        let mut output = if self.output_target == OutputTarget::Native16 {
            OwnedDecodedPixels::U16(try_alloc_vec(count, "native output")?)
        } else if self.output_target.is_f32() {
            OwnedDecodedPixels::F32(try_alloc_vec(count, "native output")?)
        } else {
            OwnedDecodedPixels::U8(try_alloc_vec(count, "native output")?)
        };
        let mut cmyk_row: Vec<scanline::Cmyk16> = try_alloc_vec(
            if row_format == PixelFormat::Cmyk {
                width as usize
            } else {
                0
            },
            "native CMYK row",
        )?;
        let mut row: Vec<rgb::RGB<u16>> = try_alloc_vec(width as usize, "native RGB row")?;
        let max = ((1u32 << precision) - 1) as u16;
        for y in 0..height as usize {
            stop.check()?;
            let dest = &mut samples[..];
            if row_format == PixelFormat::Gray {
                reader.read_rows_gray16(imgref::ImgRefMut::new(dest, width as usize, 1))?;
            } else if row_format == PixelFormat::Cmyk {
                reader.read_rows_cmyk16(imgref::ImgRefMut::new(
                    &mut cmyk_row,
                    width as usize,
                    1,
                ))?;
                for (dst, p) in dest.as_chunks_mut::<4>().0.iter_mut().zip(&cmyk_row) {
                    dst.copy_from_slice(&[p.c, p.m, p.y, p.k]);
                }
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
        let gain_map = if self.gain_map != GainMapHandling::Discard {
            self.extract_gain_map(parser, data, stop)?
        } else {
            None
        };
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
        result.gain_map = gain_map;
        Ok(result)
    }
}
