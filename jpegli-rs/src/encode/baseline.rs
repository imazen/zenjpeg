//! Baseline JPEG encoding implementation.
//!
//! This module contains the baseline (non-progressive) encoding functions
//! for both YCbCr and XYB color modes.

use super::*;

impl Encoder {
    /// Encodes as baseline JPEG.
    pub(super) fn encode_baseline(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut output = Vec::with_capacity(data.len() / 4);

        if self.config.use_xyb {
            self.encode_baseline_xyb(data, &mut output)
        } else {
            self.encode_baseline_ycbcr(data, &mut output)
        }
    }

    /// Encodes using standard YCbCr color space.
    fn encode_baseline_ycbcr(&self, data: &[u8], output: &mut Vec<u8>) -> Result<Vec<u8>> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;

        // Check if internal_pipeline specifies gamma-aware downsampling
        if let Some(ref pipeline) = self.config.internal_pipeline {
            match pipeline.downsampling {
                DownsamplingMethod::GammaAwareF32 => {
                    // Use f32 gamma-aware single-pass path
                    let (y_plane, cb_plane_final, cr_plane_final, c_width, c_height) =
                        match self.config.subsampling {
                            Subsampling::S420 => chroma::convert_gamma_aware_420(
                                data,
                                width,
                                height,
                                self.config.pixel_format,
                            )?,
                            Subsampling::S422 => chroma::convert_gamma_aware_422(
                                data,
                                width,
                                height,
                                self.config.pixel_format,
                            )?,
                            Subsampling::S440 => chroma::convert_gamma_aware_440(
                                data,
                                width,
                                height,
                                self.config.pixel_format,
                            )?,
                            Subsampling::S444 => {
                                // Should not happen - validation prevents this
                                return Err(Error::InvalidColorFormat {
                                    reason: "GammaAwareF32 not valid for 4:4:4",
                                });
                            }
                        };
                    return self.encode_baseline_ycbcr_with_planes(
                        output,
                        y_plane,
                        cb_plane_final,
                        cr_plane_final,
                        c_width,
                        c_height,
                    );
                }
                DownsamplingMethod::GammaAwareIterative => {
                    // Use f32 gamma-aware iterative path (Sharp YUV style optimization)
                    let (y_plane, cb_plane_final, cr_plane_final, c_width, c_height) =
                        match self.config.subsampling {
                            Subsampling::S420 => chroma::convert_gamma_aware_iterative_420(
                                data,
                                width,
                                height,
                                self.config.pixel_format,
                            )?,
                            Subsampling::S422 => chroma::convert_gamma_aware_iterative_422(
                                data,
                                width,
                                height,
                                self.config.pixel_format,
                            )?,
                            Subsampling::S440 => chroma::convert_gamma_aware_iterative_440(
                                data,
                                width,
                                height,
                                self.config.pixel_format,
                            )?,
                            Subsampling::S444 => {
                                // Should not happen - validation prevents this
                                return Err(Error::InvalidColorFormat {
                                    reason: "GammaAwareIterative not valid for 4:4:4",
                                });
                            }
                        };
                    return self.encode_baseline_ycbcr_with_planes(
                        output,
                        y_plane,
                        cb_plane_final,
                        cr_plane_final,
                        c_width,
                        c_height,
                    );
                }
                DownsamplingMethod::Sharp => {
                    // Use yuv crate Sharp YUV path
                    match self.config.subsampling {
                        Subsampling::S420 => {
                            let (y_plane, cb_plane_final, cr_plane_final, c_width, c_height) =
                                self.convert_yuv_crate_420(data, true)?;
                            return self.encode_baseline_ycbcr_with_planes(
                                output,
                                y_plane,
                                cb_plane_final,
                                cr_plane_final,
                                c_width,
                                c_height,
                            );
                        }
                        Subsampling::S422 => {
                            let (y_plane, cb_plane_final, cr_plane_final, c_width, c_height) =
                                self.convert_yuv_crate_422(data, true)?;
                            return self.encode_baseline_ycbcr_with_planes(
                                output,
                                y_plane,
                                cb_plane_final,
                                cr_plane_final,
                                c_width,
                                c_height,
                            );
                        }
                        Subsampling::S440 => {
                            // yuv crate doesn't support 4:4:0, fall through
                        }
                        Subsampling::S444 => {
                            // No downsampling needed
                        }
                    }
                }
                DownsamplingMethod::Box => {
                    // Use yuv crate Box filter path (non-sharp)
                    match self.config.subsampling {
                        Subsampling::S420 => {
                            let (y_plane, cb_plane_final, cr_plane_final, c_width, c_height) =
                                self.convert_yuv_crate_420(data, false)?;
                            return self.encode_baseline_ycbcr_with_planes(
                                output,
                                y_plane,
                                cb_plane_final,
                                cr_plane_final,
                                c_width,
                                c_height,
                            );
                        }
                        Subsampling::S422 => {
                            let (y_plane, cb_plane_final, cr_plane_final, c_width, c_height) =
                                self.convert_yuv_crate_422(data, false)?;
                            return self.encode_baseline_ycbcr_with_planes(
                                output,
                                y_plane,
                                cb_plane_final,
                                cr_plane_final,
                                c_width,
                                c_height,
                            );
                        }
                        Subsampling::S440 => {
                            // yuv crate doesn't support 4:4:0, fall through
                        }
                        Subsampling::S444 => {
                            // No downsampling needed
                        }
                    }
                }
                _ => {}
            }
        }

        // Resolve Auto to concrete method based on subsampling
        let chroma_method = self
            .config
            .chroma_conversion
            .resolve(self.config.subsampling);

        // yuv crate path (Sharp or Fast): performs color conversion + downsampling in one step
        if matches!(
            chroma_method,
            ChromaConversion::Sharp | ChromaConversion::Fast
        ) {
            let use_sharp = matches!(chroma_method, ChromaConversion::Sharp);
            match self.config.subsampling {
                Subsampling::S420 => {
                    let (y_plane, cb_plane_final, cr_plane_final, c_width, c_height) =
                        self.convert_yuv_crate_420(data, use_sharp)?;
                    return self.encode_baseline_ycbcr_with_planes(
                        output,
                        y_plane,
                        cb_plane_final,
                        cr_plane_final,
                        c_width,
                        c_height,
                    );
                }
                Subsampling::S422 => {
                    let (y_plane, cb_plane_final, cr_plane_final, c_width, c_height) =
                        self.convert_yuv_crate_422(data, use_sharp)?;
                    return self.encode_baseline_ycbcr_with_planes(
                        output,
                        y_plane,
                        cb_plane_final,
                        cr_plane_final,
                        c_width,
                        c_height,
                    );
                }
                // yuv crate doesn't support 4:4:0 or 4:4:4, fall through to Intrinsic path
                _ => {}
            }
        }

        // Intrinsic path: convert to YCbCr using f32 precision throughout (matches C++ jpegli)
        let (y_plane, cb_plane, cr_plane) = self.convert_to_ycbcr_f32(data)?;

        // Handle chroma subsampling (with optional input smoothing)
        let (cb_plane_final, cr_plane_final, c_width, c_height) = match self.config.subsampling {
            Subsampling::S420 => {
                // 4:2:0: Apply smoothing then downsample both Cb and Cr by 2x2
                let cb_smooth = self.apply_input_smoothing(&cb_plane, width, height)?;
                let cr_smooth = self.apply_input_smoothing(&cr_plane, width, height)?;
                let cb_down = self.downsample_2x2_f32(&cb_smooth, width, height)?;
                let cr_down = self.downsample_2x2_f32(&cr_smooth, width, height)?;
                let c_w = (width + 1) / 2;
                let c_h = (height + 1) / 2;
                (cb_down, cr_down, c_w, c_h)
            }
            Subsampling::S422 => {
                // 4:2:2: Apply smoothing then downsample horizontally only
                let cb_smooth = self.apply_input_smoothing(&cb_plane, width, height)?;
                let cr_smooth = self.apply_input_smoothing(&cr_plane, width, height)?;
                let cb_down = self.downsample_2x1_f32(&cb_smooth, width, height)?;
                let cr_down = self.downsample_2x1_f32(&cr_smooth, width, height)?;
                let c_w = (width + 1) / 2;
                (cb_down, cr_down, c_w, height)
            }
            Subsampling::S440 => {
                // 4:4:0: Apply smoothing then downsample vertically only
                let cb_smooth = self.apply_input_smoothing(&cb_plane, width, height)?;
                let cr_smooth = self.apply_input_smoothing(&cr_plane, width, height)?;
                let cb_down = self.downsample_1x2_f32(&cb_smooth, width, height)?;
                let cr_down = self.downsample_1x2_f32(&cr_smooth, width, height)?;
                let c_h = (height + 1) / 2;
                (cb_down, cr_down, width, c_h)
            }
            Subsampling::S444 => {
                // 4:4:4: No subsampling, no smoothing needed
                (cb_plane, cr_plane, width, height)
            }
        };

        self.encode_baseline_ycbcr_with_planes(
            output,
            y_plane,
            cb_plane_final,
            cr_plane_final,
            c_width,
            c_height,
        )
    }

    /// Encodes YCbCr planes to JPEG (shared by standard and Sharp YUV paths).
    pub(super) fn encode_baseline_ycbcr_with_planes(
        &self,
        output: &mut Vec<u8>,
        y_plane: Vec<f32>,
        cb_plane_final: Vec<f32>,
        cr_plane_final: Vec<f32>,
        c_width: usize,
        c_height: usize,
    ) -> Result<Vec<u8>> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;

        // Generate quantization tables (3 separate tables like C++ cjpegli)
        // Apply 4:2:0 quality compensation if using 4:2:0 subsampling
        let is_420 = self.config.subsampling == Subsampling::S420;
        let y_quant = self.gen_quant_table(0, false, is_420);
        let cb_quant = self.gen_quant_table(1, false, is_420);
        let cr_quant = self.gen_quant_table(2, false, is_420);

        // Quantize all blocks first (needed for both standard and optimized encoding)
        let (y_blocks, cb_blocks, cr_blocks) = self.quantize_all_blocks_subsampled(
            &y_plane,
            width,
            height,
            &cb_plane_final,
            &cr_plane_final,
            c_width,
            c_height,
            &y_quant,
            &cb_quant,
            &cr_quant,
        )?;
        let is_color = self.config.pixel_format != PixelFormat::Gray;

        // Write JPEG structure
        self.write_header(output)?;
        self.write_quant_tables(output, &y_quant, &cb_quant, &cr_quant)?;
        self.write_frame_header(output)?;

        // For optimized Huffman, build tables from block frequencies before writing DHT
        let scan_data = if self.config.optimize_huffman {
            let tables =
                self.build_optimized_tables(&y_blocks, &cb_blocks, &cr_blocks, is_color)?;
            self.write_huffman_tables_optimized(output, &tables)?;

            if self.config.restart_interval > 0 {
                self.write_restart_interval(output)?;
            }
            self.write_scan_header(output)?;

            // Encode with optimized tables
            self.encode_with_tables(&y_blocks, &cb_blocks, &cr_blocks, is_color, &tables)?
        } else {
            self.write_huffman_tables(output)?;

            if self.config.restart_interval > 0 {
                self.write_restart_interval(output)?;
            }
            self.write_scan_header(output)?;

            // Encode with standard tables
            self.encode_blocks_standard(&y_blocks, &cb_blocks, &cr_blocks, is_color)?
        };

        output.extend_from_slice(&scan_data);

        // Write EOI
        output.push(0xFF);
        output.push(MARKER_EOI);

        Ok(std::mem::take(output))
    }

    /// Encodes using XYB mode (perceptually optimized color space).
    ///
    /// XYB encoding pipeline:
    /// 1. sRGB → linear RGB → XYB → scaled XYB (values in [0, 1])
    /// 2. Multiply by 255 for JPEG sample range
    /// 3. Level shift by subtracting 128 for DCT
    fn encode_baseline_xyb(&self, data: &[u8], output: &mut Vec<u8>) -> Result<Vec<u8>> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;

        // Convert sRGB to scaled XYB (full color conversion pipeline)
        let (x_plane, y_plane, b_plane) = self.convert_to_scaled_xyb(data)?;

        // Downsample B channel (XYB subsamples B to 1/4 resolution)
        // Apply input smoothing before downsampling (matches C++ jpegli behavior)
        let b_smooth = self.apply_input_smoothing(&b_plane, width, height)?;
        let b_downsampled = self.downsample_2x2_f32(&b_smooth, width, height)?;
        let b_width = (width + 1) / 2;
        let b_height = (height + 1) / 2;

        // Generate XYB quantization tables (one per component)
        // XYB mode doesn't use 4:2:0 quality compensation
        let x_quant = self.gen_quant_table(0, true, false); // X component
        let y_quant = self.gen_quant_table(1, true, false); // Y component (luma-like)
        let b_quant = self.gen_quant_table(2, true, false); // B component

        // Compute AQ map from Y plane (XYB's Y is the luma-like channel)
        // Scale Y plane from [0,1] to [0,255] range for AQ computation (SIMD)
        let y_plane_scaled = crate::encode_simd::scale_f32_slice_simd(&y_plane, 255.0);
        let y_quant_01 = y_quant.values[1];
        #[cfg(feature = "experimental-hybrid-trellis")]
        let aq_map =
            hybrid::get_aq_map_or_compute(&self.config, &y_plane_scaled, width, height, y_quant_01);
        #[cfg(not(feature = "experimental-hybrid-trellis"))]
        let aq_map = compute_aq_strength_map(&y_plane_scaled, width, height, y_quant_01);

        // Zero-bias parameters for XYB (use YCbCr tables as approximation)
        // X and Y are luma-like (full-res), B is chroma-like (downsampled)
        let effective_distance = quant::quant_vals_to_distance(&x_quant, &y_quant, &b_quant);
        let x_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 0); // X uses luma params
        let y_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 0); // Y uses luma params
        let b_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 1); // B uses chroma params

        // Create hybrid quantization context if enabled
        #[cfg(feature = "experimental-hybrid-trellis")]
        let hybrid_ctx = hybrid::create_hybrid_ctx(&self.config);

        // Write JPEG structure for XYB mode (no JFIF, just ICC profile)
        self.write_header_xyb(output)?;
        // Write APP14 Adobe marker for RGB colorspace (required by some decoders)
        // See: https://github.com/google/jpegli/pull/135
        self.write_app14_adobe(output, 0)?; // 0 = RGB (no transform)
                                            // Write XYB ICC profile so decoders can interpret the colors correctly
        self.write_icc_profile(output, &XYB_ICC_PROFILE)?;
        self.write_quant_tables_xyb(output, &x_quant, &y_quant, &b_quant)?;
        self.write_frame_header_xyb(output)?;

        // For optimized Huffman, quantize all blocks first to collect frequencies
        let scan_data = if self.config.optimize_huffman {
            #[cfg(feature = "experimental-hybrid-trellis")]
            let (x_blocks, y_blocks, b_blocks) = hybrid::quantize_all_blocks_xyb_with_aq(
                &x_plane,
                &y_plane,
                &b_downsampled,
                width,
                height,
                b_width,
                b_height,
                &x_quant,
                &y_quant,
                &b_quant,
                &aq_map,
                hybrid_ctx.as_ref(),
            );
            #[cfg(not(feature = "experimental-hybrid-trellis"))]
            let (x_blocks, y_blocks, b_blocks) = self.quantize_all_blocks_xyb_with_aq_simple(
                &x_plane,
                &y_plane,
                &b_downsampled,
                width,
                height,
                b_width,
                b_height,
                &x_quant,
                &y_quant,
                &b_quant,
                &aq_map,
                &x_zero_bias,
                &y_zero_bias,
                &b_zero_bias,
            );
            let (dc_table, ac_table) =
                self.build_optimized_tables_xyb(&x_blocks, &y_blocks, &b_blocks)?;
            self.write_huffman_tables_xyb_optimized(output, &dc_table, &ac_table);

            if self.config.restart_interval > 0 {
                self.write_restart_interval(output)?;
            }
            self.write_scan_header_xyb(output)?;

            // Encode with optimized tables
            self.encode_with_tables_xyb(&x_blocks, &y_blocks, &b_blocks, &dc_table, &ac_table)?
        } else {
            self.write_huffman_tables(output)?;

            if self.config.restart_interval > 0 {
                self.write_restart_interval(output)?;
            }
            self.write_scan_header_xyb(output)?;

            // Encode with standard tables
            self.encode_scan_xyb_float(
                &x_plane,
                &y_plane,
                &b_downsampled,
                width,
                height,
                b_width,
                b_height,
                &x_quant,
                &y_quant,
                &b_quant,
            )?
        };

        output.extend_from_slice(&scan_data);

        // Write EOI
        output.push(0xFF);
        output.push(MARKER_EOI);

        Ok(std::mem::take(output))
    }
}
