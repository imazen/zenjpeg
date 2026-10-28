//! Component-independent native DCT rows. The uncommon sampling/component
//! layouts use the same entropy decoder, IDCT and interpolation kernels as the
//! optimized three-plane reader. Three MCU strips retain vertical context.
use super::*;
use crate::decode::{ChromaUpsampling, DecodedCoefficients};
use crate::entropy::{EntropyDecoder, EntropyDecoderState};
use crate::error::ScanRead;
use crate::foundation::alloc::{try_alloc_vec, try_with_capacity};

struct Plane {
    h: usize,
    v: usize,
    width: usize,
    height: usize,
    stride: usize,
    strip_rows: usize,
    samples: Vec<i16>,
    quant: [u16; 64],
}

pub(in crate::decode) struct ComponentRows {
    planes: Vec<Plane>,
    coefficients: Option<DecodedCoefficients>,
    dc: HuffmanTables,
    ac: HuffmanTables,
    mapping: [(usize, usize); 4],
    order: [usize; 4],
    start: usize,
    state: Option<EntropyDecoderState>,
    restart: usize,
    next_restart: u8,
    mcus: usize,
    columns: usize,
    strip_height: usize,
    strips: usize,
    loaded: usize,
    width: usize,
    height: usize,
    fancy: bool,
    precision: u8,
    pub row: Vec<u16>,
    scratch: Vec<i16>,
    pub current_row: usize,
}

impl JpegParser<'_> {
    pub(in crate::decode) fn component_rows(
        &mut self,
        fancy: ChromaUpsampling,
        stop: &dyn enough::Stop,
    ) -> Result<ComponentRows> {
        if self.height == 0 {
            return Err(Error::unsupported_feature("component rows with DNL height"));
        }
        let n = self.num_components as usize;
        let max_h = self.components[..n]
            .iter()
            .map(|c| c.h_samp_factor as usize)
            .max()
            .unwrap();
        let max_v = self.components[..n]
            .iter()
            .map(|c| c.v_samp_factor as usize)
            .max()
            .unwrap();
        if self.components[..n]
            .iter()
            .any(|c| max_h % c.h_samp_factor as usize != 0 || max_v % c.v_samp_factor as usize != 0)
        {
            return Err(Error::unsupported_feature(
                "fractional component sampling ratios",
            ));
        }
        let width = self.width as usize;
        let height = self.height as usize;
        let columns = width.div_ceil(max_h * 8);
        let mut planes = try_with_capacity(n, "component strips")?;
        // Parse tables before reserving component storage.
        let coefficients;
        let scan;
        if self.interleaved && matches!(self.mode, JpegMode::Baseline | JpegMode::Extended) {
            let header = self.find_scan_info()?;
            if !header.sequential_valid || header.num_components != self.num_components {
                return Err(Error::invalid_jpeg_data(
                    "invalid sequential component scan",
                ));
            }
            scan = Some(header);
            coefficients = None;
        } else {
            self.native_precision = true;
            self.decode_mode = DecodeMode::Coefficient;
            self.decode(&stop)?;
            coefficients = Some(self.extract_coefficients()?);
            scan = None;
        }
        for i in 0..n {
            let c = &self.components[i];
            let (h, v) = (c.h_samp_factor as usize, c.v_samp_factor as usize);
            let (stride, strip_rows) = (columns * h * 8, v * 8);
            let quant = if let Some(coeffs) = &coefficients {
                coeffs.quant_tables[coeffs.components[i].quant_table_idx as usize]
            } else {
                self.quant_tables[c.quant_table_idx as usize]
            }
            .ok_or_else(|| Error::invalid_jpeg_data("missing component quantization table"))?;
            self.charge_memory(stride * strip_rows * 6, "component strips")?;
            planes.push(Plane {
                h,
                v,
                width: (width * h).div_ceil(max_h),
                height: (height * v).div_ceil(max_v),
                stride,
                strip_rows,
                samples: try_alloc_vec(stride * strip_rows * 3, "component strips")?,
                quant,
            });
        }
        self.charge_memory(width * (n + 1) * 2, "component row")?;
        Ok(ComponentRows {
            planes,
            coefficients,
            dc: self.dc_tables.clone(),
            ac: self.ac_tables.clone(),
            mapping: scan.as_ref().map_or([(0, 0); 4], |s| s.table_mapping),
            order: scan.as_ref().map_or([0; 4], |s| s.order),
            start: scan.map_or(0, |s| s.data_start),
            state: None,
            restart: self.restart_interval as usize,
            next_restart: 0,
            mcus: 0,
            columns,
            strip_height: max_v * 8,
            strips: height.div_ceil(max_v * 8),
            loaded: 0,
            width,
            height,
            fancy: fancy != ChromaUpsampling::NearestNeighbor,
            precision: self.precision,
            row: try_alloc_vec(width * n, "component row")?,
            scratch: try_alloc_vec(width, "component interpolation")?,
            current_row: 0,
        })
    }
}

impl ComponentRows {
    fn load(&mut self, data: &[u8], stop: &dyn enough::Stop) -> Result<()> {
        let mut entropy = EntropyDecoder::new(&data[self.start..]);
        for i in 0..MAX_HUFFMAN_TABLES {
            if let Some(t) = &self.dc[i] {
                entropy.set_dc_table(i, t);
            }
            if let Some(t) = &self.ac[i] {
                entropy.set_ac_table(i, t);
            }
        }
        if let Some(s) = self.state {
            entropy.restore_state(s);
        }
        let mut block = [0i16; 64];
        let mut dequant = [0i32; 64];
        let mut pixels = [0i16; 64];
        for x in 0..self.columns {
            stop.check()?;
            if self.coefficients.is_none()
                && self.restart != 0
                && self.mcus != 0
                && self.mcus % self.restart == 0
            {
                entropy.align_to_byte();
                let _ = entropy.read_restart_marker_tolerant(self.next_restart)?;
                self.next_restart = (self.next_restart + 1) & 7;
                entropy.reset_dc();
            }
            for index in 0..self.planes.len() {
                let i = if self.coefficients.is_some() {
                    index
                } else {
                    self.order[index]
                };
                let plane = &mut self.planes[i];
                for by in 0..plane.v {
                    for bx in 0..plane.h {
                        if let Some(coeffs) = &self.coefficients {
                            let comp = &coeffs.components[i];
                            block.copy_from_slice(comp.block(
                                (self.loaded * plane.v + by) * comp.blocks_wide + x * plane.h + bx,
                            ));
                        } else {
                            block.fill(0);
                            let (dc, ac) = self.mapping[i];
                            if !matches!(
                                entropy.decode_block_into(&mut block, 64, i, dc, ac)?,
                                ScanRead::Value(_)
                            ) {
                                return Err(Error::invalid_jpeg_data("truncated component scan"));
                            }
                        }
                        crate::quant::dequantize_unzigzag_i32_into_partial(
                            &block,
                            &plane.quant,
                            &mut dequant,
                            64,
                        );
                        if crate::decode::precision::is_12_bit(self.precision) {
                            crate::decode::idct_int::idct_libjpeg_precision::<12>(
                                &dequant,
                                &mut pixels,
                                8,
                            );
                        } else {
                            crate::decode::idct_int::idct_libjpeg_precision::<8>(
                                &dequant,
                                &mut pixels,
                                8,
                            );
                        }
                        let base = (self.loaded % 3 * plane.strip_rows + by * 8) * plane.stride
                            + (x * plane.h + bx) * 8;
                        for row in 0..8 {
                            plane.samples[base + row * plane.stride..base + row * plane.stride + 8]
                                .copy_from_slice(&pixels[row * 8..row * 8 + 8]);
                        }
                    }
                }
            }
            self.mcus += 1;
        }
        self.state = Some(entropy.save_state());
        self.loaded += 1;
        Ok(())
    }

    pub fn next_row(&mut self, data: &[u8], stop: &dyn enough::Stop) -> Result<()> {
        if self.current_row >= self.height {
            return Err(Error::invalid_jpeg_data("component row past image"));
        }
        let y = self.current_row;
        let needed = (y / self.strip_height + 2).min(self.strips);
        while self.loaded < needed {
            self.load(data, stop)?;
        }
        let max_h = self.planes.iter().map(|p| p.h).max().unwrap();
        let max_v = self.planes.iter().map(|p| p.v).max().unwrap();
        let channels = self.planes.len();
        for (i, p) in self.planes.iter().enumerate() {
            let (hr, vr) = (max_h / p.h, max_v / p.v);
            let yy = (y / vr).min(p.height - 1);
            let row = |r: usize| {
                let start = (r % (3 * p.strip_rows)) * p.stride;
                &p.samples[start..start + p.width]
            };
            let near = row(yy);
            let fancy = self.fancy && (hr == 1 || (hr == 2 && p.width > 2)) && vr <= 2;
            if fancy && vr == 2 {
                let far_y = if y % 2 == 0 {
                    yy.saturating_sub(1)
                } else {
                    (yy + 1).min(p.height - 1)
                };
                let far = row(far_y);
                if hr == 2 {
                    crate::decode::upsample::upsample_h2v2_libjpeg_row_scalar(
                        near,
                        far,
                        &mut self.scratch,
                        p.width,
                        self.width,
                        crate::decode::upsample::H2v2Bias::Turbo,
                    );
                } else {
                    crate::decode::upsample::upsample_h1v2_row_ext(
                        near,
                        far,
                        &mut self.scratch,
                        self.width,
                        if y % 2 == 0 { 1 } else { 2 },
                    );
                }
            } else {
                for x in 0..self.width {
                    let xx = (x / hr).min(p.width - 1);
                    self.scratch[x] = if fancy && hr == 2 {
                        let other = if x % 2 == 0 {
                            xx.saturating_sub(1)
                        } else {
                            (xx + 1).min(p.width - 1)
                        };
                        ((3 * near[xx] as i32
                            + near[other] as i32
                            + if x % 2 == 0 { 1 } else { 2 })
                            >> 2) as i16
                    } else {
                        near[xx]
                    };
                }
            }
            for x in 0..self.width {
                self.row[x * channels + i] = self.scratch[x] as u16;
            }
        }
        self.current_row += 1;
        Ok(())
    }
}
