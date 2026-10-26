//! SOF3 prediction over bounded row buffers. Marker and entropy parsing remain
//! shared with DCT JPEG. Each scan retains independent entropy/predictor state,
//! so separate component scans can still be delivered a row at a time.

use super::JpegParser;
use crate::container::marker::{self, MarkerKind};
use crate::entropy::{EntropyDecoder, EntropyDecoderState};
use crate::error::{Error, Result, ScanRead};
use crate::foundation::alloc::try_alloc_vec;
use crate::huffman::HuffmanDecodeTable;
use enough::Stop;

pub(in crate::decode) struct PredictiveReader {
    scans: Vec<Scan>,
    pub row: Vec<u16>,
    width: usize,
    channels: usize,
    precision: u8,
    current_row: usize,
    height: usize,
}

struct Scan {
    start: usize,
    end: usize,
    components: Vec<(usize, usize)>,
    tables: [Option<HuffmanDecodeTable>; 4],
    state: Option<EntropyDecoderState>,
    previous: Vec<u16>,
    predictor: u8,
    point_transform: u8,
    restart_rows: usize,
    next_restart: u8,
}

impl JpegParser<'_> {
    pub(in crate::decode) fn collect_native_metadata(&mut self, stop: &dyn Stop) -> Result<()> {
        for span in marker::iter(self.data) {
            if span.offset < self.position {
                continue;
            }
            stop.check()?;
            self.position = span.offset + 2;
            match span.kind {
                MarkerKind::App(n) => self.process_app_or_com(0xe0 + n)?,
                MarkerKind::Com => self.process_app_or_com(0xfe)?,
                MarkerKind::Eoi => break,
                _ => (),
            }
        }
        self.extract_mpf_secondary_images()?;
        Ok(())
    }

    pub(in crate::decode) fn predictive_reader(
        &mut self,
        stop: &dyn Stop,
    ) -> Result<PredictiveReader> {
        let width = self.width as usize;
        let channels = self.num_components as usize;
        self.charge_memory(width * channels * 4, "lossless row buffers")?;
        if self.height == 0 {
            return Err(Error::unsupported_feature("lossless JPEG with DNL height"));
        }
        if self.components[..channels]
            .iter()
            .any(|c| c.h_samp_factor != 1 || c.v_samp_factor != 1)
        {
            return Err(Error::unsupported_feature("subsampled lossless JPEG"));
        }
        let mut scans = Vec::new();
        scans.try_reserve_exact(channels).map_err(|_| {
            Error::allocation_failed(channels * core::mem::size_of::<Scan>(), "lossless scans")
        })?;
        let mut seen = [false; crate::foundation::consts::MAX_COMPONENTS];
        let mut ended = false;
        // The canonical marker walker skips entropy byte stuffing and restarts.
        for span in marker::iter(self.data) {
            stop.check()?;
            if span.offset < self.position {
                continue;
            }
            self.position = span.offset + 2;
            match span.kind {
                MarkerKind::Dht => self.parse_huffman_table()?,
                MarkerKind::Dri => self.parse_restart_interval()?,
                MarkerKind::App(n) => self.process_app_or_com(0xe0 + n)?,
                MarkerKind::Com => self.process_app_or_com(0xfe)?,
                MarkerKind::Sos => {
                    let length = self.read_u16()? as usize;
                    let count = self.read_u8()? as usize;
                    if count == 0 || count > channels || count > 4 || length != 6 + 2 * count {
                        return Err(Error::invalid_jpeg_data(
                            "invalid lossless scan component count or length",
                        ));
                    }
                    let mut components = Vec::new();
                    components.try_reserve_exact(count).map_err(|_| {
                        Error::allocation_failed(
                            count * core::mem::size_of::<(usize, usize)>(),
                            "lossless scan components",
                        )
                    })?;
                    for _ in 0..count {
                        let id = self.read_u8()?;
                        let selector = self.read_u8()?;
                        let index = self.components[..channels]
                            .iter()
                            .position(|c| c.id == id)
                            .ok_or_else(|| {
                                Error::invalid_jpeg_data("unknown lossless scan component")
                            })?;
                        if seen[index] || selector & 15 != 0 || selector >> 4 >= 4 {
                            return Err(Error::invalid_jpeg_data(
                                "duplicate component or invalid lossless table selector",
                            ));
                        }
                        seen[index] = true;
                        let table = (selector >> 4) as usize;
                        if self.dc_tables[table].is_none() {
                            return Err(Error::invalid_jpeg_data("missing lossless Huffman table"));
                        }
                        components.push((index, table));
                    }
                    let predictor = self.read_u8()?;
                    let se = self.read_u8()?;
                    let ahal = self.read_u8()?;
                    if !(1..=7).contains(&predictor) || se != 0 || ahal >= self.precision {
                        return Err(Error::invalid_jpeg_data(
                            "invalid lossless predictor or point transform",
                        ));
                    }
                    if self.restart_interval as usize % width != 0 {
                        return Err(Error::invalid_jpeg_data(
                            "lossless restart interval is not row aligned",
                        ));
                    }
                    scans.push(Scan {
                        start: self.position,
                        end: span.offset + span.length,
                        components,
                        tables: self.dc_tables.clone(),
                        state: None,
                        previous: try_alloc_vec(width * count, "lossless previous row")?,
                        predictor,
                        point_transform: ahal,
                        restart_rows: self.restart_interval as usize / width,
                        next_restart: 0,
                    });
                    self.position = span.offset + span.length;
                }
                MarkerKind::Eoi => {
                    ended = true;
                    break;
                }
                MarkerKind::Sof(_) => {
                    return Err(Error::invalid_jpeg_data("multiple lossless frame headers"));
                }
                _ => self.skip_segment()?,
            }
        }
        if !ended || seen[..channels].contains(&false) {
            return Err(Error::invalid_jpeg_data("incomplete lossless SOF3 scans"));
        }
        Ok(PredictiveReader {
            scans,
            row: try_alloc_vec(width * channels, "lossless output row")?,
            width,
            channels,
            precision: self.precision,
            current_row: 0,
            height: self.height as usize,
        })
    }
}

impl PredictiveReader {
    pub(in crate::decode) fn current_row(&self) -> usize {
        self.current_row
    }
    pub(in crate::decode) fn next_row(&mut self, data: &[u8]) -> Result<()> {
        if self.current_row >= self.height {
            return Err(Error::invalid_jpeg_data("read past lossless image"));
        }
        for scan in &mut self.scans {
            let mut decoder = EntropyDecoder::new(&data[scan.start..scan.end]);
            for (i, table) in scan.tables.iter().enumerate() {
                if let Some(table) = table {
                    decoder.set_dc_table(i, table);
                }
            }
            if let Some(state) = scan.state {
                decoder.restore_state(state);
            }
            let restart = scan.restart_rows != 0 && self.current_row % scan.restart_rows == 0;
            if restart && self.current_row != 0 {
                decoder.align_to_byte();
                decoder.read_restart_marker(scan.next_restart)?;
                scan.next_restart = (scan.next_restart + 1) & 7;
            }
            let first = self.current_row == 0 || restart;
            let count = scan.components.len();
            let mut left = [0i32; 4];
            let mut upper_left = [0i32; 4];
            for x in 0..self.width {
                for (slot, &(component, table)) in scan.components.iter().enumerate() {
                    let above = scan.previous[x * count + slot] as i32;
                    let prediction = if first {
                        if x == 0 {
                            1 << (self.precision - scan.point_transform - 1)
                        } else {
                            left[slot]
                        }
                    } else if x == 0 {
                        above
                    } else {
                        predict(scan.predictor, left[slot], above, upper_left[slot])
                    };
                    let difference = match decoder.decode_difference::<true>(table)? {
                        ScanRead::Value(v) => v,
                        _ => return Err(Error::invalid_jpeg_data("truncated lossless difference")),
                    };
                    let sample = prediction.wrapping_add(difference) as u16;
                    scan.previous[x * count + slot] = sample;
                    upper_left[slot] = above;
                    left[slot] = sample as i32;
                    self.row[x * self.channels + component] =
                        sample.wrapping_shl(scan.point_transform as u32);
                }
            }
            scan.state = Some(decoder.save_state());
        }
        self.current_row += 1;
        Ok(())
    }
}

fn predict(selector: u8, a: i32, b: i32, c: i32) -> i32 {
    match selector {
        1 => a,
        2 => b,
        3 => c,
        4 => a + b - c,
        5 => a + ((b - c) >> 1),
        6 => b + ((a - c) >> 1),
        7 => (a + b) >> 1,
        _ => unreachable!("validated predictor"),
    }
}
