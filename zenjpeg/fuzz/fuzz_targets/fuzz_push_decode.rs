//! Fuzz target for the push/streaming decode paths.
//!
//! Exercises decode_rows (u8) and decode_rows_f32 (f32) callbacks,
//! which use the ScanlineReader internally but manage buffers for the caller.
#![no_main]

use libfuzzer_sys::fuzz_target;
use zenjpeg::decoder::{Decoder, PixelFormat};

fuzz_target!(|data: &[u8]| {
    let max_px = 4_000_000u64;

    if let Ok(mut reader) = Decoder::new().max_pixels(max_px).scanline_reader(data) {
        let width = reader.width() as usize;
        let stride = width + 3;
        let mut row = vec![rgb::RGB::new(0u16, 0, 0); stride];
        while !reader.is_finished() {
            match reader.read_rows_rgb16(imgref::ImgRefMut::new_stride(&mut row, width, 1, stride)) {
                Ok(0) | Err(_) => break,
                Ok(_) => (),
            }
        }
    }

    if let Ok(mut reader) = Decoder::new().max_pixels(max_px).max_memory(64 * 1024 * 1024).scanline_reader(data) {
        match reader.component_ids().len() {
            1 => components::<1>(&mut reader),
            2 => components::<2>(&mut reader),
            3 => components::<3>(&mut reader),
            4 => components::<4>(&mut reader),
            5 => components::<5>(&mut reader),
            6 => components::<6>(&mut reader),
            7 => components::<7>(&mut reader),
            8 => components::<8>(&mut reader),
            9 => components::<9>(&mut reader),
            10 => components::<10>(&mut reader),
            _ => unreachable!("validated component count"),
        }
    }

    // u8 row callbacks — all formats
    for format in [
        PixelFormat::Rgb, PixelFormat::Bgr,
        PixelFormat::Rgba, PixelFormat::Bgra,
        PixelFormat::Gray,
    ] {
        let _ = Decoder::new().max_pixels(max_px)
            .decode_rows(data, format, |_row| Ok(()), enough::Unstoppable);
    }

    // f32 row callbacks
    for format in [PixelFormat::RgbaF32, PixelFormat::GrayF32] {
        let _ = Decoder::new().max_pixels(max_px)
            .decode_rows_f32(data, format, |_row| Ok(()), enough::Unstoppable);
    }
});

fn components<const N: usize>(reader: &mut zenjpeg::decoder::ScanlineReader<'_>) {
    let width=reader.width() as usize;
    let mut row=vec![[0u16;N];width+3];
    while !reader.is_finished() {
        match reader.read_rows_components16(imgref::ImgRefMut::new_stride(&mut row,width,1,width+3)) {
            Ok(0) | Err(_) => break,
            Ok(_) => (),
        }
    }
}
