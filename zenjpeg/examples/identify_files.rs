//! Identify the quantiser behind each JPEG given on the command line.
use zenjpeg::quant::identify::{IJG_PRESET_NAMES, TableId, identify_luma_table};
fn dqt(data: &[u8]) -> Option<[u16; 64]> {
    const ZZ: [usize; 64] = [
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27,
        20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    ];
    let mut i = 2usize;
    while i + 3 < data.len() {
        if data[i] != 0xFF {
            i += 1;
            continue;
        }
        let m = data[i + 1];
        if m == 0xD8 || m == 0x01 || (0xD0..=0xD7).contains(&m) {
            i += 2;
            continue;
        }
        if m == 0xD9 {
            break;
        }
        let len = ((data[i + 2] as usize) << 8) | data[i + 3] as usize;
        if i + 2 + len > data.len() {
            break;
        }
        if m == 0xDB {
            let seg = &data[i + 4..i + 2 + len];
            if seg[0] == 0 && seg.len() >= 65 {
                let mut o = [0u16; 64];
                for k in 0..64 {
                    o[ZZ[k]] = u16::from(seg[1 + k]);
                }
                return Some(o);
            }
        }
        if m == 0xDA {
            break;
        }
        i += 2 + len;
    }
    None
}
fn main() {
    for f in std::env::args().skip(1) {
        let Ok(d) = std::fs::read(&f) else { continue };
        let Some(t) = dqt(&d) else {
            println!("{f}\tno luma DQT");
            continue;
        };
        let name = f.rsplit('/').next().unwrap_or(&f).to_string();
        match identify_luma_table(&t, 1) {
            TableId::IjgPreset {
                preset,
                quality,
                exact,
            } => println!(
                "{name}\t{} q{quality}{}",
                IJG_PRESET_NAMES[preset as usize],
                if exact { "" } else { " (tol)" }
            ),
            TableId::JpegliDistance { distance, exact } => println!(
                "{name}\tjpegli d={distance:.3}{}", if exact { "" } else { " (±1)" }),
            TableId::Unknown => println!("{name}\tUNKNOWN"),
            _ => println!("{name}\t?"),
        }
    }
}
