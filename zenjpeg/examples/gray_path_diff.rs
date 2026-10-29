//! zenjpeg#154 bisect: decode a 1-component (grayscale) JPEG with Gray
//! output (buffered coefficient path — libjpeg-turbo-exact per the issue's
//! reference arbitration) and with Rgb output (streaming strip path), and
//! report where and how the Y values diverge.
//!
//! Usage: gray_path_diff <file.jpg>

use enough::Unstoppable;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: gray_path_diff <file.jpg>");
    let data = std::fs::read(&path).unwrap();

    let gray = zenjpeg::decoder::Decoder::new()
        .output_format(zenjpeg::decoder::PixelFormat::Gray)
        .auto_orient(false)
        .decode(&data, Unstoppable)
        .expect("gray decode");
    let rgb = zenjpeg::decoder::Decoder::new()
        .output_format(zenjpeg::decoder::PixelFormat::Rgb)
        .auto_orient(false)
        .decode(&data, Unstoppable)
        .expect("rgb decode");
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    assert_eq!((rgb.width(), rgb.height()), (gray.width(), gray.height()));
    let g = gray.pixels_u8().unwrap();
    let r = rgb.pixels_u8().unwrap();

    let mut n = 0usize;
    let mut maxd = 0u8;
    let mut first: Option<(usize, usize, u8, u8)> = None;
    let mut hist = [0usize; 3]; // diff of -1, +1, other (by r-g sign)
    for y in 0..h {
        for x in 0..w {
            let gv = g[y * w + x];
            let rv = r[(y * w + x) * 3];
            if gv != rv {
                n += 1;
                let d = gv.abs_diff(rv);
                maxd = maxd.max(d);
                if d == 1 {
                    if rv > gv {
                        hist[1] += 1; // streaming higher
                    } else {
                        hist[0] += 1; // streaming lower
                    }
                } else {
                    hist[2] += 1;
                }
                if first.is_none() {
                    first = Some((x, y, gv, rv));
                }
            }
        }
    }
    println!(
        "{}x{}: {n} bytes differ (max abs {maxd}); streaming-lower={} streaming-higher={} other={}",
        w, h, hist[0], hist[1], hist[2]
    );
    if let Some((x, y, gv, rv)) = first {
        println!(
            "first diff at ({x},{y}) block=({},{}) in-block=({},{}) gray(buffered)={gv} rgb(streaming)={rv}",
            x / 8,
            y / 8,
            x % 8,
            y % 8
        );
        // Dump the 8x8 block from both for inspection.
        let bx = (x / 8) * 8;
        let by = (y / 8) * 8;
        for row in by..(by + 8).min(h) {
            let gs: Vec<String> = (bx..(bx + 8).min(w))
                .map(|c| format!("{:3}", g[row * w + c]))
                .collect();
            let rs: Vec<String> = (bx..(bx + 8).min(w))
                .map(|c| format!("{:3}", r[(row * w + c) * 3]))
                .collect();
            println!("  buf[{row:4}] {}   strm {}", gs.join(" "), rs.join(" "));
        }
    }
}
