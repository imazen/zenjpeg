// Compare progressive JPEG structure between jpegli-rs and mozjpeg reference
use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};

fn parse_scans(jpeg_data: &[u8]) -> Vec<ScanInfo> {
    let mut scans = Vec::new();
    let mut i = 0;
    while i < jpeg_data.len() - 1 {
        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xDA {
            let len = ((jpeg_data[i + 2] as usize) << 8) | (jpeg_data[i + 3] as usize);
            let num_comp = jpeg_data[i + 4];

            let mut comp_info = Vec::new();
            for c in 0..num_comp as usize {
                let comp_id = jpeg_data[i + 5 + c * 2];
                let table_sel = jpeg_data[i + 6 + c * 2];
                comp_info.push((comp_id, table_sel >> 4, table_sel & 0xF));
            }

            let base = i + 5 + num_comp as usize * 2;
            let ss = jpeg_data[base];
            let se = jpeg_data[base + 1];
            let ah_al = jpeg_data[base + 2];

            let scan_start = i + 2 + len;
            let mut scan_end = scan_start;
            while scan_end < jpeg_data.len() - 1 {
                if jpeg_data[scan_end] == 0xFF && jpeg_data[scan_end + 1] != 0x00 {
                    break;
                }
                scan_end += 1;
            }

            scans.push(ScanInfo {
                components: comp_info,
                ss,
                se,
                ah: ah_al >> 4,
                al: ah_al & 0xF,
                data: jpeg_data[scan_start..scan_end].to_vec(),
            });
            i = scan_end;
        } else {
            i += 1;
        }
    }
    scans
}

struct ScanInfo {
    components: Vec<(u8, u8, u8)>, // (id, dc_table, ac_table)
    ss: u8,
    se: u8,
    ah: u8,
    al: u8,
    data: Vec<u8>,
}

impl ScanInfo {
    fn scan_type(&self) -> &'static str {
        if self.ss == 0 && self.se == 0 {
            if self.ah > 0 {
                "DC refine"
            } else {
                "DC first"
            }
        } else if self.ah > 0 {
            "AC refine"
        } else {
            "AC first"
        }
    }

    fn print_detailed(&self, idx: usize) {
        println!(
            "  Scan {}: comp={:?} Ss={} Se={} Ah={} Al={} [{}] data={}B",
            idx,
            self.components,
            self.ss,
            self.se,
            self.ah,
            self.al,
            self.scan_type(),
            self.data.len()
        );
        if self.data.len() <= 16 {
            let hex: String = self.data.iter().map(|b| format!("{:02X} ", b)).collect();
            let bits: String = self
                .data
                .iter()
                .flat_map(|b| {
                    (0..8)
                        .rev()
                        .map(move |bit| if (b >> bit) & 1 == 1 { '1' } else { '0' })
                })
                .collect();
            println!("       Hex: {}", hex.trim());
            println!("       Bits: {}", bits);
        }
    }
}

fn encode_mozjpeg(rgb: &[u8], width: usize, height: usize) -> Vec<u8> {
    use mozjpeg::{ColorSpace, Compress, ScanMode};

    let mut compress = Compress::new(ColorSpace::JCS_RGB);
    compress.set_size(width, height);
    compress.set_quality(90.0);
    compress.set_scan_optimization_mode(ScanMode::AllComponentsTogether);
    compress.set_progressive_mode();
    compress.set_optimize_scans(false);

    let mut comp = compress.start_compress(Vec::new()).unwrap();
    comp.write_scanlines(rgb).unwrap();
    comp.finish().unwrap()
}

fn main() {
    // Create the same 8x8 RGB gradient that fails
    let rgb_grad: Vec<u8> = (0..64).flat_map(|i| vec![i as u8 * 4, 128, 64]).collect();

    println!("=== MOZJPEG Reference ===");
    let moz_jpeg = encode_mozjpeg(&rgb_grad, 8, 8);
    std::fs::write("/tmp/mozjpeg_prog_ref.jpg", &moz_jpeg).ok();
    println!("Total size: {} bytes", moz_jpeg.len());

    let moz_scans = parse_scans(&moz_jpeg);
    for (i, scan) in moz_scans.iter().enumerate() {
        scan.print_detailed(i);
    }

    // Try decoding mozjpeg output
    match jpeg_decoder::Decoder::new(&moz_jpeg[..]).decode() {
        Ok(_) => println!("Decode: OK\n"),
        Err(e) => println!("Decode: FAILED - {:?}\n", e),
    }

    println!("=== JPEGLI-RS Output ===");
    let encoder = Encoder::new()
        .width(8)
        .height(8)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(90.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive);

    let jpegli_jpeg = encoder.encode(&rgb_grad).unwrap();
    std::fs::write("/tmp/jpegli_prog.jpg", &jpegli_jpeg).ok();
    println!("Total size: {} bytes", jpegli_jpeg.len());

    let jpegli_scans = parse_scans(&jpegli_jpeg);
    for (i, scan) in jpegli_scans.iter().enumerate() {
        scan.print_detailed(i);
    }

    // Try decoding jpegli output
    match jpeg_decoder::Decoder::new(&jpegli_jpeg[..]).decode() {
        Ok(_) => println!("Decode: OK"),
        Err(e) => println!("Decode: FAILED - {:?}", e),
    }

    // Compare refinement scans directly
    println!("\n=== REFINEMENT SCAN COMPARISON ===");
    let moz_refines: Vec<_> = moz_scans.iter().filter(|s| s.ah > 0).collect();
    let jpegli_refines: Vec<_> = jpegli_scans.iter().filter(|s| s.ah > 0).collect();

    println!("mozjpeg refinement scans: {}", moz_refines.len());
    println!("jpegli refinement scans: {}", jpegli_refines.len());

    for (mi, ji) in moz_refines.iter().zip(jpegli_refines.iter()) {
        if mi.ss == ji.ss && mi.se == ji.se && mi.ah == ji.ah && mi.al == ji.al {
            println!(
                "\nComparing Ss={} Se={} Ah={} Al={}:",
                mi.ss, mi.se, mi.ah, mi.al
            );
            println!("  mozjpeg: {} bytes", mi.data.len());
            println!("  jpegli:  {} bytes", ji.data.len());
            if mi.data == ji.data {
                println!("  Data: MATCH");
            } else {
                println!("  Data: DIFFER");
                // Show first difference
                for (pos, (a, b)) in mi.data.iter().zip(ji.data.iter()).enumerate() {
                    if a != b {
                        println!(
                            "  First diff at byte {}: moz={:02X} jpegli={:02X}",
                            pos, a, b
                        );
                        break;
                    }
                }
            }
        }
    }
}
