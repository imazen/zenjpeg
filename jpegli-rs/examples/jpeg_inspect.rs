//! JPEG bitstream inspection and validation tool.
//!
//! Analyzes JPEG file structure: markers, segments, Huffman tables,
//! quantization tables, scan data, and more. Can also validate JPEGs
//! with multiple decoders.
//!
//! Usage:
//!   cargo run --release --example jpeg_inspect -- [OPTIONS] <file.jpg>
//!
//! Options:
//!   --markers       Show all markers and their positions
//!   --huffman       Dump Huffman table details
//!   --quant         Dump quantization tables
//!   --scans         Analyze scan structure (progressive)
//!   --validate      Test decoding with multiple decoders
//!   --compare <f2>  Compare with another JPEG
//!   --all           Show everything (includes validation)
//!
//! Examples:
//!   # Quick overview of JPEG structure
//!   cargo run --release --example jpeg_inspect -- image.jpg
//!
//!   # Validate JPEG with multiple decoders
//!   cargo run --release --example jpeg_inspect -- --validate image.jpg
//!
//!   # Compare two JPEGs
//!   cargo run --release --example jpeg_inspect -- --compare other.jpg image.jpg
//!
//!   # Detailed Huffman analysis
//!   cargo run --release --example jpeg_inspect -- --huffman image.jpg

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
struct JpegMarker {
    offset: usize,
    marker: u8,
    length: Option<u16>,
    name: &'static str,
}

#[derive(Debug)]
struct JpegAnalysis {
    markers: Vec<JpegMarker>,
    quant_tables: Vec<QuantTable>,
    huffman_tables: Vec<HuffmanTable>,
    scans: Vec<ScanInfo>,
    width: u16,
    height: u16,
    components: u8,
    is_progressive: bool,
}

#[derive(Debug)]
struct QuantTable {
    id: u8,
    precision: u8, // 0 = 8-bit, 1 = 16-bit
    values: [u16; 64],
}

#[derive(Debug)]
struct HuffmanTable {
    class: u8, // 0 = DC, 1 = AC
    id: u8,
    bits: [u8; 16],
    values: Vec<u8>,
    code_count: usize,
}

#[derive(Debug)]
struct ScanInfo {
    offset: usize,
    components: Vec<ScanComponent>,
    ss: u8, // Spectral selection start
    se: u8, // Spectral selection end
    ah: u8, // Successive approximation high
    al: u8, // Successive approximation low
    data_length: usize,
}

#[derive(Debug)]
struct ScanComponent {
    id: u8,
    dc_table: u8,
    ac_table: u8,
}

fn marker_name(marker: u8) -> &'static str {
    match marker {
        0xD8 => "SOI (Start of Image)",
        0xD9 => "EOI (End of Image)",
        0xE0 => "APP0 (JFIF)",
        0xE1 => "APP1 (EXIF)",
        0xE2 => "APP2 (ICC Profile)",
        0xE3..=0xEF => "APPn",
        0xDB => "DQT (Quant Tables)",
        0xC0 => "SOF0 (Baseline DCT)",
        0xC1 => "SOF1 (Extended Sequential)",
        0xC2 => "SOF2 (Progressive DCT)",
        0xC3 => "SOF3 (Lossless)",
        0xC4 => "DHT (Huffman Tables)",
        0xDA => "SOS (Start of Scan)",
        0xDD => "DRI (Restart Interval)",
        0xD0..=0xD7 => "RSTn (Restart)",
        0xFE => "COM (Comment)",
        0xDC => "DNL (Define Number of Lines)",
        0xDE => "DHP (Define Hierarchical Progression)",
        _ => "Unknown",
    }
}

fn analyze_jpeg(data: &[u8]) -> Result<JpegAnalysis, String> {
    let mut markers = Vec::new();
    let mut quant_tables = Vec::new();
    let mut huffman_tables = Vec::new();
    let mut scans = Vec::new();
    let mut width = 0u16;
    let mut height = 0u16;
    let mut components = 0u8;
    let mut is_progressive = false;

    let mut pos = 0;

    while pos < data.len() - 1 {
        if data[pos] != 0xFF {
            pos += 1;
            continue;
        }

        let marker = data[pos + 1];

        // Skip fill bytes
        if marker == 0xFF || marker == 0x00 {
            pos += 1;
            continue;
        }

        let name = marker_name(marker);
        let mut length = None;

        // Markers with length field
        if marker != 0xD8 && marker != 0xD9 && !(0xD0..=0xD7).contains(&marker) {
            if pos + 3 < data.len() {
                length = Some(u16::from_be_bytes([data[pos + 2], data[pos + 3]]));
            }
        }

        markers.push(JpegMarker {
            offset: pos,
            marker,
            length,
            name,
        });

        match marker {
            0xDB => {
                // DQT - Quantization tables
                if let Some(len) = length {
                    parse_dqt(&data[pos + 4..pos + 2 + len as usize], &mut quant_tables);
                }
            }
            0xC4 => {
                // DHT - Huffman tables
                if let Some(len) = length {
                    parse_dht(&data[pos + 4..pos + 2 + len as usize], &mut huffman_tables);
                }
            }
            0xC0 | 0xC1 | 0xC2 | 0xC3 => {
                // SOF - Start of Frame
                is_progressive = marker == 0xC2;
                if pos + 9 < data.len() {
                    height = u16::from_be_bytes([data[pos + 5], data[pos + 6]]);
                    width = u16::from_be_bytes([data[pos + 7], data[pos + 8]]);
                    components = data[pos + 9];
                }
            }
            0xDA => {
                // SOS - Start of Scan
                if let Some(len) = length {
                    let scan = parse_sos(&data[pos..], len as usize);
                    scans.push(scan);
                }
            }
            _ => {}
        }

        // Advance position
        if let Some(len) = length {
            pos += 2 + len as usize;
        } else {
            pos += 2;
        }
    }

    Ok(JpegAnalysis {
        markers,
        quant_tables,
        huffman_tables,
        scans,
        width,
        height,
        components,
        is_progressive,
    })
}

fn parse_dqt(data: &[u8], tables: &mut Vec<QuantTable>) {
    let mut pos = 0;
    while pos < data.len() {
        let pq = (data[pos] >> 4) & 0x0F; // Precision
        let tq = data[pos] & 0x0F; // Table ID
        pos += 1;

        let mut values = [0u16; 64];
        for i in 0..64 {
            if pq == 0 {
                values[i] = data[pos] as u16;
                pos += 1;
            } else {
                values[i] = u16::from_be_bytes([data[pos], data[pos + 1]]);
                pos += 2;
            }
        }

        tables.push(QuantTable {
            id: tq,
            precision: pq,
            values,
        });
    }
}

fn parse_dht(data: &[u8], tables: &mut Vec<HuffmanTable>) {
    let mut pos = 0;
    while pos < data.len() {
        let tc = (data[pos] >> 4) & 0x0F; // Table class (0=DC, 1=AC)
        let th = data[pos] & 0x0F; // Table ID
        pos += 1;

        let mut bits = [0u8; 16];
        let mut total_codes = 0usize;
        for i in 0..16 {
            bits[i] = data[pos + i];
            total_codes += bits[i] as usize;
        }
        pos += 16;

        let values: Vec<u8> = data[pos..pos + total_codes].to_vec();
        pos += total_codes;

        tables.push(HuffmanTable {
            class: tc,
            id: th,
            bits,
            values,
            code_count: total_codes,
        });
    }
}

fn parse_sos(data: &[u8], header_len: usize) -> ScanInfo {
    let offset = 0;
    let ns = data[4] as usize; // Number of components

    let mut components = Vec::new();
    for i in 0..ns {
        let idx = 5 + i * 2;
        components.push(ScanComponent {
            id: data[idx],
            dc_table: (data[idx + 1] >> 4) & 0x0F,
            ac_table: data[idx + 1] & 0x0F,
        });
    }

    let spec_idx = 5 + ns * 2;
    let ss = data[spec_idx];
    let se = data[spec_idx + 1];
    let ah = (data[spec_idx + 2] >> 4) & 0x0F;
    let al = data[spec_idx + 2] & 0x0F;

    // Find end of scan data (next marker or EOI)
    let scan_start = 2 + header_len;
    let mut scan_end = scan_start;
    while scan_end < data.len() - 1 {
        if data[scan_end] == 0xFF && data[scan_end + 1] != 0x00 && data[scan_end + 1] != 0xFF {
            break;
        }
        scan_end += 1;
    }

    ScanInfo {
        offset,
        components,
        ss,
        se,
        ah,
        al,
        data_length: scan_end - scan_start,
    }
}

fn print_markers(analysis: &JpegAnalysis) {
    println!("\n=== JPEG Markers ===");
    println!(
        "{:>8}  {:>4}  {:>6}  {}",
        "Offset", "0xFF", "Length", "Description"
    );
    println!("{}", "-".repeat(60));

    for m in &analysis.markers {
        let len_str = m.length.map(|l| format!("{}", l)).unwrap_or_default();
        println!(
            "{:>8}  0x{:02X}  {:>6}  {}",
            m.offset, m.marker, len_str, m.name
        );
    }
}

fn print_quant_tables(analysis: &JpegAnalysis) {
    println!("\n=== Quantization Tables ===");

    for qt in &analysis.quant_tables {
        println!(
            "\nTable {} ({}bit):",
            qt.id,
            if qt.precision == 0 { 8 } else { 16 }
        );

        // Print as 8x8 matrix
        for row in 0..8 {
            print!("  ");
            for col in 0..8 {
                print!("{:4}", qt.values[row * 8 + col]);
            }
            println!();
        }

        // Statistics
        let min = qt.values.iter().min().unwrap();
        let max = qt.values.iter().max().unwrap();
        let sum: u32 = qt.values.iter().map(|&v| v as u32).sum();
        let avg = sum as f64 / 64.0;
        println!("  Min: {}, Max: {}, Avg: {:.1}", min, max, avg);
    }
}

fn print_huffman_tables(analysis: &JpegAnalysis) {
    println!("\n=== Huffman Tables ===");

    for ht in &analysis.huffman_tables {
        let class = if ht.class == 0 { "DC" } else { "AC" };
        println!("\n{} Table {} ({} codes):", class, ht.id, ht.code_count);

        println!("  Bit lengths: {:?}", &ht.bits[..]);

        // Show code distribution
        let mut code_count_by_len = [0u32; 16];
        for (i, &count) in ht.bits.iter().enumerate() {
            code_count_by_len[i] = count as u32;
        }

        print!("  Codes per length: ");
        for (i, &count) in code_count_by_len.iter().enumerate() {
            if count > 0 {
                print!("{}:{} ", i + 1, count);
            }
        }
        println!();

        // For AC tables, show run/size distribution
        if ht.class == 1 {
            let mut run_size_counts: HashMap<(u8, u8), usize> = HashMap::new();
            for &val in &ht.values {
                let run = val >> 4;
                let size = val & 0x0F;
                *run_size_counts.entry((run, size)).or_default() += 1;
            }

            println!("  Run/Size pairs: {} unique", run_size_counts.len());
        }
    }
}

fn print_scans(analysis: &JpegAnalysis) {
    println!("\n=== Scan Structure ===");
    println!("Progressive: {}", analysis.is_progressive);
    println!("Total scans: {}", analysis.scans.len());

    for (i, scan) in analysis.scans.iter().enumerate() {
        println!("\nScan {}:", i + 1);
        println!(
            "  Components: {:?}",
            scan.components
                .iter()
                .map(|c| format!("{}(DC:{},AC:{})", c.id, c.dc_table, c.ac_table))
                .collect::<Vec<_>>()
        );
        println!("  Spectral: {} - {}", scan.ss, scan.se);
        println!("  Successive approx: high={}, low={}", scan.ah, scan.al);
        println!("  Data length: {} bytes", scan.data_length);

        // Describe scan type
        let scan_type = if scan.ss == 0 && scan.se == 0 {
            "DC only"
        } else if scan.ss == 0 {
            "DC + AC"
        } else if scan.ah == 0 {
            "AC first"
        } else {
            "AC refine"
        };
        println!("  Type: {}", scan_type);
    }
}

fn print_summary(analysis: &JpegAnalysis, path: &str) {
    println!("=== JPEG Summary: {} ===", path);
    println!("  Dimensions: {}x{}", analysis.width, analysis.height);
    println!("  Components: {}", analysis.components);
    println!("  Progressive: {}", analysis.is_progressive);
    println!("  Markers: {}", analysis.markers.len());
    println!("  Quant tables: {}", analysis.quant_tables.len());
    println!("  Huffman tables: {}", analysis.huffman_tables.len());
    println!("  Scans: {}", analysis.scans.len());
}

fn validate_jpeg(data: &[u8], path: &str) {
    println!("\n=== Decoder Validation ===");
    println!("File: {} ({} bytes)", path, data.len());

    // Test with zune-jpeg
    print!("  zune-jpeg:  ");
    match zune_jpeg::JpegDecoder::new(std::io::Cursor::new(data)).decode() {
        Ok(pixels) => println!("OK ({} bytes decoded)", pixels.len()),
        Err(e) => println!("ERROR: {:?}", e),
    }

    // Test with jpegli-rs decoder
    print!("  jpegli-rs:  ");
    match jpegli::Decoder::new().decode(data) {
        Ok(img) => println!(
            "OK ({}x{}, {} bytes)",
            img.width,
            img.height,
            img.data.len()
        ),
        Err(e) => println!("ERROR: {}", e),
    }

}

fn validate_directory(dir: &Path) {
    let mut files: Vec<_> = fs::read_dir(dir)
        .expect("Failed to read directory")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| {
                    let ext = ext.to_ascii_lowercase();
                    ext == "jpg" || ext == "jpeg"
                })
                .unwrap_or(false)
        })
        .collect();
    files.sort_by_key(|e| e.path());

    println!("=== Validating {} JPEG files ===\n", files.len());

    for entry in &files {
        let path = entry.path();
        let data = match fs::read(&path) {
            Ok(d) => d,
            Err(e) => {
                println!("{}: Read error: {}\n", path.display(), e);
                continue;
            }
        };
        validate_jpeg(&data, &path.display().to_string());
        println!();
    }
}

fn compare_jpegs(path1: &str, path2: &str) -> Result<(), String> {
    let data1 = fs::read(path1).map_err(|e| format!("Failed to read {}: {}", path1, e))?;
    let data2 = fs::read(path2).map_err(|e| format!("Failed to read {}: {}", path2, e))?;

    let analysis1 = analyze_jpeg(&data1)?;
    let analysis2 = analyze_jpeg(&data2)?;

    println!("=== Comparing JPEGs ===\n");
    println!("{:30} {:>15} {:>15}", "", path1, path2);
    println!("{}", "-".repeat(60));
    println!(
        "{:30} {:>15} {:>15}",
        "Size (bytes)",
        data1.len(),
        data2.len()
    );
    println!(
        "{:30} {:>15} {:>15}",
        "Dimensions",
        format!("{}x{}", analysis1.width, analysis1.height),
        format!("{}x{}", analysis2.width, analysis2.height)
    );
    println!(
        "{:30} {:>15} {:>15}",
        "Progressive", analysis1.is_progressive, analysis2.is_progressive
    );
    println!(
        "{:30} {:>15} {:>15}",
        "Markers",
        analysis1.markers.len(),
        analysis2.markers.len()
    );
    println!(
        "{:30} {:>15} {:>15}",
        "Quant tables",
        analysis1.quant_tables.len(),
        analysis2.quant_tables.len()
    );
    println!(
        "{:30} {:>15} {:>15}",
        "Huffman tables",
        analysis1.huffman_tables.len(),
        analysis2.huffman_tables.len()
    );
    println!(
        "{:30} {:>15} {:>15}",
        "Scans",
        analysis1.scans.len(),
        analysis2.scans.len()
    );

    // Compare quant tables
    if analysis1.quant_tables.len() == analysis2.quant_tables.len() {
        println!("\n--- Quantization Table Comparison ---");
        for (qt1, qt2) in analysis1.quant_tables.iter().zip(&analysis2.quant_tables) {
            let diff: i32 = qt1
                .values
                .iter()
                .zip(&qt2.values)
                .map(|(&a, &b)| (a as i32 - b as i32).abs())
                .sum();
            let max_diff = qt1
                .values
                .iter()
                .zip(&qt2.values)
                .map(|(&a, &b)| (a as i32 - b as i32).abs())
                .max()
                .unwrap_or(0);
            println!(
                "Table {}: total diff={}, max diff={}",
                qt1.id, diff, max_diff
            );
        }
    }

    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut show_markers = false;
    let mut show_huffman = false;
    let mut show_quant = false;
    let mut show_scans = false;
    let mut show_validate = false;
    let mut show_all = false;
    let mut compare_file: Option<String> = None;
    let mut jpeg_path: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--markers" => show_markers = true,
            "--huffman" => show_huffman = true,
            "--quant" => show_quant = true,
            "--scans" => show_scans = true,
            "--validate" => show_validate = true,
            "--all" => show_all = true,
            "--compare" => {
                i += 1;
                if i < args.len() {
                    compare_file = Some(args[i].clone());
                }
            }
            arg if !arg.starts_with('-') => {
                jpeg_path = Some(arg.to_string());
            }
            _ => eprintln!("Unknown argument: {}", args[i]),
        }
        i += 1;
    }

    let jpeg_path = match jpeg_path {
        Some(p) => p,
        None => {
            eprintln!("Usage: jpeg_inspect [OPTIONS] <file.jpg|dir>");
            eprintln!("  --markers   Show all markers");
            eprintln!("  --huffman   Dump Huffman tables");
            eprintln!("  --quant     Dump quantization tables");
            eprintln!("  --scans     Analyze progressive scans");
            eprintln!("  --validate  Test with multiple decoders");
            eprintln!("  --compare   Compare with another JPEG");
            eprintln!("  --all       Show everything (includes validation)");
            std::process::exit(1);
        }
    };

    // Handle comparison mode
    if let Some(other) = compare_file {
        if let Err(e) = compare_jpegs(&jpeg_path, &other) {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
        return;
    }

    // Handle directory for validation mode
    let path = Path::new(&jpeg_path);
    if path.is_dir() {
        if !show_validate && !show_all {
            eprintln!("Directory mode only supported with --validate or --all");
            std::process::exit(1);
        }
        validate_directory(path);
        return;
    }

    // Normal analysis
    let data = match fs::read(&jpeg_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to read {}: {}", jpeg_path, e);
            std::process::exit(1);
        }
    };

    let analysis = match analyze_jpeg(&data) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Failed to analyze JPEG: {}", e);
            std::process::exit(1);
        }
    };

    // Default: show summary
    if !show_markers && !show_huffman && !show_quant && !show_scans && !show_validate && !show_all {
        print_summary(&analysis, &jpeg_path);
        println!("\nUse --markers, --huffman, --quant, --scans, --validate, or --all for details");
        return;
    }

    print_summary(&analysis, &jpeg_path);

    if show_all || show_markers {
        print_markers(&analysis);
    }

    if show_all || show_quant {
        print_quant_tables(&analysis);
    }

    if show_all || show_huffman {
        print_huffman_tables(&analysis);
    }

    if show_all || show_scans {
        print_scans(&analysis);
    }

    if show_all || show_validate {
        validate_jpeg(&data, &jpeg_path);
    }
}
