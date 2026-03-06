//! AC Refinement Scan C++ Comparison Test
//!
//! This test compares Rust AC refinement tokenization against C++ jpegli output.
//! The C++ output is generated using instrumentation in entropy_coding.cc with
//! the DUMP_AC_REFINEMENT environment variable.

use std::fs;
use std::path::Path;
use zenjpeg::encoder::ChromaSubsampling;
use zenjpeg::encoder::{EncoderConfig, PixelLayout};

fn encode_rgb(width: u32, height: u32, data: &[u8], config: &EncoderConfig) -> Vec<u8> {
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("create encoder");
    enc.push_packed(data, enough::Unstoppable)
        .expect("push data");
    enc.finish().expect("finish")
}

/// Parsed token from C++ output
#[derive(Debug, Clone, PartialEq, Eq)]
struct CppToken {
    symbol: u8,
    refbits: u8,
}

/// Parsed scan info from C++ output
#[derive(Debug, Clone)]
struct CppScanInfo {
    scan_index: u32,
    ss: u8,
    se: u8,
    ah: u8,
    al: u8,
    num_blocks: usize,
    num_tokens: usize,
    num_refbits: usize,
    num_eobruns: usize,
    tokens: Vec<CppToken>,
    refbits: Vec<u8>,
    eobruns: Vec<u16>,
}

/// Parses C++ AC refinement dump file
fn parse_cpp_dump(content: &str) -> Vec<CppScanInfo> {
    let mut scans = Vec::new();
    let mut current_scan: Option<CppScanInfo> = None;
    let mut in_tokens = false;
    let mut in_refbits = false;
    let mut in_eobruns = false;
    let mut refbit_str = String::new();

    for line in content.lines() {
        let line = line.trim();

        if line.starts_with("=== AC Refinement Scan") {
            // Parse scan index
            if let Some(idx_str) = line
                .strip_prefix("=== AC Refinement Scan ")
                .and_then(|s| s.strip_suffix(" ==="))
                && let Ok(idx) = idx_str.parse::<u32>()
            {
                current_scan = Some(CppScanInfo {
                    scan_index: idx,
                    ss: 0,
                    se: 0,
                    ah: 0,
                    al: 0,
                    num_blocks: 0,
                    num_tokens: 0,
                    num_refbits: 0,
                    num_eobruns: 0,
                    tokens: Vec::new(),
                    refbits: Vec::new(),
                    eobruns: Vec::new(),
                });
                in_tokens = false;
                in_refbits = false;
                in_eobruns = false;
                refbit_str.clear();
            }
        } else if line.starts_with("=== End AC Refinement Scan") {
            // Finalize refbits parsing
            if !refbit_str.is_empty()
                && let Some(ref mut scan) = current_scan
            {
                for c in refbit_str.chars() {
                    if c == '0' {
                        scan.refbits.push(0);
                    } else if c == '1' {
                        scan.refbits.push(1);
                    }
                }
            }
            if let Some(scan) = current_scan.take() {
                scans.push(scan);
            }
        } else if line.starts_with("Ss=") {
            // Parse Ss=3 Se=63 Ah=2 Al=1
            if let Some(ref mut scan) = current_scan {
                for part in line.split_whitespace() {
                    if let Some(val) = part.strip_prefix("Ss=") {
                        scan.ss = val.parse().unwrap_or(0);
                    } else if let Some(val) = part.strip_prefix("Se=") {
                        scan.se = val.parse().unwrap_or(0);
                    } else if let Some(val) = part.strip_prefix("Ah=") {
                        scan.ah = val.parse().unwrap_or(0);
                    } else if let Some(val) = part.strip_prefix("Al=") {
                        scan.al = val.parse().unwrap_or(0);
                    }
                }
            }
        } else if line.starts_with("num_blocks=") {
            // Parse num_blocks=16 num_tokens=180 num_refbits=626 num_eobruns=0
            if let Some(ref mut scan) = current_scan {
                for part in line.split_whitespace() {
                    if let Some(val) = part.strip_prefix("num_blocks=") {
                        scan.num_blocks = val.parse().unwrap_or(0);
                    } else if let Some(val) = part.strip_prefix("num_tokens=") {
                        scan.num_tokens = val.parse().unwrap_or(0);
                    } else if let Some(val) = part.strip_prefix("num_refbits=") {
                        scan.num_refbits = val.parse().unwrap_or(0);
                    } else if let Some(val) = part.strip_prefix("num_eobruns=") {
                        scan.num_eobruns = val.parse().unwrap_or(0);
                    }
                }
            }
        } else if line == "TOKENS:" {
            in_tokens = true;
            in_refbits = false;
            in_eobruns = false;
        } else if line == "REFBITS:" {
            in_tokens = false;
            in_refbits = true;
            in_eobruns = false;
        } else if line == "EOBRUNS:" {
            in_tokens = false;
            in_refbits = false;
            in_eobruns = true;
        } else if in_tokens && line.starts_with('[') {
            // Parse [0] symbol=0x03 refbits=6
            if let Some(ref mut scan) = current_scan {
                let mut symbol: Option<u8> = None;
                let mut refbits: Option<u8> = None;

                for part in line.split_whitespace() {
                    if let Some(val) = part.strip_prefix("symbol=0x") {
                        symbol = u8::from_str_radix(val, 16).ok();
                    } else if let Some(val) = part.strip_prefix("refbits=") {
                        refbits = val.parse().ok();
                    }
                }

                if let (Some(s), Some(r)) = (symbol, refbits) {
                    scan.tokens.push(CppToken {
                        symbol: s,
                        refbits: r,
                    });
                }
            }
        } else if in_tokens && line.starts_with("...") {
            // Skip "... (80 more tokens)" line
        } else if in_refbits {
            // Collect refbit strings
            if !line.starts_with("...") {
                refbit_str.push_str(line.trim());
            }
        } else if in_eobruns && !line.is_empty() {
            // Parse EOB run values
            if let Some(ref mut scan) = current_scan {
                for val in line.split_whitespace() {
                    if let Ok(v) = val.parse::<u16>() {
                        scan.eobruns.push(v);
                    }
                }
            }
        }
    }

    scans
}

/// Decodes what a token means
fn describe_token(symbol: u8) -> String {
    let run = symbol >> 4;
    let cat = symbol & 0x0F;

    if symbol == 0xF0 {
        "ZRL (16 zeros)".to_string()
    } else if cat == 0 {
        if run == 0 {
            "EOB".to_string()
        } else {
            format!("EOB run (2^{} + extra)", run)
        }
    } else if cat == 1 {
        format!("NEW_NZ neg, {} zeros", run)
    } else if cat == 3 {
        format!("NEW_NZ pos, {} zeros", run)
    } else {
        format!("??? symbol=0x{:02x}", symbol)
    }
}

#[test]
fn test_parse_cpp_dump() {
    let testdata_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../internal/jpegli-cpp/testdata/ac_refinement/noise32_tokens.txt");

    if !testdata_path.exists() {
        eprintln!(
            "Skipping test: C++ testdata not found at {:?}",
            testdata_path
        );
        eprintln!(
            "Generate it with: DUMP_AC_REFINEMENT=1 ./build/tools/cjpegli test.png out.jpg -p 2"
        );
        return;
    }

    let content = fs::read_to_string(&testdata_path).expect("Failed to read testdata");
    let scans = parse_cpp_dump(&content);

    println!("\n=== C++ AC Refinement Scan Analysis ===\n");

    for scan in &scans {
        println!(
            "Scan {}: Ss={} Se={} Ah={} Al={}",
            scan.scan_index, scan.ss, scan.se, scan.ah, scan.al
        );
        println!(
            "  {} blocks, {} tokens, {} refbits, {} eobruns",
            scan.num_blocks, scan.num_tokens, scan.num_refbits, scan.num_eobruns
        );
        println!(
            "  Parsed {} tokens, {} refbits, {} eobruns",
            scan.tokens.len(),
            scan.refbits.len(),
            scan.eobruns.len()
        );

        // Count token types
        let mut eob_count = 0;
        let mut eob_run_count = 0;
        let mut zrl_count = 0;
        let mut new_nz_pos = 0;
        let mut new_nz_neg = 0;

        for token in &scan.tokens {
            let run = token.symbol >> 4;
            let cat = token.symbol & 0x0F;

            if token.symbol == 0xF0 {
                zrl_count += 1;
            } else if cat == 0 {
                if run == 0 {
                    eob_count += 1;
                } else {
                    eob_run_count += 1;
                }
            } else if cat == 1 {
                new_nz_neg += 1;
            } else if cat == 3 {
                new_nz_pos += 1;
            }
        }

        println!("  Token breakdown:");
        println!(
            "    EOB: {}, EOB runs: {}, ZRL: {}",
            eob_count, eob_run_count, zrl_count
        );
        println!(
            "    NEW_NZ positive: {}, NEW_NZ negative: {}",
            new_nz_pos, new_nz_neg
        );

        // Show first few tokens
        println!("  First 10 tokens:");
        for (i, token) in scan.tokens.iter().take(10).enumerate() {
            println!(
                "    [{}] 0x{:02x} refbits={} - {}",
                i,
                token.symbol,
                token.refbits,
                describe_token(token.symbol)
            );
        }

        // Show first few refbits
        if !scan.refbits.is_empty() {
            let first_bits: String = scan
                .refbits
                .iter()
                .take(32)
                .map(|b| if *b == 0 { '0' } else { '1' })
                .collect();
            println!("  First 32 refbits: {}", first_bits);
        }

        println!();
    }

    // Summary statistics
    let total_tokens: usize = scans.iter().map(|s| s.tokens.len()).sum();
    let total_refbits: usize = scans.iter().map(|s| s.refbits.len()).sum();

    println!(
        "Total: {} scans, {} tokens, {} refbits",
        scans.len(),
        total_tokens,
        total_refbits
    );

    assert!(!scans.is_empty(), "Should have parsed at least one scan");
}

/// Analyzes token patterns to identify potential issues
#[test]
fn test_analyze_token_patterns() {
    let testdata_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../internal/jpegli-cpp/testdata/ac_refinement/noise32_tokens.txt");

    if !testdata_path.exists() {
        return;
    }

    let content = fs::read_to_string(&testdata_path).expect("Failed to read testdata");
    let scans = parse_cpp_dump(&content);

    println!("\n=== Token Pattern Analysis ===\n");

    for scan in &scans {
        println!("Scan {} (Ah={} Al={}):", scan.scan_index, scan.ah, scan.al);

        // Check for suspicious patterns
        let mut consecutive_eob = 0;
        let mut max_consecutive_eob = 0;
        let mut high_refbit_tokens = 0;

        for token in &scan.tokens {
            if token.symbol == 0x00 {
                consecutive_eob += 1;
                max_consecutive_eob = max_consecutive_eob.max(consecutive_eob);
            } else {
                consecutive_eob = 0;
            }

            if token.refbits > 20 {
                high_refbit_tokens += 1;
            }
        }

        println!("  Max consecutive EOB: {}", max_consecutive_eob);
        println!("  Tokens with >20 refbits: {}", high_refbit_tokens);

        // Compute average refbits per token
        let total_refbits: u32 = scan.tokens.iter().map(|t| t.refbits as u32).sum();
        let avg_refbits = if scan.tokens.is_empty() {
            0.0
        } else {
            total_refbits as f64 / scan.tokens.len() as f64
        };
        println!("  Average refbits per token: {:.2}", avg_refbits);

        // Distribution of zero runs
        let mut run_counts = [0u32; 16];
        for token in &scan.tokens {
            if token.symbol != 0xF0 && (token.symbol & 0x0F) != 0 {
                let run = (token.symbol >> 4) as usize;
                if run < 16 {
                    run_counts[run] += 1;
                }
            }
        }
        println!("  Zero run distribution (for NEW_NZ tokens):");
        for (run, count) in run_counts.iter().enumerate() {
            if *count > 0 {
                println!("    run={}: {} tokens", run, count);
            }
        }

        println!();
    }
}

/// Compares Rust tokenization against C++ (placeholder for full implementation)
#[test]
#[ignore] // Enable when coefficient capture is implemented
fn test_token_by_token_comparison() {
    // This test would:
    // 1. Load the test image
    // 2. Run Rust encoder to get quantized coefficients
    // 3. Run Rust AC refinement tokenization
    // 4. Compare against C++ tokens
    //
    // Currently blocked on capturing intermediate coefficients from Rust encoder.

    println!("\n=== Token-by-Token Comparison ===\n");
    println!("TODO: Implement coefficient capture from Rust encoder");
}

/// Test progressive encoding file size comparison
#[test]
#[ignore = "requires C++ cjpegli build"]
fn test_progressive_filesize_comparison() {
    use std::process::Command;

    // Use the noise test image
    let png_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../internal/jpegli-cpp/testdata/ac_refinement/test32_noise.png");

    if !png_path.exists() {
        eprintln!("Test image not found: {:?}", png_path);
        eprintln!(
            "Generate it with: convert -size 32x32 xc: +noise Random PNG24:/path/to/test32_noise.png"
        );
        return;
    }

    // Load PNG
    let img = zenjpeg_bench_utils::load_png(&png_path).expect("Failed to load PNG");
    let img_width = img.width() as u32;
    let img_height = img.height() as u32;
    let rgb: Vec<u8> = img.buf().iter().flat_map(|p| [p.r, p.g, p.b]).collect();

    // Save as PPM for C++
    let ppm_path = "/tmp/test_noise32.ppm";
    write_ppm(ppm_path, &rgb, img_width as usize, img_height as usize).unwrap();

    println!("\n=== Progressive Encoding File Size Comparison ===\n");
    println!("Image: {}x{}", img_width, img_height);

    // Find cjpegli
    let cjpegli_path = match zenjpeg::test_utils::find_cjpegli() {
        Some(p) => p,
        None => {
            eprintln!("cjpegli not found");
            return;
        }
    };

    // Test configurations
    let configs = [
        (
            "Sequential, no AQ",
            vec!["--noadaptive_quantization", "-p", "0"],
        ),
        (
            "Progressive p2, no AQ",
            vec!["--noadaptive_quantization", "-p", "2"],
        ),
        ("Progressive p2, with AQ", vec!["-p", "2"]),
    ];

    // Also test with flower image if available
    let flower_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../internal/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png");

    let images: Vec<(&str, std::path::PathBuf)> =
        vec![("noise32", png_path.clone()), ("flower_small", flower_path)];

    for (img_name, img_path) in &images {
        if !img_path.exists() {
            continue;
        }

        // Load PNG
        let img = match zenjpeg_bench_utils::load_png(img_path) {
            Ok(img) => img,
            Err(_) => continue,
        };
        let img_w = img.width() as u32;
        let img_h = img.height() as u32;
        let rgb: Vec<u8> = img.buf().iter().flat_map(|p| [p.r, p.g, p.b]).collect();

        // Save as PPM for C++
        let ppm_path = format!("/tmp/test_{}.ppm", img_name);
        write_ppm(&ppm_path, &rgb, img_w as usize, img_h as usize).unwrap();

        println!("\n=== {} ({}x{}) ===", img_name, img_w, img_h);

        for quality in [90, 80] {
            println!("\nQuality {}:", quality);

            for (name, args) in &configs {
                // Encode with C++
                let cpp_output = format!("/tmp/cpp_prog_q{}.jpg", quality);
                let quality_str = quality.to_string();
                let mut cmd_args = args.clone();
                cmd_args.extend([ppm_path.as_str(), &cpp_output, "-q", &quality_str]);

                let output = Command::new(&cjpegli_path)
                    .args(&cmd_args)
                    .output()
                    .expect("Failed to run cjpegli");

                if !output.status.success() {
                    eprintln!("  {}: C++ encoding failed", name);
                    continue;
                }

                let cpp_size = fs::metadata(&cpp_output).map(|m| m.len()).unwrap_or(0);

                // Encode with Rust
                // Check if progressive by looking for "-p" followed by "2"
                let is_progressive = args.windows(2).any(|w| w[0] == "-p" && w[1] == "2");

                // Note: Rust encoder always uses AQ, so we can only test with-AQ case for comparison
                let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
                    .quality(quality as f32)
                    .progressive(is_progressive);
                let rust_jpeg = encode_rgb(img_w, img_h, &rgb, &config);

                // Verify the JPEG is actually progressive by checking for SOF2 marker
                let is_progressive_jpeg = rust_jpeg.windows(2).any(|w| w == [0xFF, 0xC2]);
                let is_baseline_jpeg = rust_jpeg.windows(2).any(|w| w == [0xFF, 0xC0]);
                let sof_type = if is_progressive_jpeg {
                    "progressive"
                } else if is_baseline_jpeg {
                    "baseline"
                } else {
                    "unknown"
                };

                let rust_size = rust_jpeg.len();
                let diff_pct = if cpp_size > 0 {
                    100.0 * (rust_size as f64 - cpp_size as f64) / cpp_size as f64
                } else {
                    0.0
                };

                println!(
                    "  {}: C++={} Rust={} ({:+.1}%) [Rust: {}]",
                    name, cpp_size, rust_size, diff_pct, sof_type
                );

                // Also save Rust output for manual inspection
                let rust_output = format!("/tmp/rust_prog_q{}.jpg", quality);
                fs::write(&rust_output, &rust_jpeg).ok();
            }
        }
    } // end image loop
}

/// Write PPM file for C++ cjpegli
fn write_ppm(path: &str, rgb: &[u8], width: usize, height: usize) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = fs::File::create(path)?;
    writeln!(file, "P6")?;
    writeln!(file, "{} {}", width, height)?;
    writeln!(file, "255")?;
    file.write_all(rgb)?;
    Ok(())
}
