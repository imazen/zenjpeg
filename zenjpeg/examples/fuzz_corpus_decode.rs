//! Decode all JPEGs in a directory and copy failures to an output folder.
//!
//! Usage:
//!   cargo run --release --features decoder --example fuzz_corpus_decode -- <input_dir> <fail_dir>

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use zenjpeg::decode::DecodeConfig;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <input_dir> <fail_dir>", args[0]);
        std::process::exit(1);
    }

    let input_dir = PathBuf::from(&args[1]);
    let fail_dir = PathBuf::from(&args[2]);

    if !input_dir.is_dir() {
        eprintln!("Input directory does not exist: {}", input_dir.display());
        std::process::exit(1);
    }

    fs::create_dir_all(&fail_dir).expect("Failed to create fail directory");

    // Collect all .jpg/.jpeg files
    let mut files: Vec<PathBuf> = fs::read_dir(&input_dir)
        .expect("Failed to read input directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .map(|ext| {
                    let s = ext.to_ascii_lowercase();
                    s == "jpg" || s == "jpeg"
                })
                .unwrap_or(false)
        })
        .collect();
    files.sort();

    println!("Found {} JPEG files in {}", files.len(), input_dir.display());
    println!("Failures will be copied to {}", fail_dir.display());
    println!();

    let decoder = DecodeConfig::new();

    let start = Instant::now();
    let mut ok_count = 0u32;
    let mut fail_count = 0u32;
    let mut warn_count = 0u32;
    let mut total_bytes = 0u64;
    let mut failures: Vec<(PathBuf, String)> = Vec::new();

    for (i, path) in files.iter().enumerate() {
        let data = match fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                let msg = format!("IO error: {e}");
                eprintln!("[{}/{}] FAIL (IO) {}: {msg}", i + 1, files.len(), path.display());
                failures.push((path.clone(), msg));
                fail_count += 1;
                continue;
            }
        };
        total_bytes += data.len() as u64;

        match decoder.decode(&data, enough::Unstoppable) {
            Ok(result) => {
                ok_count += 1;
                if result.has_warnings() {
                    warn_count += 1;
                    let warnings: Vec<String> =
                        result.warnings().iter().map(|w| format!("{w:?}")).collect();
                    println!(
                        "[{}/{}] OK (warnings: {}) {}",
                        i + 1,
                        files.len(),
                        warnings.join(", "),
                        path.file_name().unwrap().to_string_lossy()
                    );
                }
            }
            Err(e) => {
                let msg = format!("{e}");
                eprintln!(
                    "[{}/{}] FAIL {}: {msg}",
                    i + 1,
                    files.len(),
                    path.file_name().unwrap().to_string_lossy()
                );
                failures.push((path.clone(), msg));
                fail_count += 1;
            }
        }

        // Progress every 50 files
        if (i + 1) % 50 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = (i + 1) as f64 / elapsed;
            println!(
                "  ... {}/{} ({:.0} files/sec, {:.1} MB processed)",
                i + 1,
                files.len(),
                rate,
                total_bytes as f64 / 1_048_576.0
            );
        }
    }

    // Copy failures
    for (path, _) in &failures {
        let dest = fail_dir.join(path.file_name().unwrap());
        if let Err(e) = fs::copy(path, &dest) {
            eprintln!("Failed to copy {}: {e}", path.display());
        }
    }

    let elapsed = start.elapsed();
    println!();
    println!("=== Results ===");
    println!("Total:    {} files ({:.1} MB)", files.len(), total_bytes as f64 / 1_048_576.0);
    println!("OK:       {} ({} with warnings)", ok_count, warn_count);
    println!("Failed:   {}", fail_count);
    println!("Time:     {:.2}s ({:.0} files/sec)", elapsed.as_secs_f64(), files.len() as f64 / elapsed.as_secs_f64());
    println!();

    if !failures.is_empty() {
        println!("=== Failures ===");
        for (path, msg) in &failures {
            println!("  {}: {msg}", path.file_name().unwrap().to_string_lossy());
        }
        println!();
        println!("Failed files copied to: {}", fail_dir.display());
    }
}
