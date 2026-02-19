use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Expand input arguments into a list of file paths.
///
/// Handles both literal paths and glob patterns. Deduplicates results.
pub fn expand_inputs(inputs: &[String]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for input in inputs {
        // Try as glob first
        if input.contains('*') || input.contains('?') || input.contains('[') {
            let paths = glob::glob(input)
                .with_context(|| format!("invalid glob pattern: {input}"))?;
            for entry in paths {
                let path = entry.with_context(|| format!("glob error for pattern: {input}"))?;
                if path.is_file() && is_jpeg(&path) {
                    let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
                    if seen.insert(canonical) {
                        files.push(path);
                    }
                }
            }
        } else {
            let path = PathBuf::from(input);
            if path.is_dir() {
                // Recursively find JPEGs in directory
                for_each_jpeg_in_dir(&path, &mut |p| {
                    let canonical = p.canonicalize().unwrap_or_else(|_| p.clone());
                    if seen.insert(canonical) {
                        files.push(p);
                    }
                })?;
            } else if path.is_file() {
                let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
                if seen.insert(canonical) {
                    files.push(path);
                }
            } else {
                anyhow::bail!("not found: {input}");
            }
        }
    }

    // Sort by size descending for better load balancing in parallel processing
    files.sort_by(|a, b| {
        let sa = std::fs::metadata(a).map(|m| m.len()).unwrap_or(0);
        let sb = std::fs::metadata(b).map(|m| m.len()).unwrap_or(0);
        sb.cmp(&sa)
    });

    Ok(files)
}

/// Check if a path looks like a JPEG file.
fn is_jpeg(path: &Path) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "jpe" | "jfif"),
        None => false,
    }
}

/// Recursively find JPEG files in a directory.
fn for_each_jpeg_in_dir(dir: &Path, cb: &mut dyn FnMut(PathBuf)) -> Result<()> {
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("failed to read directory '{}'", dir.display()))?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            for_each_jpeg_in_dir(&path, cb)?;
        } else if path.is_file() && is_jpeg(&path) {
            cb(path);
        }
    }
    Ok(())
}

/// Summary of processing results for reporting.
#[derive(Default)]
pub struct BatchSummary {
    pub results: Vec<FileResult>,
}

pub struct FileResult {
    pub path: PathBuf,
    pub input_size: u64,
    pub output_size: Option<u64>,
    pub skipped: bool,
    pub error: Option<String>,
}

impl BatchSummary {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, result: FileResult) {
        self.results.push(result);
    }

    pub fn print_report(&self) {
        if self.results.is_empty() {
            return;
        }

        let max_path_len = self
            .results
            .iter()
            .map(|r| r.path.display().to_string().len())
            .max()
            .unwrap_or(10)
            .min(60);

        println!();
        println!(
            "{:<width$}  {:>10}  {:>10}  {:>8}  Status",
            "File",
            "Input",
            "Output",
            "Change",
            width = max_path_len
        );
        println!("{}", "-".repeat(max_path_len + 45));

        let mut total_input: u64 = 0;
        let mut total_output: u64 = 0;
        let mut count_ok = 0u32;
        let mut count_skip = 0u32;
        let mut count_err = 0u32;

        for r in &self.results {
            let path_str = r.path.display().to_string();
            let path_display = if path_str.len() > max_path_len {
                format!("...{}", &path_str[path_str.len() - max_path_len + 3..])
            } else {
                path_str
            };

            total_input += r.input_size;

            if let Some(ref err) = r.error {
                count_err += 1;
                println!(
                    "{:<width$}  {:>10}  {:>10}  {:>8}  ERROR: {}",
                    path_display,
                    format_size(r.input_size),
                    "-",
                    "-",
                    err,
                    width = max_path_len
                );
            } else if r.skipped {
                count_skip += 1;
                total_output += r.input_size;
                println!(
                    "{:<width$}  {:>10}  {:>10}  {:>8}  skipped (larger)",
                    path_display,
                    format_size(r.input_size),
                    "-",
                    "-",
                    width = max_path_len
                );
            } else if let Some(out_size) = r.output_size {
                count_ok += 1;
                total_output += out_size;
                let change = if r.input_size > 0 {
                    let pct =
                        (out_size as f64 - r.input_size as f64) / r.input_size as f64 * 100.0;
                    format!("{pct:+.1}%")
                } else {
                    "-".into()
                };
                println!(
                    "{:<width$}  {:>10}  {:>10}  {:>8}  ok",
                    path_display,
                    format_size(r.input_size),
                    format_size(out_size),
                    change,
                    width = max_path_len
                );
            }
        }

        println!("{}", "-".repeat(max_path_len + 45));
        let total_change = if total_input > 0 {
            let pct = (total_output as f64 - total_input as f64) / total_input as f64 * 100.0;
            format!("{pct:+.1}%")
        } else {
            "-".into()
        };
        println!(
            "{:<width$}  {:>10}  {:>10}  {:>8}  {} ok, {} skipped, {} errors",
            "TOTAL",
            format_size(total_input),
            format_size(total_output),
            total_change,
            count_ok,
            count_skip,
            count_err,
            width = max_path_len
        );
    }

    pub fn write_csv(&self, path: &std::path::Path) -> Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)
            .with_context(|| format!("failed to create CSV '{}'", path.display()))?;
        writeln!(f, "file,input_bytes,output_bytes,change_pct,status")?;
        for r in &self.results {
            let status = if r.error.is_some() {
                "error"
            } else if r.skipped {
                "skipped"
            } else {
                "ok"
            };
            let out = r.output_size.unwrap_or(0);
            let change = if r.input_size > 0 && r.output_size.is_some() {
                (out as f64 - r.input_size as f64) / r.input_size as f64 * 100.0
            } else {
                0.0
            };
            writeln!(
                f,
                "\"{}\",{},{},{:.2},{}",
                r.path.display(),
                r.input_size,
                out,
                change,
                status
            )?;
        }
        Ok(())
    }
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_048_576 {
        format!("{:.1}MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes}B")
    }
}
