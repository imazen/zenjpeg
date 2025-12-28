//! Corpus reduction tool for zenjpeg testing.
//!
//! Reduces a test corpus while maintaining coverage of unique encoding behaviors.
//! Images with similar rate-distortion curves are grouped together, and only the
//! smallest representative from each group is kept.
//!
//! Usage:
//!   cargo run --release --example corpus_reduction -- <corpus_dir> [output_dir]
//!
//! Example:
//!   cargo run --release --example corpus_reduction -- ../codec-eval/codec-corpus/kodak reduced_corpus

use dssim::Dssim;
use rgb::RGBA8;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use zenjpeg::{Encoder, Quality};

/// Quality levels to sample for curve generation
const QUALITY_LEVELS: [u8; 5] = [20, 40, 60, 80, 95];

/// Threshold for considering curves "similar" (normalized difference)
const CURVE_SIMILARITY_THRESHOLD: f64 = 0.10;

/// Rate-distortion curve for an image
#[derive(Debug, Clone)]
struct RdCurve {
    /// Image path
    path: PathBuf,
    /// Image dimensions
    width: usize,
    height: usize,
    /// File size in bytes (original PNG)
    original_size: usize,
    /// Points: (quality, bpp, dssim)
    points: Vec<(u8, f64, f64)>,
}

impl RdCurve {
    /// Compute similarity to another curve (0.0 = identical, higher = more different)
    fn distance(&self, other: &RdCurve) -> f64 {
        if self.points.len() != other.points.len() {
            return f64::MAX;
        }

        let mut total_diff = 0.0;
        let mut total_weight = 0.0;

        for ((q1, bpp1, dssim1), (q2, bpp2, dssim2)) in self.points.iter().zip(other.points.iter())
        {
            if q1 != q2 {
                return f64::MAX;
            }

            // Normalize differences by average value
            let avg_bpp = (bpp1 + bpp2) / 2.0;
            let avg_dssim = (dssim1 + dssim2) / 2.0;

            if avg_bpp > 0.0 {
                total_diff += ((bpp1 - bpp2) / avg_bpp).abs();
                total_weight += 1.0;
            }
            if avg_dssim > 0.0 {
                total_diff += ((dssim1 - dssim2) / avg_dssim).abs();
                total_weight += 1.0;
            }
        }

        if total_weight > 0.0 {
            total_diff / total_weight
        } else {
            f64::MAX
        }
    }

    /// Get the curve "signature" for quick comparison
    fn signature(&self) -> String {
        // Quantize the curve to create a coarse signature
        let sig: Vec<String> = self
            .points
            .iter()
            .map(|(q, bpp, dssim)| {
                let bpp_bucket = (bpp * 10.0).round() as i32;
                let dssim_bucket = (dssim * 10000.0).round() as i32;
                format!("{}:{}/{}", q, bpp_bucket, dssim_bucket)
            })
            .collect();
        sig.join("_")
    }
}

/// Cluster of images with similar curves
#[derive(Debug)]
struct Cluster {
    /// Representative image (smallest in the cluster)
    representative: RdCurve,
    /// All images in the cluster
    members: Vec<RdCurve>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <corpus_dir> [output_dir]", args[0]);
        eprintln!("  corpus_dir: Directory containing PNG/JPEG images");
        eprintln!("  output_dir: Where to write the reduced corpus (default: reduced_corpus)");
        std::process::exit(1);
    }

    let corpus_dir = PathBuf::from(&args[1]);
    let output_dir = if args.len() > 2 {
        PathBuf::from(&args[2])
    } else {
        PathBuf::from("reduced_corpus")
    };

    println!("=== Corpus Reduction Tool ===");
    println!("Input: {:?}", corpus_dir);
    println!("Output: {:?}", output_dir);
    println!();

    // Find all images
    let image_paths = find_images(&corpus_dir)?;
    println!("Found {} images", image_paths.len());

    if image_paths.is_empty() {
        eprintln!("No images found in {:?}", corpus_dir);
        std::process::exit(1);
    }

    // Compute RD curves for all images
    println!("\nComputing rate-distortion curves...");
    let mut curves: Vec<RdCurve> = Vec::new();

    for (i, path) in image_paths.iter().enumerate() {
        print!(
            "\r  [{}/{}] {}",
            i + 1,
            image_paths.len(),
            path.file_name().unwrap().to_string_lossy()
        );
        std::io::Write::flush(&mut std::io::stdout())?;

        match compute_rd_curve(path) {
            Ok(curve) => curves.push(curve),
            Err(e) => eprintln!("\n  Warning: Failed to process {:?}: {}", path, e),
        }
    }
    println!("\n  Computed {} curves", curves.len());

    // Cluster curves by similarity
    println!("\nClustering by curve similarity...");
    let clusters = cluster_curves(&curves, CURVE_SIMILARITY_THRESHOLD);
    println!("  Found {} clusters", clusters.len());

    // Print cluster summary
    println!("\nCluster Summary:");
    println!(
        "{:>4} | {:>5} | {:>8} | {:>8} | Representative",
        "#", "Size", "OrigSize", "AvgBPP"
    );
    println!("{}", "-".repeat(60));

    let mut total_original_size = 0;
    let mut total_reduced_size = 0;

    for (i, cluster) in clusters.iter().enumerate() {
        let rep = &cluster.representative;
        let avg_bpp: f64 =
            rep.points.iter().map(|(_, bpp, _)| bpp).sum::<f64>() / rep.points.len() as f64;

        let cluster_original_size: usize = cluster.members.iter().map(|c| c.original_size).sum();
        total_original_size += cluster_original_size;
        total_reduced_size += rep.original_size;

        println!(
            "{:>4} | {:>5} | {:>7}K | {:>8.2} | {}",
            i + 1,
            cluster.members.len(),
            cluster_original_size / 1024,
            avg_bpp,
            rep.path.file_name().unwrap().to_string_lossy()
        );

        // Print cluster members if more than 1
        if cluster.members.len() > 1 {
            for member in &cluster.members {
                if member.path != rep.path {
                    println!(
                        "      |       |          |          | -> {}",
                        member.path.file_name().unwrap().to_string_lossy()
                    );
                }
            }
        }
    }

    println!("{}", "-".repeat(60));
    println!(
        "Total: {} images -> {} representatives",
        curves.len(),
        clusters.len()
    );
    println!(
        "Size reduction: {}K -> {}K ({:.1}%)",
        total_original_size / 1024,
        total_reduced_size / 1024,
        (1.0 - total_reduced_size as f64 / total_original_size as f64) * 100.0
    );

    // Create output directory and copy representatives
    fs::create_dir_all(&output_dir)?;

    println!("\nWriting reduced corpus to {:?}...", output_dir);
    for cluster in &clusters {
        let src = &cluster.representative.path;
        let dst = output_dir.join(src.file_name().unwrap());
        fs::copy(src, &dst)?;
    }

    // Write curve data for reference
    write_curve_data(&output_dir, &clusters)?;

    println!("Done!");
    Ok(())
}

/// Find all PNG/JPEG images in a directory
fn find_images(dir: &Path) -> Result<Vec<PathBuf>, std::io::Error> {
    let mut images = Vec::new();

    if dir.is_file() {
        // Single file
        let ext = dir.extension().map(|e| e.to_string_lossy().to_lowercase());
        if matches!(ext.as_deref(), Some("png") | Some("jpg") | Some("jpeg")) {
            images.push(dir.to_path_buf());
        }
    } else {
        // Directory
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase());
                if matches!(ext.as_deref(), Some("png") | Some("jpg") | Some("jpeg")) {
                    images.push(path);
                }
            }
        }
    }

    images.sort();
    Ok(images)
}

/// Load an image and return (rgb_data, width, height)
fn load_image(path: &Path) -> Result<(Vec<u8>, usize, usize), Box<dyn std::error::Error>> {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "png" => {
            let decoder = png::Decoder::new(fs::File::open(path)?);
            let mut reader = decoder.read_info()?;
            let mut buf = vec![0; reader.output_buffer_size()];
            let info = reader.next_frame(&mut buf)?;
            let bytes = &buf[..info.buffer_size()];

            let rgb = match info.color_type {
                png::ColorType::Rgb => bytes.to_vec(),
                png::ColorType::Rgba => bytes
                    .chunks_exact(4)
                    .flat_map(|c| [c[0], c[1], c[2]])
                    .collect(),
                png::ColorType::Grayscale => bytes.iter().flat_map(|&g| [g, g, g]).collect(),
                png::ColorType::GrayscaleAlpha => bytes
                    .chunks_exact(2)
                    .flat_map(|c| [c[0], c[0], c[0]])
                    .collect(),
                _ => return Err(format!("Unsupported color type: {:?}", info.color_type).into()),
            };

            Ok((rgb, info.width as usize, info.height as usize))
        }
        "jpg" | "jpeg" => {
            let data = fs::read(path)?;
            let mut decoder = jpeg_decoder::Decoder::new(&data[..]);
            let pixels = decoder.decode()?;
            let info = decoder.info().unwrap();

            let rgb = match info.pixel_format {
                jpeg_decoder::PixelFormat::RGB24 => pixels,
                jpeg_decoder::PixelFormat::L8 => pixels.iter().flat_map(|&g| [g, g, g]).collect(),
                _ => return Err("Unsupported JPEG pixel format".into()),
            };

            Ok((rgb, info.width as usize, info.height as usize))
        }
        _ => Err(format!("Unknown image format: {}", ext).into()),
    }
}

/// Compute rate-distortion curve for an image
fn compute_rd_curve(path: &Path) -> Result<RdCurve, Box<dyn std::error::Error>> {
    let original_size = fs::metadata(path)?.len() as usize;
    let (original, width, height) = load_image(path)?;

    let mut points = Vec::new();
    let attr = Dssim::new();

    for q in QUALITY_LEVELS {
        // Encode with zenjpeg
        let encoder = Encoder::new().quality(Quality::Standard(q));
        let jpeg_data = encoder.encode_rgb(&original, width, height)?;

        // Decode
        let mut decoder = jpeg_decoder::Decoder::new(&jpeg_data[..]);
        let decoded = decoder.decode()?;

        // Compute DSSIM
        let orig_rgba: Vec<RGBA8> = original
            .chunks(3)
            .map(|c| RGBA8::new(c[0], c[1], c[2], 255))
            .collect();
        let dec_rgba: Vec<RGBA8> = decoded
            .chunks(3)
            .map(|c| RGBA8::new(c[0], c[1], c[2], 255))
            .collect();

        let orig_img = attr.create_image_rgba(&orig_rgba, width, height).unwrap();
        let dec_img = attr.create_image_rgba(&dec_rgba, width, height).unwrap();
        let (dssim, _) = attr.compare(&orig_img, dec_img);

        let bpp = (jpeg_data.len() as f64 * 8.0) / (width * height) as f64;
        points.push((q, bpp, dssim.into()));
    }

    Ok(RdCurve {
        path: path.to_path_buf(),
        width,
        height,
        original_size,
        points,
    })
}

/// Cluster curves by similarity using simple greedy clustering
fn cluster_curves(curves: &[RdCurve], threshold: f64) -> Vec<Cluster> {
    let mut clusters: Vec<Cluster> = Vec::new();
    let mut assigned = vec![false; curves.len()];

    for i in 0..curves.len() {
        if assigned[i] {
            continue;
        }

        // Start a new cluster with this curve
        let mut cluster = Cluster {
            representative: curves[i].clone(),
            members: vec![curves[i].clone()],
        };

        // Find all similar curves
        for j in (i + 1)..curves.len() {
            if assigned[j] {
                continue;
            }

            let distance = curves[i].distance(&curves[j]);
            if distance < threshold {
                assigned[j] = true;
                cluster.members.push(curves[j].clone());

                // Update representative to smallest
                if curves[j].original_size < cluster.representative.original_size {
                    cluster.representative = curves[j].clone();
                }
            }
        }

        assigned[i] = true;
        clusters.push(cluster);
    }

    // Sort clusters by representative size (smallest first)
    clusters.sort_by_key(|c| c.representative.original_size);

    clusters
}

/// Write curve data for reference
fn write_curve_data(output_dir: &Path, clusters: &[Cluster]) -> Result<(), std::io::Error> {
    let mut csv = String::from("image,cluster,width,height,q20_bpp,q20_dssim,q40_bpp,q40_dssim,q60_bpp,q60_dssim,q80_bpp,q80_dssim,q95_bpp,q95_dssim\n");

    for (i, cluster) in clusters.iter().enumerate() {
        for member in &cluster.members {
            let name = member.path.file_name().unwrap().to_string_lossy();
            let mut row = format!("{},{},{},{}", name, i, member.width, member.height);

            for (_, bpp, dssim) in &member.points {
                row.push_str(&format!(",{:.4},{:.6}", bpp, dssim));
            }

            csv.push_str(&row);
            csv.push('\n');
        }
    }

    fs::write(output_dir.join("curve_data.csv"), csv)?;

    // Also write a summary
    let mut summary = String::from("# Corpus Reduction Summary\n\n");
    summary.push_str(&format!(
        "Original images: {}\n",
        clusters.iter().map(|c| c.members.len()).sum::<usize>()
    ));
    summary.push_str(&format!(
        "Reduced to: {} representatives\n\n",
        clusters.len()
    ));

    summary.push_str("## Clusters\n\n");
    for (i, cluster) in clusters.iter().enumerate() {
        let rep_name = cluster
            .representative
            .path
            .file_name()
            .unwrap()
            .to_string_lossy();
        summary.push_str(&format!(
            "### Cluster {} ({})\n",
            i + 1,
            cluster.members.len()
        ));
        summary.push_str(&format!("Representative: {}\n", rep_name));

        if cluster.members.len() > 1 {
            summary.push_str("Members:\n");
            for member in &cluster.members {
                let name = member.path.file_name().unwrap().to_string_lossy();
                summary.push_str(&format!("- {}\n", name));
            }
        }
        summary.push('\n');
    }

    fs::write(output_dir.join("README.md"), summary)?;

    Ok(())
}
