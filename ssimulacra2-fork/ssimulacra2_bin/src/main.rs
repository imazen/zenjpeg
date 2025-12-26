#[cfg(feature = "video")]
mod video;

#[cfg(feature = "video")]
use self::video::*;
use clap::{Parser, Subcommand};
#[cfg(feature = "video")]
use ssimulacra2::MatrixCoefficients;
use ssimulacra2::{compute_frame_ssimulacra2, ColorPrimaries, Rgb, TransferCharacteristic};
use std::path::{Path, PathBuf};

#[cfg(feature = "jpegli-icc")]
mod jpegli_icc_support {
    use std::path::Path;

    /// Decode JPEG with ICC profile support using jpegli
    pub fn decode_jpeg_with_icc(path: &Path) -> Option<(Vec<[f32; 3]>, usize, usize)> {
        let data = std::fs::read(path).ok()?;
        let (pixels, width, height) = jpegli::icc::decode_jpeg_with_icc(&data).ok()?;

        let rgb_f32: Vec<[f32; 3]> = pixels
            .chunks_exact(3)
            .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
            .collect();

        Some((rgb_f32, width, height))
    }

    /// Check if file is a JPEG
    pub fn is_jpeg(path: &Path) -> bool {
        path.extension()
            .map(|e| {
                let e = e.to_string_lossy().to_lowercase();
                e == "jpg" || e == "jpeg"
            })
            .unwrap_or(false)
    }
}

#[cfg(feature = "icc-moxcms")]
mod moxcms_icc_support {
    use std::path::Path;

    /// Decode JPEG with ICC profile support using moxcms
    pub fn decode_jpeg_with_icc(path: &Path) -> Option<(Vec<[f32; 3]>, usize, usize)> {
        use std::fs::File;
        use std::io::BufReader;

        let file = File::open(path).ok()?;
        let mut decoder = jpeg_decoder::Decoder::new(BufReader::new(file));
        let pixels = decoder.decode().ok()?;
        let info = decoder.info()?;

        let width = info.width as usize;
        let height = info.height as usize;

        let icc_profile = decoder.icc_profile();

        let rgb_u8: Vec<u8> = match info.pixel_format {
            jpeg_decoder::PixelFormat::RGB24 => pixels,
            jpeg_decoder::PixelFormat::L8 => {
                pixels.iter().flat_map(|&g| [g, g, g]).collect()
            }
            _ => return None,
        };

        let final_rgb = if let Some(icc_data) = icc_profile {
            apply_icc_moxcms(&rgb_u8, &icc_data)?
        } else {
            rgb_u8
        };

        let rgb_f32: Vec<[f32; 3]> = final_rgb
            .chunks_exact(3)
            .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
            .collect();

        Some((rgb_f32, width, height))
    }

    fn apply_icc_moxcms(rgb: &[u8], icc_data: &[u8]) -> Option<Vec<u8>> {
        use moxcms::{ColorProfile, Layout, TransformOptions};

        let input_profile = ColorProfile::new_from_slice(icc_data).ok()?;
        let srgb = ColorProfile::new_srgb();

        // Use default (Linear) interpolation - closest match to skcms
        // See EXPERIMENTS.md for comparison results
        let transform = input_profile
            .create_transform_8bit(Layout::Rgb, &srgb, Layout::Rgb, TransformOptions::default())
            .ok()?;

        let mut output = vec![0u8; rgb.len()];
        transform.transform(rgb, &mut output).ok()?;

        Some(output)
    }

    pub fn is_jpeg(path: &Path) -> bool {
        path.extension()
            .map(|e| {
                let e = e.to_string_lossy().to_lowercase();
                e == "jpg" || e == "jpeg"
            })
            .unwrap_or(false)
    }
}

#[cfg(all(feature = "icc", not(feature = "jpegli-icc"), not(feature = "icc-moxcms")))]
mod icc_support {
    use std::path::Path;

    /// Decode JPEG with ICC profile support, returning RGB f32 pixels
    pub fn decode_jpeg_with_icc(path: &Path) -> Option<(Vec<[f32; 3]>, usize, usize)> {
        use std::fs::File;
        use std::io::BufReader;

        let file = File::open(path).ok()?;
        let mut decoder = jpeg_decoder::Decoder::new(BufReader::new(file));
        let pixels = decoder.decode().ok()?;
        let info = decoder.info()?;

        let width = info.width as usize;
        let height = info.height as usize;

        let icc_profile = decoder.icc_profile();

        let rgb_u8: Vec<u8> = match info.pixel_format {
            jpeg_decoder::PixelFormat::RGB24 => pixels,
            jpeg_decoder::PixelFormat::L8 => {
                pixels.iter().flat_map(|&g| [g, g, g]).collect()
            }
            _ => return None,
        };

        // Apply ICC profile if present
        let final_rgb = if let Some(icc_data) = icc_profile {
            apply_icc_profile(&rgb_u8, &icc_data).unwrap_or(rgb_u8)
        } else {
            rgb_u8
        };

        // Convert to f32
        let rgb_f32: Vec<[f32; 3]> = final_rgb
            .chunks_exact(3)
            .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
            .collect();

        Some((rgb_f32, width, height))
    }

    fn apply_icc_profile(rgb: &[u8], icc_data: &[u8]) -> Option<Vec<u8>> {
        use lcms2::*;

        // Parse source ICC profile
        let src_profile = Profile::new_icc(icc_data).ok()?;

        // Create sRGB destination profile
        let dst_profile = Profile::new_srgb();

        // Create transform
        let transform = Transform::new(
            &src_profile,
            PixelFormat::RGB_8,
            &dst_profile,
            PixelFormat::RGB_8,
            Intent::Perceptual,
        )
        .ok()?;

        // Apply transform
        let mut output = vec![0u8; rgb.len()];
        transform.transform_pixels(rgb, &mut output);

        Some(output)
    }

    /// Check if file is a JPEG
    pub fn is_jpeg(path: &Path) -> bool {
        path.extension()
            .map(|e| {
                let e = e.to_string_lossy().to_lowercase();
                e == "jpg" || e == "jpeg"
            })
            .unwrap_or(false)
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
#[allow(clippy::large_enum_variant)]
enum Commands {
    /// Compare two still images. Resolutions must be identical.
    Image {
        /// Source image
        #[arg(help = "Original unmodified image", value_hint = clap::ValueHint::FilePath)]
        source: PathBuf,

        /// Distorted image
        #[arg(help = "Distorted image", value_hint = clap::ValueHint::FilePath)]
        distorted: PathBuf,
    },
    /// Compare two videos. Resolutions and frame counts must be identical.
    #[cfg(feature = "video")]
    Video {
        /// Source video
        #[arg(help = "Original unmodified video", value_hint = clap::ValueHint::FilePath)]
        source: String,

        /// Distorted video
        #[arg(help = "Distorted video", value_hint = clap::ValueHint::FilePath)]
        distorted: String,

        /// How many worker threads to use for decoding & calculating scores.
        /// Note: Memory usage increases linearly with the number of workers.
        #[arg(long, short, verbatim_doc_comment)]
        frame_threads: Option<usize>,

        /// The amount of frames to skip.
        #[arg(long, default_value_t = 0)]
        skip_frames: usize,

        /// Limit the amount of frames to compare.
        #[arg(long)]
        frames: Option<usize>,

        /// How to increment current frame count; e.g. 10 will read every 10th frame.
        #[arg(long, short)]
        increment: Option<usize>,

        /// Whether to output a frame-by-frame graph of scores.
        #[arg(long, short)]
        graph: bool,

        /// Will output scores for every frame followed by the average at the end.
        #[arg(long, short)]
        verbose: bool,

        /// Source color matrix
        #[arg(long)]
        src_matrix: Option<String>,

        /// Source transfer characteristics
        #[arg(long)]
        src_transfer: Option<String>,

        /// Source color primaries
        #[arg(long)]
        src_primaries: Option<String>,

        /// The source is using full-range data
        #[arg(long)]
        src_full_range: bool,

        /// Distorted color matrix
        #[arg(long)]
        dst_matrix: Option<String>,

        /// Distorted transfer characteristics
        #[arg(long)]
        dst_transfer: Option<String>,

        /// Distorted color primaries
        #[arg(long)]
        dst_primaries: Option<String>,

        /// The distorted video is using full-range data
        #[arg(long)]
        dst_full_range: bool,
    },
}

fn main() {
    match Cli::parse().command {
        Commands::Image { source, distorted } => compare_images(&source, &distorted),
        #[cfg(feature = "video")]
        Commands::Video {
            source,
            distorted,
            frame_threads,
            skip_frames,
            frames,
            increment,
            graph,
            verbose,
            src_matrix,
            src_transfer,
            src_primaries,
            src_full_range,
            dst_matrix,
            dst_transfer,
            dst_primaries,
            dst_full_range,
        } => {
            let frame_threads = frame_threads.unwrap_or(1).max(1);
            let inc = increment.unwrap_or(1).max(1);
            let src_matrix = src_matrix
                .map(|i| parse_matrix(&i))
                .unwrap_or(MatrixCoefficients::Unspecified);
            let src_transfer = src_transfer
                .map(|i| parse_transfer(&i))
                .unwrap_or(TransferCharacteristic::Unspecified);
            let src_primaries = src_primaries
                .map(|i| parse_primaries(&i))
                .unwrap_or(ColorPrimaries::Unspecified);
            let dst_matrix = dst_matrix
                .map(|i| parse_matrix(&i))
                .unwrap_or(MatrixCoefficients::Unspecified);
            let dst_transfer = dst_transfer
                .map(|i| parse_transfer(&i))
                .unwrap_or(TransferCharacteristic::Unspecified);
            let dst_primaries = dst_primaries
                .map(|i| parse_primaries(&i))
                .unwrap_or(ColorPrimaries::Unspecified);
            compare_videos(
                &source,
                &distorted,
                frame_threads,
                skip_frames,
                frames,
                inc,
                graph,
                verbose,
                src_matrix,
                src_transfer,
                src_primaries,
                src_full_range,
                dst_matrix,
                dst_transfer,
                dst_primaries,
                dst_full_range,
            )
        }
    }
}

/// Load image with ICC profile support for JPEGs
fn load_image_with_icc(path: &Path) -> (Vec<[f32; 3]>, usize, usize) {
    // Try jpegli ICC-aware decoding first (best quality)
    #[cfg(any(feature = "jpegli-icc", feature = "jpegli-icc-moxcms"))]
    {
        if jpegli_icc_support::is_jpeg(path) {
            if let Some((data, width, height)) = jpegli_icc_support::decode_jpeg_with_icc(path) {
                return (data, width, height);
            }
        }
    }

    // Try moxcms ICC-aware decoding
    #[cfg(feature = "icc-moxcms")]
    {
        if moxcms_icc_support::is_jpeg(path) {
            if let Some((data, width, height)) = moxcms_icc_support::decode_jpeg_with_icc(path) {
                return (data, width, height);
            }
        }
    }

    // Try ICC-aware JPEG decoding with jpeg-decoder + lcms2
    #[cfg(all(feature = "icc", not(feature = "jpegli-icc"), not(feature = "icc-moxcms")))]
    {
        if icc_support::is_jpeg(path) {
            if let Some((data, width, height)) = icc_support::decode_jpeg_with_icc(path) {
                return (data, width, height);
            }
        }
    }

    // Fallback to image crate (assumes sRGB)
    let img = image::open(path).expect("Failed to open image file");
    let data = img
        .to_rgb32f()
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect::<Vec<_>>();
    (data, img.width() as usize, img.height() as usize)
}

fn compare_images(source: &Path, distorted: &Path) {
    let (source_data, source_w, source_h) = load_image_with_icc(source);
    let (distorted_data, distorted_w, distorted_h) = load_image_with_icc(distorted);

    let source_data = Rgb::new(
        source_data,
        source_w,
        source_h,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .expect("Failed to process source_data into RGB");

    let distorted_data = Rgb::new(
        distorted_data,
        distorted_w,
        distorted_h,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .expect("Failed to process distorted_data into RGB");

    let result = compute_frame_ssimulacra2(source_data, distorted_data)
        .expect("Failed to calculate ssimulacra2");

    println!("Score: {result:.8}");
}
