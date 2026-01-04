//! Simulated annealing optimizer for jpegli quantization matrices.
//!
//! Optimizes base quantization matrices to maximize SSIMULACRA2 quality
//! at a given file size (or minimize file size at a given quality).
//!
//! The search space includes:
//! - Base YCbCr matrix (192 values: Y[64], Cb[64], Cr[64])
//! - Global scale factor
//! - Frequency exponents (64 values)
//!
//! Usage:
//!   cargo run --release --example optimize_quant_matrices -- <corpus_dir> [options]
//!
//! Options:
//!   --quality <N>        Target quality level (default: 85)
//!   --iterations <N>     SA iterations (default: 10000)
//!   --output <file>      Output file for best matrices (JSON)
//!   --resume <file>      Resume from checkpoint
//!   --seed <N>           Random seed for reproducibility

use jpegli::quant::{CustomQuantMatrices, Quality};
use jpegli::{Encoder, PixelFormat};
use rayon::prelude::*;
use ssimulacra2::{ColorPrimaries, Rgb, Ssim2Reference, TransferCharacteristic};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// Simple PRNG (xoshiro256++) to avoid dependency on rand crate
struct Rng {
    state: [u64; 4],
}

impl Rng {
    fn new(seed: u64) -> Self {
        // SplitMix64 to initialize state from seed
        let mut s = seed;
        let mut state = [0u64; 4];
        for i in 0..4 {
            s = s.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            state[i] = z ^ (z >> 31);
        }
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    fn gen_range(&mut self, range: std::ops::Range<usize>) -> usize {
        let len = range.end - range.start;
        if len == 0 {
            return range.start;
        }
        range.start + (self.next_u64() as usize % len)
    }

    fn gen_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn gen_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn gen_range_f32(&mut self, range: std::ops::Range<f32>) -> f32 {
        range.start + self.gen_f32() * (range.end - range.start)
    }
}

/// Default base YCbCr matrix from jpegli (192 values)
const BASE_QUANT_MATRIX_YCBCR: [f32; 192] = [
    // Channel 0 (Y - Luminance)
    1.239_740_9,
    1.722_711_5,
    2.921_216_7,
    2.812_737_4,
    3.339_819_7,
    3.463_603_8,
    3.840_915_2,
    3.869_56,
    1.722_711_5,
    2.092_889_4,
    2.845_676,
    2.704_506_8,
    3.440_767_4,
    3.166_232_4,
    4.025_208_7,
    4.035_324_5,
    2.921_216_7,
    2.845_676,
    2.958_740_4,
    3.386_295,
    3.619_523_8,
    3.904_628,
    3.757_835_8,
    4.237_447_5,
    2.812_737_4,
    2.704_506_8,
    3.386_295,
    3.380_058_8,
    4.167_986_7,
    4.805_510_6,
    4.784_259,
    4.605_934,
    3.339_819_7,
    3.440_767_4,
    3.619_523_8,
    4.167_986_7,
    4.579_851_3,
    4.923_237,
    5.574_107,
    5.485_333_4,
    3.463_603_8,
    3.166_232_4,
    3.904_628,
    4.805_510_6,
    4.923_237,
    5.439_36,
    5.093_895_7,
    6.087_225_4,
    3.840_915_2,
    4.025_208_7,
    3.757_835_8,
    4.784_259,
    5.574_107,
    5.093_895_7,
    5.438_461,
    5.403_736,
    3.869_56,
    4.035_324_5,
    4.237_447_5,
    4.605_934,
    5.485_333_4,
    6.087_225_4,
    5.403_736,
    4.377_871,
    // Channel 1 (Cb - Blue difference)
    2.823_619_8,
    6.495_639_4,
    9.310_489,
    10.647_479,
    11.074_191,
    17.146_39,
    18.463_982,
    29.087_002,
    6.495_639_4,
    8.890_104,
    8.976_895_8,
    13.666_27,
    16.547_072,
    16.638_714,
    26.778_397,
    21.330_343,
    9.310_489,
    8.976_895_8,
    11.087_377,
    18.205_482,
    19.752_482,
    23.985_66,
    102.645_74,
    24.450_989,
    10.647_479,
    13.666_27,
    18.205_482,
    18.628_012,
    16.042_51,
    25.049_183,
    25.017_14,
    35.797_89,
    11.074_191,
    16.547_072,
    19.752_482,
    16.042_51,
    19.373_483,
    14.677_53,
    19.946_96,
    51.094_112,
    17.146_39,
    16.638_714,
    23.985_66,
    25.049_183,
    14.677_53,
    31.320_412,
    46.357_234,
    67.481_11,
    18.463_982,
    26.778_397,
    102.645_74,
    25.017_14,
    19.946_96,
    46.357_234,
    61.315_765,
    88.346_65,
    29.087_002,
    21.330_343,
    24.450_989,
    35.797_89,
    51.094_112,
    67.481_11,
    88.346_65,
    112.160_99,
    // Channel 2 (Cr - Red difference)
    2.921_725_5,
    4.497_681,
    7.356_344_5,
    6.583_891_5,
    8.535_608_7,
    8.799_434_4,
    9.188_341_5,
    9.482_7,
    4.497_681,
    6.309_548_9,
    7.024_609,
    7.156_445_3,
    8.049_059_2,
    7.012_429,
    6.711_923_2,
    8.380_308,
    7.356_344_5,
    7.024_609,
    6.892_101_2,
    6.882_82,
    8.782_226,
    6.877_475,
    7.885_817_6,
    8.679_09,
    6.583_891_5,
    7.156_445_3,
    6.882_82,
    7.003_073,
    7.722_346_5,
    7.955_425_7,
    7.473_411,
    8.362_933,
    8.535_608_7,
    8.049_059_2,
    8.782_226,
    7.722_346_5,
    6.778_005_9,
    9.484_922_7,
    9.043_702_7,
    8.053_178_2,
    8.799_434_4,
    7.012_429,
    6.877_475,
    7.955_425_7,
    9.484_922_7,
    8.607_606_5,
    9.922_697_4,
    64.251_35,
    9.188_341_5,
    6.711_923_2,
    7.885_817_6,
    7.473_411,
    9.043_702_7,
    9.922_697_4,
    63.184_937,
    83.352_94,
    9.482_7,
    8.380_308,
    8.679_09,
    8.362_933,
    8.053_178_2,
    64.251_35,
    83.352_94,
    114.892_02,
];

/// Default frequency exponents from jpegli
const FREQUENCY_EXPONENT: [f32; 64] = [
    1.00, 0.51, 0.67, 0.74, 1.00, 1.00, 1.00, 1.00, 0.51, 0.66, 0.69, 0.87, 1.00, 1.00, 1.00, 1.00,
    0.67, 0.69, 0.84, 0.83, 0.96, 1.00, 1.00, 1.00, 0.74, 0.87, 0.83, 1.00, 1.00, 0.91, 0.91, 1.00,
    1.00, 1.00, 0.96, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.91, 1.00, 1.00, 1.00, 1.00,
    1.00, 1.00, 1.00, 0.91, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
];

/// Default global scale for YCbCr
const GLOBAL_SCALE_YCBCR: f32 = 1.739_660_1;

/// Optimization state
#[derive(Clone)]
struct OptState {
    /// Base matrix values (192 for YCbCr)
    base_matrix: [f32; 192],
    /// Frequency exponents (64 values)
    freq_exp: [f32; 64],
    /// Global scale factor
    global_scale: f32,
}

impl OptState {
    fn new() -> Self {
        Self {
            base_matrix: BASE_QUANT_MATRIX_YCBCR,
            freq_exp: FREQUENCY_EXPONENT,
            global_scale: GLOBAL_SCALE_YCBCR,
        }
    }

    fn to_custom_matrices(&self) -> CustomQuantMatrices {
        CustomQuantMatrices::new()
            .with_ycbcr(self.base_matrix)
            .with_global_scale_ycbcr(self.global_scale)
            .with_frequency_exponents(self.freq_exp)
    }

    /// Serialize to JSON for checkpointing
    fn to_json(&self) -> String {
        let mut json = String::from("{\n");
        json.push_str("  \"base_matrix\": [");
        for (i, v) in self.base_matrix.iter().enumerate() {
            if i > 0 {
                json.push_str(", ");
            }
            if i % 8 == 0 {
                json.push_str("\n    ");
            }
            json.push_str(&format!("{:.6}", v));
        }
        json.push_str("\n  ],\n");

        json.push_str("  \"freq_exp\": [");
        for (i, v) in self.freq_exp.iter().enumerate() {
            if i > 0 {
                json.push_str(", ");
            }
            if i % 8 == 0 {
                json.push_str("\n    ");
            }
            json.push_str(&format!("{:.2}", v));
        }
        json.push_str("\n  ],\n");

        json.push_str(&format!("  \"global_scale\": {:.6}\n", self.global_scale));
        json.push_str("}\n");
        json
    }

    /// Load from JSON
    fn from_json(json: &str) -> Option<Self> {
        // Simple JSON parser for our format
        let mut state = Self::new();

        // Parse base_matrix
        if let Some(start) = json.find("\"base_matrix\": [") {
            let start = start + 16;
            if let Some(end) = json[start..].find(']') {
                let arr_str = &json[start..start + end];
                let values: Vec<f32> = arr_str
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
                if values.len() == 192 {
                    state.base_matrix.copy_from_slice(&values);
                }
            }
        }

        // Parse freq_exp
        if let Some(start) = json.find("\"freq_exp\": [") {
            let start = start + 13;
            if let Some(end) = json[start..].find(']') {
                let arr_str = &json[start..start + end];
                let values: Vec<f32> = arr_str
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
                if values.len() == 64 {
                    state.freq_exp.copy_from_slice(&values);
                }
            }
        }

        // Parse global_scale
        if let Some(start) = json.find("\"global_scale\": ") {
            let start = start + 16;
            if let Some(end) = json[start..].find(['\n', ',', '}']) {
                if let Ok(v) = json[start..start + end].trim().parse() {
                    state.global_scale = v;
                }
            }
        }

        Some(state)
    }
}

/// Perturbation strategy
enum Perturbation {
    /// Modify a single base matrix value
    SingleBase { idx: usize, delta: f32 },
    /// Modify a frequency exponent
    FreqExp { idx: usize, delta: f32 },
    /// Modify global scale
    GlobalScale { delta: f32 },
    /// Modify a block of adjacent values
    BlockBase {
        start: usize,
        count: usize,
        delta: f32,
    },
    /// Scale an entire component (Y, Cb, or Cr)
    ComponentScale { component: usize, factor: f32 },
}

/// Apply perturbation to state
fn apply_perturbation(state: &mut OptState, pert: &Perturbation) {
    match pert {
        Perturbation::SingleBase { idx, delta } => {
            state.base_matrix[*idx] = (state.base_matrix[*idx] + delta).clamp(0.1, 200.0);
        }
        Perturbation::FreqExp { idx, delta } => {
            state.freq_exp[*idx] = (state.freq_exp[*idx] + delta).clamp(0.3, 1.5);
        }
        Perturbation::GlobalScale { delta } => {
            state.global_scale = (state.global_scale + delta).clamp(0.5, 3.0);
        }
        Perturbation::BlockBase {
            start,
            count,
            delta,
        } => {
            for i in *start..(*start + count).min(192) {
                state.base_matrix[i] = (state.base_matrix[i] + delta).clamp(0.1, 200.0);
            }
        }
        Perturbation::ComponentScale { component, factor } => {
            let offset = component * 64;
            for i in offset..offset + 64 {
                state.base_matrix[i] = (state.base_matrix[i] * factor).clamp(0.1, 200.0);
            }
        }
    }
}

/// Generate random perturbation - balanced exploration/exploitation
fn random_perturbation(rng: &mut Rng, temperature: f64) -> Perturbation {
    // Scale with temperature: larger at high temp, smaller at low temp
    let scale = (temperature.sqrt() * 2.0) as f32;

    match rng.gen_range(0..100) {
        0..=49 => {
            // Single base matrix value (50%)
            Perturbation::SingleBase {
                idx: rng.gen_range(0..192),
                delta: rng.gen_range_f32(-1.5..1.5) * scale,
            }
        }
        50..=69 => {
            // Frequency exponent (20%)
            Perturbation::FreqExp {
                idx: rng.gen_range(0..64),
                delta: rng.gen_range_f32(-0.08..0.08) * scale,
            }
        }
        70..=79 => {
            // Global scale (10%)
            Perturbation::GlobalScale {
                delta: rng.gen_range_f32(-0.08..0.08) * scale,
            }
        }
        80..=89 => {
            // Block of values (10%)
            let start = rng.gen_range(0..192);
            Perturbation::BlockBase {
                start,
                count: rng.gen_range(2..8),
                delta: rng.gen_range_f32(-1.0..1.0) * scale,
            }
        }
        _ => {
            // Component scale (10%)
            Perturbation::ComponentScale {
                component: rng.gen_range(0..3),
                factor: 1.0 + rng.gen_range_f32(-0.05..0.05) * scale,
            }
        }
    }
}

/// Test image data with precomputed SSIM2 reference
struct TestImage {
    rgb: Vec<u8>,
    width: usize,
    height: usize,
    name: String,
    /// Precomputed SSIM2 reference with all scales (provides ~2x speedup)
    ssim2_ref: Ssim2Reference,
}

/// Load PNG image
fn load_png(path: &Path) -> Option<TestImage> {
    let file = fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;

    let (width, height) = (info.width as usize, info.height as usize);

    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..width * height * 3].to_vec(),
        png::ColorType::Rgba => buf[..width * height * 4]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        png::ColorType::Grayscale => buf[..width * height]
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect(),
        png::ColorType::GrayscaleAlpha => buf[..width * height * 2]
            .chunks(2)
            .flat_map(|c| [c[0], c[0], c[0]])
            .collect(),
        _ => return None,
    };

    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    // Precompute SSIM2 reference once at load time (includes blur, variance at all scales)
    let rgb_frame = Rgb::new(
        rgb.chunks(3)
            .map(|c| {
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                ]
            })
            .collect(),
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .ok()?;
    let ssim2_ref = Ssim2Reference::new(rgb_frame).ok()?;

    Some(TestImage {
        rgb,
        width,
        height,
        name,
        ssim2_ref,
    })
}

/// Compute SSIMULACRA2 score using precomputed reference (higher = better, 100 = identical)
fn compute_ssim2_with_ref(
    reference: &Ssim2Reference,
    decoded: &[u8],
    width: usize,
    height: usize,
) -> f64 {
    let decoded_rgb = Rgb::new(
        decoded
            .chunks(3)
            .map(|c| {
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                ]
            })
            .collect(),
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();

    reference.compare(decoded_rgb).unwrap_or(0.0)
}

/// Decode JPEG using zune-jpeg (faster than jpeg-decoder)
fn decode_jpeg(data: &[u8]) -> Option<Vec<u8>> {
    use std::io::Cursor;
    use zune_jpeg::JpegDecoder;
    let mut decoder = JpegDecoder::new(Cursor::new(data));
    decoder.decode().ok()
}

/// Profiling stats for the hot loop (thread-safe with atomics)
struct ProfileStats {
    encode_ns: AtomicU64,
    decode_ns: AtomicU64,
    ssim2_ns: AtomicU64,
    count: AtomicU64,
}

impl Default for ProfileStats {
    fn default() -> Self {
        Self {
            encode_ns: AtomicU64::new(0),
            decode_ns: AtomicU64::new(0),
            ssim2_ns: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }
}

impl ProfileStats {
    fn add_encode(&self, ns: u64) {
        self.encode_ns.fetch_add(ns, Ordering::Relaxed);
    }
    fn add_decode(&self, ns: u64) {
        self.decode_ns.fetch_add(ns, Ordering::Relaxed);
    }
    fn add_ssim2(&self, ns: u64) {
        self.ssim2_ns.fetch_add(ns, Ordering::Relaxed);
    }
    fn inc_count(&self) {
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    fn report(&self) {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return;
        }
        let encode = self.encode_ns.load(Ordering::Relaxed) as f64;
        let decode = self.decode_ns.load(Ordering::Relaxed) as f64;
        let ssim2 = self.ssim2_ns.load(Ordering::Relaxed) as f64;
        let total = encode + decode + ssim2;

        println!(
            "\n=== Hot Loop Profile ({} evaluations, {} threads) ===",
            count,
            rayon::current_num_threads()
        );
        println!(
            "  Encode:  {:>7.2}ms ({:>5.1}%)",
            encode / 1_000_000.0,
            100.0 * encode / total
        );
        println!(
            "  Decode:  {:>7.2}ms ({:>5.1}%)",
            decode / 1_000_000.0,
            100.0 * decode / total
        );
        println!(
            "  SSIM2:   {:>7.2}ms ({:>5.1}%)",
            ssim2 / 1_000_000.0,
            100.0 * ssim2 / total
        );
        println!("  Total CPU time: {:>7.2}ms", total / 1_000_000.0);
        println!(
            "  Per-eval (wall): {:.2}ms",
            total / 1_000_000.0 / count as f64 / rayon::current_num_threads() as f64
        );
    }
}

/// Encode with custom matrices and measure quality/size (parallel across images)
fn evaluate_state_profiled(
    state: &OptState,
    images: &[TestImage],
    quality: u8,
    stats: &ProfileStats,
) -> (f64, usize) {
    let custom = state.to_custom_matrices();

    // Process all images in parallel
    let results: Vec<(f64, usize)> = images
        .par_iter()
        .map(|img| {
            // Encode
            let t0 = Instant::now();
            let jpeg = Encoder::new()
                .width(img.width as u32)
                .height(img.height as u32)
                .pixel_format(PixelFormat::Rgb)
                .jpegli_quality(Quality::from_quality(quality.into()))
                .custom_quant_matrices(custom.clone())
                .encode(&img.rgb)
                .expect("encode failed");
            stats.add_encode(t0.elapsed().as_nanos() as u64);

            // Decode (zune-jpeg is ~3x faster than jpeg-decoder)
            let t1 = Instant::now();
            let decoded = decode_jpeg(&jpeg).expect("decode failed");
            stats.add_decode(t1.elapsed().as_nanos() as u64);

            // Measure quality (using precomputed reference)
            let t2 = Instant::now();
            let quality_score =
                compute_ssim2_with_ref(&img.ssim2_ref, &decoded, img.width, img.height);
            stats.add_ssim2(t2.elapsed().as_nanos() as u64);

            (quality_score, jpeg.len())
        })
        .collect();

    // Sum up results
    let (total_quality, total_size) = results
        .iter()
        .fold((0.0, 0), |(q, s), (qi, si)| (q + qi, s + si));

    stats.inc_count();
    (total_quality / images.len() as f64, total_size)
}

/// Baseline Pareto curve data points (bpp, ssim2) from jpegli at various Q levels
/// These are measured once at startup and used to compute Pareto distance
#[derive(Clone)]
struct ParetoCurve {
    /// (bpp, ssim2) points sorted by bpp ascending
    points: Vec<(f64, f64)>,
}

impl ParetoCurve {
    fn new() -> Self {
        Self { points: Vec::new() }
    }

    fn add_point(&mut self, bpp: f64, ssim2: f64) {
        self.points.push((bpp, ssim2));
        self.points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    }

    /// Load Pareto curve from zenjpeg benchmark CSV
    /// CSV format: source_hash,source_name,width,height,variance,edge_density,
    ///             chroma_complexity,uniform_block_fraction,config_key,quality,
    ///             cache_version,size_bytes,bpp,butteraugli,ssimulacra2,dssim,
    ///             encode_time_ms,timestamp
    fn from_zenjpeg_csv(path: &Path, config_key: &str) -> Option<Self> {
        use std::collections::HashMap;
        use std::io::{BufRead, BufReader};

        let file = fs::File::open(path).ok()?;
        let reader = BufReader::new(file);

        // Aggregate by quality level
        let mut by_quality: HashMap<u8, (f64, f64, usize)> = HashMap::new();

        for line in reader.lines().filter_map(|l| l.ok()) {
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() < 15 {
                continue;
            }

            // Check config matches
            if fields[8] != config_key {
                continue;
            }

            let quality: u8 = fields[9].parse().ok()?;
            let bpp: f64 = fields[12].parse().ok()?;
            let ssim2: f64 = fields[14].parse().ok()?;

            let entry = by_quality.entry(quality).or_insert((0.0, 0.0, 0));
            entry.0 += bpp;
            entry.1 += ssim2;
            entry.2 += 1;
        }

        if by_quality.is_empty() {
            return None;
        }

        let mut curve = ParetoCurve::new();
        for (_, (bpp_sum, ssim2_sum, count)) in by_quality {
            let avg_bpp = bpp_sum / count as f64;
            let avg_ssim2 = ssim2_sum / count as f64;
            curve.add_point(avg_bpp, avg_ssim2);
        }

        println!(
            "Loaded {} Pareto points from CSV ({})",
            curve.points.len(),
            config_key
        );
        Some(curve)
    }

    /// Get expected SSIM2 at given bpp by linear interpolation on Pareto curve
    fn expected_ssim2_at_bpp(&self, bpp: f64) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }

        // Find bracketing points
        let mut lower = None;
        let mut upper = None;

        for &(b, s) in &self.points {
            if b <= bpp {
                lower = Some((b, s));
            }
            if b >= bpp && upper.is_none() {
                upper = Some((b, s));
            }
        }

        match (lower, upper) {
            (Some((b1, s1)), Some((b2, s2))) if (b2 - b1).abs() > 0.001 => {
                // Interpolate
                let t = (bpp - b1) / (b2 - b1);
                s1 + t * (s2 - s1)
            }
            (Some((_, s)), None) => s, // Beyond upper bound
            (None, Some((_, s))) => s, // Below lower bound
            (Some((_, s)), Some(_)) => s, // Same point
            _ => 0.0,
        }
    }

    /// Compute vertical distance from Pareto curve (positive = above/better)
    fn distance_above_pareto(&self, bpp: f64, ssim2: f64) -> f64 {
        let expected = self.expected_ssim2_at_bpp(bpp);
        ssim2 - expected
    }
}

/// Fitness function: Distance above the Pareto curve with bpp penalty
/// Positive = better than baseline, negative = worse
/// Penalizes moving to very different bpp to prevent exploiting curve shape
fn fitness_pareto(ssim2: f64, bpp: f64, pareto: &ParetoCurve, target_bpp: f64) -> f64 {
    let pareto_dist = pareto.distance_above_pareto(bpp, ssim2);

    // Penalize deviation from target bpp to encourage staying at same operating point
    let bpp_penalty = 5.0 * (bpp - target_bpp).abs();

    pareto_dist - bpp_penalty
}

/// Legacy fitness function (for backward compatibility)
fn fitness(ssim2: f64, size: usize, total_pixels: usize, target_bpp: f64) -> f64 {
    let bpp = (size * 8) as f64 / total_pixels as f64;

    // Pure quality mode if target_bpp is 0
    if target_bpp <= 0.0 {
        return ssim2;
    }

    // Symmetric penalty with hard cap
    let bpp_diff = bpp - target_bpp;
    let lambda = 20.0;

    if bpp_diff > 0.0 {
        ssim2 - lambda * bpp_diff - 50.0 * bpp_diff * bpp_diff
    } else {
        ssim2 - lambda * 0.3 * bpp_diff.abs()
    }
}

/// Compute total pixels across all images
fn total_pixels(images: &[TestImage]) -> usize {
    images.iter().map(|img| img.width * img.height).sum()
}

/// Measure baseline Pareto curve by encoding at various Q levels
fn measure_pareto_curve(images: &[TestImage], q_levels: &[u8]) -> ParetoCurve {
    let pixels = total_pixels(images);
    let mut pareto = ParetoCurve::new();

    println!("\nMeasuring baseline Pareto curve:");
    println!("  Q  |  SSIM2  |  bpp  |  size");
    println!("-----|---------|-------|--------");

    for &q in q_levels {
        let mut total_ssim2 = 0.0;
        let mut total_size = 0;

        for img in images {
            let jpeg = Encoder::new()
                .width(img.width as u32)
                .height(img.height as u32)
                .pixel_format(PixelFormat::Rgb)
                .jpegli_quality(Quality::from_quality(q.into()))
                .encode(&img.rgb)
                .expect("encode failed");

            let decoded = decode_jpeg(&jpeg).expect("decode failed");
            let ssim2 = compute_ssim2_with_ref(&img.ssim2_ref, &decoded, img.width, img.height);

            total_ssim2 += ssim2;
            total_size += jpeg.len();
        }

        let avg_ssim2 = total_ssim2 / images.len() as f64;
        let bpp = (total_size * 8) as f64 / pixels as f64;

        println!(" Q{:2} | {:.4} | {:.3} | {}", q, avg_ssim2, bpp, total_size);
        pareto.add_point(bpp, avg_ssim2);
    }

    pareto
}

/// Load images from corpus directory
fn load_corpus(corpus_dir: &Path, max_images: usize) -> Vec<TestImage> {
    let mut images = Vec::new();

    let entries: Vec<_> = fs::read_dir(corpus_dir)
        .expect("Failed to read corpus directory")
        .filter_map(|e| e.ok())
        .collect();

    for entry in entries {
        if images.len() >= max_images {
            break;
        }

        let path = entry.path();
        if path.extension().map(|e| e == "png").unwrap_or(false) {
            if let Some(img) = load_png(&path) {
                println!("  Loaded: {} ({}x{})", img.name, img.width, img.height);
                images.push(img);
            }
        }
    }

    images
}

/// Simulated annealing optimizer with Pareto-distance fitness
fn optimize(
    images: &[TestImage],
    quality: u8,
    iterations: usize,
    seed: u64,
    pareto: &ParetoCurve,
    checkpoint_path: Option<&Path>,
    initial_state: Option<OptState>,
) -> OptState {
    let mut rng = Rng::new(seed);
    let profile_stats = ProfileStats::default();
    let pixels = total_pixels(images);

    // Initialize state
    let mut current = initial_state.unwrap_or_else(OptState::new);
    let (current_ssim2, current_size) =
        evaluate_state_profiled(&current, images, quality, &profile_stats);
    let current_bpp = (current_size * 8) as f64 / pixels as f64;

    // Show baseline comparison
    let baseline = OptState::new();
    let (baseline_ssim2, baseline_size) =
        evaluate_state_profiled(&baseline, images, quality, &profile_stats);
    let baseline_bpp = (baseline_size * 8) as f64 / pixels as f64;
    let baseline_dist = pareto.distance_above_pareto(baseline_bpp, baseline_ssim2);
    println!(
        "\nBaseline Q{}: SSIM2={:.4}, bpp={:.3}, pareto_dist={:+.4}",
        quality, baseline_ssim2, baseline_bpp, baseline_dist
    );

    // Use baseline bpp as target to stay at same operating point
    let target_bpp = baseline_bpp;
    let mut current_fitness = fitness_pareto(current_ssim2, current_bpp, pareto, target_bpp);
    let mut best = current.clone();
    let mut best_fitness = current_fitness;
    let mut best_ssim2 = current_ssim2;
    let mut best_size = current_size;
    let mut best_bpp = current_bpp;

    println!(
        "Initial: SSIM2={:.4}, bpp={:.3}, pareto_dist={:+.4}",
        current_ssim2, best_bpp, current_fitness
    );

    // Annealing schedule - lower temp for fine-tuning mode
    let initial_temp: f64 = 1.0;
    let final_temp: f64 = 0.001;
    let cooling_rate = (final_temp / initial_temp).powf(1.0 / iterations as f64);

    let mut temperature = initial_temp;
    let mut accepted = 0;
    let mut improved = 0;
    let mut stagnant = 0; // Track iterations without improvement
    let reheat_threshold = 500; // Longer threshold for fine-tuning
    let reheat_temp = 0.5; // Lower reheat temp for fine-tuning

    let start = Instant::now();
    let checkpoint_interval = 100;

    for i in 0..iterations {
        // Generate neighbor
        let pert = random_perturbation(&mut rng, temperature);
        let mut candidate = current.clone();
        apply_perturbation(&mut candidate, &pert);

        // Evaluate
        let (cand_ssim2, cand_size) =
            evaluate_state_profiled(&candidate, images, quality, &profile_stats);
        let cand_bpp = (cand_size * 8) as f64 / pixels as f64;
        let cand_fitness = fitness_pareto(cand_ssim2, cand_bpp, pareto, target_bpp);

        // Accept or reject
        let delta = cand_fitness - current_fitness;
        let accept = if delta > 0.0 {
            true
        } else {
            let prob = (delta / temperature).exp();
            rng.gen_f64() < prob
        };

        if accept {
            current = candidate;
            current_fitness = cand_fitness;
            accepted += 1;

            if cand_fitness > best_fitness {
                best = current.clone();
                best_fitness = cand_fitness;
                best_ssim2 = cand_ssim2;
                best_size = cand_size;
                best_bpp = cand_bpp;
                improved += 1;
                stagnant = 0;
            } else {
                stagnant += 1;
            }
        } else {
            stagnant += 1;
        }

        // Reheat if stuck for too long
        if stagnant >= reheat_threshold && temperature < reheat_temp {
            temperature = reheat_temp;
            stagnant = 0;
            println!("  [Reheating to T={:.2} at iteration {}]", temperature, i + 1);
        }

        // Cool down
        temperature *= cooling_rate;

        // Progress report
        if (i + 1) % 100 == 0 || i == iterations - 1 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = (i + 1) as f64 / elapsed;
            let eta = (iterations - i - 1) as f64 / rate;

            println!(
                "[{:5}/{:5}] T={:.4} SSIM2={:.4} bpp={:.3} pareto_dist={:+.4} accept={:.1}% improve={} ETA={:.0}s",
                i + 1,
                iterations,
                temperature,
                best_ssim2,
                best_bpp,
                best_fitness,
                100.0 * accepted as f64 / (i + 1) as f64,
                improved,
                eta
            );
        }

        // Checkpoint
        if let Some(path) = checkpoint_path {
            if (i + 1) % checkpoint_interval == 0 {
                let json = best.to_json();
                let _ = fs::write(path, &json);
            }
        }
    }

    println!("\n=== Optimization Complete ===");
    println!(
        "Best: SSIM2={:.4}, bpp={:.3}, pareto_dist={:+.4}",
        best_ssim2, best_bpp, best_fitness
    );
    println!(
        "vs baseline: SSIM2 {:+.4}, bpp {:+.3}",
        best_ssim2 - baseline_ssim2,
        best_bpp - baseline_bpp
    );
    println!(
        "Accepted: {}/{} ({:.1}%)",
        accepted,
        iterations,
        100.0 * accepted as f64 / iterations as f64
    );
    println!("Improved: {} times", improved);

    // Print profiling stats
    profile_stats.report();

    best
}

fn print_usage() {
    eprintln!("Usage: optimize_quant_matrices <corpus_dir> [options]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --quality <N>      Target quality level (default: 85)");
    eprintln!("  --iterations <N>   SA iterations (default: 10000)");
    eprintln!("  --max-images <N>   Max images to load (default: 20)");
    eprintln!("  --output <file>    Output file for best matrices (JSON)");
    eprintln!("  --resume <file>    Resume from checkpoint");
    eprintln!("  --seed <N>         Random seed (default: 42)");
    eprintln!();
    eprintln!("Fitness is Pareto distance: positive = above baseline curve (better)");
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    let corpus_dir = PathBuf::from(&args[1]);
    if !corpus_dir.is_dir() {
        eprintln!("Error: {} is not a directory", corpus_dir.display());
        std::process::exit(1);
    }

    // Parse options
    let mut quality: u8 = 85;
    let mut iterations: usize = 10000;
    let mut max_images: usize = 20;
    let mut output_path: Option<PathBuf> = None;
    let mut resume_path: Option<PathBuf> = None;
    let mut seed: u64 = 42;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--quality" => {
                quality = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(85);
                i += 2;
            }
            "--iterations" => {
                iterations = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(10000);
                i += 2;
            }
            "--max-images" => {
                max_images = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(20);
                i += 2;
            }
            "--output" => {
                output_path = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--resume" => {
                resume_path = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--seed" => {
                seed = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(42);
                i += 2;
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                print_usage();
                std::process::exit(1);
            }
        }
    }

    println!("=== Quantization Matrix Optimizer ===");
    println!("Corpus: {}", corpus_dir.display());
    println!("Quality: {}", quality);
    println!("Iterations: {}", iterations);
    println!("Max images: {}", max_images);
    println!("Seed: {}", seed);
    println!();

    // Load corpus
    println!("Loading images...");
    let images = load_corpus(&corpus_dir, max_images);
    if images.is_empty() {
        eprintln!("Error: No PNG images found in corpus");
        std::process::exit(1);
    }
    println!("Loaded {} images\n", images.len());

    // Load initial state from resume file if provided
    let initial_state = resume_path.as_ref().and_then(|path| {
        println!("Resuming from: {}", path.display());
        fs::read_to_string(path)
            .ok()
            .and_then(|json| OptState::from_json(&json))
    });

    // Try loading Pareto curve from zenjpeg benchmark CSV, fall back to measuring
    let zenjpeg_csv = PathBuf::from("/home/lilith/work/zenjpeg/heuristic_outputs/results.csv");
    let pareto = if zenjpeg_csv.exists() {
        ParetoCurve::from_zenjpeg_csv(&zenjpeg_csv, "jpegli-444")
            .unwrap_or_else(|| {
                println!("Failed to load from CSV, measuring locally...");
                let q_levels: Vec<u8> = (60..=98).step_by(2).collect();
                measure_pareto_curve(&images, &q_levels)
            })
    } else {
        println!("Zenjpeg CSV not found, measuring Pareto curve...");
        let q_levels: Vec<u8> = (60..=98).step_by(2).collect();
        measure_pareto_curve(&images, &q_levels)
    };
    println!("Pareto curve has {} points", pareto.points.len());

    // Checkpoint path
    let checkpoint_path = output_path
        .as_ref()
        .map(|p| p.with_extension("checkpoint.json"));

    // Run optimization
    let best = optimize(
        &images,
        quality,
        iterations,
        seed,
        &pareto,
        checkpoint_path.as_deref(),
        initial_state,
    );

    // Save best result
    if let Some(path) = output_path {
        let json = best.to_json();
        fs::write(&path, &json).expect("Failed to write output");
        println!("\nBest matrices saved to: {}", path.display());
    }

    // Print the optimized matrices for copy-paste
    println!("\n=== Optimized Base Matrix (paste into consts.rs) ===");
    println!("pub const OPTIMIZED_QUANT_MATRIX_YCBCR: [f32; 192] = [");
    for component in 0..3 {
        let name = ["Y", "Cb", "Cr"][component];
        println!("    // Channel {} ({})", component, name);
        for row in 0..8 {
            print!("    ");
            for col in 0..8 {
                let idx = component * 64 + row * 8 + col;
                print!("{:.6}, ", best.base_matrix[idx]);
            }
            println!();
        }
    }
    println!("];");
    println!(
        "\npub const OPTIMIZED_GLOBAL_SCALE: f32 = {:.6};",
        best.global_scale
    );

    println!("\npub const OPTIMIZED_FREQ_EXP: [f32; 64] = [");
    for row in 0..8 {
        print!("    ");
        for col in 0..8 {
            let idx = row * 8 + col;
            print!("{:.2}, ", best.freq_exp[idx]);
        }
        println!();
    }
    println!("];");
}
