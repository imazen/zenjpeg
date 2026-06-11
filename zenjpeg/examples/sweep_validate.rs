//! Empirical validation of the curated sweep axes (`encode::sweep`).
//!
//! Encodes the default stratum plus every single-deviation stratum of
//! [`SweepAxes::modes_full`] on a small mixed corpus (CID22-512 photos,
//! synthetic noise / complex / checkerboard, one 64×64 tiny) and checks:
//!
//! 1. **Fingerprint contract** — equal fingerprint ⇒ byte-identical
//!    output, on real encodes of the documented alias pairs (including
//!    the `speed_mode` exclusion, attacked on high-entropy noise), plus
//!    a distinct-fingerprint negative control.
//! 2. **No inert step** — every curated scalar step changes output
//!    bytes vs the default stratum somewhere in the subset, and the
//!    within-axis probe pairs (λ₂ 16.0 vs 17.0, delta-DC vs default
//!    trellis, exponent-2 vs linear coupling, asymmetric chroma
//!    [1,2] vs [2,1]) are mutually distinct.
//! 3. **Documented directions** — λ₁ ladder monotone in size, coupling
//!    −4 < +4, pre-blur and 2× chroma-distance shrink files, 0.5×
//!    grows them (soft checks: reported, non-fatal).
//! 4. **Queue ordering invariants** on the emitted plan.
//! 5. **ssim2 sanity floor** at q85 (catches corrupt pixel paths).
//!
//! Run:
//! ```bash
//! GIT_COMMIT=$(git rev-parse --short HEAD) cargo run --release \
//!   --example sweep_validate --features __expert -- \
//!   --out benchmarks/sweep_validate_$(date +%F).tsv
//! ```
//!
//! Exit code is non-zero on any hard failure (contract violation,
//! inert scalar step, ordering breakage, encode error).

use std::collections::HashMap;
use std::io::Write as _;

use rgb::ComponentBytes;
use zenjpeg::decoder::Decoder;
use zenjpeg::encode::sweep::{
    QualityGrid, SweepAxes, SweepBuilder, fingerprint, trellis_auto_shape, trellis_lambda,
};
use zenjpeg::encode::trellis::{TrellisConfig, TrellisSpeedMode};
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout, QuantTableConfig};
use zenjpeg_bench_utils::{
    RgbImage, codec_corpus_dir, generate_checkerboard, generate_complex, generate_noise,
    generate_photo_like, load_png,
};

const DEFAULT_BASE: &str = "jp3_t0_small_420";
const Q_GRID: [f32; 6] = [10.0, 30.0, 50.0, 70.0, 85.0, 95.0];

fn fnv64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn image_bytes(img: &RgbImage) -> &[u8] {
    assert_eq!(img.stride(), img.width(), "harness expects tight buffers");
    img.buf().as_bytes()
}

fn encode(cfg: &EncoderConfig, img: &RgbImage) -> Vec<u8> {
    cfg.encode_bytes(
        image_bytes(img),
        img.width() as u32,
        img.height() as u32,
        PixelLayout::Rgb8Srgb,
    )
    .unwrap_or_else(|e| panic!("encode failed: {e}"))
}

fn ssim2(orig: &RgbImage, jpeg: &[u8]) -> f64 {
    use fast_ssim2::{LinearRgbImage, compute_ssimulacra2, srgb_u8_to_linear};
    let decoded = match Decoder::new().decode(jpeg, enough::Unstoppable) {
        Ok(d) => d,
        Err(_) => return f64::NAN,
    };
    let Some(pixels) = decoded.pixels_u8() else {
        return f64::NAN;
    };
    let to_linear = |bytes: &[u8]| {
        let px: Vec<[f32; 3]> = bytes
            .chunks_exact(3)
            .map(|c| {
                [
                    srgb_u8_to_linear(c[0]),
                    srgb_u8_to_linear(c[1]),
                    srgb_u8_to_linear(c[2]),
                ]
            })
            .collect();
        LinearRgbImage::new(px, orig.width(), orig.height())
    };
    compute_ssimulacra2(to_linear(image_bytes(orig)), to_linear(pixels)).unwrap_or(f64::NAN)
}

/// Strip the `_q…` suffix; return (base_id, q-token).
fn split_q(id: &str) -> (&str, &str) {
    let at = id.rfind("_q").expect("cell id must end in _q<q>");
    (&id[..at], &id[at + 2..])
}

/// Diff a base id against the default stratum's tokens. Returns
/// (deviation count, label of the deviating token(s)).
///
/// The fourth token carries the color mode plus '-'-joined flag
/// suffixes ("444-noaq-blur0.4"); the color and each flag are separate
/// axes, so each counts as its own deviation.
fn parse_label(base: &str) -> (usize, String) {
    let def: Vec<&str> = DEFAULT_BASE.splitn(4, '_').collect();
    let got: Vec<&str> = base.splitn(4, '_').collect();
    assert_eq!(got.len(), 4, "unparseable id {base}");
    let mut devs = Vec::new();
    for (d, g) in def.iter().zip(&got).take(3) {
        if d != g {
            devs.push((*g).to_string());
        }
    }
    let mut colflags = got[3].split('-');
    let col = colflags.next().unwrap_or_default();
    if col != def[3] {
        devs.push(col.to_string());
    }
    for flag in colflags {
        devs.push(flag.to_string());
    }
    (devs.len(), devs.join("+"))
}

struct Measure {
    bytes: usize,
    hash: u64,
    ssim2: f64,
}

fn main() {
    let out_path = {
        let args: Vec<String> = std::env::args().collect();
        args.iter()
            .position(|a| a == "--out")
            .and_then(|i| args.get(i + 1).cloned())
            .unwrap_or_else(|| "benchmarks/sweep_validate.tsv".to_string())
    };
    let mut hard_failures: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    // ------------------------------------------------------------------
    // Corpus: 3 CID22-512 photos + 3 synthetic 512s + one 64×64 tiny.
    // ------------------------------------------------------------------
    let mut images: Vec<(String, RgbImage)> = Vec::new();
    let cid_dir = codec_corpus_dir()
        .expect("codec corpus not found")
        .join("CID22/CID22-512/validation");
    let mut cid: Vec<_> = std::fs::read_dir(&cid_dir)
        .expect("CID22-512/validation missing")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "png"))
        .collect();
    cid.sort();
    for p in cid.iter().take(3) {
        let name = format!("cid_{}", p.file_stem().unwrap().to_string_lossy());
        images.push((name, load_png(p).expect("png load")));
    }
    images.push(("noise512".into(), generate_noise(512, 512, 42)));
    images.push(("complex512".into(), generate_complex(512, 512)));
    images.push(("checker512".into(), generate_checkerboard(512, 512, 8)));
    images.push(("tiny64".into(), generate_photo_like(64, 64)));

    // ------------------------------------------------------------------
    // Plan + ordering invariants.
    // ------------------------------------------------------------------
    let plan = SweepBuilder::new(
        SweepAxes::modes_full(),
        QualityGrid::Explicit(Q_GRID.to_vec()),
    )
    .plan();
    println!(
        "plan: {} cells, {} merged aliases, {} invalid strata",
        plan.cells.len(),
        plan.duplicates_merged,
        plan.invalid_skipped.len()
    );
    if plan.cells[0].deviations != 0 || !plan.cells[0].id.starts_with(DEFAULT_BASE) {
        hard_failures.push(format!(
            "ordering: first cell is not the default stratum ({})",
            plan.cells[0].id
        ));
    }
    if plan
        .cells
        .windows(2)
        .any(|w| w[1].deviations < w[0].deviations)
    {
        hard_failures.push("ordering: deviations not non-decreasing".into());
    }
    {
        let mut seen = std::collections::HashSet::new();
        for c in &plan.cells {
            for id in std::iter::once(&c.id).chain(c.aliases.iter()) {
                if !seen.insert(id.clone()) {
                    hard_failures.push(format!("duplicate cell id {id}"));
                }
            }
        }
    }

    // dev≤1 prefix (sorted ⇒ contiguous), plus id → canonical-index map
    // covering aliases so merged spellings resolve to their encoder.
    let subset: Vec<usize> = plan
        .cells
        .iter()
        .enumerate()
        .take_while(|(_, c)| c.deviations <= 1)
        .map(|(i, _)| i)
        .collect();
    let mut resolve: HashMap<String, usize> = HashMap::new();
    for (i, c) in plan.cells.iter().enumerate() {
        resolve.insert(c.id.clone(), i);
        for a in &c.aliases {
            resolve.insert(a.clone(), i);
        }
    }
    println!(
        "subset: {} canonical cells (dev<=1) x {} images",
        subset.len(),
        images.len()
    );
    for (i, c) in plan.cells.iter().enumerate().take(subset.len()) {
        let (base, _) = split_q(&c.id);
        let (n, _) = parse_label(base);
        if n != c.deviations as usize {
            hard_failures.push(format!(
                "id/deviation mismatch: {} parses to {} deviations, cell says {}",
                c.id, n, c.deviations
            ));
        }
        let _ = i;
    }

    // ------------------------------------------------------------------
    // Encode the subset.
    // ------------------------------------------------------------------
    let t0 = std::time::Instant::now();
    let mut measures: HashMap<(usize, usize), Measure> = HashMap::new();
    for (ii, (iname, img)) in images.iter().enumerate() {
        for &ci in &subset {
            let cell = &plan.cells[ci];
            let jpeg = encode(&cell.config, img);
            let score = ssim2(img, &jpeg);
            measures.insert(
                (ci, ii),
                Measure {
                    bytes: jpeg.len(),
                    hash: fnv64(&jpeg),
                    ssim2: score,
                },
            );
        }
        println!("  encoded {} cells on {iname}", subset.len());
    }
    println!("encode+score: {:.1}s", t0.elapsed().as_secs_f64());

    // ------------------------------------------------------------------
    // TSV.
    // ------------------------------------------------------------------
    if let Some(dir) = std::path::Path::new(&out_path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut tsv = std::fs::File::create(&out_path).expect("tsv create");
    writeln!(
        tsv,
        "# sweep_validate: modes_full dev<=1 subset, q={Q_GRID:?}\n# git_commit: {}\n# images: {}",
        std::env::var("GIT_COMMIT").unwrap_or_else(|_| "unknown".into()),
        images
            .iter()
            .map(|(n, i)| format!("{n}({}x{})", i.width(), i.height()))
            .collect::<Vec<_>>()
            .join(", ")
    )
    .unwrap();
    writeln!(
        tsv,
        "image\tbase_id\tlabel\tdeviations\tq\tbytes\tssim2\tfingerprint\tbytes_fnv"
    )
    .unwrap();
    for (ii, (iname, _)) in images.iter().enumerate() {
        for &ci in &subset {
            let c = &plan.cells[ci];
            let m = &measures[&(ci, ii)];
            let (base, q) = split_q(&c.id);
            let (_, label) = parse_label(base);
            writeln!(
                tsv,
                "{iname}\t{base}\t{label}\t{}\t{q}\t{}\t{:.3}\t{:016x}\t{:016x}",
                c.deviations, m.bytes, m.ssim2, c.fingerprint, m.hash
            )
            .unwrap();
        }
    }
    println!("wrote {out_path}");

    // ------------------------------------------------------------------
    // Per-label aggregates vs the default stratum.
    // ------------------------------------------------------------------
    let baseline = |ii: usize, q: f32| -> &Measure {
        let id = format!("{DEFAULT_BASE}_q{q}");
        &measures[&(resolve[&id], ii)]
    };
    // Collect every dev-1 base id present in the subset (canonical or alias).
    let mut bases: Vec<String> = Vec::new();
    for &ci in &subset {
        let c = &plan.cells[ci];
        for id in std::iter::once(&c.id).chain(c.aliases.iter()) {
            let (base, _) = split_q(id);
            let (n, _) = parse_label(base);
            if n == 1 && !bases.iter().any(|b| b == base) {
                bases.push(base.to_string());
            }
        }
    }
    struct Agg {
        label: String,
        n: usize,
        differing: usize,
        dsize_sum: f64,
        dsize_min: f64,
        dsize_max: f64,
        dssim_sum: f64,
        dssim_n: usize,
    }
    let mut aggs: Vec<Agg> = Vec::new();
    for base in &bases {
        let (_, label) = parse_label(base);
        let mut a = Agg {
            label: label.clone(),
            n: 0,
            differing: 0,
            dsize_sum: 0.0,
            dsize_min: f64::INFINITY,
            dsize_max: f64::NEG_INFINITY,
            dssim_sum: 0.0,
            dssim_n: 0,
        };
        for (ii, _) in images.iter().enumerate() {
            for &q in &Q_GRID {
                let id = format!("{base}_q{q}");
                let Some(&ci) = resolve.get(&id) else {
                    continue; // stratum invalid at this q (not expected for dev-1)
                };
                let Some(m) = measures.get(&(ci, ii)) else {
                    hard_failures.push(format!(
                        "dev-1 spelling {id} resolved to un-encoded cell {} (dev {})",
                        plan.cells[ci].id, plan.cells[ci].deviations
                    ));
                    continue;
                };
                let b = baseline(ii, q);
                a.n += 1;
                if m.hash != b.hash {
                    a.differing += 1;
                }
                let d = (m.bytes as f64 - b.bytes as f64) / b.bytes as f64 * 100.0;
                a.dsize_sum += d;
                a.dsize_min = a.dsize_min.min(d);
                a.dsize_max = a.dsize_max.max(d);
                if m.ssim2.is_finite() && b.ssim2.is_finite() {
                    a.dssim_sum += m.ssim2 - b.ssim2;
                    a.dssim_n += 1;
                }
            }
        }
        aggs.push(a);
    }
    aggs.sort_by(|x, y| x.label.cmp(&y.label));
    println!(
        "\n{:<28} {:>5} {:>6} {:>9} {:>9} {:>9} {:>8}",
        "label", "n", "diff%", "dsize%", "min", "max", "dssim2"
    );
    for a in &aggs {
        println!(
            "{:<28} {:>5} {:>5.0}% {:>8.2}% {:>8.2}% {:>8.2}% {:>+8.2}",
            a.label,
            a.n,
            a.differing as f64 / a.n as f64 * 100.0,
            a.dsize_sum / a.n as f64,
            a.dsize_min,
            a.dsize_max,
            a.dssim_sum / a.dssim_n.max(1) as f64
        );
    }

    // Hard inert check: every curated scalar/mode step must change bytes
    // somewhere. Scan modes are exempt (Smallest legitimately equals the
    // winning explicit mode); they get a WARN instead.
    let soft_labels = ["prog", "base", "smsrch", "pmoz", "psrch"];
    for a in &aggs {
        if a.differing == 0 {
            if soft_labels.contains(&a.label.as_str()) {
                warnings.push(format!(
                    "scan mode {} byte-identical to Smallest everywhere (expected when the trial picks it)",
                    a.label
                ));
            } else {
                hard_failures.push(format!(
                    "INERT STEP: {} never changed output bytes across {} (image,q) pairs",
                    a.label, a.n
                ));
            }
        }
    }

    // ------------------------------------------------------------------
    // Within-axis probe distinctness (must differ somewhere).
    // ------------------------------------------------------------------
    let pair_differs = |base_a: &str, base_b: &str| -> (bool, usize) {
        let mut n = 0;
        let mut differs = false;
        for (ii, _) in images.iter().enumerate() {
            for &q in &Q_GRID {
                let (Some(&ca), Some(&cb)) = (
                    resolve.get(&format!("{base_a}_q{q}")),
                    resolve.get(&format!("{base_b}_q{q}")),
                ) else {
                    continue;
                };
                n += 1;
                if measures[&(ca, ii)].hash != measures[&(cb, ii)].hash {
                    differs = true;
                }
            }
        }
        (differs, n)
    };
    let must_differ = [
        (
            "jp3_tr14.75l216_small_420",
            "jp3_tr14.75l217_small_420",
            "lambda2 16.0 vs 17.0",
        ),
        (
            "jp3_tr14.75+dcddc1_small_420",
            "jp3_tr14.75+dc_small_420",
            "delta_dc 1.0 vs 0.0",
        ),
        (
            "jp3_tr14.75cpl-4e2cl1_small_420",
            "jp3_tr14.75cpl-4cl1_small_420",
            "coupling exponent 2 vs 1",
        ),
        (
            "jp3_tr14.75cpl-8cl1_small_420",
            "jp3_tr14.75cpl-4cl1_small_420",
            "coupling -8 vs -4 (both clamped)",
        ),
        (
            "jp3[1,2]_t0_small_420",
            "jp3[2,1]_t0_small_420",
            "asymmetric chroma [1,2] vs [2,1]",
        ),
        (
            "jp3_tr13.5_small_420",
            "jp3_tr14_small_420",
            "lambda 13.5 vs 14.0",
        ),
        (
            "jp3_tr15.5_small_420",
            "jp3_tr16_small_420",
            "lambda 15.5 vs 16.0",
        ),
    ];
    println!();
    for (a, b, what) in must_differ {
        let (differs, n) = pair_differs(a, b);
        if n == 0 {
            hard_failures.push(format!("probe pair missing from plan: {what} ({a} / {b})"));
        } else if differs {
            println!("PASS distinct: {what} (over {n} pairs)");
        } else {
            hard_failures.push(format!(
                "INERT PROBE: {what} byte-identical across all {n} (image,q) pairs"
            ));
        }
    }

    // ------------------------------------------------------------------
    // Fingerprint contract on real encodes.
    // ------------------------------------------------------------------
    let photo = &images[0].1;
    let noise = &images[3].1;
    let byte_pair = |what: &str,
                     a: &EncoderConfig,
                     b: &EncoderConfig,
                     img: &RgbImage,
                     expect_equal: bool,
                     hard_failures: &mut Vec<String>| {
        let fa = fingerprint(a);
        let fb = fingerprint(b);
        let ea = encode(a, img);
        let eb = encode(b, img);
        let bytes_equal = ea == eb;
        if expect_equal {
            if fa != fb {
                hard_failures.push(format!("{what}: fingerprints differ but should alias"));
            }
            if bytes_equal {
                println!("PASS alias bytes: {what}");
            } else {
                hard_failures.push(format!(
                    "FINGERPRINT CONTRACT VIOLATION: {what} — equal fingerprint, bytes differ ({} vs {} bytes)",
                    ea.len(),
                    eb.len()
                ));
            }
        } else {
            if fa == fb {
                hard_failures.push(format!("{what}: fingerprints equal but should differ"));
            }
            if bytes_equal {
                hard_failures.push(format!("{what}: control pair produced identical bytes"));
            } else {
                println!("PASS control distinct: {what}");
            }
        }
    };
    println!();
    byte_pair(
        "auto_optimize vs explicit trellis(14.5)+progressive @q85",
        &EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter).auto_optimize(true),
        &EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .progressive(true)
            .trellis(trellis_auto_shape()),
        photo,
        true,
        &mut hard_failures,
    );
    byte_pair(
        "allow_16bit false vs true @q95 (tables fit 8-bit)",
        &EncoderConfig::ycbcr(95, ChromaSubsampling::Quarter).allow_16bit_quant_tables(false),
        &EncoderConfig::ycbcr(95, ChromaSubsampling::Quarter).allow_16bit_quant_tables(true),
        photo,
        true,
        &mut hard_failures,
    );
    byte_pair(
        "Glassa q70 vs q90 (anchor clamp)",
        &EncoderConfig::ycbcr(70, ChromaSubsampling::Quarter)
            .quant_table_config(QuantTableConfig::GlassaLowBpp),
        &EncoderConfig::ycbcr(90, ChromaSubsampling::Quarter)
            .quant_table_config(QuantTableConfig::GlassaLowBpp),
        photo,
        true,
        &mut hard_failures,
    );
    // speed_mode bounds the trellis search, so it changes bytes on
    // high-entropy content — the fingerprint must separate it. (An
    // earlier fingerprint excluded it as "output-neutral"; this run
    // falsified that with a 582-byte delta on noise.)
    byte_pair(
        "speed_mode Adaptive vs Thorough @q95 on noise512 (search-bound knob)",
        &EncoderConfig::ycbcr(95, ChromaSubsampling::Quarter).trellis(TrellisConfig::default()),
        &EncoderConfig::ycbcr(95, ChromaSubsampling::Quarter).trellis(TrellisConfig {
            speed_mode: TrellisSpeedMode::Thorough,
            ..TrellisConfig::default()
        }),
        noise,
        false,
        &mut hard_failures,
    );
    byte_pair(
        "negative control: lambda 13.5 vs 16.0 @q85",
        &EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter).trellis(trellis_lambda(13.5)),
        &EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter).trellis(trellis_lambda(16.0)),
        photo,
        false,
        &mut hard_failures,
    );

    // ------------------------------------------------------------------
    // Soft direction checks (512px images only, mean bytes per label).
    // ------------------------------------------------------------------
    let mean_bytes = |base: &str| -> f64 {
        let mut sum = 0f64;
        let mut n = 0usize;
        for (ii, (iname, _)) in images.iter().enumerate() {
            if iname == "tiny64" {
                continue;
            }
            for &q in &Q_GRID {
                if let Some(&ci) = resolve.get(&format!("{base}_q{q}")) {
                    sum += measures[&(ci, ii)].bytes as f64;
                    n += 1;
                }
            }
        }
        sum / n.max(1) as f64
    };
    println!();
    let ladder = [
        ("jp3_tr13.5_small_420", 13.5),
        ("jp3_tr14_small_420", 14.0),
        ("jp3_tr14.5_small_420", 14.5),
        ("jp3_tr15.5_small_420", 15.5),
        ("jp3_tr16_small_420", 16.0),
    ];
    let sizes: Vec<f64> = ladder.iter().map(|(b, _)| mean_bytes(b)).collect();
    let monotone = sizes.windows(2).all(|w| w[0] < w[1]);
    let t0_size = mean_bytes(DEFAULT_BASE);
    println!(
        "lambda ladder mean bytes: {} vs no-trellis {:.0}",
        ladder
            .iter()
            .zip(&sizes)
            .map(|((_, l), s)| format!("λ{l}={s:.0}"))
            .collect::<Vec<_>>()
            .join(" "),
        t0_size
    );
    // Note: higher λ keeps MORE coefficients than zero-bias rounding —
    // the ladder does not converge to no-trellis; it brackets it.
    if monotone {
        println!("PASS direction: lambda ladder strictly monotone in size");
    } else {
        warnings.push(format!("lambda ladder not monotone: {sizes:?}"));
    }
    let checks = [
        (
            "coupling -4 smaller than +4",
            mean_bytes("jp3_tr14.75cpl-4cl1_small_420"),
            mean_bytes("jp3_tr14.75cpl+4cl1_small_420"),
        ),
        (
            "pre_blur 0.4 shrinks vs default",
            mean_bytes("jp3_t0_small_420-blur0.4"),
            t0_size,
        ),
        (
            "chroma 2x shrinks vs default",
            mean_bytes("jp3[2,2]_t0_small_420"),
            t0_size,
        ),
        (
            "default smaller than chroma 0.5x",
            t0_size,
            mean_bytes("jp3[0.5,0.5]_t0_small_420"),
        ),
    ];
    for (what, a, b) in checks {
        if a < b {
            println!("PASS direction: {what} ({a:.0} < {b:.0})");
        } else {
            warnings.push(format!("direction: {what} FAILED ({a:.0} >= {b:.0})"));
        }
    }

    // ssim2 sanity floor at q85 on 512px content. Pure noise legitimately
    // scores in the low-to-mid 20s at 4:2:0 q85 (incompressible content);
    // its floor exists to catch genuinely corrupt pixel paths and the
    // coupling-destruction mode (which scored NEGATIVE before the clamp).
    for (ii, (iname, _)) in images.iter().enumerate() {
        if iname == "tiny64" {
            continue;
        }
        let floor = if iname == "noise512" { 15.0 } else { 30.0 };
        for &ci in &subset {
            let c = &plan.cells[ci];
            if c.quality != 85.0 {
                continue;
            }
            let s = measures[&(ci, ii)].ssim2;
            if !s.is_finite() || s < floor {
                hard_failures.push(format!(
                    "ssim2 sanity: {} on {iname} scored {s:.1} at q85 (floor {floor})",
                    c.id
                ));
            }
        }
    }

    // ------------------------------------------------------------------
    // Exact-minimizer contracts. Smallest = min(prog-default-script,
    // restart-free sequential, tiny) with the sequential trial gated at
    // 32 KiB of progressive output; SmallestSearch additionally owns
    // script space (search winner + canonical mozjpeg + default script).
    // The swept `base` cells carry the default 4-MCU-row restart
    // markers, which sit OUTSIDE Smallest's candidate set — restart
    // resets re-base DC prediction and can rarely net out cheaper than
    // the marker bytes, so they are compared as a WARN, not a contract.
    // ------------------------------------------------------------------
    const GATE: usize = 32 * 1024;
    let cell_for = |base: &str, q: f32, ii: usize| -> Option<&Measure> {
        resolve
            .get(&format!("{base}_q{q}"))
            .and_then(|&ci| measures.get(&(ci, ii)))
    };
    for (ii, (iname, img)) in images.iter().enumerate() {
        for &q in &Q_GRID {
            let (Some(small), Some(prog), Some(base), Some(pmoz), Some(psrch), Some(smsrch)) = (
                cell_for(DEFAULT_BASE, q, ii),
                cell_for("jp3_t0_prog_420", q, ii),
                cell_for("jp3_t0_base_420", q, ii),
                cell_for("jp3_t0_pmoz_420", q, ii),
                cell_for("jp3_t0_psrch_420", q, ii),
                cell_for("jp3_t0_smsrch_420", q, ii),
            ) else {
                hard_failures.push(format!("scan-mode cells missing on {iname} q{q}"));
                continue;
            };
            if prog.bytes <= GATE {
                // The candidate Smallest actually trials: restart-free
                // sequential (the swept base uses the 4-row default).
                let small_cell = &plan.cells[resolve[&format!("{DEFAULT_BASE}_q{q}")]];
                let base_rf = encode(
                    &small_cell
                        .config
                        .clone()
                        .progressive(zenjpeg::encode::ProgressiveScanMode::Baseline)
                        .restart_mcu_rows(0),
                    img,
                )
                .len();
                if small.bytes > prog.bytes.min(base_rf) {
                    hard_failures.push(format!(
                        "Smallest not minimal under gate on {iname} q{q}: {} vs prog {} / restart-free base {base_rf}",
                        small.bytes, prog.bytes
                    ));
                }
                if smsrch.bytes > base_rf {
                    hard_failures.push(format!(
                        "SmallestSearch lost to restart-free baseline under gate on {iname} q{q}: {} vs {base_rf}",
                        smsrch.bytes
                    ));
                }
                if base.bytes < small.bytes {
                    warnings.push(format!(
                        "restart-marker DC-reset anomaly on {iname} q{q}: 4-row-interval baseline {} beats Smallest {} by {} bytes",
                        base.bytes,
                        small.bytes,
                        small.bytes - base.bytes
                    ));
                }
            } else if small.hash != prog.hash {
                hard_failures.push(format!(
                    "Smallest above gate must emit the progressive stream verbatim on {iname} q{q}"
                ));
            }
            let script_floor = prog.bytes.min(pmoz.bytes).min(psrch.bytes).min(small.bytes);
            if smsrch.bytes > script_floor {
                hard_failures.push(format!(
                    "SmallestSearch lost in script space on {iname} q{q}: {} vs floor {script_floor}",
                    smsrch.bytes
                ));
            }
        }
    }
    println!("contracts evaluated: Smallest exact-min under gate, SmallestSearch script-space sup");

    // ------------------------------------------------------------------
    // Verdict.
    // ------------------------------------------------------------------
    println!();
    for w in &warnings {
        println!("WARN {w}");
    }
    if hard_failures.is_empty() {
        println!("\nALL HARD CHECKS PASSED ({} warnings)", warnings.len());
    } else {
        println!("\n{} HARD FAILURES:", hard_failures.len());
        for f in &hard_failures {
            println!("FAIL {f}");
        }
        std::process::exit(1);
    }
}
