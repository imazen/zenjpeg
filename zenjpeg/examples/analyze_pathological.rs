//! Analyze pathological images that pass heuristics but produce poor tables.
//!
//! These images pass entropy/coverage heuristics at 25% but still produce
//! Huffman tables with >4% overhead.
//!
//! Run with: cargo run --release -p zenjpeg --features test-utils --example analyze_pathological

use zenjpeg::encode::encoder_types::Quality;
use zenjpeg::encode::streaming::StreamingEncoder;
use zenjpeg::types::Subsampling;

fn main() {
    let pathological = [
        "/home/lilith/work/codec-corpus/clic2025/validation/5e5ce43575fa67fdc0dd37146d7f479e.png",
        "/home/lilith/work/codec-corpus/clic2025/validation/d79d465ac77c36518e0f0d626bf97ec4.png",
    ];

    // Include a "good" image that benefits from safety valve
    let good = [
        "/home/lilith/work/codec-corpus/clic2025/validation/aed95e005df28e790519eefb6eb1e565.png",
    ];

    eprintln!("=== Analyzing pathological images (pass heuristics at 25%, but poor tables) ===\n");

    for path in pathological.iter() {
        analyze_image(path, true);
    }

    eprintln!("\n=== Comparing with good image (benefits from safety valve) ===\n");

    for path in good.iter() {
        analyze_image(path, false);
    }
}

fn load_png(path: &str) -> (u32, u32, Vec<u8>) {
    let decoder = png::Decoder::new(std::fs::File::open(path).expect("Failed to open file"));
    let mut reader = decoder.read_info().expect("Failed to read PNG info");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("Failed to decode PNG");
    buf.truncate(info.buffer_size());
    (info.width, info.height, buf)
}

fn analyze_image(path: &str, is_pathological: bool) {
    let name = std::path::Path::new(path)
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap();

    let (width, height, pixels) = load_png(path);

    eprintln!(
        "Image: {} ({}x{}) [{}]",
        &name[..16],
        width,
        height,
        if is_pathological {
            "PATHOLOGICAL"
        } else {
            "GOOD"
        }
    );

    // Measure heuristics at different thresholds
    let thresholds = [15, 20, 25, 30, 35, 40, 50, 75, 100];

    eprintln!(
        "{:>8} {:>10} {:>10} {:>10} {:>10}",
        "Thresh%", "AC Cov", "AC Ent", "Overhead", "Note"
    );
    eprintln!("{}", "-".repeat(55));

    // First get baseline size (full buffering)
    let baseline_size = {
        let mut encoder = StreamingEncoder::new(width, height)
            .quality(Quality::ApproxJpegli(85.0))
            .subsampling(Subsampling::S420)
            .progressive(false)
            .start()
            .unwrap();
        let row_size = width as usize * 3;
        for y in 0..height as usize {
            let start = y * row_size;
            let end = start + row_size;
            encoder.push_row(&pixels[start..end]).unwrap();
        }
        encoder.finish().unwrap().len()
    };

    for &thresh in &thresholds {
        let (overhead, ac_cov, ac_ent) =
            encode_with_threshold(width, height, &pixels, thresh, baseline_size);

        // Check if heuristics would pass (entropy >= 4.0, coverage >= 30%)
        let heuristics_pass = ac_ent >= 4.0 && ac_cov >= 30.0;
        let note = if heuristics_pass && overhead > 4.0 {
            "PASS+BAD"
        } else if heuristics_pass {
            "pass"
        } else {
            "wait"
        };

        eprintln!(
            "{:>7}% {:>9.1}% {:>9.2} {:>9.2}% {:>10}",
            thresh, ac_cov, ac_ent, overhead, note
        );
    }

    eprintln!();
}

fn encode_with_threshold(
    width: u32,
    height: u32,
    pixels: &[u8],
    threshold_percent: usize,
    baseline_size: usize,
) -> (f64, f64, f64) {
    let threshold_rows = (height as usize * threshold_percent) / 100;

    let mut encoder = StreamingEncoder::new(width, height)
        .quality(Quality::ApproxJpegli(85.0))
        .subsampling(Subsampling::S420)
        .progressive(false)
        .transition_after_rows(threshold_rows)
        .start()
        .unwrap();

    // Track heuristics at transition point
    let mut ac_cov_at_transition = 0.0;
    let mut ac_ent_at_transition = 0.0;
    let mut captured = false;

    let row_size = width as usize * 3;
    for y in 0..height as usize {
        let start = y * row_size;
        let end = start + row_size;
        encoder.push_row(&pixels[start..end]).unwrap();

        // Capture heuristics right before transition
        if !captured && y + 1 >= threshold_rows {
            let (cov, ent, _, _) = encoder.frequency_heuristics();
            ac_cov_at_transition = cov;
            ac_ent_at_transition = ent;
            captured = true;
        }
    }

    let result = encoder.finish().unwrap();
    let overhead = 100.0 * (result.len() as f64 - baseline_size as f64) / baseline_size as f64;

    (overhead, ac_cov_at_transition * 100.0, ac_ent_at_transition)
}
