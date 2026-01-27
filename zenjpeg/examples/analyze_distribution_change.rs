//! Analyze how frequency distribution changes over the course of encoding.
//!
//! For pathological images, the distribution at 25% is NOT representative.
//! This tool shows how much the distribution changes between early and late encoding.
//!
//! Run with: cargo run --release -p zenjpeg --features test-utils --example analyze_distribution_change

use zenjpeg::encode::encoder_types::Quality;
use zenjpeg::encode::streaming::StreamingEncoder;
use zenjpeg::types::Subsampling;

fn main() {
    let images = [
        // Pathological - pass heuristics early but have bad tables
        ("/home/lilith/work/codec-corpus/clic2025/validation/5e5ce43575fa67fdc0dd37146d7f479e.png", "pathological1"),
        ("/home/lilith/work/codec-corpus/clic2025/validation/d79d465ac77c36518e0f0d626bf97ec4.png", "pathological2"),
        // Good - fails heuristics early (gradient sky)
        ("/home/lilith/work/codec-corpus/clic2025/validation/aed95e005df28e790519eefb6eb1e565.png", "good-gradient"),
        // Normal image for comparison
        ("/home/lilith/work/codec-corpus/clic2025/validation/2c1f84548ef99faec2b4f9bf12227c83.png", "normal"),
    ];

    eprintln!("=== Analyzing distribution stability over encoding ===\n");

    for (path, label) in images.iter() {
        if std::path::Path::new(path).exists() {
            analyze_distribution_change(path, label);
        }
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

fn analyze_distribution_change(path: &str, label: &str) {
    let (width, height, pixels) = load_png(path);

    eprintln!("Image: {} ({}x{})", label, width, height);

    // Create encoder and track distribution at different points
    let mut encoder = StreamingEncoder::new(width, height)
        .quality(Quality::ApproxJpegli(85.0))
        .subsampling(Subsampling::S420)
        .progressive(false)
        .start()
        .unwrap();

    let row_size = width as usize * 3;
    let checkpoints = [15, 25, 35, 50, 75, 100];
    let checkpoint_rows: Vec<_> = checkpoints
        .iter()
        .map(|p| (height as usize * p) / 100)
        .collect();

    let mut distributions: Vec<(usize, Vec<i64>, f64, f64)> = Vec::new();

    for y in 0..height as usize {
        let start = y * row_size;
        let end = start + row_size;
        encoder.push_row(&pixels[start..end]).unwrap();

        // Record distribution at checkpoints
        if let Some(idx) = checkpoint_rows.iter().position(|&r| r == y + 1) {
            let (_, ac_luma, _, _) = encoder.frequency_counters();
            let counts: Vec<i64> = (0..256).map(|i| ac_luma.get_count(i as u8)).collect();
            let (cov, ent, _, _) = encoder.frequency_heuristics();
            distributions.push((checkpoints[idx], counts, cov * 100.0, ent));
        }
    }

    // Analyze how distribution changed from 15% to each later checkpoint
    eprintln!(
        "{:>8} {:>10} {:>10} {:>12} {:>12}",
        "Point", "Coverage", "Entropy", "KL vs 15%", "Top10 Change"
    );
    eprintln!("{}", "-".repeat(60));

    let baseline = &distributions[0];
    for (i, (pct, counts, cov, ent)) in distributions.iter().enumerate() {
        let kl = if i == 0 {
            0.0
        } else {
            kl_divergence(&baseline.1, counts)
        };
        let top10_change = if i == 0 {
            0.0
        } else {
            top_n_frequency_change(&baseline.1, counts, 10)
        };

        eprintln!(
            "{:>7}% {:>9.1}% {:>9.2} {:>11.4} {:>11.2}%",
            pct, cov, ent, kl, top10_change * 100.0
        );
    }

    eprintln!();
}

/// KL divergence between two frequency distributions.
/// Returns infinity if Q has zero where P has non-zero.
fn kl_divergence(p: &[i64], q: &[i64]) -> f64 {
    let p_total: f64 = p.iter().map(|&x| x as f64).sum();
    let q_total: f64 = q.iter().map(|&x| x as f64).sum();

    if p_total == 0.0 || q_total == 0.0 {
        return 0.0;
    }

    let mut kl = 0.0;
    for (&p_count, &q_count) in p.iter().zip(q.iter()) {
        if p_count == 0 {
            continue;
        }
        let p_prob = p_count as f64 / p_total;
        let q_prob = (q_count as f64 + 1.0) / (q_total + 256.0); // Add smoothing
        kl += p_prob * (p_prob / q_prob).ln();
    }
    kl
}

/// Measure how much the top N symbols' relative frequencies changed.
fn top_n_frequency_change(early: &[i64], late: &[i64], n: usize) -> f64 {
    let early_total: f64 = early.iter().map(|&x| x as f64).sum();
    let late_total: f64 = late.iter().map(|&x| x as f64).sum();

    if early_total == 0.0 || late_total == 0.0 {
        return 0.0;
    }

    // Find top N symbols in the LATE distribution (what matters for final tables)
    let mut late_indexed: Vec<_> = late.iter().enumerate().collect();
    late_indexed.sort_by(|a, b| b.1.cmp(a.1));

    let mut change = 0.0;
    for &(idx, _) in late_indexed.iter().take(n) {
        let early_freq = early[idx] as f64 / early_total;
        let late_freq = late[idx] as f64 / late_total;
        change += (late_freq - early_freq).abs();
    }

    change / n as f64
}
