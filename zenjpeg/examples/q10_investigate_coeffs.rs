//! Decode both Q10 JPEGs at the coefficient level and compare distributions.
//! Goal: figure out what's different between zen's and cpp's quantized
//! coefficients that leads to 2.4KB more scan bytes.

use std::fs;

fn main() {
    let z = fs::read("/tmp/q10_zen.jpg").unwrap();
    let c = fs::read("/tmp/q10_cpp_baseline.jpg").unwrap();
    println!("zen {}B  cpp {}B  Δ={:+}", z.len(), c.len(), z.len() as i64 - c.len() as i64);

    let zc = decode_coeffs(&z, "zen");
    let cc = decode_coeffs(&c, "cpp");

    // Compare using built-in compare
    let cmp = zc.compare(&cc);
    println!("\nzc.compare(cc):");
    println!("  total_blocks={} differing_blocks={} total_diff_coeffs={} max_diff={}",
        cmp.total_blocks, cmp.differing_blocks, cmp.total_diff_coeffs, cmp.max_diff);
    println!("  diff_by_position (zigzag index 0-63):");
    for (i, n) in cmp.diff_by_position.iter().enumerate() {
        if *n > 0 {
            print!("    [{i:2}]={n}");
            if i % 4 == 3 { println!(); }
        }
    }
    println!();

    // Per-component stats
    println!("\nComponent-level stats:");
    println!("{:<4} {:<6} {:>10} {:>12} {:>12} {:>14} {:>14}",
             "src", "comp", "blocks", "nz_blks(Y)", "nz_blks(0)", "tot_nonzero_ac", "tot_eob_pos_sum");
    for (label, coefs) in [("zen", &zc), ("cpp", &cc)] {
        for (i, comp) in coefs.components.iter().enumerate() {
            let (nz_blocks, total_ac_nz, eob_pos_sum) = summarize(comp);
            let total_blocks = comp.num_blocks();
            println!("{:<4} {:<6} {:>10} {:>12} {:>12} {:>14} {:>14}",
                label, i, total_blocks, nz_blocks, total_blocks - nz_blocks, total_ac_nz, eob_pos_sum);
        }
    }

    // Histogram of AC magnitudes for component 0 (Y)
    println!("\nY-channel AC magnitude (category) histogram:");
    print_ac_hist("zen", &zc.components[0]);
    print_ac_hist("cpp", &cc.components[0]);

    // Bit-size estimate per block: sum over nonzero AC of (category+1) + #runs+EOB*avg
    // Simpler: compute a rough Rice/JPEG-ish predicted bit cost
    println!("\nRough JPEG-entropy bit estimate (DC + AC):");
    for (lbl, coefs) in [("zen", &zc), ("cpp", &cc)] {
        let mut total_bits: u64 = 0;
        for comp in &coefs.components {
            for b in 0..comp.num_blocks() {
                let blk = comp.block(b);
                total_bits += estimate_block_bits(blk);
            }
        }
        println!("  {lbl}: {} bits ≈ {} bytes (Huffman-pessimistic)", total_bits, total_bits / 8);
    }
}

fn decode_coeffs(data: &[u8], label: &str) -> zenjpeg::decoder::DecodedCoefficients {
    use zenjpeg::decoder::Decoder;
    let decoder = Decoder::new();
    let coefs = decoder
        .decode_coefficients(data, enough::Unstoppable)
        .expect("decode_coefficients failed");
    println!(
        "{label}: {}x{}  components={}",
        coefs.width,
        coefs.height,
        coefs.num_components()
    );
    coefs
}

fn summarize(comp: &zenjpeg::decode::ComponentCoefficients) -> (usize, u64, u64) {
    let mut nz_blocks = 0usize;
    let mut total_ac_nz: u64 = 0;
    let mut eob_pos_sum: u64 = 0;
    for b in 0..comp.num_blocks() {
        let blk = comp.block(b);
        let mut last_nz = 0usize;
        let mut any_ac = false;
        for k in 1..64 {
            if blk[k] != 0 {
                any_ac = true;
                last_nz = k;
                total_ac_nz += 1;
            }
        }
        if any_ac { nz_blocks += 1; }
        eob_pos_sum += last_nz as u64;
    }
    (nz_blocks, total_ac_nz, eob_pos_sum)
}

fn print_ac_hist(lbl: &str, comp: &zenjpeg::decode::ComponentCoefficients) {
    let mut hist = [0u64; 16];
    for b in 0..comp.num_blocks() {
        let blk = comp.block(b);
        for k in 1..64 {
            let v = blk[k];
            if v == 0 { continue; }
            let cat = 16 - (v.unsigned_abs()).leading_zeros() as usize;
            hist[cat.min(15)] += 1;
        }
    }
    let tot: u64 = hist.iter().sum();
    println!("  {lbl}: total AC nz = {tot}");
    for (c, n) in hist.iter().enumerate() {
        if *n == 0 { continue; }
        println!("    cat{:<2}: {:>10} ({:>5.2}%)  approx_bits/nz = ~{} (huff_code) + {} (value)",
                 c, n, *n as f64 * 100.0 / tot as f64, 4, c);
    }
}

/// Pessimistic baseline-JPEG block bit cost for comparison.
/// Assumes typical Huffman code lengths: DC code ~4 bits + DC value bits = cat; AC symbol ~5 bits + AC value bits = cat.
/// EOB ~4 bits. This is a *proxy*, not the real optimized Huffman cost.
fn estimate_block_bits(blk: &[i16]) -> u64 {
    let dc_cat = if blk[0] == 0 { 0 } else { 16 - blk[0].unsigned_abs().leading_zeros() as u64 };
    let mut bits = 4 + dc_cat; // DC huff + DC value

    let mut last_nz = 0usize;
    for k in 1..64 {
        if blk[k] != 0 { last_nz = k; }
    }
    if last_nz == 0 {
        bits += 4; // EOB
        return bits;
    }

    let mut run = 0;
    for k in 1..=last_nz {
        let v = blk[k];
        if v == 0 {
            run += 1;
            if run == 16 {
                bits += 11; // ZRL marker
                run = 0;
            }
        } else {
            let cat = 16 - v.unsigned_abs().leading_zeros() as u64;
            bits += 5 + cat; // AC huff symbol + value bits
            run = 0;
        }
    }
    if last_nz < 63 {
        bits += 4; // EOB
    }
    bits
}
