//! Debug: Compare wide vs autovec for individual AQ functions

fn main() {
    use jpegli::quant::aq::{simd, autovec};
    
    let block_w = 4;
    let height = 8;
    let stride = block_w * 8 + 1;
    
    let input: Vec<f32> = (0..stride * height)
        .map(|i| ((i % 256) as f32 * 1.7) % 255.0)
        .collect();
    
    println!("Testing individual functions on block 0:");
    let block = &input[0..];
    
    let gamma_wide = simd::gamma_modulation_sum_8x8(block, stride, 0, 0, block_w*8, height);
    let gamma_autovec = autovec::gamma_modulation_sum_8x8_autovec(block, stride, 0, height);
    println!("gamma: wide={:.6}, autovec={:.6}, diff={:.6}", 
             gamma_wide, gamma_autovec, (gamma_wide - gamma_autovec).abs());
    
    let hf_wide = simd::hf_modulation_sum_8x8(block, stride, 0, 0, block_w*8, height);
    let hf_autovec = autovec::hf_modulation_sum_8x8_autovec(block, stride, 0, height);
    println!("hf: wide={:.6}, autovec={:.6}, diff={:.6}",
             hf_wide, hf_autovec, (hf_wide - hf_autovec).abs());
    
    println!("\nTesting per_block_modulations:");
    let mut aq_wide = vec![0.5f32; block_w];
    let mut aq_autovec = vec![0.5f32; block_w];
    
    simd::per_block_modulations_row(
        &input, stride, block_w * 8, height, 0, block_w, &mut aq_wide, 0.841, 0.1
    );
    autovec::per_block_modulations_row_autovec(
        &input, stride, block_w * 8, height, 0, block_w, &mut aq_autovec, 0.841, 0.1
    );
    
    for bx in 0..block_w {
        let diff = (aq_wide[bx] - aq_autovec[bx]).abs();
        println!("  block {}: wide={:.6}, autovec={:.6}, diff={:.6}", 
                 bx, aq_wide[bx], aq_autovec[bx], diff);
    }
}
