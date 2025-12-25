//! Test butteraugli with uniform images

use butteraugli::{compute_butteraugli, ButteraugliParams};

fn main() {
    // Uniform gray images differing by 10
    let width = 64;
    let height = 64;
    
    let rgb1: Vec<u8> = vec![128; width * height * 3];
    let rgb2: Vec<u8> = vec![138; width * height * 3]; // +10 difference
    
    let params = ButteraugliParams::default();
    let result = compute_butteraugli(&rgb1, &rgb2, width, height, &params);
    
    println!("Uniform images (128 vs 138), 64x64:");
    println!("  butteraugli score: {:.4}", result.score);
    
    if let Some(dm) = result.diffmap.as_ref() {
        let sum: f64 = (0..dm.height()).flat_map(|y| (0..dm.width()).map(move |x| dm.get(x, y) as f64)).sum();
        let mean = sum / (dm.width() * dm.height()) as f64;
        println!("  mean diffmap: {:.6}", mean);
        
        // Sample some values
        println!("  diffmap[32,32]: {:.6}", dm.get(32, 32));
    }
    
    // Try smaller difference
    let rgb3: Vec<u8> = vec![129; width * height * 3]; // +1 difference
    let result2 = compute_butteraugli(&rgb1, &rgb3, width, height, &params);
    println!("\nUniform images (128 vs 129), 64x64:");
    println!("  butteraugli score: {:.4}", result2.score);
    
    // Very different (black vs white)
    let rgb_black: Vec<u8> = vec![0; width * height * 3];
    let rgb_white: Vec<u8> = vec![255; width * height * 3];
    let result3 = compute_butteraugli(&rgb_black, &rgb_white, width, height, &params);
    println!("\nBlack vs White, 64x64:");
    println!("  butteraugli score: {:.4}", result3.score);
}
