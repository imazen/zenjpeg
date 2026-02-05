//! Verify arithmetic decoder produces identical coefficients to Huffman transcode.
use enough::Unstoppable;

use zenjpeg::decode::Decoder;

const TESTIMGARI_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../internal/jpegli-cpp/third_party/libjpeg-turbo/testimages/testimgari.jpg"
);

#[test]
fn verify_arithmetic_coefficients_match_huffman_transcode() {
    // Decode original arithmetic JPEG
    let ari_data = std::fs::read(TESTIMGARI_PATH).expect("failed to read arithmetic file");
    let decoder = Decoder::new();
    let ari_coeffs = decoder.decode_coefficients(&ari_data, Unstoppable).expect("failed to decode arithmetic");
    
    // Decode Huffman-transcoded version (jpegtran -copy none)
    let huff_data = std::fs::read("/tmp/testimgari_huffman.jpg").expect("failed to read huffman file");
    let huff_coeffs = decoder.decode_coefficients(&huff_data, Unstoppable).expect("failed to decode huffman");
    
    // Compare coefficient counts
    assert_eq!(ari_coeffs.components.len(), huff_coeffs.components.len(), 
               "component count mismatch");
    
    // Compare each component
    for (ci, (ari_comp, huff_comp)) in ari_coeffs.components.iter()
        .zip(huff_coeffs.components.iter()).enumerate() 
    {
        assert_eq!(ari_comp.coeffs.len(), huff_comp.coeffs.len(),
                   "component {} coefficient count mismatch", ci);
        
        let mut diff_count = 0;
        let mut max_diff = 0i16;
        
        for (i, (&a, &h)) in ari_comp.coeffs.iter().zip(huff_comp.coeffs.iter()).enumerate() {
            let diff = (a as i32 - h as i32).abs() as i16;
            if diff != 0 {
                diff_count += 1;
                max_diff = max_diff.max(diff);
                if diff_count <= 10 {
                    let block = i / 64;
                    let pos = i % 64;
                    println!("Component {}, Block {}, Position {}: ari={}, huff={}, diff={}", 
                             ci, block, pos, a, h, diff);
                }
            }
        }
        
        println!("Component {}: {} total diffs, max diff = {}", ci, diff_count, max_diff);
        
        // Coefficients should be IDENTICAL for lossless arithmetic decoding
        assert_eq!(diff_count, 0, 
                   "Component {} has {} coefficient differences (max {})", 
                   ci, diff_count, max_diff);
    }
    
    println!("All coefficients match perfectly!");
}
