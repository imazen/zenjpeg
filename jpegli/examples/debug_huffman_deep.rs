//! Deep debug of Huffman table generation

use jpegli::huffman_opt::{generate_optimal_table, FrequencyCounter};

fn main() {
    // Simulate a simple case that should work
    println!("=== Test 1: Simple case ===");
    let mut freq = [0i64; 257];
    freq[0] = 100; // symbol 0
    freq[1] = 50;  // symbol 1
    freq[2] = 25;  // symbol 2
    freq[3] = 10;  // symbol 3

    match generate_optimal_table(&mut freq) {
        Ok((bits, values)) => {
            println!("bits: {:?}", bits);
            println!("values: {:?}", values);
            validate_huffman(&bits, &values);
        }
        Err(e) => println!("Error: {:?}", e),
    }

    // Test with a FrequencyCounter
    println!("\n=== Test 2: FrequencyCounter ===");
    let mut counter = FrequencyCounter::new();
    for _ in 0..100 { counter.count(0); }
    for _ in 0..50 { counter.count(1); }
    for _ in 0..25 { counter.count(2); }
    for _ in 0..10 { counter.count(3); }

    match counter.generate_table_with_dht() {
        Ok(table) => {
            println!("bits: {:?}", table.bits);
            println!("values: {:?}", table.values);
            validate_huffman(&table.bits, &table.values);
        }
        Err(e) => println!("Error: {:?}", e),
    }

    // Test standard DC luminance for comparison
    println!("\n=== Standard DC Luminance ===");
    let std_bits = [0u8, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
    let std_values = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    validate_huffman(&std_bits, &std_values);
}

fn validate_huffman(bits: &[u8; 16], values: &[u8]) {
    // Check Kraft inequality: sum(2^(-length)) <= 1
    let mut kraft_sum = 0.0f64;
    let mut total_symbols = 0usize;

    for (i, &count) in bits.iter().enumerate() {
        let length = i + 1;
        kraft_sum += count as f64 / (1u64 << length) as f64;
        total_symbols += count as usize;
    }

    println!("Kraft sum: {} (should be <= 1.0)", kraft_sum);
    println!("Total symbols: {} (from bits), {} (from values)", total_symbols, values.len());

    if total_symbols != values.len() {
        println!("ERROR: Symbol count mismatch!");
    }

    if kraft_sum > 1.0 + 1e-9 {
        println!("ERROR: Kraft inequality violated!");
    }

    // Check JPEG validity: can we build codes from this?
    let mut code: u32 = 0;
    let mut valid = true;
    for (i, &count) in bits.iter().enumerate() {
        let length = i + 1;
        // After assigning 'count' codes at this length, shift to next length
        code += count as u32;
        if code > (1u32 << length) {
            println!("ERROR at length {}: {} codes assigned but only {} possible",
                     length, code, 1u32 << length);
            valid = false;
        }
        code <<= 1;
    }

    if valid {
        println!("JPEG validity: OK");
    }
}
