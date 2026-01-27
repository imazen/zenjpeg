//! Vec utilization analysis example.
//!
//! Run with:
//!   cargo run --release --example vec_utilization --features alloc-instrument
//!
//! This demonstrates InstrumentedVec for profiling allocation patterns.
//! To profile the actual encoder, key Vec allocations would need to be
//! replaced with InstrumentedVec (or use a custom global allocator).

use zenjpeg::foundation::instrumented_vec::InstrumentedVec;

fn main() {
    eprintln!("=== Vec Utilization Analysis ===\n");

    // Simulate encoder allocation patterns

    // 1. Well-sized allocation (knows exact size)
    eprintln!("1. Well-sized (exact capacity):");
    {
        let mut v: InstrumentedVec<u8> = InstrumentedVec::with_capacity(1000, "well_sized");
        for i in 0..1000u16 {
            v.push(i as u8);
        }
        // Stats logged on drop
    }

    // 2. Over-allocated (common anti-pattern)
    eprintln!("\n2. Over-allocated (100x estimate):");
    {
        let mut v: InstrumentedVec<u8> = InstrumentedVec::with_capacity(100_000, "over_allocated");
        for i in 0..1000u16 {
            v.push(i as u8);
        }
        // Shows 1% utilization
    }

    // 3. Under-allocated (causes reallocations)
    eprintln!("\n3. Under-allocated (causes reallocations):");
    {
        let mut v: InstrumentedVec<u8> = InstrumentedVec::with_capacity(10, "under_allocated");
        for i in 0..1000u16 {
            v.push(i as u8);
        }
        // Shows realloc count
    }

    // 4. Simulating entropy encoder pattern (3 bytes/block estimate)
    eprintln!("\n4. Entropy encoder simulation:");
    {
        let blocks = 32400; // 1080p: 240*135
        let estimate = blocks * 3; // Our new estimate

        // Typical photo content: ~5-8 bytes/block
        let actual_bytes = blocks * 6;

        let mut v: InstrumentedVec<u8> =
            InstrumentedVec::with_capacity(estimate, "entropy_encoder_3x");
        for _ in 0..actual_bytes {
            v.push(0u8);
        }
        // Shows that 3 bytes/block estimate causes 1 realloc for typical content
    }

    // 5. Old entropy encoder pattern (100 bytes/block estimate)
    eprintln!("\n5. OLD entropy encoder simulation (100 bytes/block):");
    {
        let blocks = 32400;
        let estimate = blocks * 100; // Old over-estimate

        let actual_bytes = blocks * 6;

        let mut v: InstrumentedVec<u8> =
            InstrumentedVec::with_capacity(estimate, "entropy_encoder_100x");
        for _ in 0..actual_bytes {
            v.push(0u8);
        }
        // Shows massive waste
    }

    eprintln!("\n=== Summary ===");
    eprintln!("The entropy encoder fix (100→3 bytes/block) reduces waste from");
    eprintln!("~3MB to ~0 for typical 1080p images, with occasional reallocations");
    eprintln!("for high-frequency content.");
}
