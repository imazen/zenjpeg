//! Benchmark comparing Vec vs SmallVec vs ArrayVec for the
//! block_refbits pattern in tokenize_ac_refinement_scan.
//!
//! Pattern: small buffer (max 63 elements) reused across many iterations,
//! cleared between uses, elements pushed one at a time.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use smallvec::SmallVec;
use tinyvec::ArrayVec;

const NUM_BLOCKS: usize = 10_000;
const MAX_REFBITS_PER_BLOCK: usize = 64;

/// Simulates the pattern: create Vec inside loop (original hot path)
fn bench_vec_inside_loop(c: &mut Criterion) {
    let mut group = c.benchmark_group("vec_variants");
    group.throughput(Throughput::Elements(NUM_BLOCKS as u64));

    // Simulate varying refbit counts per block (0-20 typical)
    let refbit_counts: Vec<usize> = (0..NUM_BLOCKS)
        .map(|i| (i * 7 + 3) % 21) // 0-20 refbits per block
        .collect();

    group.bench_function("vec_inside_loop", |b| {
        b.iter(|| {
            let mut total = 0u64;
            for &count in &refbit_counts {
                let mut v: Vec<u8> = Vec::new(); // Alloc per block!
                for i in 0..count {
                    v.push((i & 0xFF) as u8);
                }
                total += v.iter().map(|&x| x as u64).sum::<u64>();
            }
            black_box(total)
        })
    });

    group.bench_function("vec_outside_loop", |b| {
        b.iter(|| {
            let mut total = 0u64;
            let mut v: Vec<u8> = Vec::with_capacity(MAX_REFBITS_PER_BLOCK);
            for &count in &refbit_counts {
                v.clear();
                for i in 0..count {
                    v.push((i & 0xFF) as u8);
                }
                total += v.iter().map(|&x| x as u64).sum::<u64>();
            }
            black_box(total)
        })
    });

    group.bench_function("smallvec_inside_loop", |b| {
        b.iter(|| {
            let mut total = 0u64;
            for &count in &refbit_counts {
                let mut v: SmallVec<[u8; 64]> = SmallVec::new();
                for i in 0..count {
                    v.push((i & 0xFF) as u8);
                }
                total += v.iter().map(|&x| x as u64).sum::<u64>();
            }
            black_box(total)
        })
    });

    group.bench_function("smallvec_outside_loop", |b| {
        b.iter(|| {
            let mut total = 0u64;
            let mut v: SmallVec<[u8; 64]> = SmallVec::new();
            for &count in &refbit_counts {
                v.clear();
                for i in 0..count {
                    v.push((i & 0xFF) as u8);
                }
                total += v.iter().map(|&x| x as u64).sum::<u64>();
            }
            black_box(total)
        })
    });

    group.bench_function("arrayvec_inside_loop", |b| {
        b.iter(|| {
            let mut total = 0u64;
            for &count in &refbit_counts {
                let mut v: ArrayVec<[u8; 64]> = ArrayVec::new();
                for i in 0..count {
                    v.push((i & 0xFF) as u8);
                }
                total += v.iter().map(|&x| x as u64).sum::<u64>();
            }
            black_box(total)
        })
    });

    group.bench_function("arrayvec_outside_loop", |b| {
        b.iter(|| {
            let mut total = 0u64;
            let mut v: ArrayVec<[u8; 64]> = ArrayVec::new();
            for &count in &refbit_counts {
                v.clear();
                for i in 0..count {
                    v.push((i & 0xFF) as u8);
                }
                total += v.iter().map(|&x| x as u64).sum::<u64>();
            }
            black_box(total)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_vec_inside_loop);
criterion_main!(benches);
