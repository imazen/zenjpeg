//! `container::metadata` walk cost against the existing container scanners.
//!
//! Per representative file: the bare marker walk (`marker::iter` to EOI),
//! the single-pass `probe(Wants::ALL)`, `JpegMetadata::parse` (every item,
//! trailers included), and parse plus the three reassembly accessors.
//! `AllocProfiler` reports heap allocations per iteration alongside time.
//!
//! Then the whole codec-corpus JPEG set as one group (aggregate throughput),
//! and a per-file measurement outside zenbench: each corpus file parsed 15
//! times, its median kept, and the distribution of those medians printed.
//!
//! Run: `cargo bench -p zenjpeg --features extended-metadata --bench metadata_walk`

use std::path::PathBuf;
use std::time::Instant;

use enough::Unstoppable;
use zenbench::prelude::*;
use zenjpeg::container::metadata::JpegMetadata;
use zenjpeg::container::{Wants, probe};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

#[global_allocator]
static ALLOC: zenbench::AllocProfiler = zenbench::AllocProfiler::system();

fn corpus_dir(dataset: &str) -> PathBuf {
    codec_corpus::Corpus::new()
        .expect("codec-corpus init")
        .get(dataset)
        .unwrap_or_else(|e| panic!("codec-corpus {dataset}: {e}"))
}

fn leak(v: Vec<u8>) -> &'static [u8] {
    Box::leak(v.into_boxed_slice())
}

/// A 256x256 noise-patch JPEG with no metadata (fixed per-call cost).
fn small_synthetic() -> Vec<u8> {
    let (w, h) = (256usize, 256usize);
    let mut rgb = vec![0u8; w * h * 3];
    let mut s: u32 = 0x9e37_79b9;
    for (i, px) in rgb.as_chunks_mut::<3>().0.iter_mut().enumerate() {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let patch = (((i % w) / 32 + (i / w) / 32) & 3) as u8;
        *px = [
            ((s >> 24) as u8).wrapping_add(patch.wrapping_mul(40)),
            ((s >> 16) as u8).wrapping_add(patch.wrapping_mul(80)),
            ((s >> 8) as u8).wrapping_add(patch.wrapping_mul(120)),
        ];
    }
    let mut enc = EncoderConfig::ycbcr(80.0, ChromaSubsampling::Quarter)
        .encode_from_bytes(w as u32, h as u32, PixelLayout::Rgb8Srgb)
        .expect("encoder");
    enc.push_packed(&rgb, Unstoppable).expect("push");
    enc.finish().expect("finish")
}

fn inputs() -> Vec<(&'static str, &'static [u8])> {
    let read = |dataset: &str, file: &str| {
        let p = corpus_dir(dataset).join(file);
        leak(std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display())))
    };
    vec![
        ("synthetic_256", leak(small_synthetic())),
        (
            "canon_40d_exif",
            read("jpeg-conformance", "valid/Canon_40D.jpg"),
        ),
        (
            "ycck_icc18_psir",
            read("jpeg-conformance", "valid/ycck.jpg"),
        ),
        (
            "pixel_ultrahdr_mpf",
            read(
                "ultrahdr-conformance",
                "valid/jpeg/pixel-ultrahdr/Ultra_HDR_Samples_Originals_02.jpg",
            ),
        ),
        (
            "mountains_extxmp",
            read(
                "ultrahdr-conformance",
                "valid/jpeg/libultrahdr-benchmark/mountains.jpg",
            ),
        ),
    ]
}

fn bench_files(suite: &mut Suite) {
    for (label, data) in inputs() {
        suite.group(format!("metadata/{label}"), |g| {
            g.throughput(Throughput::Bytes(data.len() as u64));
            g.bench("marker_iter_to_eoi", move |b| {
                b.iter(|| zenjpeg::container::marker::iter(black_box(data)).count())
            });
            g.bench("probe_all", move |b| {
                b.iter(|| probe(black_box(data), Wants::ALL).scan_count())
            });
            g.bench("jpeg_metadata_parse", move |b| {
                b.iter(|| {
                    let m = JpegMetadata::parse(black_box(data)).unwrap();
                    m.items().len()
                })
            });
            g.bench("parse_and_reassemble", move |b| {
                b.iter(|| {
                    let m = JpegMetadata::parse(black_box(data)).unwrap();
                    let icc = m.icc_profile().ok().flatten().map_or(0, |p| p.len());
                    let xmp = m.extended_xmp().ok().flatten().map_or(0, |p| p.len());
                    let ps = m.photoshop().ok().flatten().map_or(0, |p| p.data().len());
                    icc + xmp + ps
                })
            });
        });
    }
}

fn corpus_files() -> Vec<Vec<u8>> {
    let mut paths = Vec::new();
    for dataset in [
        "jpeg-conformance",
        "ultrahdr-conformance",
        "imageflow/test_inputs",
        "mozjpeg",
        "zune/test-images/jpeg",
        "zune/fuzz-corpus/jpeg",
        "image-rs/tests/images/jpg",
    ] {
        collect(&corpus_dir(dataset), &mut paths);
    }
    paths.sort();
    paths.iter().map(|p| std::fs::read(p).unwrap()).collect()
}

fn collect(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let p = entry.unwrap().path();
        if p.is_dir() {
            collect(&p, out);
            continue;
        }
        let ext = p
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        let jpeg = match ext.as_deref() {
            Some("jpg" | "jpeg" | "jfif") => true,
            None => std::fs::read(&p).is_ok_and(|d| d.starts_with(&[0xFF, 0xD8])),
            _ => false,
        };
        if jpeg {
            out.push(p);
        }
    }
}

fn bench_corpus(suite: &mut Suite) {
    let files: &'static [Vec<u8>] = Box::leak(corpus_files().into_boxed_slice());
    let total: usize = files.iter().map(Vec::len).sum();
    per_file_medians(files);
    suite.group(format!("metadata/corpus_{}_files", files.len()), |g| {
        g.throughput(Throughput::Bytes(total as u64));
        g.bench("marker_iter_to_eoi", move |b| {
            b.iter(|| {
                files
                    .iter()
                    .map(|f| zenjpeg::container::marker::iter(black_box(f)).count())
                    .sum::<usize>()
            })
        });
        g.bench("jpeg_metadata_parse", move |b| {
            b.iter(|| {
                files
                    .iter()
                    .map(|f| JpegMetadata::parse(black_box(f)).unwrap().items().len())
                    .sum::<usize>()
            })
        });
    });
}

/// Median parse time of each corpus file (15 runs each), summarized.
fn per_file_medians(files: &[Vec<u8>]) {
    let mut medians: Vec<(f64, usize)> = files
        .iter()
        .map(|f| {
            let mut t: Vec<f64> = (0..15)
                .map(|_| {
                    let start = Instant::now();
                    let m = JpegMetadata::parse(black_box(f)).unwrap();
                    black_box(m.items().len());
                    start.elapsed().as_secs_f64() * 1e9
                })
                .collect();
            t.sort_by(f64::total_cmp);
            (t[7], f.len())
        })
        .collect();
    medians.sort_by(|a, b| a.0.total_cmp(&b.0));
    let pct = |p: f64| medians[((medians.len() - 1) as f64 * p) as usize].0;
    let mut per_kib: Vec<f64> = medians
        .iter()
        .map(|&(ns, len)| ns / (len.max(1) as f64 / 1024.0))
        .collect();
    per_kib.sort_by(f64::total_cmp);
    println!(
        "per-file JpegMetadata::parse medians over {} files: p50 {:.0} ns, p90 {:.0} ns, \
         p99 {:.0} ns, max {:.0} ns; median {:.1} ns/KiB",
        medians.len(),
        pct(0.5),
        pct(0.9),
        pct(0.99),
        medians.last().unwrap().0,
        per_kib[per_kib.len() / 2]
    );
}

zenbench::main!(bench_files, bench_corpus);
