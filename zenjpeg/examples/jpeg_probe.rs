//! Resource probe for JPEG encode/decode calibration
//! (`scripts/jpeg_resource_calibrate.py`).
//!
//! Measures the marginal working set (`VmHWM` delta), wall time, and
//! user/sys CPU of a single encode OR decode call, isolated to the codec
//! call. Encode and decode are separate invocations (clean per-op `VmHWM`
//! peak). Single-thread (no `parallel` feature) so wall ≈ user.
//!
//! Input is raw packed RGB8 bytes (the harness writes `PIL.tobytes()`), so
//! the probe needs no PNG/image dependency.
//!
//! For the vCPU sweep, set `RAYON_NUM_THREADS=N` in the environment and build
//! with `--features parallel,boundary-rd`: zenjpeg's strip DCT parallelises
//! over the GLOBAL rayon pool (the encoder has no per-call thread knob), so
//! the env var is the thread control. The probe reads it only to LABEL the row
//! (`threads=`); the est_* columns are `heuristics::estimate_encode` (thread-
//! independent) for prediction-vs-measurement in one record.
//!
//! Usage:
//!   jpeg_probe <raw_rgb> <w> <h> encode <quality> <trellis 0|1> <brd 0|1> <out.jpg>
//!   jpeg_probe <raw_rgb> <w> <h> decode <quality> <trellis 0|1> <brd 0|1> <in.jpg>
//! Prints (encode): `delta_kb=<n> peak_kb=<n> wall_ms=<f> user_ms=<f> sys_ms=<f> \
//!   bytes=<n> threads=<n> est_min_kb=<n> est_typ_kb=<n> est_max_kb=<n> est_time_ms=<f>`

use std::fs;
use std::time::Instant;

use enough::Unstoppable;
use zenjpeg::decoder::Decoder;
use zenjpeg::encode::trellis::TrellisConfig;
use zenjpeg::encoder::{
    BoundaryRd, BoundaryRdConfig, ChromaSubsampling, EncoderConfig, PixelLayout,
};

fn vmhwm_kb() -> u64 {
    let s = fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            return rest
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .unwrap_or(0);
        }
    }
    0
}

fn cpu_ticks() -> (u64, u64) {
    let s = fs::read_to_string("/proc/self/stat").unwrap_or_default();
    if let Some(p) = s.rfind(')') {
        let f: Vec<&str> = s[p + 1..].split_whitespace().collect();
        if f.len() > 12 {
            return (f[11].parse().unwrap_or(0), f[12].parse().unwrap_or(0));
        }
    }
    (0, 0)
}
const TICK_MS: f64 = 10.0;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 9 {
        eprintln!(
            "usage: jpeg_probe <raw_rgb> <w> <h> <encode|decode> <quality> <trellis> <brd> <out.jpg>"
        );
        std::process::exit(2);
    }
    let (raw, w, h, mode, quality, trellis, brd, outp) = (
        &a[1],
        a[2].parse::<u32>().unwrap(),
        a[3].parse::<u32>().unwrap(),
        &a[4],
        a[5].parse::<u8>().unwrap(),
        a[6].parse::<u8>().unwrap(),
        a[7].parse::<u8>().unwrap(),
        &a[8],
    );

    if mode == "encode" {
        let rgb = fs::read(raw).expect("read raw rgb");
        let mut cfg = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter);
        // Set trellis explicitly both ways so the contrast is real
        // (the ycbcr default does not enable it).
        cfg = if trellis == 1 {
            cfg.trellis(TrellisConfig::default())
        } else {
            cfg.trellis(TrellisConfig::default().ac_trellis(false).dc_trellis(false))
        };
        cfg = if brd == 1 {
            cfg.boundary_rd(BoundaryRd::On(BoundaryRdConfig::new()))
        } else {
            cfg.boundary_rd(BoundaryRd::Off)
        };

        // Thread count is the global rayon pool size (RAYON_NUM_THREADS); the
        // encoder has no per-call knob, so we read it only to label the row.
        let threads: usize = std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let est = zenjpeg::heuristics::estimate_encode(w, h, &cfg);
        let (est_min, est_typ, est_max, est_t) = (
            est.peak_memory_bytes_min / 1024,
            est.peak_memory_bytes / 1024,
            est.peak_memory_bytes_max / 1024,
            est.time_ms,
        );

        let (b0, t0) = (vmhwm_kb(), Instant::now());
        let (cu0, cs0) = cpu_ticks();
        let jpeg = cfg
            .encode_bytes(&rgb, w, h, PixelLayout::Rgb8Srgb)
            .expect("encode failed");
        let wall = t0.elapsed();
        let (cu1, cs1) = cpu_ticks();
        let peak = vmhwm_kb();
        fs::write(outp, &jpeg).expect("write jpg");
        println!(
            "delta_kb={} peak_kb={} wall_ms={:.1} user_ms={:.1} sys_ms={:.1} bytes={} \
             threads={} est_min_kb={} est_typ_kb={} est_max_kb={} est_time_ms={:.1}",
            peak.saturating_sub(b0),
            peak,
            wall.as_secs_f64() * 1000.0,
            (cu1 - cu0) as f64 * TICK_MS,
            (cs1 - cs0) as f64 * TICK_MS,
            jpeg.len(),
            threads,
            est_min,
            est_typ,
            est_max,
            est_t,
        );
    } else {
        let data = fs::read(outp).expect("read jpg");
        let (b0, t0) = (vmhwm_kb(), Instant::now());
        let (cu0, cs0) = cpu_ticks();
        let res = Decoder::new()
            .decode(&data, Unstoppable)
            .expect("decode failed");
        let wall = t0.elapsed();
        let (cu1, cs1) = cpu_ticks();
        let peak = vmhwm_kb();
        let px = (res.width() as u64) * (res.height() as u64);
        println!(
            "delta_kb={} peak_kb={} wall_ms={:.1} user_ms={:.1} sys_ms={:.1} bytes={}",
            peak.saturating_sub(b0),
            peak,
            wall.as_secs_f64() * 1000.0,
            (cu1 - cu0) as f64 * TICK_MS,
            (cs1 - cs0) as f64 * TICK_MS,
            px
        );
    }
}
