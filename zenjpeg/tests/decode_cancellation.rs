//! #190 behavioural gate: every public decode entry point is a thin
//! `impl Stop` shim over a non-generic `&dyn Stop` body. The shims must
//! still deliver cancellation from the caller's concrete token through the
//! `dyn` boundary, and a token that never fires must be indistinguishable
//! from `Unstoppable`.

use core::sync::atomic::{AtomicUsize, Ordering};

use enough::{Stop, StopReason, Unstoppable};
use zenjpeg::decoder::{Decoder, ErrorKind, PixelFormat};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

/// Fires `Cancelled` on the `fire_at`-th `check()`; counts every check.
struct CountingStop {
    checks: AtomicUsize,
    fire_at: usize,
}

impl CountingStop {
    fn firing_at(fire_at: usize) -> Self {
        Self {
            checks: AtomicUsize::new(0),
            fire_at,
        }
    }
    fn never() -> Self {
        Self::firing_at(usize::MAX)
    }
    fn checks(&self) -> usize {
        self.checks.load(Ordering::Relaxed)
    }
}

impl Stop for CountingStop {
    fn check(&self) -> Result<(), StopReason> {
        let n = self.checks.fetch_add(1, Ordering::Relaxed) + 1;
        if n >= self.fire_at {
            Err(StopReason::Cancelled)
        } else {
            Ok(())
        }
    }
}

fn test_jpeg(progressive: bool) -> (Vec<u8>, u32, u32) {
    let (w, h) = (256u32, 192u32);
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    let mut seed = 0x1234_5678u32;
    for px in rgb.as_chunks_mut::<3>().0 {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        px[0] = (seed >> 24) as u8;
        px[1] = (seed >> 16) as u8;
        px[2] = (seed >> 8) as u8;
    }
    // Restart markers give the baseline streaming path its per-segment
    // cancellation checks.
    let cfg = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
        .progressive(progressive)
        .restart_mcu_rows(4);
    let mut enc = cfg
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encoder");
    enc.push_packed(&rgb, Unstoppable).expect("push");
    (enc.finish().expect("finish"), w, h)
}

fn is_cancelled<T>(r: &Result<T, zenjpeg::decoder::Error>) -> bool {
    matches!(r, Err(e) if matches!(e.kind(), ErrorKind::Cancelled(_)))
}

#[test]
fn never_firing_token_matches_unstoppable_and_is_consulted() {
    for progressive in [false, true] {
        let (jpeg, _, _) = test_jpeg(progressive);
        let stop = CountingStop::never();
        let a = Decoder::new()
            .decode(&jpeg, &stop)
            .expect("decode with token");
        let b = Decoder::new()
            .decode(&jpeg, Unstoppable)
            .expect("decode unstoppable");
        assert_eq!(a.pixels_u8(), b.pixels_u8(), "progressive={progressive}");
        assert!(
            stop.checks() > 0,
            "the caller's token was never consulted through the dyn boundary (progressive={progressive})"
        );
    }
}

#[test]
fn cancellation_reaches_every_entry_point() {
    for progressive in [false, true] {
        let (jpeg, w, h) = test_jpeg(progressive);
        let fire = 1;

        let r = Decoder::new().decode(&jpeg, CountingStop::firing_at(fire));
        assert!(is_cancelled(&r), "decode progressive={progressive}: {r:?}");

        let mut dst = vec![0u8; (w * h * 3) as usize];
        let r = Decoder::new().decode_into(
            &jpeg,
            PixelFormat::Rgb,
            &mut dst,
            CountingStop::firing_at(fire),
        );
        assert!(
            is_cancelled(&r),
            "decode_into progressive={progressive}: {r:?}"
        );

        let r = Decoder::new().decode_rows(
            &jpeg,
            PixelFormat::Rgb,
            |_| Ok(()),
            CountingStop::firing_at(fire),
        );
        assert!(
            is_cancelled(&r),
            "decode_rows progressive={progressive}: {r:?}"
        );

        let r = Decoder::new().decode_rows_f32(
            &jpeg,
            PixelFormat::RgbaF32,
            |_| Ok(()),
            CountingStop::firing_at(fire),
        );
        assert!(
            is_cancelled(&r),
            "decode_rows_f32 progressive={progressive}: {r:?}"
        );

        let r = Decoder::new().decode_coefficients(&jpeg, CountingStop::firing_at(fire));
        assert!(
            is_cancelled(&r),
            "decode_coefficients progressive={progressive}: {r:?}"
        );

        let r = Decoder::new().decode_to_ycbcr_f32(&jpeg, CountingStop::firing_at(fire));
        assert!(
            is_cancelled(&r),
            "decode_to_ycbcr_f32 progressive={progressive}: {r:?}"
        );

        // Auto-orient path (decode_upright_then_orient) is a separate route.
        let r = Decoder::new()
            .auto_orient(true)
            .decode(&jpeg, CountingStop::firing_at(fire));
        assert!(
            is_cancelled(&r),
            "decode(auto_orient) progressive={progressive}: {r:?}"
        );
    }
}
