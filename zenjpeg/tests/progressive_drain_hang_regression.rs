//! Regression for a fuzz-farm DoS (zenpipe#47): a small progressive JPEG with a
//! restart interval but a missing/marker-less RST hung the decoder forever in
//! the AC-scan restart-drain loop. Past EOF, `BitReader::refill()` keeps claiming
//! synthetic zero bits (overread), so `bits_available() >= 32` stayed true while
//! `marker_found()` never fired — `while marker_found().is_none()` spun. Both the
//! first-scan and refine-scan drains now also break on `is_exhausted()`.
//!
//! The bug was an infinite loop, so the test simply requires `decode` to RETURN
//! (Ok or Err both fine). If the fix regresses, CI's per-test timeout catches the
//! hang.

/// The exact fuzz repro: a 292×259 progressive JPEG (3 scans, malformed DRIs).
const REPRO: &[u8] = include_bytes!("../fuzz/regression/timeout-progressive-restart-drain-hang");

#[test]
fn progressive_restart_drain_does_not_hang_47() {
    // Must return promptly instead of spinning forever.
    let _ = zenjpeg::decode::DecodeConfig::new().decode(REPRO, &enough::Unstoppable);
}
