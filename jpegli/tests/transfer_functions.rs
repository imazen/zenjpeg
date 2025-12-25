//! Transfer function tests matching C++ jpegli thresholds.
//!
//! These tests validate that the PQ and HLG implementations match the C++
//! jpegli reference implementation within the exact error thresholds specified
//! in lib/cms/transfer_functions_test.cc.

use jpegli::transfer_functions::{hlg_display_from_encoded, hlg_encoded_from_display, PQ};

// ============================================================================
// C++ Thresholds (from lib/cms/transfer_functions_test.cc)
// ============================================================================

/// TestPqEncodedFromDisplay absolute error threshold
const PQ_ENCODE_ERROR: f64 = 6e-7;
/// TestHlgEncodedFromDisplay absolute error threshold
const HLG_ENCODE_ERROR: f64 = 4e-7;
/// TestPqDisplayFromEncoded absolute error threshold
const PQ_DECODE_ERROR: f64 = 3e-6;
/// TestHlgDisplayFromEncoded absolute error threshold
const HLG_DECODE_ERROR: f64 = 6e-7;

/// Simple LCG PRNG matching the C++ Rng class for test reproducibility.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn uniform_u32(&mut self) -> u32 {
        // LCG constants from C++ Rng
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 32) as u32
    }

    fn uniform_f(&mut self, min: f32, max: f32) -> f32 {
        let t = (self.uniform_u32() as f64) / (u32::MAX as f64);
        min + (max - min) * t as f32
    }
}

/// Test PQ EncodedFromDisplay matches C++ within threshold.
///
/// From transfer_functions_test.cc:
/// - Error threshold: 6e-7
/// - Intensity range: 11000 +/- 150 nits
/// - Input range: [0, 1]
#[test]
fn test_pq_encoded_from_display() {
    const NUM_TRIALS: usize = 1 << 20; // 1M trials (reduced from C++ 8M for speed)
    let mut rng = Rng::new(1);
    let mut max_abs_err: f64 = 0.0;

    for _ in 0..NUM_TRIALS {
        let intensity = 11000.0 + rng.uniform_f(-150.0, 150.0);
        let pq = PQ::new(intensity);

        let f = rng.uniform_f(0.0, 1.0) as f64;
        let actual = pq.encoded_from_display(f);

        // Reference implementation (same as our implementation, but serves as baseline)
        let expected = pq_reference_encoded_from_display(intensity as f64, f);

        let abs_err = (expected - actual).abs();
        max_abs_err = max_abs_err.max(abs_err);

        // C++ threshold is 6e-7
        assert!(
            abs_err < PQ_ENCODE_ERROR,
            "PQ encode error {} exceeds threshold {} at f={}, intensity={}",
            abs_err,
            PQ_ENCODE_ERROR,
            f,
            intensity
        );
    }

    println!("PQ encode max abs err: {:.2e}", max_abs_err);
}

/// Test PQ DisplayFromEncoded matches C++ within threshold.
///
/// From transfer_functions_test.cc:
/// - Error threshold: 3e-6
/// - Intensity range: 11000 +/- 150 nits
/// - Input range: [0, 1]
#[test]
fn test_pq_display_from_encoded() {
    const NUM_TRIALS: usize = 1 << 20;
    let mut rng = Rng::new(1);
    let mut max_abs_err: f64 = 0.0;

    for _ in 0..NUM_TRIALS {
        let intensity = 11000.0 + rng.uniform_f(-150.0, 150.0);
        let pq = PQ::new(intensity);

        let f = rng.uniform_f(0.0, 1.0) as f64;
        let actual = pq.display_from_encoded(f);

        // Reference
        let expected = pq_reference_display_from_encoded(intensity as f64, f);

        let abs_err = (expected - actual).abs();
        max_abs_err = max_abs_err.max(abs_err);

        // C++ threshold is 3e-6
        assert!(
            abs_err < PQ_DECODE_ERROR,
            "PQ decode error {} exceeds threshold {} at f={}, intensity={}",
            abs_err,
            PQ_DECODE_ERROR,
            f,
            intensity
        );
    }

    println!("PQ decode max abs err: {:.2e}", max_abs_err);
}

/// Test HLG EncodedFromDisplay matches C++ within threshold.
///
/// From transfer_functions_test.cc:
/// - Error threshold: 4e-7
/// - Input range: [0, 1]
#[test]
fn test_hlg_encoded_from_display() {
    const NUM_TRIALS: usize = 1 << 20;
    let mut rng = Rng::new(1);
    let mut max_abs_err: f64 = 0.0;

    for _ in 0..NUM_TRIALS {
        let f = rng.uniform_f(0.0, 1.0) as f64;
        let actual = hlg_encoded_from_display(f);
        let expected = hlg_reference_encoded_from_display(f);

        let abs_err = (expected - actual).abs();
        max_abs_err = max_abs_err.max(abs_err);

        // C++ threshold is 4e-7
        assert!(
            abs_err < HLG_ENCODE_ERROR,
            "HLG encode error {} exceeds threshold {} at f={}",
            abs_err,
            HLG_ENCODE_ERROR,
            f
        );
    }

    println!("HLG encode max abs err: {:.2e}", max_abs_err);
}

/// Test HLG DisplayFromEncoded matches C++ within threshold.
///
/// From transfer_functions_test.cc:
/// - Error threshold: 6e-7
/// - Input range: [0, 1]
#[test]
fn test_hlg_display_from_encoded() {
    const NUM_TRIALS: usize = 1 << 20;
    let mut rng = Rng::new(1);
    let mut max_abs_err: f64 = 0.0;

    for _ in 0..NUM_TRIALS {
        let f = rng.uniform_f(0.0, 1.0) as f64;
        let actual = hlg_display_from_encoded(f);
        let expected = hlg_reference_display_from_encoded(f);

        let abs_err = (expected - actual).abs();
        max_abs_err = max_abs_err.max(abs_err);

        // C++ threshold is 6e-7
        assert!(
            abs_err < HLG_DECODE_ERROR,
            "HLG decode error {} exceeds threshold {} at f={}",
            abs_err,
            HLG_DECODE_ERROR,
            f
        );
    }

    println!("HLG decode max abs err: {:.2e}", max_abs_err);
}

/// Test roundtrip accuracy for PQ.
#[test]
fn test_pq_roundtrip_accuracy() {
    const NUM_TRIALS: usize = 10000;
    let mut rng = Rng::new(42);
    let mut max_err: f64 = 0.0;

    for _ in 0..NUM_TRIALS {
        let intensity = 11000.0 + rng.uniform_f(-150.0, 150.0);
        let pq = PQ::new(intensity);

        let d = rng.uniform_f(0.0, 1.0) as f64;
        let encoded = pq.encoded_from_display(d);
        let decoded = pq.display_from_encoded(encoded);

        let err = (d - decoded).abs();
        max_err = max_err.max(err);

        // Roundtrip should be very accurate
        assert!(
            err < 1e-10,
            "PQ roundtrip error {} at d={}, intensity={}",
            err,
            d,
            intensity
        );
    }

    println!("PQ roundtrip max error: {:.2e}", max_err);
}

/// Test roundtrip accuracy for HLG.
#[test]
fn test_hlg_roundtrip_accuracy() {
    const NUM_TRIALS: usize = 10000;
    let mut rng = Rng::new(42);
    let mut max_err: f64 = 0.0;

    for _ in 0..NUM_TRIALS {
        let d = rng.uniform_f(0.0, 1.0) as f64;
        let encoded = hlg_encoded_from_display(d);
        let decoded = hlg_display_from_encoded(encoded);

        let err = (d - decoded).abs();
        max_err = max_err.max(err);

        assert!(err < 1e-10, "HLG roundtrip error {} at d={}", err, d);
    }

    println!("HLG roundtrip max error: {:.2e}", max_err);
}

/// Test negative input handling (unbounded transfer functions).
#[test]
fn test_unbounded_inputs() {
    // PQ negative
    let pq = PQ::new(11000.0);
    let d_neg = -0.5;
    let e_neg = pq.encoded_from_display(d_neg);
    assert!(e_neg < 0.0, "PQ should preserve negative sign");
    let d_back = pq.display_from_encoded(e_neg);
    assert!((d_neg - d_back).abs() < 1e-10, "PQ negative roundtrip failed");

    // HLG negative
    let e_hlg_neg = hlg_encoded_from_display(-0.5);
    assert!(e_hlg_neg < 0.0, "HLG should preserve negative sign");
    let d_hlg_back = hlg_display_from_encoded(e_hlg_neg);
    assert!(
        (-0.5 - d_hlg_back).abs() < 1e-10,
        "HLG negative roundtrip failed"
    );

    // Above 1.0
    let d_high = 1.5;
    let e_high = pq.encoded_from_display(d_high);
    let d_high_back = pq.display_from_encoded(e_high);
    assert!(
        (d_high - d_high_back).abs() < 1e-10,
        "PQ above-1 roundtrip failed"
    );
}

/// Test specific known values.
#[test]
fn test_known_values() {
    let pq = PQ::new(10000.0); // Standard SDR intensity

    // Zero should map to zero
    assert_eq!(pq.encoded_from_display(0.0), 0.0);
    assert_eq!(pq.display_from_encoded(0.0), 0.0);
    assert_eq!(hlg_encoded_from_display(0.0), 0.0);
    assert_eq!(hlg_display_from_encoded(0.0), 0.0);

    // Middle values should be in (0, 1)
    let pq_mid = pq.encoded_from_display(0.5);
    assert!(pq_mid > 0.0 && pq_mid < 1.0, "PQ mid value out of range");

    let hlg_mid = hlg_encoded_from_display(0.5);
    assert!(hlg_mid > 0.0 && hlg_mid < 1.0, "HLG mid value out of range");
}

// ============================================================================
// Reference Implementations (matching C++ exactly)
// ============================================================================

// PQ constants from C++
const PQ_M1: f64 = 2610.0 / 16384.0;
const PQ_M2: f64 = (2523.0 / 4096.0) * 128.0;
const PQ_C1: f64 = 3424.0 / 4096.0;
const PQ_C2: f64 = (2413.0 / 4096.0) * 32.0;
const PQ_C3: f64 = (2392.0 / 4096.0) * 32.0;

fn pq_reference_encoded_from_display(intensity_target: f64, d: f64) -> f64 {
    if d == 0.0 {
        return 0.0;
    }
    let original_sign = d.signum();
    let d_abs = d.abs();

    let xp = (d_abs * (intensity_target / 10000.0)).powf(PQ_M1);
    let num = PQ_C1 + xp * PQ_C2;
    let den = 1.0 + xp * PQ_C3;
    let e = (num / den).powf(PQ_M2);

    original_sign * e
}

fn pq_reference_display_from_encoded(intensity_target: f64, e: f64) -> f64 {
    if e == 0.0 {
        return 0.0;
    }
    let original_sign = e.signum();
    let e_abs = e.abs();

    let xp = e_abs.powf(1.0 / PQ_M2);
    let num = (xp - PQ_C1).max(0.0);
    let den = PQ_C2 - PQ_C3 * xp;
    let d = if den != 0.0 {
        (num / den).powf(1.0 / PQ_M1)
    } else {
        0.0
    };

    original_sign * d * (10000.0 / intensity_target)
}

// HLG constants from C++
const HLG_A: f64 = 0.17883277;
const HLG_RA: f64 = 1.0 / HLG_A;
const HLG_B: f64 = 1.0 - 4.0 * HLG_A;
const HLG_C: f64 = 0.5599107295;
const HLG_INV12: f64 = 1.0 / 12.0;

fn hlg_reference_oetf(s: f64) -> f64 {
    if s == 0.0 {
        return 0.0;
    }
    let original_sign = s.signum();
    let s_abs = s.abs();

    let e = if s_abs <= HLG_INV12 {
        (3.0 * s_abs).sqrt()
    } else {
        HLG_A * (12.0 * s_abs - HLG_B).ln() + HLG_C
    };

    original_sign * e
}

fn hlg_reference_inv_oetf(e: f64) -> f64 {
    if e == 0.0 {
        return 0.0;
    }
    let original_sign = e.signum();
    let e_abs = e.abs();

    let s = if e_abs <= 0.5 {
        e_abs * e_abs / 3.0
    } else {
        (((e_abs - HLG_C) * HLG_RA).exp() + HLG_B) * HLG_INV12
    };

    original_sign * s
}

fn hlg_reference_encoded_from_display(d: f64) -> f64 {
    hlg_reference_oetf(d) // OOTF is identity at gamma=1
}

fn hlg_reference_display_from_encoded(e: f64) -> f64 {
    hlg_reference_inv_oetf(e) // OOTF is identity at gamma=1
}
