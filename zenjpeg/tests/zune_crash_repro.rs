//! Regression tests for zune-jpeg crash bugs.
//!
//! These tests cover known panic/crash bugs from the zune-jpeg project
//! (etemesi254/zune-image) that may affect zenjpeg since it shares ancestry.
//!
//! Each test loads a malformed JPEG that triggered a panic in zune-jpeg and
//! verifies that zenjpeg handles it gracefully (returns an error, not a panic).
//!
//! Bug sources:
//! - image-rs/image-tiff issues #314, #315, #316
//! - etemesi254/zune-image issues #218, #219, #236, #257, #262, #297,
//!   #300, #301, #302, #309, #324, #331

use enough::Unstoppable;
use std::panic::{self, AssertUnwindSafe};
use zenjpeg::decode::ChromaUpsampling;
use zenjpeg::decode::Decoder;

/// Helper: decode must not panic. Returns Ok if decode succeeded or returned
/// an error; returns Err with the panic message if it panicked.
fn must_not_panic(data: &[u8]) -> Result<(), String> {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        Decoder::new().decode(data, Unstoppable)
    }));
    match result {
        Ok(Ok(_)) | Ok(Err(_)) => Ok(()),
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "unknown panic".to_string());
            Err(msg)
        }
    }
}

/// Helper: also test with different pixel formats and options to catch
/// panics that only manifest in specific decode paths.
fn must_not_panic_any_config(data: &[u8]) -> Result<(), String> {
    // Default config
    must_not_panic(data)?;

    // With fancy upsampling disabled
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        Decoder::new()
            .chroma_upsampling(ChromaUpsampling::NearestNeighbor)
            .decode(data, Unstoppable)
    }));
    if let Err(e) = result {
        let msg = e
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "unknown panic".to_string());
        return Err(format!("panic with fancy_upsampling(false): {}", msg));
    }

    // With fancy upsampling enabled
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        Decoder::new().decode(data, Unstoppable)
    }));
    if let Err(e) = result {
        let msg = e
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "unknown panic".to_string());
        return Err(format!("panic with fancy_upsampling(true): {}", msg));
    }

    Ok(())
}

// ============================================================================
// image-rs/image-tiff issue #314: mcu.rs index out of bounds
// JPEG extracted from TIFF wrapper. Range end index 5248 out of range for
// slice of length 5120 in mcu.rs:596.
// ============================================================================

#[test]
fn crash_314_mcu_oob() {
    let data = include_bytes!("crash_repro/crash_314_mcu_oob.jpg");
    let result = must_not_panic_any_config(data);
    assert!(
        result.is_ok(),
        "Panic on issue #314 data: {:?}",
        result.err()
    );
}

// ============================================================================
// image-rs/image-tiff issue #315: upsampler assertion failure
// assertion `left == right` failed in scalar upsampler, left: 768, right: 384
// ============================================================================

#[test]
fn crash_315_upsampler_assert() {
    let data = include_bytes!("crash_repro/crash_315_upsampler_assert.jpg");
    let result = must_not_panic_any_config(data);
    assert!(
        result.is_ok(),
        "Panic on issue #315 data: {:?}",
        result.err()
    );
}

// ============================================================================
// image-rs/image-tiff issue #316: bitstream assertion failure
// assertion failed: self.bits_left >= n in bitstream.rs:403
// ============================================================================

#[test]
fn crash_316_bitstream_assert() {
    let data = include_bytes!("crash_repro/crash_316_bitstream_assert.jpg");
    let result = must_not_panic_any_config(data);
    assert!(
        result.is_ok(),
        "Panic on issue #316 data: {:?}",
        result.err()
    );
}

// ============================================================================
// zune-image issue #218: mcu_prog.rs index out of bounds
// index out of bounds in mcu_prog.rs:391
// ============================================================================

#[test]
fn crash_218_mcu_prog_oob() {
    let data = include_bytes!("crash_repro/crash_218_mcu_prog_oob.jpg");
    let result = must_not_panic_any_config(data);
    assert!(
        result.is_ok(),
        "Panic on issue #218 data: {:?}",
        result.err()
    );
}

// ============================================================================
// zune-image issue #236: upsampler/scalar.rs assertion failure
// assertion `left == right` failed at scalar.rs:59
// Open since September 2024.
// ============================================================================

#[test]
fn crash_236_upsampler_assert() {
    let data = include_bytes!("crash_repro/crash_236_upsampler_assert.jpg");
    let result = must_not_panic_any_config(data);
    assert!(
        result.is_ok(),
        "Panic on issue #236 data: {:?}",
        result.err()
    );
}

// ============================================================================
// zune-image issue #257: parse_entropy_coded_data mcu_prog.rs OOB
// ============================================================================

#[test]
fn crash_257_mcu_prog_oob() {
    let data = include_bytes!("crash_repro/crash_257_mcu_prog_oob.jpg");
    let result = must_not_panic_any_config(data);
    assert!(
        result.is_ok(),
        "Panic on issue #257 data: {:?}",
        result.err()
    );
}

// ============================================================================
// zune-image issue #262: three distinct panics from fuzzing
// ============================================================================

#[test]
fn crash_262_panic1_mcu_assert() {
    let data = include_bytes!("crash_repro/crash_262_panic1_mcu_assert.jpg");
    let result = must_not_panic_any_config(data);
    assert!(
        result.is_ok(),
        "Panic on issue #262 panic1: {:?}",
        result.err()
    );
}

#[test]
fn crash_262_panic2_mcu_prog_unwrap() {
    let data = include_bytes!("crash_repro/crash_262_panic2_mcu_prog_unwrap.jpg");
    let result = must_not_panic_any_config(data);
    assert!(
        result.is_ok(),
        "Panic on issue #262 panic2: {:?}",
        result.err()
    );
}

#[test]
fn crash_262_panic3_mcu_prog_range() {
    let data = include_bytes!("crash_repro/crash_262_panic3_mcu_prog_range.jpg");
    let result = must_not_panic_any_config(data);
    assert!(
        result.is_ok(),
        "Panic on issue #262 panic3: {:?}",
        result.err()
    );
}

// ============================================================================
// zune-image issue #297: divide by zero in decoder.rs
// ============================================================================

#[test]
fn crash_297_div_by_zero() {
    let data = include_bytes!("crash_repro/crash_297_div_by_zero.jpg");
    let result = must_not_panic_any_config(data);
    assert!(
        result.is_ok(),
        "Panic on issue #297 data: {:?}",
        result.err()
    );
}

// ============================================================================
// zune-image issue #300: bitstream.rs assertion (bits_left >= n)
// ============================================================================

#[test]
fn crash_300_bitstream_assert() {
    let data = include_bytes!("crash_repro/crash_300_bitstream_assert.jpg");
    let result = must_not_panic_any_config(data);
    assert!(
        result.is_ok(),
        "Panic on issue #300 data: {:?}",
        result.err()
    );
}

// ============================================================================
// zune-image issue #301: mcu.rs range end index 3200 out of range
// ============================================================================

#[test]
fn crash_301_mcu_range_3200() {
    let data = include_bytes!("crash_repro/crash_301_mcu_range_3200.jpg");
    let result = must_not_panic_any_config(data);
    assert!(
        result.is_ok(),
        "Panic on issue #301 data: {:?}",
        result.err()
    );
}

// ============================================================================
// zune-image issue #302: bitstream.rs assertion (bits_left >= n)
// ============================================================================

#[test]
fn crash_302_bitstream_assert() {
    let data = include_bytes!("crash_repro/crash_302_bitstream_assert.jpg");
    let result = must_not_panic_any_config(data);
    assert!(
        result.is_ok(),
        "Panic on issue #302 data: {:?}",
        result.err()
    );
}

// ============================================================================
// zune-image issue #309: mcu.rs range end index 2176 out of range
// ============================================================================

#[test]
fn crash_309_mcu_range_2176() {
    let data = include_bytes!("crash_repro/crash_309_mcu_range_2176.jpg");
    let result = must_not_panic_any_config(data);
    assert!(
        result.is_ok(),
        "Panic on issue #309 data: {:?}",
        result.err()
    );
}

// ============================================================================
// zune-image issue #324: idct/scalar.rs multiply overflow
// ============================================================================

#[test]
fn crash_324_idct_mul_overflow() {
    let data = include_bytes!("crash_repro/crash_324_idct_mul_overflow.jpg");
    let result = must_not_panic_any_config(data);
    assert!(
        result.is_ok(),
        "Panic on issue #324 data: {:?}",
        result.err()
    );
}

// ============================================================================
// zune-image issue #331: worker.rs color_convert_ycbcr panic
// ============================================================================

#[test]
fn crash_331_worker_color_convert() {
    let data = include_bytes!("crash_repro/crash_331_worker_color_convert.jpg");
    let result = must_not_panic_any_config(data);
    assert!(
        result.is_ok(),
        "Panic on issue #331 data: {:?}",
        result.err()
    );
}

// ============================================================================
// zune-image issue #219: batch of 20 crash files covering ~7 distinct panic
// locations. These were found by extensive fuzzing.
// ============================================================================

macro_rules! crash_219_test {
    ($name:ident, $file:literal) => {
        #[test]
        fn $name() {
            let data = include_bytes!(concat!("crash_repro/", $file));
            let result = must_not_panic_any_config(data);
            assert!(result.is_ok(), "Panic on {}: {:?}", $file, result.err());
        }
    };
}

crash_219_test!(
    crash_219_15daf076,
    "crash_219_15daf076cac75fc71d88b5b1475da54a56c336a9.jpg"
);
crash_219_test!(
    crash_219_2b148807,
    "crash_219_2b1488070639567997cb0e6953f000b3867e0e54.jpg"
);
crash_219_test!(
    crash_219_30e20103,
    "crash_219_30e20103ed9b2acbad03aa54e91344df6a256739.jpg"
);
crash_219_test!(
    crash_219_40fd8bf0,
    "crash_219_40fd8bf0a55bd09915973099ae7df3785d590077.jpg"
);
crash_219_test!(
    crash_219_430fcba3,
    "crash_219_430fcba35c5e3db14ff0aafe87ddb81a6c5bdd8b.jpg"
);
crash_219_test!(
    crash_219_4383a0c6,
    "crash_219_4383a0c6805d99c4aa7bcc48c07dc719ff72cba4.jpg"
);
crash_219_test!(
    crash_219_48210322,
    "crash_219_482103221bc18230f3a41b364da02bb298770806.jpg"
);
crash_219_test!(
    crash_219_5316ef2d,
    "crash_219_5316ef2d8fa08ce11477f5008de84f488ec6740b.jpg"
);
crash_219_test!(
    crash_219_754f6933,
    "crash_219_754f6933e294e8016d0c8764783bbb90b2e23515.jpg"
);
crash_219_test!(
    crash_219_7e8ef95a,
    "crash_219_7e8ef95a03083f33c82be7c19fc9bbad3f1d9a4c.jpg"
);
crash_219_test!(
    crash_219_8bb50809,
    "crash_219_8bb50809d589e6ec4b555ce9cf69b27d8b36c528.jpg"
);
crash_219_test!(
    crash_219_9e9b55d9,
    "crash_219_9e9b55d900bf047d5cf3edecc0e62250828f7663.jpg"
);
crash_219_test!(
    crash_219_a9909f74,
    "crash_219_a9909f747e700fcd01431330769ae9faba7ac420.jpg"
);
crash_219_test!(
    crash_219_c8a925e3,
    "crash_219_c8a925e39b0ad2589e588a4dbd51f234b3299e65.jpg"
);
crash_219_test!(
    crash_219_ca049ac4,
    "crash_219_ca049ac4657a1ff2cb8a5f5ccdc774391d76679f.jpg"
);
crash_219_test!(
    crash_219_de2b0aac,
    "crash_219_de2b0aacb3431b6425eb292bae7a1991bc99370e.jpg"
);
crash_219_test!(
    crash_219_f0d5fdfa,
    "crash_219_f0d5fdfaa0f43174a7e6ce64761606538c2f7e65.jpg"
);
crash_219_test!(
    crash_219_f2a37464,
    "crash_219_f2a374644f9e64c0eb3cc81cdda99c7fdd1f5797.jpg"
);
crash_219_test!(
    crash_219_f69e5129,
    "crash_219_f69e5129fcba4f79dc03570f98ab0fbebae1e1d2.jpg"
);
crash_219_test!(
    crash_219_fb5c7664,
    "crash_219_fb5c7664dbc9117c998c2f6e76c392e6cc481048.jpg"
);
