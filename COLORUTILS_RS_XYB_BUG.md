# colorutils-rs XYB Implementation Bug

**Library**: [colorutils-rs](https://github.com/awxkee/colorutils-rs) by awxkee
**Version**: 0.7.5
**Severity**: Critical - XYB is completely broken
**Status**: Unfixed as of 2025-12-27

## Bug Summary

The XYB color space implementation in colorutils-rs v0.7.5 produces completely incorrect results. The `Xyb::from_rgb()` and `Xyb::to_rgb()` functions are broken.

## Symptoms

1. **Colors with r=0 all produce identical XYB values**
   - Black (0,0,0), Green (0,255,0), Blue (0,0,255) all return the same XYB
   - The XYB values returned match what Red (255,0,0) should produce

2. **Round-trip is completely broken**
   - Most colors return (255, 0, 0) after XYB→RGB conversion
   - Only White (255,255,255) round-trips correctly

3. **Even colors with r>0 produce wrong values**
   - Red's XYB values don't match the JPEG XL specification

## Test Results

```
=== colorutils-rs XYB Round-Trip ===

White: Input RGB: r=255, g=255, b=255
       XYB: x=+0.000000, y=0.845309, b=+0.000000
       Out RGB: r=255, g=255, b=255
       Error: 0  ✓ (only color that works)

Black: Input RGB: r=0, g=0, b=0
       XYB: x=+0.028370, y=0.485109, b=+0.011694  ← WRONG (should be 0,0,0)
       Out RGB: r=255, g=0, b=0  ← WRONG
       Error: 255

Gray: Input RGB: r=128, g=128, b=128
       XYB: x=+0.016576, y=0.586872, b=-0.047011
       Out RGB: r=255, g=0, b=0  ← WRONG
       Error: 128

Red: Input RGB: r=255, g=0, b=0
       XYB: x=+0.000000, y=0.817339, b=-0.208012  ← WRONG
       Out RGB: r=255, g=223, b=0  ← WRONG
       Error: 223

Green: Input RGB: r=0, g=255, b=0
       XYB: x=+0.028370, y=0.485109, b=+0.011694  ← Same as Black!
       Out RGB: r=255, g=0, b=0
       Error: 255

Blue: Input RGB: r=0, g=0, b=255
       XYB: x=+0.028370, y=0.485109, b=+0.011694  ← Same as Black!
       Out RGB: r=255, g=0, b=0
       Error: 255
```

## Expected Values (Correct Implementation)

From our verified XYB implementation with perfect round-trip:

```
Red   [255,  0,  0] -> XYB(+0.0281, 0.4882, +0.0116) -> [255,  0,  0] ✓
Green [  0,255,  0] -> XYB(-0.0154, 0.7148, -0.2931) -> [  0,255,  0] ✓
Blue  [  0,  0,255] -> XYB(+0.0000, 0.2781, +0.3880) -> [  0,  0,255] ✓
White [255,255,255] -> XYB(+0.0000, 0.8453, +0.0000) -> [255,255,255] ✓
Black [  0,  0,  0] -> XYB(+0.0000, 0.0000, +0.0000) -> [  0,  0,  0] ✓
Gray  [128,128,128] -> XYB(+0.0000, 0.4474, +0.0000) -> [128,128,128] ✓
```

## Root Cause Analysis

The bug pattern suggests **channel confusion** in the implementation:

1. colorutils-rs Black produces XYB values that match our correct Red values
2. All colors with r=0 produce identical output
3. This suggests the `from_rgb` function may be:
   - Only reading the R channel
   - Having incorrect struct field ordering
   - Using uninitialized memory for G and B

## Reproduction Code

```rust
use colorutils_rs::{Xyb, Rgb, TransferFunction};

fn main() {
    // These all produce the SAME XYB values (bug)
    let black = Xyb::from_rgb(Rgb::<u8>::new(0, 0, 0), TransferFunction::Srgb);
    let green = Xyb::from_rgb(Rgb::<u8>::new(0, 255, 0), TransferFunction::Srgb);
    let blue = Xyb::from_rgb(Rgb::<u8>::new(0, 0, 255), TransferFunction::Srgb);

    println!("Black: x={}, y={}, b={}", black.x, black.y, black.b);
    println!("Green: x={}, y={}, b={}", green.x, green.y, green.b);
    println!("Blue:  x={}, y={}, b={}", blue.x, blue.y, blue.b);

    // All three will print the same values!

    // Round-trip test
    let rgb_back = black.to_rgb(TransferFunction::Srgb);
    println!("Black round-trip: r={}, g={}, b={}", rgb_back.r, rgb_back.g, rgb_back.b);
    // Will print: r=255, g=0, b=0  (completely wrong)
}
```

## What Needs To Be Fixed

### 1. Investigate `from_rgb` / `from_linear_rgb`

Location: `src/xyb.rs` in colorutils-rs

Check for:
- Incorrect field access in `Rgb<T>` struct
- SIMD code that may have incorrect lane ordering
- Copy-paste errors in RGB→LMS matrix application

### 2. Verify RGB→LMS Matrix

The correct matrix (from JPEG XL spec):
```rust
let l = 0.3 * r + 0.622 * g + 0.078 * b;
let m = 0.23 * r + 0.692 * g + 0.078 * b;
let s = 0.24342268924547819 * r + 0.20476744424496821 * g + 0.55180986650955360 * b;
```

### 3. Verify Cube Root with Bias

```rust
const BIAS: f32 = 0.00379307325527544933;
const BIAS_CBRT: f32 = 0.155954200549248620;

let l_gamma = (l + BIAS).cbrt() - BIAS_CBRT;
let m_gamma = (m + BIAS).cbrt() - BIAS_CBRT;
let s_gamma = (s + BIAS).cbrt() - BIAS_CBRT;
```

### 4. Verify LMS→XYB Transform

```rust
let x = (l_gamma - m_gamma) * 0.5;
let y = (l_gamma + m_gamma) * 0.5;
let b = s_gamma - m_gamma;
```

### 5. Fix `to_rgb` / `to_linear_rgb`

The inverse transform also appears broken. Verify:
- XYB→LMS inverse
- Cube operation (inverse of cube root)
- LMS→RGB inverse matrix

## Test Suite for Verification

Add these tests to colorutils-rs:

```rust
#[test]
fn test_xyb_black_is_zero() {
    let black = Xyb::from_rgb(Rgb::<u8>::new(0, 0, 0), TransferFunction::Srgb);
    assert!((black.x).abs() < 0.001, "Black X should be ~0, got {}", black.x);
    assert!((black.y).abs() < 0.001, "Black Y should be ~0, got {}", black.y);
    assert!((black.b).abs() < 0.001, "Black B should be ~0, got {}", black.b);
}

#[test]
fn test_xyb_colors_are_distinct() {
    let black = Xyb::from_rgb(Rgb::<u8>::new(0, 0, 0), TransferFunction::Srgb);
    let green = Xyb::from_rgb(Rgb::<u8>::new(0, 255, 0), TransferFunction::Srgb);
    let blue = Xyb::from_rgb(Rgb::<u8>::new(0, 0, 255), TransferFunction::Srgb);

    // These should all be different!
    assert!((black.y - green.y).abs() > 0.1, "Black and Green should differ");
    assert!((black.y - blue.y).abs() > 0.1, "Black and Blue should differ");
    assert!((green.b - blue.b).abs() > 0.1, "Green and Blue should differ in B channel");
}

#[test]
fn test_xyb_round_trip_primaries() {
    let test_colors = [
        (255, 0, 0, "Red"),
        (0, 255, 0, "Green"),
        (0, 0, 255, "Blue"),
        (255, 255, 255, "White"),
        (0, 0, 0, "Black"),
        (128, 128, 128, "Gray"),
    ];

    for (r, g, b, name) in test_colors {
        let rgb = Rgb::<u8>::new(r, g, b);
        let xyb = Xyb::from_rgb(rgb, TransferFunction::Srgb);
        let rgb2 = xyb.to_rgb(TransferFunction::Srgb);

        let max_error = (r as i32 - rgb2.r as i32).abs()
            .max((g as i32 - rgb2.g as i32).abs())
            .max((b as i32 - rgb2.b as i32).abs());

        assert!(max_error <= 1, "{} round-trip failed: [{},{},{}] -> [{},{},{}], error={}",
            name, r, g, b, rgb2.r, rgb2.g, rgb2.b, max_error);
    }
}

#[test]
fn test_xyb_neutral_colors_have_zero_opponents() {
    // Neutral colors (grays) should have X ≈ 0 and B ≈ 0
    for v in [0, 64, 128, 192, 255] {
        let gray = Xyb::from_rgb(Rgb::<u8>::new(v, v, v), TransferFunction::Srgb);
        assert!(gray.x.abs() < 0.001, "Gray {} X should be ~0, got {}", v, gray.x);
        assert!(gray.b.abs() < 0.001, "Gray {} B should be ~0, got {}", v, gray.b);
    }
}

#[test]
fn test_xyb_red_green_opponent() {
    let red = Xyb::from_rgb(Rgb::<u8>::new(255, 0, 0), TransferFunction::Srgb);
    let green = Xyb::from_rgb(Rgb::<u8>::new(0, 255, 0), TransferFunction::Srgb);

    // Red should have positive X (L > M)
    assert!(red.x > 0.0, "Red X should be positive, got {}", red.x);
    // Green should have negative X (M > L)
    assert!(green.x < 0.0, "Green X should be negative, got {}", green.x);
}

#[test]
fn test_xyb_blue_channel() {
    let blue = Xyb::from_rgb(Rgb::<u8>::new(0, 0, 255), TransferFunction::Srgb);

    // Pure blue should have high positive B channel
    assert!(blue.b > 0.3, "Blue B should be > 0.3, got {}", blue.b);
}
```

## Correct Reference Implementation

See `XYB_ICC_PROFILE.md` in this directory for verified correct formulas, or use this Rust implementation:

```rust
pub mod xyb {
    pub const BIAS: f64 = 0.00379307325527544933;
    pub const BIAS_CBRT: f64 = 0.155954200549248620;

    fn srgb_to_linear(v: f64) -> f64 {
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    }

    fn linear_to_srgb(v: f64) -> f64 {
        if v <= 0.0031308 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
    }

    fn cbrt(x: f64) -> f64 {
        if x >= 0.0 { x.cbrt() } else { -(-x).cbrt() }
    }

    pub fn srgb_to_xyb(r: u8, g: u8, b: u8) -> (f64, f64, f64) {
        let r_lin = srgb_to_linear(r as f64 / 255.0);
        let g_lin = srgb_to_linear(g as f64 / 255.0);
        let b_lin = srgb_to_linear(b as f64 / 255.0);

        let l = 0.3 * r_lin + 0.622 * g_lin + 0.078 * b_lin;
        let m = 0.23 * r_lin + 0.692 * g_lin + 0.078 * b_lin;
        let s = 0.24342268924547819 * r_lin
              + 0.20476744424496821 * g_lin
              + 0.55180986650955360 * b_lin;

        let l_g = cbrt(l + BIAS) - BIAS_CBRT;
        let m_g = cbrt(m + BIAS) - BIAS_CBRT;
        let s_g = cbrt(s + BIAS) - BIAS_CBRT;

        let x = (l_g - m_g) * 0.5;
        let y = (l_g + m_g) * 0.5;
        let b = s_g - m_g;

        (x, y, b)
    }

    pub fn xyb_to_srgb(x: f64, y: f64, b: f64) -> (u8, u8, u8) {
        let l_g = x + y + BIAS_CBRT;
        let m_g = -x + y + BIAS_CBRT;
        let s_g = -x + y + b + BIAS_CBRT;

        let l = l_g.powi(3) - BIAS;
        let m = m_g.powi(3) - BIAS;
        let s = s_g.powi(3) - BIAS;

        let r_lin = 11.031566901960783 * l - 9.866943921568629 * m - 0.16462299647058826 * s;
        let g_lin = -3.254147380392157 * l + 4.418770392156863 * m - 0.16462299647058826 * s;
        let b_lin = -3.6588512862745097 * l + 2.7129230470588235 * m + 1.9459282392156863 * s;

        let r = (linear_to_srgb(r_lin.clamp(0.0, 1.0)) * 255.0).round() as u8;
        let g = (linear_to_srgb(g_lin.clamp(0.0, 1.0)) * 255.0).round() as u8;
        let b = (linear_to_srgb(b_lin.clamp(0.0, 1.0)) * 255.0).round() as u8;

        (r, g, b)
    }
}
```

## Filing an Issue

Consider filing an issue at https://github.com/awxkee/colorutils-rs/issues with:

1. This bug report
2. The reproduction code
3. Expected vs actual values
4. The failing test suite

Note: awxkee is also the author of moxcms, so they may be responsive to color science issues.
