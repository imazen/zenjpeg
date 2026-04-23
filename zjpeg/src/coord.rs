//! Coordinate value parsing for the `process` subcommand.
//!
//! Supports pixel values, percentages (`%` or `pct`), and calc expressions
//! (`50%+20`, `50pct-10`). Also provides CSS TRBL shorthand and dimension
//! parsers.

use anyhow::{Result, bail};

/// A coordinate value with optional percent and pixel components.
///
/// Resolved against a source dimension: `result = percent * dim + pixels`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub struct CoordValue {
    pub percent: f32,
    pub pixels: i32,
}

impl CoordValue {
    pub const fn px(pixels: i32) -> Self {
        Self {
            percent: 0.0,
            pixels,
        }
    }

    #[cfg(test)]
    pub const fn pct(percent: f32) -> Self {
        Self { percent, pixels: 0 }
    }

    #[cfg(test)]
    pub const fn pct_px(percent: f32, pixels: i32) -> Self {
        Self { percent, pixels }
    }

    /// Resolve against a source dimension.
    #[cfg(test)]
    pub fn resolve(self, dim: u32) -> i32 {
        (self.percent * dim as f32).round() as i32 + self.pixels
    }
}

/// Parse a coordinate value string.
///
/// Formats:
/// - `100` → 100 pixels
/// - `10%` or `10pct` → 10%
/// - `50%+20` or `50pct+20` → 50% + 20px
/// - `50%-10` or `50pct-10` → 50% - 10px
pub fn parse_coord(s: &str) -> Result<CoordValue> {
    let s = s.trim();
    if s.is_empty() {
        bail!("empty coordinate value");
    }

    // Check for percent indicator
    if let Some(pos) = s.find('%') {
        let pct_str = &s[..pos];
        let pct: f32 = pct_str
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid percent value: '{pct_str}'"))?;
        let remainder = &s[pos + 1..];
        let pixels = parse_pixel_offset(remainder)?;
        return Ok(CoordValue {
            percent: pct / 100.0,
            pixels,
        });
    }

    if let Some(pos) = s.find("pct") {
        let pct_str = &s[..pos];
        let pct: f32 = pct_str
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid percent value: '{pct_str}'"))?;
        let remainder = &s[pos + 3..];
        let pixels = parse_pixel_offset(remainder)?;
        return Ok(CoordValue {
            percent: pct / 100.0,
            pixels,
        });
    }

    // Pure pixel value
    let pixels: i32 = s
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid coordinate value: '{s}'"))?;
    Ok(CoordValue::px(pixels))
}

/// Parse an optional `+N` or `-N` pixel offset after the percent part.
fn parse_pixel_offset(s: &str) -> Result<i32> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(0);
    }
    s.parse::<i32>()
        .map_err(|_| anyhow::anyhow!("invalid pixel offset: '{s}'"))
}

/// CSS TRBL (top, right, bottom, left) shorthand values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Trbl {
    pub top: CoordValue,
    pub right: CoordValue,
    pub bottom: CoordValue,
    pub left: CoordValue,
}

/// Parse CSS TRBL shorthand.
///
/// - 1 value: all four sides
/// - 2 values: vertical, horizontal
/// - 3 values: top, horizontal, bottom
/// - 4 values: top, right, bottom, left
pub fn parse_trbl(s: &str) -> Result<Trbl> {
    let parts: Vec<&str> = s.split(',').map(|p| p.trim()).collect();
    match parts.len() {
        1 => {
            let v = parse_coord(parts[0])?;
            Ok(Trbl {
                top: v,
                right: v,
                bottom: v,
                left: v,
            })
        }
        2 => {
            let v = parse_coord(parts[0])?;
            let h = parse_coord(parts[1])?;
            Ok(Trbl {
                top: v,
                right: h,
                bottom: v,
                left: h,
            })
        }
        3 => {
            let t = parse_coord(parts[0])?;
            let h = parse_coord(parts[1])?;
            let b = parse_coord(parts[2])?;
            Ok(Trbl {
                top: t,
                right: h,
                bottom: b,
                left: h,
            })
        }
        4 => {
            let t = parse_coord(parts[0])?;
            let r = parse_coord(parts[1])?;
            let b = parse_coord(parts[2])?;
            let l = parse_coord(parts[3])?;
            Ok(Trbl {
                top: t,
                right: r,
                bottom: b,
                left: l,
            })
        }
        _ => bail!(
            "TRBL shorthand requires 1-4 comma-separated values, got {}",
            parts.len()
        ),
    }
}

/// Parse a dimension string like `800x600`, `800`, or `x600`.
///
/// Returns `(width, height)` where either can be `None`.
pub fn parse_dimensions(s: &str) -> Result<(Option<u32>, Option<u32>)> {
    let s = s.trim();
    if let Some((w_str, h_str)) = s.split_once('x') {
        let w = if w_str.is_empty() {
            None
        } else {
            Some(
                w_str
                    .parse()
                    .map_err(|_| anyhow::anyhow!("invalid width: '{w_str}'"))?,
            )
        };
        let h = if h_str.is_empty() {
            None
        } else {
            Some(
                h_str
                    .parse()
                    .map_err(|_| anyhow::anyhow!("invalid height: '{h_str}'"))?,
            )
        };
        if w.is_none() && h.is_none() {
            bail!("dimension must specify at least width or height");
        }
        Ok((w, h))
    } else {
        // Single value = width
        let w: u32 = s
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid dimension: '{s}'"))?;
        Ok((Some(w), None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_coord_pixels() {
        assert_eq!(parse_coord("100").unwrap(), CoordValue::px(100));
        assert_eq!(parse_coord("-50").unwrap(), CoordValue::px(-50));
        assert_eq!(parse_coord("0").unwrap(), CoordValue::px(0));
    }

    #[test]
    fn parse_coord_percent() {
        let v = parse_coord("10%").unwrap();
        assert!((v.percent - 0.1).abs() < 1e-6);
        assert_eq!(v.pixels, 0);

        let v = parse_coord("10pct").unwrap();
        assert!((v.percent - 0.1).abs() < 1e-6);
        assert_eq!(v.pixels, 0);

        let v = parse_coord("100%").unwrap();
        assert!((v.percent - 1.0).abs() < 1e-6);
    }

    #[test]
    fn parse_coord_calc() {
        let v = parse_coord("50%+20").unwrap();
        assert!((v.percent - 0.5).abs() < 1e-6);
        assert_eq!(v.pixels, 20);

        let v = parse_coord("50%-10").unwrap();
        assert!((v.percent - 0.5).abs() < 1e-6);
        assert_eq!(v.pixels, -10);

        let v = parse_coord("50pct+20").unwrap();
        assert!((v.percent - 0.5).abs() < 1e-6);
        assert_eq!(v.pixels, 20);
    }

    #[test]
    fn parse_coord_resolve() {
        assert_eq!(CoordValue::px(100).resolve(1000), 100);
        assert_eq!(CoordValue::pct(0.5).resolve(1000), 500);
        assert_eq!(CoordValue::pct_px(0.5, 20).resolve(1000), 520);
        assert_eq!(CoordValue::pct_px(0.5, -10).resolve(1000), 490);
    }

    #[test]
    fn parse_trbl_one() {
        let t = parse_trbl("10").unwrap();
        assert_eq!(t.top, CoordValue::px(10));
        assert_eq!(t.right, CoordValue::px(10));
        assert_eq!(t.bottom, CoordValue::px(10));
        assert_eq!(t.left, CoordValue::px(10));
    }

    #[test]
    fn parse_trbl_two() {
        let t = parse_trbl("10,20").unwrap();
        assert_eq!(t.top, CoordValue::px(10));
        assert_eq!(t.right, CoordValue::px(20));
        assert_eq!(t.bottom, CoordValue::px(10));
        assert_eq!(t.left, CoordValue::px(20));
    }

    #[test]
    fn parse_trbl_three() {
        let t = parse_trbl("10,20,30").unwrap();
        assert_eq!(t.top, CoordValue::px(10));
        assert_eq!(t.right, CoordValue::px(20));
        assert_eq!(t.bottom, CoordValue::px(30));
        assert_eq!(t.left, CoordValue::px(20));
    }

    #[test]
    fn parse_trbl_four() {
        let t = parse_trbl("10,20,30,40").unwrap();
        assert_eq!(t.top, CoordValue::px(10));
        assert_eq!(t.right, CoordValue::px(20));
        assert_eq!(t.bottom, CoordValue::px(30));
        assert_eq!(t.left, CoordValue::px(40));
    }

    #[test]
    fn parse_trbl_percent() {
        let t = parse_trbl("10%").unwrap();
        assert!((t.top.percent - 0.1).abs() < 1e-6);
        assert!((t.right.percent - 0.1).abs() < 1e-6);
    }

    #[test]
    fn parse_dimensions_both() {
        assert_eq!(parse_dimensions("800x600").unwrap(), (Some(800), Some(600)));
    }

    #[test]
    fn parse_dimensions_width_only() {
        assert_eq!(parse_dimensions("800").unwrap(), (Some(800), None));
        assert_eq!(parse_dimensions("800x").unwrap(), (Some(800), None));
    }

    #[test]
    fn parse_dimensions_height_only() {
        assert_eq!(parse_dimensions("x600").unwrap(), (None, Some(600)));
    }
}
