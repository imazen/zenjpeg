//! Coordinate value parsing for the `process` subcommand.
//!
//! Supports pixel values, percentages (`%` or `pct`), and calc expressions
//! (`50%+20`, `50pct-10`). Also provides CSS TRBL shorthand, crop rect,
//! aspect ratio, position, and dimension parsers.

use anyhow::{bail, Result};
use zenlayout::{Gravity, RegionCoord};

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

    pub const fn pct(percent: f32) -> Self {
        Self { percent, pixels: 0 }
    }

    pub const fn pct_px(percent: f32, pixels: i32) -> Self {
        Self { percent, pixels }
    }

    /// Resolve against a source dimension.
    pub fn resolve(self, dim: u32) -> i32 {
        (self.percent * dim as f32).round() as i32 + self.pixels
    }

    /// Convert to a `RegionCoord`.
    pub fn to_region_coord(self) -> RegionCoord {
        RegionCoord::pct_px(self.percent, self.pixels)
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

/// Parse a crop rect: `x,y,w,h` where each can be px or %.
pub fn parse_crop_rect(s: &str) -> Result<[CoordValue; 4]> {
    let parts: Vec<&str> = s.split(',').map(|p| p.trim()).collect();
    if parts.len() != 4 {
        bail!(
            "crop rect requires 4 comma-separated values (x,y,w,h), got {}",
            parts.len()
        );
    }
    Ok([
        parse_coord(parts[0])?,
        parse_coord(parts[1])?,
        parse_coord(parts[2])?,
        parse_coord(parts[3])?,
    ])
}

/// Parse an aspect ratio like `16:9` or `4:3`.
pub fn parse_aspect_ratio(s: &str) -> Result<(u32, u32)> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 2 {
        bail!("aspect ratio must be W:H (e.g. 16:9), got '{s}'");
    }
    let w: u32 = parts[0]
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid aspect width: '{}'", parts[0]))?;
    let h: u32 = parts[1]
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid aspect height: '{}'", parts[1]))?;
    if w == 0 || h == 0 {
        bail!("aspect ratio values must be positive");
    }
    Ok((w, h))
}

/// Parse a position string into a `Gravity`.
///
/// Accepts:
/// - Named positions: `center`, `top-left`, `top`, `top-right`, `left`, `right`,
///   `bottom-left`, `bottom`, `bottom-right`
/// - Percentage pair: `30%,70%` or `30pct,70pct`
pub fn parse_position(s: &str) -> Result<Gravity> {
    let s = s.trim().to_ascii_lowercase();
    match s.as_str() {
        "center" => Ok(Gravity::Center),
        "top-left" | "topleft" => Ok(Gravity::Percentage(0.0, 0.0)),
        "top" | "top-center" => Ok(Gravity::Percentage(0.5, 0.0)),
        "top-right" | "topright" => Ok(Gravity::Percentage(1.0, 0.0)),
        "left" | "center-left" => Ok(Gravity::Percentage(0.0, 0.5)),
        "right" | "center-right" => Ok(Gravity::Percentage(1.0, 0.5)),
        "bottom-left" | "bottomleft" => Ok(Gravity::Percentage(0.0, 1.0)),
        "bottom" | "bottom-center" => Ok(Gravity::Percentage(0.5, 1.0)),
        "bottom-right" | "bottomright" => Ok(Gravity::Percentage(1.0, 1.0)),
        _ => {
            // Try percent pair
            let parts: Vec<&str> = s.split(',').collect();
            if parts.len() == 2 {
                let x = parse_pct_value(parts[0])?;
                let y = parse_pct_value(parts[1])?;
                Ok(Gravity::Percentage(x, y))
            } else {
                bail!("invalid position: '{s}' (expected named position or X%,Y%)")
            }
        }
    }
}

/// Parse a percentage value like `30%` or `30pct` → 0.3.
fn parse_pct_value(s: &str) -> Result<f32> {
    let s = s.trim();
    if let Some(num) = s.strip_suffix('%') {
        let v: f32 = num
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid percent: '{s}'"))?;
        Ok(v / 100.0)
    } else if let Some(num) = s.strip_suffix("pct") {
        let v: f32 = num
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid percent: '{s}'"))?;
        Ok(v / 100.0)
    } else {
        bail!("position values must use % or pct suffix: '{s}'")
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

/// Parse a region string: `left,top,right,bottom` where each can be px, %, or calc.
pub fn parse_region(s: &str) -> Result<[CoordValue; 4]> {
    let parts: Vec<&str> = s.split(',').map(|p| p.trim()).collect();
    if parts.len() != 4 {
        bail!(
            "region requires 4 comma-separated values (left,top,right,bottom), got {}",
            parts.len()
        );
    }
    Ok([
        parse_coord(parts[0])?,
        parse_coord(parts[1])?,
        parse_coord(parts[2])?,
        parse_coord(parts[3])?,
    ])
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
    fn parse_crop_rect_basic() {
        let r = parse_crop_rect("0,0,800,600").unwrap();
        assert_eq!(r[0], CoordValue::px(0));
        assert_eq!(r[1], CoordValue::px(0));
        assert_eq!(r[2], CoordValue::px(800));
        assert_eq!(r[3], CoordValue::px(600));
    }

    #[test]
    fn parse_crop_rect_percent() {
        let r = parse_crop_rect("10%,10%,80%,80%").unwrap();
        assert!((r[0].percent - 0.1).abs() < 1e-6);
        assert!((r[2].percent - 0.8).abs() < 1e-6);
    }

    #[test]
    fn parse_aspect_basic() {
        assert_eq!(parse_aspect_ratio("16:9").unwrap(), (16, 9));
        assert_eq!(parse_aspect_ratio("4:3").unwrap(), (4, 3));
        assert_eq!(parse_aspect_ratio("1:1").unwrap(), (1, 1));
    }

    #[test]
    fn parse_aspect_zero_fails() {
        assert!(parse_aspect_ratio("0:9").is_err());
        assert!(parse_aspect_ratio("16:0").is_err());
    }

    #[test]
    fn parse_position_named() {
        assert!(matches!(parse_position("center").unwrap(), Gravity::Center));
        assert!(matches!(
            parse_position("top-left").unwrap(),
            Gravity::Percentage(x, y) if x == 0.0 && y == 0.0
        ));
        assert!(matches!(
            parse_position("bottom-right").unwrap(),
            Gravity::Percentage(x, y) if x == 1.0 && y == 1.0
        ));
    }

    #[test]
    fn parse_position_percent() {
        match parse_position("30%,70%").unwrap() {
            Gravity::Percentage(x, y) => {
                assert!((x - 0.3).abs() < 1e-6);
                assert!((y - 0.7).abs() < 1e-6);
            }
            _ => panic!("expected Percentage"),
        }
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

    #[test]
    fn parse_region_basic() {
        let r = parse_region("0,0,100%,100%").unwrap();
        assert_eq!(r[0], CoordValue::px(0));
        assert_eq!(r[1], CoordValue::px(0));
        assert!((r[2].percent - 1.0).abs() < 1e-6);
        assert!((r[3].percent - 1.0).abs() < 1e-6);
    }

    #[test]
    fn parse_region_calc() {
        let r = parse_region("-20,-20,100%+20,100%+20").unwrap();
        assert_eq!(r[0], CoordValue::px(-20));
        assert_eq!(r[1], CoordValue::px(-20));
        assert!((r[2].percent - 1.0).abs() < 1e-6);
        assert_eq!(r[2].pixels, 20);
    }
}
