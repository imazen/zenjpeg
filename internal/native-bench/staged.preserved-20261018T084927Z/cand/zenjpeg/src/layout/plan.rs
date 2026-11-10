//! Convert a command list into a concrete `zenresize` configuration.
//!
//! Replaces what used to come out of `zenlayout::compute_layout` — computes
//! the final output dimensions, the crop (if any), the resize dimensions,
//! and the post-resize orientation, from the logical user-intent sequence
//! `crop → orient → fit`.
//!
//! **Pipeline model.** zenresize 0.3 resizes in source coordinates and then
//! applies orientation post-resize:
//!
//! ```text
//!   source  --crop-->  resize  --orient-->  final output
//! ```
//!
//! The user specifies target dimensions in *post-orient* (visual) space —
//! `fit(800, 600)` means "I want the final 800×600". When the composed
//! orient swaps axes (Rotate90/270/Transpose/Transverse), the ResizeConfig
//! `out_width` / `out_height` are the transposed pre-orient dimensions.

use zenresize::{FitMode, OrientOutput, Orientation, PixelDescriptor, ResizeConfig, SourceRegion};

use super::LayoutConfig;
use super::command::{Command, active_fit, compose_orientation, explicit_crop, needs_lossy};

/// Fully-planned layout — ready to feed into a `StreamingResize`.
pub(crate) struct Plan {
    /// Resize configuration. `source_region` carries any user crop; for Cover
    /// the region is aspect-aligned so the resize produces the exact target
    /// dims without overflow.
    pub resize: ResizeConfig,
    /// Orientation to apply post-resize (feeds `with_orientation`).
    pub orient: OrientOutput,
    /// Final encoder output dimensions (post-orient).
    pub final_w: u32,
    pub final_h: u32,
    /// True when the whole pipeline is a no-op (no crop, no resize, no orient).
    pub is_identity: bool,
    /// The source rectangle the resize reads (`x, y, w, h`, pre-orient
    /// source coordinates; the whole frame when nothing crops).
    pub source_rect: (u32, u32, u32, u32),
    /// The resize output (pre-orient) dimensions.
    pub resize_out: (u32, u32),
    /// Whether the orientation swaps the axes.
    pub swaps_axes: bool,
}

impl Plan {
    /// The same geometry for a secondary image (an Ultra HDR gain map) of
    /// `sec_w`×`sec_h` covering this plan's `src_w`×`src_h` frame: the source
    /// rectangle (rounded outward) and the output size scale by the size
    /// ratio, the orientation is the same. The resize filters the stored
    /// values (`srgb()`), not linear light: a gain map holds encoded boosts,
    /// not colours, and linearizing them as sRGB shifts the boost wherever the
    /// map has detail.
    pub(crate) fn for_secondary(
        &self,
        src_w: u32,
        src_h: u32,
        sec_w: u32,
        sec_h: u32,
        layout_cfg: &LayoutConfig,
        format: PixelDescriptor,
    ) -> Plan {
        let scale_down = |v: u32, to: u32, from: u32| {
            (u64::from(v) * u64::from(to) / u64::from(from.max(1))) as u32
        };
        let scale_up = |v: u32, to: u32, from: u32| {
            u64::from(v)
                .saturating_mul(u64::from(to))
                .div_ceil(u64::from(from.max(1))) as u32
        };
        let scale_round = |v: u32, to: u32, from: u32| {
            ((u64::from(v) * u64::from(to) + u64::from(from.max(1)) / 2) / u64::from(from.max(1)))
                .max(1) as u32
        };
        let (x, y, w, h) = self.source_rect;
        let x0 = scale_down(x, sec_w, src_w).min(sec_w.saturating_sub(1));
        let y0 = scale_down(y, sec_h, src_h).min(sec_h.saturating_sub(1));
        let x1 = scale_up(x + w, sec_w, src_w).clamp(x0 + 1, sec_w);
        let y1 = scale_up(y + h, sec_h, src_h).clamp(y0 + 1, sec_h);
        let out_w = scale_round(self.resize_out.0, sec_w, src_w);
        let out_h = scale_round(self.resize_out.1, sec_h, src_h);
        let whole = (x0, y0, x1 - x0, y1 - y0) == (0, 0, sec_w, sec_h);
        let mut builder = ResizeConfig::builder(sec_w, sec_h, out_w, out_h)
            .filter(layout_cfg.filter)
            .format(format)
            .srgb();
        if !whole {
            builder = builder.source_region(SourceRegion {
                x: x0,
                y: y0,
                width: x1 - x0,
                height: y1 - y0,
            });
        }
        let (final_w, final_h) = if self.swaps_axes {
            (out_h, out_w)
        } else {
            (out_w, out_h)
        };
        Plan {
            resize: builder.build(),
            orient: self.orient,
            final_w,
            final_h,
            is_identity: self.is_identity && whole && (out_w, out_h) == (sec_w, sec_h),
            source_rect: (x0, y0, x1 - x0, y1 - y0),
            resize_out: (out_w, out_h),
            swaps_axes: self.swaps_axes,
        }
    }
}

/// Build a concrete plan from the command list.
///
/// `commands` has already had any `AutoOrient(0)` sentinels resolved against
/// the source EXIF.
pub(crate) fn plan_layout(
    commands: &[Command],
    src_w: u32,
    src_h: u32,
    layout_cfg: &LayoutConfig,
) -> Plan {
    let orient = compose_orientation(commands);
    let crop = explicit_crop(commands);
    let fit = active_fit(commands);

    // Step 1: resolve the source-side crop rectangle (in pre-orient source coords).
    let (cx, cy, cw, ch) = crop.unwrap_or((0, 0, src_w, src_h));

    // Step 2: the cropped source as seen *after* orientation — this is what
    // the user's Fit target dimensions measure against.
    let (ori_w, ori_h) = if orient.swaps_axes() {
        (ch, cw)
    } else {
        (cw, ch)
    };

    // Step 3: apply Fit. The result (final_w, final_h) is in post-orient
    // space — the dimensions the encoder emits.
    let (final_w, final_h) = match fit {
        Some((mode, tw, th)) => zenresize::fit_dims(ori_w, ori_h, tw, th, mode),
        None => (ori_w, ori_h),
    };

    // Step 4: translate back to the pre-orient dimensions the ResizeConfig
    // needs as out_width/out_height.
    let (resize_out_w, resize_out_h) = if orient.swaps_axes() {
        (final_h, final_w)
    } else {
        (final_w, final_h)
    };

    // Step 5: Cover mode wants the source further cropped to the target
    // aspect so the resize produces the exact target dims without stretching.
    // `fit_cover_source_crop` returns the aspect crop in oriented (post-orient)
    // coordinates; we need it in pre-orient source coords, combined with any
    // explicit user crop.
    let source_region = if let Some((FitMode::Cover, tw, th)) = fit {
        let (acx, acy, acw, ach) = zenresize::fit_cover_source_crop(ori_w, ori_h, tw, th);
        // The aspect crop is relative to the *oriented* cropped source. Map it
        // back through the orientation to get coordinates in pre-orient
        // cropped-source space.
        let (acx_pre, acy_pre, acw_pre, ach_pre) =
            unmap_rect_through_orient(orient, acx, acy, acw, ach, cw, ch);
        // Then shift by the explicit crop offset to get source coords.
        if acw_pre > 0 && ach_pre > 0 {
            Some(SourceRegion {
                x: cx + acx_pre,
                y: cy + acy_pre,
                width: acw_pre,
                height: ach_pre,
            })
        } else {
            crop.map(|(x, y, w, h)| SourceRegion {
                x,
                y,
                width: w,
                height: h,
            })
        }
    } else {
        crop.map(|(x, y, w, h)| SourceRegion {
            x,
            y,
            width: w,
            height: h,
        })
    };

    // Build the ResizeConfig. in_width/in_height are the full source dims;
    // the resizer applies source_region internally.
    let mut builder = ResizeConfig::builder(src_w, src_h, resize_out_w, resize_out_h)
        .filter(layout_cfg.filter)
        .format(PixelDescriptor::RGB8_SRGB)
        .linear();
    let source_rect = source_region
        .as_ref()
        .map_or((0, 0, src_w, src_h), |r| (r.x, r.y, r.width, r.height));
    if let Some(region) = source_region {
        builder = builder.source_region(region);
    }
    let resize = builder.build();

    let is_identity = orient.is_identity()
        && !needs_lossy(commands)
        && resize_out_w == src_w
        && resize_out_h == src_h;

    Plan {
        resize,
        orient: orient.into(),
        final_w,
        final_h,
        is_identity,
        source_rect,
        resize_out: (resize_out_w, resize_out_h),
        swaps_axes: orient.swaps_axes(),
    }
}

/// Map a rectangle expressed in *oriented* coordinates back through `orient`
/// to the underlying pre-orient coordinate space of size `pre_w × pre_h`.
///
/// Needed for [`FitMode::Cover`] — `fit_cover_source_crop` returns the aspect
/// crop in post-orient space but the zenresize `source_region` is interpreted
/// pre-orient.
fn unmap_rect_through_orient(
    orient: Orientation,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    pre_w: u32,
    pre_h: u32,
) -> (u32, u32, u32, u32) {
    // Map the oriented-space box corners back to pre-orient space. Apply the
    // inverse orient and rebuild a rect (after inverse, axes may be swapped
    // back, requiring a fresh axis-aligned bounding rect from the corner pair).
    let inv = orient.inverse();

    // Oriented size: if orient swapped axes, the "oriented source" dims are
    // (pre_h, pre_w); otherwise (pre_w, pre_h).
    let (ori_w, ori_h) = if orient.swaps_axes() {
        (pre_h, pre_w)
    } else {
        (pre_w, pre_h)
    };

    // Two opposite corners of the oriented rect: (x, y) and (x+w-1, y+h-1).
    let (x0, y0) = map_pixel(inv, x, y, ori_w, ori_h);
    let (x1, y1) = map_pixel(
        inv,
        x + w.saturating_sub(1),
        y + h.saturating_sub(1),
        ori_w,
        ori_h,
    );
    let xmin = x0.min(x1);
    let ymin = y0.min(y1);
    let xmax = x0.max(x1);
    let ymax = y0.max(y1);
    (xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)
}

/// Forward-map a pixel `(sx, sy)` through `orient` given source dims
/// `(w, h)`. Mirrors `zenresize::OrientOutput::map_pixel`'s formulas; kept
/// local to avoid depending on that specific API shape.
fn map_pixel(orient: Orientation, sx: u32, sy: u32, w: u32, h: u32) -> (u32, u32) {
    // Clamp defensively for zero-size inputs.
    let w = w.max(1);
    let h = h.max(1);
    match orient {
        Orientation::Identity => (sx, sy),
        Orientation::FlipH => (w - 1 - sx, sy),
        Orientation::Rotate90 => (h - 1 - sy, sx),
        Orientation::Transpose => (sy, sx),
        Orientation::Rotate180 => (w - 1 - sx, h - 1 - sy),
        Orientation::FlipV => (sx, h - 1 - sy),
        Orientation::Rotate270 => (sy, w - 1 - sx),
        Orientation::Transverse => (h - 1 - sy, w - 1 - sx),
        _ => (sx, sy),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::command::Rotation;

    fn fit(w: u32, h: u32) -> Command {
        Command::Fit {
            mode: FitMode::Fit,
            w,
            h,
        }
    }

    fn secondary(commands: &[Command], src: (u32, u32), sec: (u32, u32)) -> Plan {
        let cfg = LayoutConfig::new(85.0_f32);
        plan_layout(commands, src.0, src.1, &cfg).for_secondary(
            src.0,
            src.1,
            sec.0,
            sec.1,
            &cfg,
            PixelDescriptor::GRAY8_SRGB,
        )
    }

    #[test]
    fn secondary_size_follows_the_primary_resize() {
        // 2048² → 512²: a 512² map becomes 128².
        let p = secondary(&[fit(512, 512)], (2048, 2048), (512, 512));
        assert_eq!((p.final_w, p.final_h), (128, 128));
        // 1000x500 → 500x250: a 250x125 map becomes 125x63 (62.5 rounds up).
        let p = secondary(&[fit(500, 250)], (1000, 500), (250, 125));
        assert_eq!((p.final_w, p.final_h), (125, 63));
        // Never below one pixel.
        let p = secondary(&[fit(1, 1)], (1000, 1000), (2, 2));
        assert!(p.final_w >= 1 && p.final_h >= 1);
    }

    #[test]
    fn secondary_crop_scales_outward_and_orientation_follows() {
        // Crop the right half of a 64x32 frame; a 16x8 map reads its right
        // half too.
        let crop = Command::Crop {
            x: 32,
            y: 0,
            w: 32,
            h: 32,
        };
        let p = secondary(&[crop], (64, 32), (16, 8));
        assert_eq!(p.source_rect, (8, 0, 8, 8));
        // A crop that does not fall on the map's grid widens to cover it.
        let odd = Command::Crop {
            x: 5,
            y: 3,
            w: 10,
            h: 10,
        };
        let p = secondary(&[odd], (64, 32), (16, 8));
        assert_eq!(p.source_rect, (1, 0, 3, 4));
        // A rotation swaps the map's output axes as it does the primary's.
        let p = secondary(
            &[Command::Rotate(Rotation::Rotate90), fit(16, 32)],
            (64, 32),
            (16, 8),
        );
        assert!(p.swaps_axes);
        assert_eq!((p.final_w, p.final_h), (4, 8));
    }
}
