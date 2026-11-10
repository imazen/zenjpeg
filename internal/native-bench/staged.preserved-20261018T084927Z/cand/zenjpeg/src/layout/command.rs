//! Local command/geometry types for the layout pipeline.
//!
//! Zenjpeg-internal replacements for what used to come from `zenlayout`.
//! The constraint solver (fit/within/cover math) lives in `zenresize`
//! (`FitMode` + `fit_dims`) and the orientation group algebra lives in
//! `zenpixels` (`Orientation`, re-exported via `zenresize::Orientation`).
//! This module just carries the user's builder calls as plain data until
//! `plan::plan_layout` converts the vec into a `zenresize::ResizeConfig`
//! plus `OrientOutput`.

use alloc::vec::Vec;

pub use zenresize::{FitMode, Orientation};

/// A single builder-level operation on the layout request.
#[derive(Clone, Copy, Debug)]
pub enum Command {
    /// Apply EXIF orientation tag (1-8). A value of 0 is a sentinel meaning
    /// "read from source EXIF"; [`LayoutRequest::execute`] resolves it before
    /// planning.
    AutoOrient(u8),
    /// 90/180/270° clockwise rotation.
    Rotate(Rotation),
    /// Horizontal or vertical flip.
    Flip(FlipAxis),
    /// Explicit source-side crop in source pixel coordinates (pre-orient).
    Crop { x: u32, y: u32, w: u32, h: u32 },
    /// Aspect-ratio constraint (Fit / Within / Cover / Stretch).
    Fit { mode: FitMode, w: u32, h: u32 },
}

/// 90/180/270° clockwise rotation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rotation {
    Rotate90,
    Rotate180,
    Rotate270,
}

impl From<Rotation> for Orientation {
    fn from(r: Rotation) -> Self {
        match r {
            Rotation::Rotate90 => Orientation::Rotate90,
            Rotation::Rotate180 => Orientation::Rotate180,
            Rotation::Rotate270 => Orientation::Rotate270,
        }
    }
}

/// Horizontal or vertical flip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlipAxis {
    Horizontal,
    Vertical,
}

impl From<FlipAxis> for Orientation {
    fn from(a: FlipAxis) -> Self {
        match a {
            FlipAxis::Horizontal => Orientation::FlipH,
            FlipAxis::Vertical => Orientation::FlipV,
        }
    }
}

/// Compose the orientation produced by walking the commands in order.
///
/// Returns [`Orientation::Identity`] when no orientation-affecting command is
/// present.
pub(super) fn compose_orientation(commands: &[Command]) -> Orientation {
    let mut o = Orientation::Identity;
    for cmd in commands {
        match cmd {
            Command::AutoOrient(exif) => {
                if let Some(step) = Orientation::from_exif(*exif) {
                    o = o.compose(step);
                }
            }
            Command::Rotate(r) => o = o.compose((*r).into()),
            Command::Flip(f) => o = o.compose((*f).into()),
            Command::Crop { .. } | Command::Fit { .. } => {}
        }
    }
    o
}

/// Scan commands for explicit source-side crops and return the aggregate
/// rectangle (last crop wins, matching the old zenlayout behavior — the
/// builder only emits one).
pub(super) fn explicit_crop(commands: &[Command]) -> Option<(u32, u32, u32, u32)> {
    commands.iter().rev().find_map(|cmd| match cmd {
        Command::Crop { x, y, w, h } => Some((*x, *y, *w, *h)),
        _ => None,
    })
}

/// Scan commands for the active Fit constraint (last wins).
pub(super) fn active_fit(commands: &[Command]) -> Option<(FitMode, u32, u32)> {
    commands.iter().rev().find_map(|cmd| match cmd {
        Command::Fit { mode, w, h } => Some((*mode, *w, *h)),
        _ => None,
    })
}

/// True if the commands contain anything that forces the lossy path
/// (resize or explicit crop).
pub(super) fn needs_lossy(commands: &[Command]) -> bool {
    commands
        .iter()
        .any(|c| matches!(c, Command::Fit { .. } | Command::Crop { .. }))
}

/// Replace any `AutoOrient(0)` sentinel with the concrete EXIF tag.
pub(super) fn resolve_auto_orient(commands: &[Command], exif_orient: u8) -> Vec<Command> {
    commands
        .iter()
        .map(|cmd| match cmd {
            Command::AutoOrient(0) => Command::AutoOrient(exif_orient),
            other => *other,
        })
        .collect()
}
