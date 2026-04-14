//! Post-decode deblocking filters for JPEG artifacts.
//!
//! Three complementary strategies:
//!
//! - **Boundary 4-tap** ([`boundary`]): Pixel-domain H.264-style [1,3,3,1]/8 filter
//!   at 8×8 block boundaries. Effective across all quality levels.
//!
//! - **Knusperli** ([`knusperli`]): DCT-domain boundary correction from the
//!   [Knusperli](https://github.com/google/knusperli) project. Analytically computes
//!   boundary discontinuities and distributes corrections across low frequencies.
//!   Best at low quality (Q5–Q30).
//!
//! - **Triage** ([`triage`]): Three-category pixel-domain filter (US 7,079,703 B2)
//!   that classifies each 8×8 luma block as uniform, transitional, or busy via
//!   the 3×3 neighbourhood variance map, then selects between a 7×7 Gaussian
//!   deblock kernel, a 3×3 dering kernel, or the original pixels accordingly.
//!
//! Use [`detect::content`](crate::detect::content) to choose the right strategy
//! based on content type and quality level.

pub mod boundary;
pub mod knusperli;
pub mod triage;

pub use boundary::{
    BoundaryStrength, filter_interleaved_u8_boundary_4tap, filter_plane_boundary_4tap,
};
pub use knusperli::process_component;
pub use triage::{BlockClass, TriageConfig, classify_plane, filter_plane_triage};
