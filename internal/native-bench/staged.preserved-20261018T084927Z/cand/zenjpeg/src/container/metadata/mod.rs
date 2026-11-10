//! JPEG component filtering and optional extended metadata inspection.
//!
//! [`crate::container::metadata::filter_for_gain_map`] filters an encoded JPEG
//! component before gain-map assembly, without decoding pixels. Extended
//! segment/trailer inspection and reassembly are available with the default-off
//! `extended-metadata` feature. These operations do not require inventory or a
//! pipeline crate.

mod filter;
pub use filter::{Error, filter_for_gain_map};

#[cfg(feature = "extended-metadata")]
mod inspection;
#[cfg(feature = "extended-metadata")]
pub use inspection::*;
