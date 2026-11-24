#![forbid(unsafe_code)]

#[path = "../../../zenjpeg/src/foundation/consts.rs"]
pub mod jpeg_consts;
pub mod foundation {
    pub use crate::jpeg_consts as consts;
}
#[path = "../../../zenjpeg/src/color/xyb.rs"]
pub mod xyb;
