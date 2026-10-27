//! Compile-time admission of the optional 12-bit sample path.
//! Internal fixed-point precision and XYB processing are independent of this.
#[inline]
pub(super) fn is_12_bit(precision: u8) -> bool {
    cfg!(feature = "extended-jpeg") && precision == 12
}
