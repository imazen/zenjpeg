//! The crate's unit tests predate per-plane strides and exercise tight planes: these
//! pass tight views to the one strided form of each operation.

use crate::{Matrix, Range, SharpYuvConfig, YuvContext, decode, encode, sharp, types::Strides};
use imgref::{ImgRef, ImgRefMut};

fn plane(p: &[u8], w: usize, h: usize) -> ImgRef<'_, u8> {
    ImgRef::new(p, w, h)
}

fn plane_mut(p: &mut [u8], w: usize, h: usize) -> ImgRefMut<'_, u8> {
    ImgRefMut::new(p, w, h)
}

impl YuvContext {
    pub(crate) fn t_encode_420_u8(
        &mut self,
        rgb: &[u8],
        y: &mut [u8],
        cb: &mut [u8],
        cr: &mut [u8],
        w: usize,
        h: usize,
    ) {
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        self.encode_420_u8(
            plane(rgb, w * 3, h),
            plane_mut(y, w, h),
            plane_mut(cb, cw, ch),
            plane_mut(cr, cw, ch),
        )
        .unwrap();
    }

    pub(crate) fn t_encode_444_u8(
        &mut self,
        rgb: &[u8],
        y: &mut [u8],
        cb: &mut [u8],
        cr: &mut [u8],
        w: usize,
        h: usize,
    ) {
        self.encode_444_u8(
            plane(rgb, w * 3, h),
            plane_mut(y, w, h),
            plane_mut(cb, w, h),
            plane_mut(cr, w, h),
        )
        .unwrap();
    }

    pub(crate) fn t_decode_444_to_rgb(
        &self,
        y: &[u8],
        cb: &[u8],
        cr: &[u8],
        rgb: &mut [u8],
        w: usize,
        h: usize,
    ) {
        yuv444_to_rgb_with(y, cb, cr, rgb, w, h, self.range, self.matrix);
    }

    pub(crate) fn t_decode_420_to_rgb(
        &self,
        y: &[u8],
        cb: &[u8],
        cr: &[u8],
        rgb: &mut [u8],
        w: usize,
        h: usize,
    ) {
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        decode::yuv420_to_rgb(
            plane(y, w, h),
            plane(cb, cw, ch),
            plane(cr, cw, ch),
            plane_mut(rgb, w * 3, h),
            self.range,
            self.matrix,
        );
    }
}

pub(crate) fn rgb_to_yuv420(
    rgb: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    w: usize,
    h: usize,
) {
    encode::rgb_to_yuv420(
        rgb,
        y,
        cb,
        cr,
        w,
        h,
        Strides::tight(w, w.div_ceil(2)),
        Range::Full,
        Matrix::Bt601,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rgb_to_yuv444_with(
    rgb: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    w: usize,
    h: usize,
    range: Range,
    matrix: Matrix,
) {
    encode::rgb_to_yuv444(rgb, y, cb, cr, w, h, Strides::tight(w, w), range, matrix);
}

pub(crate) fn yuv444_to_rgb(y: &[u8], cb: &[u8], cr: &[u8], rgb: &mut [u8], w: usize, h: usize) {
    yuv444_to_rgb_with(y, cb, cr, rgb, w, h, Range::Full, Matrix::Bt601);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv444_to_rgb_with(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    rgb: &mut [u8],
    w: usize,
    h: usize,
    range: Range,
    matrix: Matrix,
) {
    decode::yuv444_to_rgb(
        plane(y, w, h),
        plane(cb, w, h),
        plane(cr, w, h),
        plane_mut(rgb, w * 3, h),
        range,
        matrix,
    );
}

pub(crate) fn yuv420_to_rgb(y: &[u8], cb: &[u8], cr: &[u8], rgb: &mut [u8], w: usize, h: usize) {
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    decode::yuv420_to_rgb(
        plane(y, w, h),
        plane(cb, cw, ch),
        plane(cr, cw, ch),
        plane_mut(rgb, w * 3, h),
        Range::Full,
        Matrix::Bt601,
    );
}

pub(crate) fn yuv422_to_rgb(y: &[u8], cb: &[u8], cr: &[u8], rgb: &mut [u8], w: usize, h: usize) {
    let cw = w.div_ceil(2);
    decode::yuv422_to_rgb(
        plane(y, w, h),
        plane(cb, cw, h),
        plane(cr, cw, h),
        plane_mut(rgb, w * 3, h),
        Range::Full,
        Matrix::Bt601,
    );
}

pub(crate) fn yuv400_to_rgb(y: &[u8], rgb: &mut [u8], w: usize, h: usize) {
    decode::yuv400_to_rgb(
        plane(y, w, h),
        plane_mut(rgb, w * 3, h),
        Range::Full,
        Matrix::Bt601,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rgb_to_yuv420_sharp(
    rgb: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    w: usize,
    h: usize,
    range: Range,
    matrix: Matrix,
    config: &SharpYuvConfig,
) {
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    sharp::rgb_to_yuv420_sharp(
        plane(rgb, w * 3, h),
        plane_mut(y, w, h),
        plane_mut(cb, cw, ch),
        plane_mut(cr, cw, ch),
        range,
        matrix,
        config,
        &mut sharp::SharpYuvWorkspace::new(cw),
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn refine_y_420_u8(
    rgb: &[u8],
    y: &mut [u8],
    cb: &[u8],
    cr: &[u8],
    w: usize,
    h: usize,
    range: Range,
    matrix: Matrix,
) -> u64 {
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    sharp::refine_y_420_u8(
        plane(rgb, w * 3, h),
        plane_mut(y, w, h),
        plane(cb, cw, ch),
        plane(cr, cw, ch),
        range,
        matrix,
    )
}
