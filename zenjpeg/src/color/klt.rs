//! KLT (Karhunen-Loève Transform) decorrelation for JPEG encoding.
//!
//! Computes a per-image optimal 3x3 color decorrelation matrix using PCA on
//! the RGB pixel data. The resulting transform concentrates energy into the
//! first channel (analogous to Y in YCbCr, but optimized per-image).
//!
//! This is inspired by AV2's CCTX (Cross-Chroma Component Transform), adapted
//! for JPEG as an encoder-only optimization with ICC profile signaling.

#![allow(dead_code)]

/// A 3x3 matrix stored in row-major order.
#[derive(Debug, Clone, Copy)]
pub struct Mat3([f32; 9]);

impl Mat3 {
    /// Identity matrix.
    pub const IDENTITY: Self = Self([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

    /// Create from row-major array.
    #[inline]
    pub const fn from_rows(rows: [[f32; 3]; 3]) -> Self {
        Self([
            rows[0][0], rows[0][1], rows[0][2], rows[1][0], rows[1][1], rows[1][2], rows[2][0],
            rows[2][1], rows[2][2],
        ])
    }

    /// Get element at (row, col).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.0[row * 3 + col]
    }

    /// Set element at (row, col).
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        self.0[row * 3 + col] = val;
    }

    /// Matrix multiply: self * other.
    pub fn mul(&self, other: &Mat3) -> Mat3 {
        let mut result = [0.0f32; 9];
        for i in 0..3 {
            for j in 0..3 {
                result[i * 3 + j] = self.get(i, 0) * other.get(0, j)
                    + self.get(i, 1) * other.get(1, j)
                    + self.get(i, 2) * other.get(2, j);
            }
        }
        Mat3(result)
    }

    /// Transpose.
    pub fn transpose(&self) -> Mat3 {
        Mat3([
            self.0[0], self.0[3], self.0[6], self.0[1], self.0[4], self.0[7], self.0[2], self.0[5],
            self.0[8],
        ])
    }

    /// Matrix-vector multiply: self * [x, y, z].
    #[inline]
    pub fn transform(&self, v: [f32; 3]) -> [f32; 3] {
        [
            self.get(0, 0) * v[0] + self.get(0, 1) * v[1] + self.get(0, 2) * v[2],
            self.get(1, 0) * v[0] + self.get(1, 1) * v[1] + self.get(1, 2) * v[2],
            self.get(2, 0) * v[0] + self.get(2, 1) * v[1] + self.get(2, 2) * v[2],
        ]
    }

    /// Compute the inverse of a 3x3 matrix. Returns None if singular.
    pub fn inverse(&self) -> Option<Mat3> {
        let a = self.0;
        let det = a[0] * (a[4] * a[8] - a[5] * a[7])
            - a[1] * (a[3] * a[8] - a[5] * a[6])
            + a[2] * (a[3] * a[7] - a[4] * a[6]);

        if det.abs() < 1e-12 {
            return None;
        }

        let inv_det = 1.0 / det;
        Some(Mat3([
            (a[4] * a[8] - a[5] * a[7]) * inv_det,
            (a[2] * a[7] - a[1] * a[8]) * inv_det,
            (a[1] * a[5] - a[2] * a[4]) * inv_det,
            (a[5] * a[6] - a[3] * a[8]) * inv_det,
            (a[0] * a[8] - a[2] * a[6]) * inv_det,
            (a[2] * a[3] - a[0] * a[5]) * inv_det,
            (a[3] * a[7] - a[4] * a[6]) * inv_det,
            (a[1] * a[6] - a[0] * a[7]) * inv_det,
            (a[0] * a[4] - a[1] * a[3]) * inv_det,
        ]))
    }

    /// Access as row-major slice.
    pub fn as_slice(&self) -> &[f32; 9] {
        &self.0
    }

    /// Get a row as [f32; 3].
    pub fn row(&self, i: usize) -> [f32; 3] {
        [self.0[i * 3], self.0[i * 3 + 1], self.0[i * 3 + 2]]
    }

    /// Get a column as [f32; 3].
    pub fn col(&self, j: usize) -> [f32; 3] {
        [self.0[j], self.0[3 + j], self.0[6 + j]]
    }

    /// Determinant.
    pub fn det(&self) -> f32 {
        let a = self.0;
        a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6])
            + a[2] * (a[3] * a[7] - a[4] * a[6])
    }

    /// Computes the output range for each row when applied to [0, max_input]^3 inputs.
    ///
    /// Returns (min, max) per row. For rows with all-positive coefficients,
    /// min=0.0. For rows with mixed signs, min<0.
    pub fn channel_ranges(&self, max_input: f32) -> [(f32, f32); 3] {
        let mut ranges = [(0.0f32, 0.0f32); 3];
        for i in 0..3 {
            let row = self.row(i);
            let mut min_val = 0.0f32;
            let mut max_val = 0.0f32;
            for &c in &row {
                if c > 0.0 {
                    max_val += c * max_input;
                } else {
                    min_val += c * max_input;
                }
            }
            ranges[i] = (min_val, max_val);
        }
        ranges
    }
}

/// Precomputed parameters for encoding KLT channels in \[0, 255\] range.
///
/// When applying the KLT forward matrix to 8-bit RGB \[0, 255\], the output
/// channels may exceed the \[0, 255\] range. These parameters provide per-channel
/// scale and offset to normalize the output.
///
/// stored_i = raw_i * scale\[i\] + offset\[i\]
#[derive(Clone, Debug)]
pub struct KltEncodeParams {
    /// Forward KLT matrix.
    pub forward: Mat3,
    /// Per-channel multiplier to apply after matrix transform.
    pub scale: [f32; 3],
    /// Per-channel additive offset to apply after scale.
    pub offset: [f32; 3],
}

impl KltEncodeParams {
    /// Computes encoding parameters from a KLT forward matrix.
    ///
    /// Scales each output channel to fit exactly in \[0, 255\] for \[0, 255\]
    /// RGB input.
    pub fn from_forward(forward: Mat3) -> Self {
        let ranges = forward.channel_ranges(255.0);
        let mut scale = [0.0f32; 3];
        let mut offset = [0.0f32; 3];
        for i in 0..3 {
            let (min_val, max_val) = ranges[i];
            let range = max_val - min_val;
            if range > 0.0 {
                scale[i] = 255.0 / range;
                offset[i] = -min_val * scale[i];
            } else {
                // Degenerate: all values collapse to the same point
                scale[i] = 1.0;
                offset[i] = 128.0;
            }
        }
        Self {
            forward,
            scale,
            offset,
        }
    }

    /// Returns per-channel (inverse_scale, inverse_offset) for decoding:
    ///   raw_i = (stored_i - offset_i) / scale_i
    pub fn inverse_scale_offset(&self) -> ([f32; 3], [f32; 3]) {
        let mut inv_scale = [0.0f32; 3];
        let mut inv_offset = [0.0f32; 3];
        for i in 0..3 {
            inv_scale[i] = 1.0 / self.scale[i];
            inv_offset[i] = -self.offset[i] / self.scale[i];
        }
        (inv_scale, inv_offset)
    }
}

/// Statistics accumulator for computing the covariance matrix of RGB pixels.
///
/// Operates in a streaming fashion — call `accumulate_row` for each row of
/// pixels, then `finish` to get the covariance matrix.
pub struct CovarianceAccumulator {
    /// Running sums: [sum_r, sum_g, sum_b]
    sum: [f64; 3],
    /// Running sums of products: [rr, rg, rb, gg, gb, bb] (upper triangle)
    sum_products: [f64; 6],
    /// Total pixel count
    count: u64,
}

impl CovarianceAccumulator {
    pub fn new() -> Self {
        Self {
            sum: [0.0; 3],
            sum_products: [0.0; 6],
            count: 0,
        }
    }

    /// Accumulate a row of RGB u8 pixels (interleaved: R, G, B, R, G, B, ...).
    ///
    /// `data` is a slice of RGB triples. `bpp` is bytes per pixel (3 for RGB,
    /// 4 for RGBA/BGRA — only the first 3 channels are used).
    pub fn accumulate_rgb_u8(&mut self, data: &[u8], width: usize, bpp: usize) {
        for x in 0..width {
            let base = x * bpp;
            let r = data[base] as f64;
            let g = data[base + 1] as f64;
            let b = data[base + 2] as f64;

            self.sum[0] += r;
            self.sum[1] += g;
            self.sum[2] += b;

            // Upper triangle: rr, rg, rb, gg, gb, bb
            self.sum_products[0] += r * r;
            self.sum_products[1] += r * g;
            self.sum_products[2] += r * b;
            self.sum_products[3] += g * g;
            self.sum_products[4] += g * b;
            self.sum_products[5] += b * b;
        }
        self.count += width as u64;
    }

    /// Accumulate a row of BGR u8 pixels.
    pub fn accumulate_bgr_u8(&mut self, data: &[u8], width: usize, bpp: usize) {
        for x in 0..width {
            let base = x * bpp;
            let b = data[base] as f64;
            let g = data[base + 1] as f64;
            let r = data[base + 2] as f64;

            self.sum[0] += r;
            self.sum[1] += g;
            self.sum[2] += b;

            self.sum_products[0] += r * r;
            self.sum_products[1] += r * g;
            self.sum_products[2] += r * b;
            self.sum_products[3] += g * g;
            self.sum_products[4] += g * b;
            self.sum_products[5] += b * b;
        }
        self.count += width as u64;
    }

    /// Accumulate a row of f32 RGB pixels (values in [0, 255] range after
    /// any input scaling — this operates in the same space as u8 values).
    pub fn accumulate_rgb_f32(&mut self, data: &[f32], width: usize, stride: usize) {
        for x in 0..width {
            let base = x * stride;
            let r = data[base] as f64;
            let g = data[base + 1] as f64;
            let b = data[base + 2] as f64;

            self.sum[0] += r;
            self.sum[1] += g;
            self.sum[2] += b;

            self.sum_products[0] += r * r;
            self.sum_products[1] += r * g;
            self.sum_products[2] += r * b;
            self.sum_products[3] += g * g;
            self.sum_products[4] += g * b;
            self.sum_products[5] += b * b;
        }
        self.count += width as u64;
    }

    /// Number of pixels accumulated so far.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Compute the 3x3 covariance matrix from accumulated statistics.
    ///
    /// Returns None if fewer than 2 pixels have been accumulated.
    pub fn covariance(&self) -> Option<[[f64; 3]; 3]> {
        if self.count < 2 {
            return None;
        }

        let n = self.count as f64;
        let mean = [self.sum[0] / n, self.sum[1] / n, self.sum[2] / n];

        // Cov(X,Y) = E[XY] - E[X]*E[Y]
        let cov_rr = self.sum_products[0] / n - mean[0] * mean[0];
        let cov_rg = self.sum_products[1] / n - mean[0] * mean[1];
        let cov_rb = self.sum_products[2] / n - mean[0] * mean[2];
        let cov_gg = self.sum_products[3] / n - mean[1] * mean[1];
        let cov_gb = self.sum_products[4] / n - mean[1] * mean[2];
        let cov_bb = self.sum_products[5] / n - mean[2] * mean[2];

        Some([
            [cov_rr, cov_rg, cov_rb],
            [cov_rg, cov_gg, cov_gb],
            [cov_rb, cov_gb, cov_bb],
        ])
    }

    /// Compute the mean RGB values.
    pub fn mean(&self) -> Option<[f64; 3]> {
        if self.count == 0 {
            return None;
        }
        let n = self.count as f64;
        Some([self.sum[0] / n, self.sum[1] / n, self.sum[2] / n])
    }
}

/// Compute eigenvectors and eigenvalues of a 3x3 symmetric matrix using
/// Jacobi iteration.
///
/// Returns (eigenvalues, eigenvectors_as_columns) sorted by descending
/// eigenvalue. The eigenvector matrix V satisfies: A = V * diag(λ) * V^T.
pub fn symmetric_eigen_3x3(mat: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    // Jacobi rotation method for 3x3 symmetric matrices.
    // Usually converges in 3-6 sweeps for well-conditioned covariance matrices.

    let mut a = mat;
    // V starts as identity — accumulates rotations
    let mut v = [[1.0f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    for _sweep in 0..50 {
        // Check convergence: sum of squared off-diagonal elements
        let off_diag = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
        if off_diag < 1e-20 {
            break;
        }

        // Rotate each off-diagonal pair
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q].abs() < 1e-15 {
                continue;
            }

            let tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = if tau.abs() > 1e15 {
                // Avoid overflow for large tau
                1.0 / (2.0 * tau)
            } else {
                let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
                sign / (tau.abs() + (1.0 + tau * tau).sqrt())
            };

            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;

            // Apply Jacobi rotation to A
            let app = a[p][p];
            let aqq = a[q][q];
            let apq = a[p][q];

            a[p][p] = app - t * apq;
            a[q][q] = aqq + t * apq;
            a[p][q] = 0.0;
            a[q][p] = 0.0;

            // Update remaining elements
            let r = 3 - p - q; // the third index
            let arp = a[r][p];
            let arq = a[r][q];
            a[r][p] = c * arp - s * arq;
            a[p][r] = a[r][p];
            a[r][q] = s * arp + c * arq;
            a[q][r] = a[r][q];

            // Accumulate rotation into V
            for i in 0..3 {
                let vip = v[i][p];
                let viq = v[i][q];
                v[i][p] = c * vip - s * viq;
                v[i][q] = s * vip + c * viq;
            }
        }
    }

    // Extract eigenvalues (diagonal of A) and sort by descending eigenvalue
    let mut eigen: [(f64, usize); 3] = [
        (a[0][0], 0),
        (a[1][1], 1),
        (a[2][2], 2),
    ];
    eigen.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let eigenvalues = [eigen[0].0, eigen[1].0, eigen[2].0];

    // Reorder eigenvector columns to match sorted eigenvalues
    let mut eigenvectors = [[0.0f64; 3]; 3];
    for row in 0..3 {
        for (new_col, item) in eigen.iter().enumerate() {
            let old_col = item.1;
            eigenvectors[row][new_col] = v[row][old_col];
        }
    }

    (eigenvalues, eigenvectors)
}

/// Result of KLT analysis on an image.
#[derive(Debug, Clone)]
pub struct KltAnalysis {
    /// The forward decorrelation matrix (RGB → decorrelated channels).
    /// Rows are the principal components, sorted by descending variance.
    pub forward: Mat3,
    /// The inverse matrix (decorrelated → RGB). Equal to forward.transpose()
    /// since the eigenvectors of a symmetric matrix are orthonormal.
    pub inverse: Mat3,
    /// Mean RGB values [0-255 scale].
    pub mean: [f32; 3],
    /// Eigenvalues (variance along each principal component), descending.
    pub eigenvalues: [f32; 3],
    /// Fraction of total variance captured by the first component (the "Y" equivalent).
    pub energy_concentration: f32,
}

/// Compute the optimal KLT decorrelation matrix from a covariance matrix.
///
/// The covariance should be computed from gamma-encoded sRGB pixel values
/// in [0, 255] range (matching JPEG's operating space).
///
/// The returned forward matrix maps RGB → decorrelated channels, with the
/// first channel capturing the most variance (like luminance).
///
/// Channel values after transform are centered around 0 and need to be shifted
/// to [0, 255] for JPEG encoding (add 128 to the chroma-like channels).
pub fn compute_klt(covariance: [[f64; 3]; 3], mean: [f64; 3]) -> KltAnalysis {
    let (eigenvalues, eigenvectors) = symmetric_eigen_3x3(covariance);

    // The forward transform has eigenvectors as ROWS (each row is a principal component).
    // eigenvectors from symmetric_eigen_3x3 are stored as COLUMNS, so we need to transpose.
    let forward = Mat3::from_rows([
        [
            eigenvectors[0][0] as f32,
            eigenvectors[1][0] as f32,
            eigenvectors[2][0] as f32,
        ],
        [
            eigenvectors[0][1] as f32,
            eigenvectors[1][1] as f32,
            eigenvectors[2][1] as f32,
        ],
        [
            eigenvectors[0][2] as f32,
            eigenvectors[1][2] as f32,
            eigenvectors[2][2] as f32,
        ],
    ]);


    let total_variance = eigenvalues[0] + eigenvalues[1] + eigenvalues[2];
    let energy_concentration = if total_variance > 0.0 {
        (eigenvalues[0] / total_variance) as f32
    } else {
        1.0
    };

    // Ensure the first principal component (luminance-like) has positive correlation
    // with perceived brightness. If the dot product with [0.299, 0.587, 0.114] is
    // negative, flip the sign of that row.
    let mut forward = forward;
    let luma_row = forward.row(0);
    let luma_dot = luma_row[0] * 0.299 + luma_row[1] * 0.587 + luma_row[2] * 0.114;
    if luma_dot < 0.0 {
        for j in 0..3 {
            let val = forward.get(0, j);
            forward.set(0, j, -val);
        }
    }

    // Recompute inverse after potential sign flip
    let inverse = forward.transpose();

    KltAnalysis {
        forward,
        inverse,
        mean: [mean[0] as f32, mean[1] as f32, mean[2] as f32],
        eigenvalues: [
            eigenvalues[0] as f32,
            eigenvalues[1] as f32,
            eigenvalues[2] as f32,
        ],
        energy_concentration,
    }
}

/// Determine if KLT decorrelation would provide meaningful benefit over
/// standard BT.601 YCbCr.
///
/// Returns true if the image's color statistics differ enough from the
/// BT.601 assumptions that a custom transform is worth the ICC profile overhead.
pub fn klt_is_beneficial(analysis: &KltAnalysis) -> bool {
    // If energy is already well-concentrated (>97%), the gain is marginal
    // and the ICC profile overhead may not be worth it.
    // But if the first component captures <95% of variance, there's room to improve.
    //
    // Also check if the forward transform differs meaningfully from BT.601.
    let bt601_luma = [0.299f32, 0.587, 0.114];
    let klt_luma = analysis.forward.row(0);

    // Cosine similarity between KLT's first row and BT.601 luma weights
    let dot: f32 =
        klt_luma[0] * bt601_luma[0] + klt_luma[1] * bt601_luma[1] + klt_luma[2] * bt601_luma[2];

    let mag_klt =
        (klt_luma[0] * klt_luma[0] + klt_luma[1] * klt_luma[1] + klt_luma[2] * klt_luma[2]).sqrt();
    let mag_bt =
        (bt601_luma[0] * bt601_luma[0] + bt601_luma[1] * bt601_luma[1] + bt601_luma[2] * bt601_luma[2]).sqrt();

    let cos_sim = dot / (mag_klt * mag_bt);

    // If the KLT luma direction is very similar to BT.601 (cos > 0.995), skip KLT.
    // The chroma decorrelation difference won't save enough bits to justify the profile.
    cos_sim < 0.995
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat3_inverse() {
        let m = Mat3::from_rows([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 2.0]]);
        let inv = m.inverse().expect("should be invertible");
        let identity = m.mul(&inv);

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (identity.get(i, j) - expected).abs() < 1e-5,
                    "identity[{i}][{j}] = {} (expected {expected})",
                    identity.get(i, j)
                );
            }
        }
    }

    #[test]
    fn test_symmetric_eigen_identity() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let (eigenvalues, _eigenvectors) = symmetric_eigen_3x3(identity);

        for &ev in &eigenvalues {
            assert!((ev - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_symmetric_eigen_diagonal() {
        let diag = [[5.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]];
        let (eigenvalues, _) = symmetric_eigen_3x3(diag);

        assert!((eigenvalues[0] - 5.0).abs() < 1e-10);
        assert!((eigenvalues[1] - 3.0).abs() < 1e-10);
        assert!((eigenvalues[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_symmetric_eigen_symmetric() {
        // A symmetric matrix with known eigenvalues
        let mat = [[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
        let (eigenvalues, eigenvectors) = symmetric_eigen_3x3(mat);

        // Verify A*v = λ*v for each eigenvector
        for col in 0..3 {
            let v = [eigenvectors[0][col], eigenvectors[1][col], eigenvectors[2][col]];
            let av = [
                mat[0][0] * v[0] + mat[0][1] * v[1] + mat[0][2] * v[2],
                mat[1][0] * v[0] + mat[1][1] * v[1] + mat[1][2] * v[2],
                mat[2][0] * v[0] + mat[2][1] * v[1] + mat[2][2] * v[2],
            ];
            let lv = [
                eigenvalues[col] * v[0],
                eigenvalues[col] * v[1],
                eigenvalues[col] * v[2],
            ];

            for i in 0..3 {
                assert!(
                    (av[i] - lv[i]).abs() < 1e-8,
                    "A*v != λ*v at component {i} for eigenvector {col}: {av:?} vs {lv:?}"
                );
            }
        }

        // Verify eigenvalues are sorted descending
        assert!(eigenvalues[0] >= eigenvalues[1]);
        assert!(eigenvalues[1] >= eigenvalues[2]);
    }

    #[test]
    fn test_covariance_accumulator() {
        let mut acc = CovarianceAccumulator::new();

        // Perfectly correlated: R=G=B
        let row: Vec<u8> = (0..=255).flat_map(|v| [v, v, v]).collect();
        acc.accumulate_rgb_u8(&row, 256, 3);

        let cov = acc.covariance().unwrap();

        // All covariances should be equal (perfect correlation)
        assert!((cov[0][0] - cov[0][1]).abs() < 1.0);
        assert!((cov[0][0] - cov[0][2]).abs() < 1.0);
        assert!((cov[1][1] - cov[1][2]).abs() < 1.0);
    }

    #[test]
    fn test_klt_grayscale_image() {
        let mut acc = CovarianceAccumulator::new();

        // Perfectly correlated grayscale: R=G=B
        let row: Vec<u8> = (0..=255).flat_map(|v| [v, v, v]).collect();
        acc.accumulate_rgb_u8(&row, 256, 3);

        let cov = acc.covariance().unwrap();
        let mean = acc.mean().unwrap();
        let analysis = compute_klt(cov, mean);

        // For perfect R=G=B correlation, first eigenvalue should capture ~100% of variance
        assert!(
            analysis.energy_concentration > 0.99,
            "energy_concentration = {} (expected >0.99)",
            analysis.energy_concentration
        );
    }

    #[test]
    fn test_klt_orthogonal_roundtrip() {
        let mut acc = CovarianceAccumulator::new();

        // Create a test image with non-trivial color distribution
        for r in (0..=255).step_by(8) {
            for g in (0..=255).step_by(8) {
                let b = ((r as u16 + g as u16) / 2) as u8;
                let row = [r, g, b];
                acc.accumulate_rgb_u8(&row, 1, 3);
            }
        }

        let cov = acc.covariance().unwrap();
        let mean = acc.mean().unwrap();
        let analysis = compute_klt(cov, mean);

        // Verify forward * inverse = identity (orthogonal matrix)
        let product = analysis.forward.mul(&analysis.inverse);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product.get(i, j) - expected).abs() < 1e-4,
                    "forward*inverse [{i}][{j}] = {} (expected {expected})",
                    product.get(i, j)
                );
            }
        }

        // Verify roundtrip: inverse(forward(pixel)) == pixel
        let test_pixel = [128.0f32, 64.0, 200.0];
        let transformed = analysis.forward.transform(test_pixel);
        let recovered = analysis.inverse.transform(transformed);
        for i in 0..3 {
            assert!(
                (recovered[i] - test_pixel[i]).abs() < 0.01,
                "roundtrip channel {i}: {} vs {}",
                recovered[i],
                test_pixel[i]
            );
        }
    }
}
