//! Gaussian blur implementation for butteraugli.
//!
//! Butteraugli uses Gaussian blurs at various scales to separate
//! frequency bands. The blur is implemented as a separable convolution.
//!
//! The C++ butteraugli uses clamp-to-edge boundary handling with
//! re-normalization for border pixels. This module matches that behavior.

use crate::image::ImageF;

/// Computes a 1D Gaussian kernel for the given sigma.
///
/// Unlike C++ which returns un-normalized weights, we return normalized
/// weights for the interior case. Border handling re-normalizes as needed.
#[must_use]
pub fn compute_kernel(sigma: f32) -> Vec<f32> {
    const M: f32 = 2.25; // Accuracy increases when m is increased
    let scaler = -1.0 / (2.0 * sigma * sigma);
    let diff = (M * sigma.abs()).max(1.0) as i32;
    let size = (2 * diff + 1) as usize;
    let mut kernel = vec![0.0f32; size];

    let mut sum = 0.0f32;
    for i in -diff..=diff {
        let weight = (scaler * (i * i) as f32).exp();
        kernel[(i + diff) as usize] = weight;
        sum += weight;
    }

    // Normalize for interior pixels (border pixels re-normalize)
    let inv_sum = 1.0 / sum;
    for k in &mut kernel {
        *k *= inv_sum;
    }

    kernel
}

/// Performs horizontal convolution on a single row with clamp boundary handling.
///
/// This matches C++ butteraugli: clamp to edge values and re-normalize
/// the kernel weights for border pixels.
fn convolve_row(input: &[f32], kernel: &[f32], output: &mut [f32]) {
    let width = input.len();
    if width == 0 {
        return;
    }
    let half = kernel.len() / 2;

    for x in 0..width {
        // Determine the valid range of kernel elements
        let minx = if x < half { 0 } else { x - half };
        let maxx = (x + half).min(width - 1);

        // For border pixels, re-normalize based on which kernel elements are used
        let mut weight_sum = 0.0f32;
        let mut sum = 0.0f32;

        for j in minx..=maxx {
            let k_idx = j + half - x;
            let k_val = kernel[k_idx];
            weight_sum += k_val;
            sum += input[j] * k_val;
        }

        // Re-normalize for border pixels (C++ approach)
        output[x] = if weight_sum > 0.0 {
            sum / weight_sum
        } else {
            0.0
        };
    }
}

/// Performs vertical convolution on a single column with clamp boundary handling.
///
/// This matches C++ butteraugli: clamp to edge values and re-normalize
/// the kernel weights for border pixels.
fn convolve_column(image: &ImageF, x: usize, kernel: &[f32], output: &mut [f32]) {
    let height = image.height();
    if height == 0 {
        return;
    }
    let half = kernel.len() / 2;

    for y in 0..height {
        // Determine the valid range of kernel elements
        let miny = if y < half { 0 } else { y - half };
        let maxy = (y + half).min(height - 1);

        // For border pixels, re-normalize based on which kernel elements are used
        let mut weight_sum = 0.0f32;
        let mut sum = 0.0f32;

        for j in miny..=maxy {
            let k_idx = j + half - y;
            let k_val = kernel[k_idx];
            weight_sum += k_val;
            sum += image.get(x, j) * k_val;
        }

        // Re-normalize for border pixels (C++ approach)
        output[y] = if weight_sum > 0.0 {
            sum / weight_sum
        } else {
            0.0
        };
    }
}

/// Applies a 2D Gaussian blur to an image.
///
/// This is implemented as two separable 1D convolutions (horizontal + vertical).
///
/// # Arguments
/// * `input` - Input image
/// * `sigma` - Standard deviation of the Gaussian
///
/// # Returns
/// Blurred image
pub fn gaussian_blur(input: &ImageF, sigma: f32) -> ImageF {
    if sigma <= 0.0 {
        return input.clone();
    }

    let kernel = compute_kernel(sigma);
    let width = input.width();
    let height = input.height();

    // Temporary buffer for horizontal pass
    let mut temp = ImageF::new(width, height);

    // Horizontal pass
    for y in 0..height {
        let row_in = input.row(y);
        let row_out = temp.row_mut(y);
        convolve_row(row_in, &kernel, row_out);
    }

    // Output buffer
    let mut output = ImageF::new(width, height);

    // Vertical pass
    let mut col_buffer = vec![0.0f32; height];
    for x in 0..width {
        convolve_column(&temp, x, &kernel, &mut col_buffer);
        for (y, &val) in col_buffer.iter().enumerate() {
            output.set(x, y, val);
        }
    }

    output
}

/// Applies blur in-place (modifies the input image).
pub fn gaussian_blur_inplace(image: &mut ImageF, sigma: f32) {
    if sigma <= 0.0 {
        return;
    }

    let blurred = gaussian_blur(image, sigma);
    image.copy_from(&blurred);
}

/// Fast blur for small sigma values (optimized 5x5 kernel).
///
/// This is faster than the general blur for sigma ~= 1.0.
/// Uses clamp-and-renormalize boundary handling like the general blur.
pub fn blur_5x5(input: &ImageF, weights: &[f32; 3]) -> ImageF {
    let width = input.width();
    let height = input.height();
    let mut output = ImageF::new(width, height);

    // Separable 5x5 kernel: [w2, w1, w0, w1, w2]
    let w0 = weights[0];
    let w1 = weights[1];
    let w2 = weights[2];
    let kernel = [w2, w1, w0, w1, w2];

    // Temporary for horizontal pass
    let mut temp = ImageF::new(width, height);

    // Horizontal pass with clamp-and-renormalize
    for y in 0..height {
        let row = input.row(y);
        let out = temp.row_mut(y);
        for x in 0..width {
            let minx = x.saturating_sub(2);
            let maxx = (x + 2).min(width - 1);

            let mut sum = 0.0f32;
            let mut weight_sum = 0.0f32;
            for j in minx..=maxx {
                let k_idx = j + 2 - x;
                let k_val = kernel[k_idx];
                weight_sum += k_val;
                sum += row[j] * k_val;
            }
            out[x] = if weight_sum > 0.0 {
                sum / weight_sum
            } else {
                0.0
            };
        }
    }

    // Vertical pass with clamp-and-renormalize
    for y in 0..height {
        let out = output.row_mut(y);
        for x in 0..width {
            let miny = y.saturating_sub(2);
            let maxy = (y + 2).min(height - 1);

            let mut sum = 0.0f32;
            let mut weight_sum = 0.0f32;
            for j in miny..=maxy {
                let k_idx = j + 2 - y;
                let k_val = kernel[k_idx];
                weight_sum += k_val;
                sum += temp.get(x, j) * k_val;
            }
            out[x] = if weight_sum > 0.0 {
                sum / weight_sum
            } else {
                0.0
            };
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_generation() {
        let kernel = compute_kernel(1.0);
        assert!(!kernel.is_empty());
        assert_eq!(kernel.len() % 2, 1); // Should be odd

        // Center should be maximum
        let center = kernel.len() / 2;
        for (i, &v) in kernel.iter().enumerate() {
            if i != center {
                assert!(v <= kernel[center]);
            }
        }

        // Should sum to ~1.0
        let sum: f32 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_blur_constant_image() {
        // Blurring a constant image should give the same constant
        let img = ImageF::filled(32, 32, 0.5);
        let blurred = gaussian_blur(&img, 2.0);

        for y in 2..30 {
            for x in 2..30 {
                assert!(
                    (blurred.get(x, y) - 0.5).abs() < 0.01,
                    "Expected 0.5, got {} at ({}, {})",
                    blurred.get(x, y),
                    x,
                    y
                );
            }
        }
    }

    #[test]
    fn test_blur_reduces_delta() {
        // A single bright pixel should spread out
        let mut img = ImageF::new(32, 32);
        img.set(16, 16, 1.0);

        let blurred = gaussian_blur(&img, 2.0);

        // Center should be lower
        assert!(blurred.get(16, 16) < 1.0);
        // Neighbors should be non-zero
        assert!(blurred.get(15, 16) > 0.0);
        assert!(blurred.get(17, 16) > 0.0);
    }
}
