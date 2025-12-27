// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file or at https://developers.google.com/open-source/licenses/bsd

// C wrapper for butteraugli - enables FFI bindings for Rust testing

#ifndef LIB_EXTRAS_BUTTERAUGLI_C_H_
#define LIB_EXTRAS_BUTTERAUGLI_C_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error codes
typedef int butteraugli_error_t;
#define BUTTERAUGLI_OK 0
#define BUTTERAUGLI_ERROR_MEMORY 1
#define BUTTERAUGLI_ERROR_INVALID_INPUT 2
#define BUTTERAUGLI_ERROR_INTERNAL 3

// Compute butteraugli score between two linear RGB images.
// Both images must be linear RGB (not sRGB) with values in [0, 1].
// Data layout: row-major, 3 channels interleaved (RGBRGBRGB...).
butteraugli_error_t butteraugli_compare(
    const float* rgb0,
    const float* rgb1,
    size_t width,
    size_t height,
    float intensity_target,
    double* out_score);

// Compute butteraugli score with full parameters.
butteraugli_error_t butteraugli_compare_full(
    const float* rgb0,
    const float* rgb1,
    size_t width,
    size_t height,
    float hf_asymmetry,
    float xmul,
    float intensity_target,
    double* out_score,
    float* out_diffmap);

// Convert sRGB u8 to linear RGB for butteraugli input.
void butteraugli_srgb_to_linear(
    const uint8_t* srgb,
    size_t width,
    size_t height,
    float* out_linear);

// Gamma function used in butteraugli
float butteraugli_gamma(float v);

// Fast log2 approximation
float butteraugli_fast_log2f(float x);

// Opsin dynamics (XYB conversion)
butteraugli_error_t butteraugli_opsin_dynamics(
    const float* linear_rgb,
    size_t width,
    size_t height,
    float intensity_target,
    float* out_xyb);

// Gaussian blur on a single plane
butteraugli_error_t butteraugli_blur(
    const float* input,
    size_t width,
    size_t height,
    float sigma,
    float* out_blurred);

// Separate frequencies
butteraugli_error_t butteraugli_separate_frequencies(
    const float* xyb,
    size_t width,
    size_t height,
    float intensity_target,
    float* out_lf_x,
    float* out_lf_y,
    float* out_lf_b,
    float* out_mf_x,
    float* out_mf_y,
    float* out_mf_b,
    float* out_hf_x,
    float* out_hf_y,
    float* out_uhf_x,
    float* out_uhf_y);

// Malta filter
butteraugli_error_t butteraugli_malta(
    const float* input,
    size_t width,
    size_t height,
    int use_lf,
    float* out_malta);

// Compute mask from HF/UHF channels
butteraugli_error_t butteraugli_compute_mask(
    const float* hf_x,
    const float* hf_y,
    const float* uhf_x,
    const float* uhf_y,
    size_t width,
    size_t height,
    float* out_mask);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIB_EXTRAS_BUTTERAUGLI_C_H_
