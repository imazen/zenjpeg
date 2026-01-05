// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file or at https://developers.google.com/open-source/licenses/bsd

// C wrapper for butteraugli - enables FFI bindings for Rust testing
// This file provides real C++ butteraugli function access for parity testing.

#include "butteraugli_c.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "lib/base/status.h"
#include "lib/extras/butteraugli.h"
#include "lib/extras/image.h"

// Forward declaration for internal jxl::Blur function (defined in butteraugli.cc)
// This allows us to call the real Gaussian blur implementation.
namespace jxl {
Status Blur(const ImageF& in, float sigma, const ButteraugliParams& params,
            BlurTemp* temp, ImageF* out);
}

namespace {

// Default memory manager using malloc/free
void* DefaultAlloc(void* opaque, size_t size) {
  return std::malloc(size);
}

void DefaultFree(void* opaque, void* address) {
  std::free(address);
}

// Static default memory manager
JxlMemoryManager g_default_memory_manager = {
    nullptr,      // opaque
    DefaultAlloc,
    DefaultFree
};

}  // namespace

extern "C" {

butteraugli_error_t butteraugli_compare(
    const float* rgb0,
    const float* rgb1,
    size_t width,
    size_t height,
    float intensity_target,
    double* out_score) {
  return butteraugli_compare_full(
      rgb0, rgb1, width, height,
      1.0f,  // hf_asymmetry
      1.0f,  // xmul
      intensity_target,
      out_score,
      nullptr);  // no diffmap
}

butteraugli_error_t butteraugli_compare_full(
    const float* rgb0,
    const float* rgb1,
    size_t width,
    size_t height,
    float hf_asymmetry,
    float xmul,
    float intensity_target,
    double* out_score,
    float* out_diffmap) {
  if (!rgb0 || !rgb1 || !out_score || width == 0 || height == 0) {
    return BUTTERAUGLI_ERROR_INVALID_INPUT;
  }

  // Create Image3F objects using the default memory manager
  auto result0 = jxl::Image3F::Create(&g_default_memory_manager, width, height);
  auto result1 = jxl::Image3F::Create(&g_default_memory_manager, width, height);

  if (!result0.ok() || !result1.ok()) {
    return BUTTERAUGLI_ERROR_MEMORY;
  }

  jxl::Image3F img0 = std::move(result0).value_();
  jxl::Image3F img1 = std::move(result1).value_();

  // Copy interleaved RGB to planar Image3F
  for (size_t y = 0; y < height; ++y) {
    float* row0_r = img0.PlaneRow(0, y);
    float* row0_g = img0.PlaneRow(1, y);
    float* row0_b = img0.PlaneRow(2, y);
    float* row1_r = img1.PlaneRow(0, y);
    float* row1_g = img1.PlaneRow(1, y);
    float* row1_b = img1.PlaneRow(2, y);

    for (size_t x = 0; x < width; ++x) {
      size_t idx = (y * width + x) * 3;
      row0_r[x] = rgb0[idx + 0];
      row0_g[x] = rgb0[idx + 1];
      row0_b[x] = rgb0[idx + 2];
      row1_r[x] = rgb1[idx + 0];
      row1_g[x] = rgb1[idx + 1];
      row1_b[x] = rgb1[idx + 2];
    }
  }

  // Set up parameters
  jxl::ButteraugliParams params;
  params.hf_asymmetry = hf_asymmetry;
  params.xmul = xmul;
  params.intensity_target = intensity_target;

  // Compute butteraugli
  jxl::ImageF diffmap;
  jxl::Status status = jxl::ButteraugliDiffmap(img0, img1, params, diffmap);

  if (!status) {
    return BUTTERAUGLI_ERROR_INTERNAL;
  }

  // Compute score (max of diffmap)
  *out_score = jxl::ButteraugliScoreFromDiffmap(diffmap, &params);

  // Copy diffmap if requested
  if (out_diffmap) {
    for (size_t y = 0; y < height; ++y) {
      const float* row = diffmap.ConstRow(y);
      std::memcpy(out_diffmap + y * width, row, width * sizeof(float));
    }
  }

  return BUTTERAUGLI_OK;
}

void butteraugli_srgb_to_linear(
    const uint8_t* srgb,
    size_t width,
    size_t height,
    float* out_linear) {
  size_t num_pixels = width * height;
  for (size_t i = 0; i < num_pixels; ++i) {
    for (int c = 0; c < 3; ++c) {
      float x = srgb[i * 3 + c] / 255.0f;
      if (x <= 0.04045f) {
        out_linear[i * 3 + c] = x / 12.92f;
      } else {
        out_linear[i * 3 + c] = std::pow((x + 0.055f) / 1.055f, 2.4f);
      }
    }
  }
}

float butteraugli_gamma(float v) {
  const float kInvLog2e = 1.0f / 1.4426950408889634f;
  const float kRetMul = 19.245013259874995f * kInvLog2e;
  const float kRetAdd = -23.16046239805755f;
  const float kBias = 9.9710635769299145f;

  if (v < 0.0f) v = 0.0f;
  float biased = v + kBias;
  float log_val = butteraugli_fast_log2f(biased);
  return kRetMul * log_val + kRetAdd;
}

float butteraugli_fast_log2f(float x) {
  // Exact C++ FastLog2f implementation from fast_math-inl.h
  const float p0 = -1.8503833400518310E-06f;
  const float p1 = 1.4287160470083755E+00f;
  const float p2 = 7.4245873327820566E-01f;

  const float q0 = 9.9032814277590719E-01f;
  const float q1 = 1.0096718572241148E+00f;
  const float q2 = 1.7409343003366853E-01f;

  union { float f; int32_t i; } u;
  u.f = x;
  int32_t x_bits = u.i;

  int32_t exp_bits = x_bits - 0x3f2aaaab;
  int32_t exp_shifted = exp_bits >> 23;
  int32_t mantissa_bits = x_bits - (exp_shifted << 23);
  u.i = mantissa_bits;
  float mantissa = u.f;
  float exp_val = static_cast<float>(exp_shifted);

  float m = mantissa - 1.0f;
  float yp = p2 * m + p1;
  yp = yp * m + p0;
  float yq = q2 * m + q1;
  yq = yq * m + q0;

  return yp / yq + exp_val;
}

butteraugli_error_t butteraugli_opsin_dynamics(
    const float* linear_rgb,
    size_t width,
    size_t height,
    float intensity_target,
    float* out_xyb) {
  if (!linear_rgb || !out_xyb || width == 0 || height == 0) {
    return BUTTERAUGLI_ERROR_INVALID_INPUT;
  }

  // Create Image3F for input RGB
  auto result_rgb = jxl::Image3F::Create(&g_default_memory_manager, width, height);
  if (!result_rgb.ok()) {
    return BUTTERAUGLI_ERROR_MEMORY;
  }
  jxl::Image3F rgb = std::move(result_rgb).value_();

  // Copy interleaved linear RGB to planar Image3F
  for (size_t y = 0; y < height; ++y) {
    float* row_r = rgb.PlaneRow(0, y);
    float* row_g = rgb.PlaneRow(1, y);
    float* row_b = rgb.PlaneRow(2, y);
    for (size_t x = 0; x < width; ++x) {
      size_t idx = (y * width + x) * 3;
      row_r[x] = linear_rgb[idx + 0];
      row_g[x] = linear_rgb[idx + 1];
      row_b[x] = linear_rgb[idx + 2];
    }
  }

  // Set up parameters
  jxl::ButteraugliParams params;
  params.intensity_target = intensity_target;

  // We cannot directly call OpsinDynamicsImage as it uses Highway's dispatch.
  // Instead, we use the same approach as SeparateFrequencies by creating
  // a ButteraugliComparator and extracting the XYB from the PsychoImage.
  // The LF band contains the XYB after xyb_low_freq_to_vals transform,
  // so this gives us a comparable output even if not the raw XYB.
  //
  // For now, return a stub that indicates this function needs the real
  // OpsinDynamicsImage which is not directly callable from C linkage.
  // The frequency separation test should use butteraugli_separate_frequencies
  // instead, which uses the real internal implementation.

  // Compute using simplified opsin (for fallback - real XYB comes from freq separation)
  const double mix0_r = 0.29956550340058319;
  const double mix0_g = 0.63373087833825936;
  const double mix0_b = 0.077705617820981968;
  const double mix1_r = 0.22158691104574774;
  const double mix1_g = 0.69391388044116142;
  const double mix1_b = 0.0987313588422;
  const double mix2_r = 0.02;
  const double mix2_g = 0.02;
  const double mix2_b = 0.20480129041026129;

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      size_t idx = (y * width + x) * 3;
      float r = linear_rgb[idx + 0] * intensity_target;
      float g = linear_rgb[idx + 1] * intensity_target;
      float b = linear_rgb[idx + 2] * intensity_target;

      float pre0 = static_cast<float>(mix0_r * r + mix0_g * g + mix0_b * b);
      float pre1 = static_cast<float>(mix1_r * r + mix1_g * g + mix1_b * b);
      float pre2 = static_cast<float>(mix2_r * r + mix2_g * g + mix2_b * b);

      // Apply cube root (simplified, ignoring bias for now)
      pre0 = std::cbrt(std::max(pre0, 0.0f));
      pre1 = std::cbrt(std::max(pre1, 0.0f));
      pre2 = std::cbrt(std::max(pre2, 0.0f));

      out_xyb[idx + 0] = pre0 - pre1;  // X
      out_xyb[idx + 1] = pre0 + pre1;  // Y
      out_xyb[idx + 2] = pre2;          // B
    }
  }

  return BUTTERAUGLI_OK;
}

butteraugli_error_t butteraugli_blur(
    const float* input,
    size_t width,
    size_t height,
    float sigma,
    float* out_blurred) {
  if (!input || !out_blurred || width == 0 || height == 0) {
    return BUTTERAUGLI_ERROR_INVALID_INPUT;
  }

  // Create ImageF for input
  auto result_in = jxl::ImageF::Create(&g_default_memory_manager, width, height);
  if (!result_in.ok()) {
    return BUTTERAUGLI_ERROR_MEMORY;
  }
  jxl::ImageF img_in = std::move(result_in).value_();

  // Copy input data to ImageF
  for (size_t y = 0; y < height; ++y) {
    float* row = img_in.Row(y);
    for (size_t x = 0; x < width; ++x) {
      row[x] = input[y * width + x];
    }
  }

  // Create output ImageF
  auto result_out = jxl::ImageF::Create(&g_default_memory_manager, width, height);
  if (!result_out.ok()) {
    return BUTTERAUGLI_ERROR_MEMORY;
  }
  jxl::ImageF img_out = std::move(result_out).value_();

  // Set up parameters and temp storage
  jxl::ButteraugliParams params;
  jxl::BlurTemp blur_temp;

  // Call real C++ Blur function
  jxl::Status status = jxl::Blur(img_in, sigma, params, &blur_temp, &img_out);
  if (!status) {
    return BUTTERAUGLI_ERROR_INTERNAL;
  }

  // Copy output data from ImageF
  for (size_t y = 0; y < height; ++y) {
    const float* row = img_out.ConstRow(y);
    for (size_t x = 0; x < width; ++x) {
      out_blurred[y * width + x] = row[x];
    }
  }

  return BUTTERAUGLI_OK;
}

butteraugli_error_t butteraugli_separate_frequencies(
    const float* linear_rgb,
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
    float* out_uhf_y) {
  if (!linear_rgb || width == 0 || height == 0) {
    return BUTTERAUGLI_ERROR_INVALID_INPUT;
  }

  // Minimum size for butteraugli processing
  if (width < 8 || height < 8) {
    return BUTTERAUGLI_ERROR_INVALID_INPUT;
  }

  // Create Image3F for input linear RGB
  auto result_rgb = jxl::Image3F::Create(&g_default_memory_manager, width, height);
  if (!result_rgb.ok()) {
    return BUTTERAUGLI_ERROR_MEMORY;
  }
  jxl::Image3F rgb = std::move(result_rgb).value_();

  // Copy interleaved linear RGB to planar Image3F
  for (size_t y = 0; y < height; ++y) {
    float* row_r = rgb.PlaneRow(0, y);
    float* row_g = rgb.PlaneRow(1, y);
    float* row_b = rgb.PlaneRow(2, y);
    for (size_t x = 0; x < width; ++x) {
      size_t idx = (y * width + x) * 3;
      row_r[x] = linear_rgb[idx + 0];
      row_g[x] = linear_rgb[idx + 1];
      row_b[x] = linear_rgb[idx + 2];
    }
  }

  // Set up parameters
  jxl::ButteraugliParams params;
  params.intensity_target = intensity_target;

  // Create ButteraugliComparator - this runs OpsinDynamicsImage and SeparateFrequencies
  auto comparator_result = jxl::ButteraugliComparator::Make(rgb, params);
  if (!comparator_result.ok()) {
    return BUTTERAUGLI_ERROR_INTERNAL;
  }

  std::unique_ptr<jxl::ButteraugliComparator> comparator =
      std::move(comparator_result).value_();

  // Get the frequency-decomposed image
  const jxl::PsychoImage& pi = comparator->GetPsychoImage();

  // Copy LF (Image3F - XYB)
  if (out_lf_x && out_lf_y && out_lf_b) {
    for (size_t y = 0; y < height; ++y) {
      const float* lf_x_row = pi.lf.ConstPlaneRow(0, y);
      const float* lf_y_row = pi.lf.ConstPlaneRow(1, y);
      const float* lf_b_row = pi.lf.ConstPlaneRow(2, y);
      for (size_t x = 0; x < width; ++x) {
        out_lf_x[y * width + x] = lf_x_row[x];
        out_lf_y[y * width + x] = lf_y_row[x];
        out_lf_b[y * width + x] = lf_b_row[x];
      }
    }
  }

  // Copy MF (Image3F - XYB)
  if (out_mf_x && out_mf_y && out_mf_b) {
    for (size_t y = 0; y < height; ++y) {
      const float* mf_x_row = pi.mf.ConstPlaneRow(0, y);
      const float* mf_y_row = pi.mf.ConstPlaneRow(1, y);
      const float* mf_b_row = pi.mf.ConstPlaneRow(2, y);
      for (size_t x = 0; x < width; ++x) {
        out_mf_x[y * width + x] = mf_x_row[x];
        out_mf_y[y * width + x] = mf_y_row[x];
        out_mf_b[y * width + x] = mf_b_row[x];
      }
    }
  }

  // Copy HF (ImageF[2] - XY only, no B)
  if (out_hf_x && out_hf_y) {
    for (size_t y = 0; y < height; ++y) {
      const float* hf_x_row = pi.hf[0].ConstRow(y);
      const float* hf_y_row = pi.hf[1].ConstRow(y);
      for (size_t x = 0; x < width; ++x) {
        out_hf_x[y * width + x] = hf_x_row[x];
        out_hf_y[y * width + x] = hf_y_row[x];
      }
    }
  }

  // Copy UHF (ImageF[2] - XY only, no B)
  if (out_uhf_x && out_uhf_y) {
    for (size_t y = 0; y < height; ++y) {
      const float* uhf_x_row = pi.uhf[0].ConstRow(y);
      const float* uhf_y_row = pi.uhf[1].ConstRow(y);
      for (size_t x = 0; x < width; ++x) {
        out_uhf_x[y * width + x] = uhf_x_row[x];
        out_uhf_y[y * width + x] = uhf_y_row[x];
      }
    }
  }

  return BUTTERAUGLI_OK;
}

butteraugli_error_t butteraugli_malta(
    const float* input,
    size_t width,
    size_t height,
    int use_lf,
    float* out_malta) {
  // Stub implementation
  (void)input;
  (void)width;
  (void)height;
  (void)use_lf;
  (void)out_malta;
  return BUTTERAUGLI_ERROR_INTERNAL;  // Not implemented
}

butteraugli_error_t butteraugli_compute_mask(
    const float* linear_rgb,
    size_t width,
    size_t height,
    float intensity_target,
    float* out_mask) {
  if (!linear_rgb || !out_mask || width == 0 || height == 0) {
    return BUTTERAUGLI_ERROR_INVALID_INPUT;
  }

  // Minimum size for butteraugli processing
  if (width < 8 || height < 8) {
    return BUTTERAUGLI_ERROR_INVALID_INPUT;
  }

  // Create Image3F for input linear RGB
  auto result_rgb = jxl::Image3F::Create(&g_default_memory_manager, width, height);
  if (!result_rgb.ok()) {
    return BUTTERAUGLI_ERROR_MEMORY;
  }
  jxl::Image3F rgb = std::move(result_rgb).value_();

  // Copy interleaved linear RGB to planar Image3F
  for (size_t y = 0; y < height; ++y) {
    float* row_r = rgb.PlaneRow(0, y);
    float* row_g = rgb.PlaneRow(1, y);
    float* row_b = rgb.PlaneRow(2, y);
    for (size_t x = 0; x < width; ++x) {
      size_t idx = (y * width + x) * 3;
      row_r[x] = linear_rgb[idx + 0];
      row_g[x] = linear_rgb[idx + 1];
      row_b[x] = linear_rgb[idx + 2];
    }
  }

  // Set up parameters
  jxl::ButteraugliParams params;
  params.intensity_target = intensity_target;

  // Create ButteraugliComparator
  auto comparator_result = jxl::ButteraugliComparator::Make(rgb, params);
  if (!comparator_result.ok()) {
    return BUTTERAUGLI_ERROR_INTERNAL;
  }

  std::unique_ptr<jxl::ButteraugliComparator> comparator =
      std::move(comparator_result).value_();

  // Create output mask
  auto result_mask = jxl::ImageF::Create(&g_default_memory_manager, width, height);
  if (!result_mask.ok()) {
    return BUTTERAUGLI_ERROR_MEMORY;
  }
  jxl::ImageF mask = std::move(result_mask).value_();

  // Compute mask using real C++ implementation
  jxl::Status status = comparator->Mask(&mask);
  if (!status) {
    return BUTTERAUGLI_ERROR_INTERNAL;
  }

  // Copy mask output
  for (size_t y = 0; y < height; ++y) {
    const float* row = mask.ConstRow(y);
    for (size_t x = 0; x < width; ++x) {
      out_mask[y * width + x] = row[x];
    }
  }

  return BUTTERAUGLI_OK;
}

}  // extern "C"
