#!/usr/bin/env python3
"""Compare XYB vs YCbCr JPEG encoding with proper ICC profile handling."""

import subprocess
import tempfile
import sys
from pathlib import Path
import io

import numpy as np
from PIL import Image, ImageCms

def load_image(path):
    """Load image as RGB numpy array."""
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)

def decode_jpeg_with_icc(jpeg_path):
    """Decode JPEG and apply ICC profile if present."""
    img = Image.open(jpeg_path)
    
    if 'icc_profile' in img.info:
        src_profile = ImageCms.ImageCmsProfile(io.BytesIO(img.info['icc_profile']))
        dst_profile = ImageCms.createProfile('sRGB')
        transform = ImageCms.buildTransform(src_profile, dst_profile, img.mode, 'RGB')
        img = ImageCms.applyTransform(img, transform)
    
    return np.array(img)

def compute_metrics(original, decoded):
    """Compute quality metrics."""
    orig = original.astype(float)
    dec = decoded.astype(float)
    
    mse = np.mean((orig - dec) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    mad = np.mean(np.abs(orig - dec))
    
    return {'psnr': psnr, 'mad': mad}

def main():
    if len(sys.argv) < 2:
        print("Usage: python xyb_vs_ycbcr_compare.py <input.png> [qualities...]")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    qualities = [int(q) for q in sys.argv[2:]] if len(sys.argv) > 2 else [50, 70, 80, 90]
    
    original = load_image(input_path)
    height, width = original.shape[:2]
    
    print(f"XYB vs YCbCr (ICC-aware decode)")
    print(f"Image: {input_path.name} ({width}x{height})")
    print()
    print(f"{'Q':>4} | {'XYB Size':>10} | {'YCbCr Size':>10} | {'Size Diff':>9} | "
          f"{'XYB PSNR':>9} | {'YCbCr PSNR':>10}")
    print("-" * 75)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for q in qualities:
            xyb_path = Path(tmpdir) / f"xyb_q{q}.jpg"
            ycbcr_path = Path(tmpdir) / f"ycbcr_q{q}.jpg"
            
            subprocess.run(['cjpegli', str(input_path), str(xyb_path), 
                           '-q', str(q), '--xyb'], capture_output=True)
            subprocess.run(['cjpegli', str(input_path), str(ycbcr_path), 
                           '-q', str(q)], capture_output=True)
            
            xyb_size = xyb_path.stat().st_size
            ycbcr_size = ycbcr_path.stat().st_size
            
            xyb_decoded = decode_jpeg_with_icc(xyb_path)
            ycbcr_decoded = decode_jpeg_with_icc(ycbcr_path)
            
            xyb_m = compute_metrics(original, xyb_decoded)
            ycbcr_m = compute_metrics(original, ycbcr_decoded)
            
            size_diff = 100 * (xyb_size - ycbcr_size) / ycbcr_size
            
            print(f"{q:>4} | {xyb_size:>10} | {ycbcr_size:>10} | {size_diff:>+8.1f}% | "
                  f"{xyb_m['psnr']:>9.2f} | {ycbcr_m['psnr']:>10.2f}")

if __name__ == '__main__':
    main()
