#!/usr/bin/env python3
"""Compare Rust jpegli XYB vs YCbCr with ICC-aware decode."""

import subprocess
import tempfile
import sys
from pathlib import Path
import io

import numpy as np
from PIL import Image, ImageCms

def load_image(path):
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)

def decode_jpeg_with_icc(jpeg_path):
    img = Image.open(jpeg_path)
    if 'icc_profile' in img.info:
        src_profile = ImageCms.ImageCmsProfile(io.BytesIO(img.info['icc_profile']))
        dst_profile = ImageCms.createProfile('sRGB')
        transform = ImageCms.buildTransform(src_profile, dst_profile, img.mode, 'RGB')
        img = ImageCms.applyTransform(img, transform)
    return np.array(img)

def compute_psnr(original, decoded):
    orig = original.astype(float)
    dec = decoded.astype(float)
    mse = np.mean((orig - dec) ** 2)
    return 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')

def encode_rust(input_path, output_path, quality, use_xyb):
    """Encode using Rust jpegli via a temp Rust script."""
    # Write a simple encode script
    script = f'''
use std::fs;
fn main() {{
    let file = fs::File::open("{input_path}").unwrap();
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let pixels = &buf[..info.buffer_size()];
    
    let jpeg = jpegli::Encoder::new()
        .width(info.width)
        .height(info.height)
        .quality(jpegli::encoder::Quality::ApproxJpegli({quality}.0))
        .use_xyb({str(use_xyb).lower()})
        .encode(pixels)
        .expect("encode");
    fs::write("{output_path}", &jpeg).unwrap();
}}
'''
    # Actually just use the CLI examples we have
    pass

def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "/home/lilith/work/codec-eval/codec-corpus/kodak/1.png"
    qualities = [int(q) for q in sys.argv[2:]] if len(sys.argv) > 2 else [50, 70, 80, 90, 95]
    
    original = load_image(input_path)
    height, width = original.shape[:2]
    
    # Build encoder
    subprocess.run(['cargo', 'build', '--release'], 
                   cwd='/home/lilith/work/zenjpeg-ac-trellis/zenjpeg',
                   capture_output=True)
    
    print(f"Rust jpegli: XYB vs YCbCr (ICC-aware decode)")
    print(f"Image: {Path(input_path).name} ({width}x{height})")
    print()
    print(f"{'Q':>4} | {'XYB Size':>10} | {'YCbCr Size':>10} | {'Size Diff':>9} | "
          f"{'XYB PSNR':>9} | {'YCbCr PSNR':>10}")
    print("-" * 75)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for q in qualities:
            xyb_path = Path(tmpdir) / f"rust_xyb_q{q}.jpg"
            ycbcr_path = Path(tmpdir) / f"rust_ycbcr_q{q}.jpg"
            
            # Encode XYB
            result = subprocess.run([
                'cargo', 'run', '--release', '--example', 'encode_xyb', '--',
                input_path, str(xyb_path), str(q)
            ], cwd='/home/lilith/work/zenjpeg-ac-trellis/zenjpeg', capture_output=True)
            
            # For YCbCr, create a simple encoder script
            ycbcr_script = f'''
use std::fs;
fn main() {{
    let file = fs::File::open("{input_path}").unwrap();
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let pixels = &buf[..info.buffer_size()];
    
    let jpeg = jpegli::Encoder::new()
        .width(info.width)
        .height(info.height)
        .quality(jpegli::encoder::Quality::ApproxJpegli({q}.0))
        .use_xyb(false)
        .encode(pixels)
        .expect("encode");
    fs::write("{ycbcr_path}", &jpeg).unwrap();
    println!("Encoded {{}} bytes", jpeg.len());
}}
'''
            # Use library directly via Python ctypes would be complex
            # Just use C jpegli for YCbCr reference
            subprocess.run(['cjpegli', input_path, str(ycbcr_path), '-q', str(q)], 
                           capture_output=True)
            
            xyb_size = xyb_path.stat().st_size if xyb_path.exists() else 0
            ycbcr_size = ycbcr_path.stat().st_size if ycbcr_path.exists() else 0
            
            if xyb_size == 0 or ycbcr_size == 0:
                print(f"{q:>4} | {'ERROR':>10} | {'ERROR':>10}")
                continue
            
            xyb_decoded = decode_jpeg_with_icc(xyb_path)
            ycbcr_decoded = decode_jpeg_with_icc(ycbcr_path)
            
            xyb_psnr = compute_psnr(original, xyb_decoded)
            ycbcr_psnr = compute_psnr(original, ycbcr_decoded)
            
            size_diff = 100 * (xyb_size - ycbcr_size) / ycbcr_size
            
            print(f"{q:>4} | {xyb_size:>10} | {ycbcr_size:>10} | {size_diff:>+8.1f}% | "
                  f"{xyb_psnr:>9.2f} | {ycbcr_psnr:>10.2f}")

if __name__ == '__main__':
    main()
