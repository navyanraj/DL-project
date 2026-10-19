"""
Wavelet-based image enhancement for chest X-rays.

Applies Discrete Wavelet Transform (DWT), boosts detail sub-bands (LH, HL, HH)
to amplify subtle disease cues, then reconstructs via inverse DWT.

This is the NOVEL LAYER in the pipeline — rarely combined with YOLO in medical imaging.
"""

import os
import numpy as np
import cv2
import pywt
from tqdm import tqdm
from pathlib import Path


def wavelet_enhance(
    image: np.ndarray,
    wavelet: str = "db2",
    level: int = 1,
    detail_boost: float = 1.5,
    hh_boost: float = 1.2,
) -> np.ndarray:
    """
    Enhance a grayscale image using Discrete Wavelet Transform.

    Process:
      1. Decompose image into LL (approx), LH, HL, HH (detail) sub-bands
      2. Boost LH and HL sub-bands by detail_boost factor (edges/textures)
      3. Optionally boost HH sub-band by hh_boost factor (fine details)
      4. Reconstruct image via inverse DWT

    Args:
        image:        Grayscale uint8 image.
        wavelet:      Wavelet family ('db2', 'haar', 'sym2', etc.).
        level:        Decomposition level (1 recommended for medical images).
        detail_boost: Multiplier for LH and HL sub-bands.
        hh_boost:     Multiplier for HH sub-band.

    Returns:
        Enhanced image (uint8).
    """
    # Convert to float for wavelet transform
    img_float = image.astype(np.float64)

    # Single-level 2D DWT
    coeffs = pywt.dwt2(img_float, wavelet)
    LL, (LH, HL, HH) = coeffs

    # Boost detail sub-bands
    LH_boosted = LH * detail_boost
    HL_boosted = HL * detail_boost
    HH_boosted = HH * hh_boost

    # Reconstruct via inverse DWT
    enhanced = pywt.idwt2((LL, (LH_boosted, HL_boosted, HH_boosted)), wavelet)

    # The reconstructed image may be slightly larger due to DWT padding.
    # Crop back to original size.
    enhanced = enhanced[: image.shape[0], : image.shape[1]]

    # Clip and convert back to uint8
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    return enhanced


def multi_level_enhance(
    image: np.ndarray,
    wavelet: str = "db2",
    level: int = 2,
    detail_boost: float = 1.5,
    hh_boost: float = 1.2,
) -> np.ndarray:
    """
    Multi-level wavelet enhancement.
    
    Decomposes at multiple levels, boosts detail at each level,
    then reconstructs bottom-up.

    Args:
        image:        Grayscale uint8 image.
        wavelet:      Wavelet family.
        level:        Number of decomposition levels.
        detail_boost: Multiplier for LH, HL sub-bands at each level.
        hh_boost:     Multiplier for HH sub-band at each level.

    Returns:
        Enhanced image (uint8).
    """
    img_float = image.astype(np.float64)

    # Multi-level decomposition
    coeffs = pywt.wavedec2(img_float, wavelet, level=level)

    # coeffs = [cA_N, (cH_N, cV_N, cD_N), ..., (cH_1, cV_1, cD_1)]
    # Boost detail coefficients at each level
    for i in range(1, len(coeffs)):
        LH, HL, HH = coeffs[i]
        coeffs[i] = (LH * detail_boost, HL * detail_boost, HH * hh_boost)

    # Reconstruct
    enhanced = pywt.waverec2(coeffs, wavelet)
    enhanced = enhanced[: image.shape[0], : image.shape[1]]
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    return enhanced


def batch_wavelet_enhance(
    input_dir: str,
    output_dir: str,
    wavelet: str = "db2",
    level: int = 1,
    detail_boost: float = 1.5,
    hh_boost: float = 1.2,
    limit: int = None,
) -> dict:
    """
    Apply wavelet enhancement to all PNG images in a directory.

    Returns:
        Stats dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    input_path = Path(input_dir)
    png_files = sorted(input_path.glob("*.png"))

    if limit is not None:
        png_files = png_files[:limit]

    stats = {"total": len(png_files), "processed": 0, "failed": 0}

    for png_file in tqdm(png_files, desc="Wavelet enhancement"):
        try:
            img = cv2.imread(str(png_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not read {png_file.name}")

            if level == 1:
                enhanced = wavelet_enhance(img, wavelet, level, detail_boost, hh_boost)
            else:
                enhanced = multi_level_enhance(img, wavelet, level, detail_boost, hh_boost)

            cv2.imwrite(os.path.join(output_dir, png_file.name), enhanced)
            stats["processed"] += 1
        except Exception as e:
            stats["failed"] += 1
            print(f"Failed: {png_file.name} — {e}")

    print(f"Wavelet enhancement done: {stats['processed']}/{stats['total']} processed.")
    return stats


def visualize_comparison(
    original: np.ndarray,
    enhanced: np.ndarray,
    save_path: str = None,
) -> None:
    """
    Show side-by-side comparison of original vs wavelet-enhanced image.
    Optionally save to file.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(enhanced, cmap="gray")
    axes[1].set_title("Wavelet Enhanced")
    axes[1].axis("off")

    # Difference map
    diff = cv2.absdiff(enhanced, original)
    axes[2].imshow(diff, cmap="hot")
    axes[2].set_title("Enhancement Difference")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparison saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wavelet enhancement for chest X-rays")
    parser.add_argument("--input", required=True, help="Input directory with PNGs")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--wavelet", default="db2", help="Wavelet family")
    parser.add_argument("--level", type=int, default=1, help="Decomposition level")
    parser.add_argument("--detail_boost", type=float, default=1.5, help="LH/HL boost factor")
    parser.add_argument("--hh_boost", type=float, default=1.2, help="HH boost factor")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    batch_wavelet_enhance(
        args.input, args.output, args.wavelet, args.level,
        args.detail_boost, args.hh_boost, args.limit,
    )
