"""
CLAHE (Contrast Limited Adaptive Histogram Equalization) enhancement
for chest X-ray images.

Improves visibility of subtle lung structures and early disease cues.
"""

import os
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE to a grayscale image.

    Args:
        image:          Grayscale uint8 image.
        clip_limit:     Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization.

    Returns:
        CLAHE-enhanced image (uint8).
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)
    return enhanced


def batch_clahe(
    input_dir: str,
    output_dir: str,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
    limit: int = None,
) -> dict:
    """
    Apply CLAHE to all PNG images in a directory.

    Args:
        input_dir:      Directory with input PNG images.
        output_dir:     Directory to save enhanced images.
        clip_limit:     CLAHE clip limit.
        tile_grid_size: CLAHE tile grid size.
        limit:          Process only first N files.

    Returns:
        Stats dict with counts.
    """
    os.makedirs(output_dir, exist_ok=True)
    input_path = Path(input_dir)
    png_files = sorted(input_path.glob("*.png"))

    if limit is not None:
        png_files = png_files[:limit]

    stats = {"total": len(png_files), "processed": 0, "failed": 0}

    for png_file in tqdm(png_files, desc="Applying CLAHE"):
        try:
            img = cv2.imread(str(png_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not read {png_file.name}")
            enhanced = apply_clahe(img, clip_limit, tile_grid_size)
            cv2.imwrite(os.path.join(output_dir, png_file.name), enhanced)
            stats["processed"] += 1
        except Exception as e:
            stats["failed"] += 1
            print(f"Failed: {png_file.name} — {e}")

    print(f"CLAHE done: {stats['processed']}/{stats['total']} processed.")
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply CLAHE to images")
    parser.add_argument("--input", required=True, help="Input directory with PNGs")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--clip_limit", type=float, default=2.0)
    parser.add_argument("--tile_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    batch_clahe(args.input, args.output, args.clip_limit, (args.tile_size, args.tile_size), args.limit)
