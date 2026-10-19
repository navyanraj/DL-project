"""
DICOM to PNG converter for VinBigData Chest X-ray dataset.

Reads DICOM files, applies windowing/normalization, resizes to target
dimensions, and saves as PNG images ready for downstream processing.
"""

import os
import numpy as np
import cv2
import pydicom
from tqdm import tqdm
from pathlib import Path


def read_dicom(dicom_path: str) -> np.ndarray:
    """
    Read a DICOM file and return the pixel array as a normalized uint8 image.
    
    Handles:
      - PhotometricInterpretation (MONOCHROME1 inversion)
      - Window center/width if present
      - Fallback to min-max normalization
    """
    ds = pydicom.dcmread(dicom_path)
    pixel_array = ds.pixel_array.astype(np.float32)

    # Invert MONOCHROME1 (white = air, black = tissue — opposite of standard)
    if hasattr(ds, "PhotometricInterpretation"):
        if ds.PhotometricInterpretation == "MONOCHROME1":
            pixel_array = pixel_array.max() - pixel_array

    # Apply windowing if available
    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        wc = ds.WindowCenter
        ww = ds.WindowWidth
        # Handle multi-value window
        if isinstance(wc, pydicom.multival.MultiValue):
            wc = float(wc[0])
            ww = float(ww[0])
        else:
            wc = float(wc)
            ww = float(ww)

        lower = wc - ww / 2
        upper = wc + ww / 2
        pixel_array = np.clip(pixel_array, lower, upper)

    # Normalize to [0, 255]
    pmin, pmax = pixel_array.min(), pixel_array.max()
    if pmax - pmin > 0:
        pixel_array = (pixel_array - pmin) / (pmax - pmin) * 255.0
    else:
        pixel_array = np.zeros_like(pixel_array)

    return pixel_array.astype(np.uint8)


def convert_dicom_to_png(
    input_dir: str,
    output_dir: str,
    image_size: int = 640,
    limit: int = None,
) -> dict:
    """
    Batch convert DICOM files to resized PNG images.

    Args:
        input_dir:  Directory containing .dicom files.
        output_dir: Directory to save .png files.
        image_size: Target size (square). Default 640.
        limit:      Process only first N files (for testing). None = all.

    Returns:
        Dict with keys: total, converted, failed, failed_files.
    """
    os.makedirs(output_dir, exist_ok=True)
    input_path = Path(input_dir)
    dicom_files = sorted(list(input_path.glob("*.dicom")) + list(input_path.glob("*.dcm")))

    if limit is not None:
        dicom_files = dicom_files[:limit]

    stats = {"total": len(dicom_files), "converted": 0, "failed": 0, "failed_files": []}

    for dcm_file in tqdm(dicom_files, desc="Converting DICOM → PNG"):
        try:
            img = read_dicom(str(dcm_file))
            img_resized = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
            out_name = dcm_file.stem + ".png"
            cv2.imwrite(os.path.join(output_dir, out_name), img_resized)
            stats["converted"] += 1
        except Exception as e:
            stats["failed"] += 1
            stats["failed_files"].append((dcm_file.name, str(e)))

    print(f"Done: {stats['converted']}/{stats['total']} converted, {stats['failed']} failed.")
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert DICOM to PNG")
    parser.add_argument("--input", required=True, help="Input directory with DICOM files")
    parser.add_argument("--output", required=True, help="Output directory for PNGs")
    parser.add_argument("--size", type=int, default=640, help="Target image size")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files")
    args = parser.parse_args()

    convert_dicom_to_png(args.input, args.output, args.size, args.limit)
