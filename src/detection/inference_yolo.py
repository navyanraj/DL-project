"""
YOLO inference and detection export.

Runs trained YOLO model on images and exports detections as JSON files
for downstream graph construction.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm


CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity",
    "Nodule/Mass", "Other lesion", "Pleural effusion", "Pleural thickening",
    "Pneumothorax", "Pulmonary fibrosis",
]


def run_inference(
    weights_path: str,
    image_dir: str,
    output_dir: str,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    image_size: int = 640,
    device: str = "0",
    save_annotated: bool = True,
    limit: int = None,
) -> dict:
    """
    Run YOLO inference on a directory of images.

    For each image, saves a JSON file with detections:
    {
        "image_id": "xxx",
        "detections": [
            {
                "class_id": int,
                "class_name": str,
                "confidence": float,
                "bbox": [x_min, y_min, x_max, y_max],  # pixel coords
                "bbox_norm": [cx, cy, w, h],  # normalized [0, 1]
            },
            ...
        ]
    }

    Returns:
        Stats dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    if save_annotated:
        os.makedirs(os.path.join(output_dir, "annotated"), exist_ok=True)

    model = YOLO(weights_path)
    image_path = Path(image_dir)
    image_files = sorted(image_path.glob("*.png"))

    if limit is not None:
        image_files = image_files[:limit]

    stats = {
        "total_images": len(image_files),
        "images_with_detections": 0,
        "total_detections": 0,
        "class_counts": {name: 0 for name in CLASS_NAMES},
    }

    for img_file in tqdm(image_files, desc="YOLO Inference"):
        image_id = img_file.stem

        # Run inference
        results = model.predict(
            str(img_file),
            conf=confidence_threshold,
            iou=iou_threshold,
            imgsz=image_size,
            device=device,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes

        detections = []
        if len(boxes) > 0:
            stats["images_with_detections"] += 1

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy()  # x_min, y_min, x_max, y_max
                xywhn = boxes.xywhn[i].cpu().numpy()  # cx, cy, w, h (normalized)

                det = {
                    "class_id": cls_id,
                    "class_name": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "Unknown",
                    "confidence": round(conf, 4),
                    "bbox": [round(float(v), 2) for v in xyxy],
                    "bbox_norm": [round(float(v), 6) for v in xywhn],
                }
                detections.append(det)
                stats["total_detections"] += 1
                if cls_id < len(CLASS_NAMES):
                    stats["class_counts"][CLASS_NAMES[cls_id]] += 1

        # Save detection JSON
        det_output = {
            "image_id": image_id,
            "num_detections": len(detections),
            "detections": detections,
        }
        json_path = os.path.join(output_dir, f"{image_id}.json")
        with open(json_path, "w") as f:
            json.dump(det_output, f, indent=2)

        # Save annotated image
        if save_annotated and len(detections) > 0:
            annotated = result.plot()
            cv2.imwrite(
                os.path.join(output_dir, "annotated", f"{image_id}.png"),
                annotated,
            )

    print(f"\nInference complete:")
    print(f"  Images processed: {stats['total_images']}")
    print(f"  Images with detections: {stats['images_with_detections']}")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"\n  Class distribution:")
    for name, count in sorted(stats["class_counts"].items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {name}: {count}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run YOLO inference")
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--images", required=True, help="Image directory")
    parser.add_argument("--output", required=True, help="Output directory for JSONs")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_inference(args.weights, args.images, args.output, args.conf, args.iou, device=args.device, limit=args.limit)
