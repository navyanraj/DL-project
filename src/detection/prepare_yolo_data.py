"""
Prepare VinBigData annotations for YOLO training.

Reads train.csv, applies consensus strategy (union/majority_vote),
converts bounding boxes to YOLO format, and creates train/val splits
with the required folder structure.

Consensus strategies:
  - union:         Keep all boxes from all radiologists (noisy, max recall)
  - majority_vote: Keep boxes where >=2 radiologists agree (IoU > threshold)
  - single_rad:    Use the radiologist with the most annotations
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from sklearn.model_selection import train_test_split


# VinBigData class names (class_id 0-13, 14 = No finding)
CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity",
    "Nodule/Mass", "Other lesion", "Pleural effusion", "Pleural thickening",
    "Pneumothorax", "Pulmonary fibrosis",
]


def compute_iou(box1: dict, box2: dict) -> float:
    """Compute IoU between two boxes (x_min, y_min, x_max, y_max)."""
    x1 = max(box1["x_min"], box2["x_min"])
    y1 = max(box1["y_min"], box2["y_min"])
    x2 = min(box1["x_max"], box2["x_max"])
    y2 = min(box1["y_max"], box2["y_max"])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1["x_max"] - box1["x_min"]) * (box1["y_max"] - box1["y_min"])
    area2 = (box2["x_max"] - box2["x_min"]) * (box2["y_max"] - box2["y_min"])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def majority_vote_consensus(
    boxes_by_rad: dict, iou_threshold: float = 0.5, min_agreement: int = 2
) -> list:
    """
    Apply majority vote: keep boxes where >= min_agreement radiologists agree.

    Groups boxes of the same class across radiologists. Two boxes "agree" if they
    have IoU > iou_threshold. Agreed boxes are averaged to produce a consensus box.

    Args:
        boxes_by_rad: Dict of rad_id → list of box dicts.
        iou_threshold: Min IoU to consider two boxes as agreeing.
        min_agreement: Min number of radiologists that must agree.

    Returns:
        List of consensus box dicts.
    """
    # Collect all boxes grouped by class_id, tagged with rad_id
    class_groups = defaultdict(list)
    for rad_id, boxes in boxes_by_rad.items():
        for box in boxes:
            class_groups[box["class_id"]].append({**box, "rad_id": rad_id})

    consensus_boxes = []

    for class_id, boxes in class_groups.items():
        # Group overlapping boxes from different radiologists
        used = [False] * len(boxes)

        for i in range(len(boxes)):
            if used[i]:
                continue
            cluster = [boxes[i]]
            used[i] = True
            rads_in_cluster = {boxes[i]["rad_id"]}

            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                # Must be from a DIFFERENT radiologist
                if boxes[j]["rad_id"] in rads_in_cluster:
                    continue
                # Check IoU with any box in the cluster
                if any(compute_iou(boxes[j], c) >= iou_threshold for c in cluster):
                    cluster.append(boxes[j])
                    rads_in_cluster.add(boxes[j]["rad_id"])
                    used[j] = True

            # Only keep if enough radiologists agree
            if len(rads_in_cluster) >= min_agreement:
                # Average the agreed boxes
                avg_box = {
                    "class_id": class_id,
                    "x_min": float(np.mean([b["x_min"] for b in cluster])),
                    "y_min": float(np.mean([b["y_min"] for b in cluster])),
                    "x_max": float(np.mean([b["x_max"] for b in cluster])),
                    "y_max": float(np.mean([b["y_max"] for b in cluster])),
                }
                consensus_boxes.append(avg_box)

    return consensus_boxes


def load_annotations(
    csv_path: str, strategy: str = "majority_vote", iou_threshold: float = 0.5
) -> dict:
    """
    Load and process VinBigData annotations.

    Args:
        csv_path:       Path to train.csv.
        strategy:       'union', 'majority_vote', or 'single_rad'.
        iou_threshold:  For majority_vote: min IoU to consider agreement.

    Returns:
        Dict mapping image_id → list of box dicts, set of no-finding IDs.
    """
    df = pd.read_csv(csv_path)

    # Filter out "No finding" (class_id == 14)
    df_lesions = df[df["class_id"] != 14].copy()

    annotations = defaultdict(list)

    if strategy == "union":
        # Keep all annotations from all radiologists
        for _, row in df_lesions.iterrows():
            annotations[row["image_id"]].append({
                "class_id": int(row["class_id"]),
                "x_min": float(row["x_min"]),
                "y_min": float(row["y_min"]),
                "x_max": float(row["x_max"]),
                "y_max": float(row["y_max"]),
            })

    elif strategy == "majority_vote":
        # Group by image, then by radiologist
        for image_id, img_df in df_lesions.groupby("image_id"):
            boxes_by_rad = defaultdict(list)
            for _, row in img_df.iterrows():
                boxes_by_rad[row["rad_id"]].append({
                    "class_id": int(row["class_id"]),
                    "x_min": float(row["x_min"]),
                    "y_min": float(row["y_min"]),
                    "x_max": float(row["x_max"]),
                    "y_max": float(row["y_max"]),
                })

            num_rads = len(boxes_by_rad)
            if num_rads < 2:
                # Only 1 radiologist annotated — keep all (no vote possible)
                for boxes in boxes_by_rad.values():
                    annotations[image_id].extend(boxes)
            else:
                consensus = majority_vote_consensus(
                    boxes_by_rad, iou_threshold=iou_threshold, min_agreement=2
                )
                if consensus:
                    annotations[image_id] = consensus

    elif strategy == "single_rad":
        top_rad = df_lesions["rad_id"].value_counts().index[0]
        df_lesions = df_lesions[df_lesions["rad_id"] == top_rad]
        for _, row in df_lesions.iterrows():
            annotations[row["image_id"]].append({
                "class_id": int(row["class_id"]),
                "x_min": float(row["x_min"]),
                "y_min": float(row["y_min"]),
                "x_max": float(row["x_max"]),
                "y_max": float(row["y_max"]),
            })
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'union', 'majority_vote', or 'single_rad'.")

    # Track images with no findings
    all_image_ids = df["image_id"].unique()
    no_finding_ids = set(all_image_ids) - set(annotations.keys())

    print(f"Consensus strategy: {strategy}")
    print(f"Total images: {len(all_image_ids)}")
    print(f"Images with lesions: {len(annotations)}")
    print(f"Images with no findings: {len(no_finding_ids)}")
    print(f"Total bounding boxes: {sum(len(v) for v in annotations.values())}")

    return annotations, no_finding_ids


def convert_to_yolo_format(
    annotations: dict,
    image_width: int,
    image_height: int,
) -> dict:
    """
    Convert absolute bounding boxes to YOLO normalized format.

    YOLO format: class_id cx cy w h (all normalized to [0, 1])

    Args:
        annotations:  Dict of image_id → list of bbox dicts.
        image_width:  Original image width (for normalization).
        image_height: Original image height (for normalization).

    Returns:
        Dict mapping image_id → list of YOLO-format strings.
    """
    yolo_labels = {}

    for image_id, boxes in annotations.items():
        lines = []
        for box in boxes:
            x_min = box["x_min"]
            y_min = box["y_min"]
            x_max = box["x_max"]
            y_max = box["y_max"]

            # Convert to center-x, center-y, width, height
            cx = (x_min + x_max) / 2.0 / image_width
            cy = (y_min + y_max) / 2.0 / image_height
            w = (x_max - x_min) / image_width
            h = (y_max - y_min) / image_height

            # Clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            lines.append(f"{box['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        yolo_labels[image_id] = lines

    return yolo_labels


def get_image_dimensions(dicom_dir: str, image_id: str) -> tuple:
    """
    Get the original dimensions of a DICOM image.
    Falls back to (1024, 1024) if file not found — typical VinBigData size.
    """
    try:
        import pydicom
        dcm_path = os.path.join(dicom_dir, f"{image_id}.dicom")
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        return int(ds.Rows), int(ds.Columns)  # height, width
    except Exception:
        return 1024, 1024  # VinBigData default resolution


def prepare_yolo_dataset(
    csv_path: str,
    image_dir: str,
    output_dir: str,
    dicom_dir: str = None,
    strategy: str = "majority_vote",
    iou_threshold: float = 0.5,
    val_ratio: float = 0.2,
    seed: int = 42,
    include_negatives: bool = True,
) -> dict:
    """
    Full pipeline: parse annotations → convert → split → write YOLO dataset.

    Args:
        csv_path:           Path to train.csv.
        image_dir:          Directory with processed PNG images.
        output_dir:         Root output directory for YOLO dataset.
        dicom_dir:          Directory with original DICOMs (for reading dimensions).
        strategy:           Annotation consensus strategy.
        val_ratio:          Validation split ratio.
        seed:               Random seed.
        include_negatives:  Whether to include images with no findings (empty label files).

    Returns:
        Stats dict.
    """
    # Load and process annotations
    annotations, no_finding_ids = load_annotations(csv_path, strategy, iou_threshold)

    # Determine image dimensions (use DICOM if available, else default 1024x1024)
    # VinBigData images are consistently ~2000-3000px, but annotations are in pixel coords
    # We need original dimensions for normalization
    sample_id = list(annotations.keys())[0]
    if dicom_dir:
        orig_h, orig_w = get_image_dimensions(dicom_dir, sample_id)
    else:
        orig_h, orig_w = 1024, 1024
    print(f"Using image dimensions: {orig_w} x {orig_h}")

    # For VinBigData, each image may have different dimensions.
    # We read per-image dimensions if DICOM dir is provided.
    yolo_labels = {}
    for image_id, boxes in annotations.items():
        if dicom_dir:
            h, w = get_image_dimensions(dicom_dir, image_id)
        else:
            h, w = orig_h, orig_w

        lines = []
        for box in boxes:
            cx = (box["x_min"] + box["x_max"]) / 2.0 / w
            cy = (box["y_min"] + box["y_max"]) / 2.0 / h
            bw = (box["x_max"] - box["x_min"]) / w
            bh = (box["y_max"] - box["y_min"]) / h
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            bw = max(0.0, min(1.0, bw))
            bh = max(0.0, min(1.0, bh))
            lines.append(f"{box['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        yolo_labels[image_id] = lines

    # Build list of all image IDs
    all_ids = list(annotations.keys())
    if include_negatives:
        all_ids += list(no_finding_ids)

    # Stratified split — use primary class of each image for stratification
    image_classes = []
    for img_id in all_ids:
        if img_id in annotations:
            # Use the most frequent class in the image
            class_counts = defaultdict(int)
            for box in annotations[img_id]:
                class_counts[box["class_id"]] += 1
            image_classes.append(max(class_counts, key=class_counts.get))
        else:
            image_classes.append(14)  # No finding

    train_ids, val_ids = train_test_split(
        all_ids, test_size=val_ratio, random_state=seed, stratify=image_classes
    )

    # Create output directories
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    # Write label files and symlink/copy images
    stats = {"train": 0, "val": 0, "train_boxes": 0, "val_boxes": 0}

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        for image_id in ids:
            # Write label file
            label_path = os.path.join(output_dir, "labels", split, f"{image_id}.txt")
            if image_id in yolo_labels:
                with open(label_path, "w") as f:
                    f.write("\n".join(yolo_labels[image_id]))
                stats[f"{split}_boxes"] += len(yolo_labels[image_id])
            else:
                # Empty label file for images with no findings
                open(label_path, "w").close()

            # Copy/symlink image
            src_img = os.path.join(image_dir, f"{image_id}.png")
            dst_img = os.path.join(output_dir, "images", split, f"{image_id}.png")
            if os.path.exists(src_img) and not os.path.exists(dst_img):
                try:
                    os.symlink(os.path.abspath(src_img), dst_img)
                except OSError:
                    # Symlinks might not work on Windows/Kaggle; fall back to copy
                    import shutil
                    shutil.copy2(src_img, dst_img)

            stats[split] += 1

    # Write dataset.yaml
    yaml_content = f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\nYOLO dataset prepared:")
    print(f"  Train: {stats['train']} images, {stats['train_boxes']} boxes")
    print(f"  Val:   {stats['val']} images, {stats['val_boxes']} boxes")
    print(f"  Config: {yaml_path}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from VinBigData")
    parser.add_argument("--csv", required=True, help="Path to train.csv")
    parser.add_argument("--images", required=True, help="Processed images directory")
    parser.add_argument("--output", required=True, help="YOLO dataset output directory")
    parser.add_argument("--dicom_dir", default=None, help="Original DICOM directory")
    parser.add_argument("--strategy", default="majority_vote", choices=["union", "majority_vote", "single_rad"])
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_yolo_dataset(
        args.csv, args.images, args.output, args.dicom_dir,
        args.strategy, args.val_ratio, args.seed,
    )
