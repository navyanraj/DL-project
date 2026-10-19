"""
Graph construction from YOLO detections.

Converts detected lesion bounding boxes into PyTorch Geometric graph objects
where nodes = lesions and edges = spatial/anatomical relationships.
This is the bridge between detection and graph reasoning.
"""

import os
import json
import numpy as np
import cv2
import torch
from torch_geometric.data import Data
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


NUM_CLASSES = 14  # VinBigData classes (0-13)


def compute_iou(box1: list, box2: list) -> float:
    """
    Compute IoU between two boxes in normalized [cx, cy, w, h] format.
    """
    # Convert to x_min, y_min, x_max, y_max
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def compute_node_features(
    detection: dict,
    image: np.ndarray = None,
) -> np.ndarray:
    """
    Compute the feature vector for a single graph node (detected lesion).

    Feature vector (22-dim):
        [cx, cy, w, h, area, aspect_ratio, mean_intensity, confidence, class_onehot(14)]

    Args:
        detection: Dict with 'bbox_norm' [cx, cy, w, h], 'confidence', 'class_id'.
        image:     Optional grayscale image for computing mean intensity within bbox.

    Returns:
        numpy array of shape (22,).
    """
    cx, cy, w, h = detection["bbox_norm"]
    area = w * h
    aspect_ratio = w / max(h, 1e-6)
    confidence = detection["confidence"]
    class_id = detection["class_id"]

    # Compute mean intensity within bounding box (if image provided)
    mean_intensity = 0.0
    if image is not None:
        img_h, img_w = image.shape[:2]
        x_min = int(max(0, (cx - w / 2) * img_w))
        y_min = int(max(0, (cy - h / 2) * img_h))
        x_max = int(min(img_w, (cx + w / 2) * img_w))
        y_max = int(min(img_h, (cy + h / 2) * img_h))
        if x_max > x_min and y_max > y_min:
            roi = image[y_min:y_max, x_min:x_max]
            mean_intensity = float(roi.mean()) / 255.0  # Normalize to [0, 1]

    # One-hot encode class
    class_onehot = np.zeros(NUM_CLASSES, dtype=np.float32)
    if 0 <= class_id < NUM_CLASSES:
        class_onehot[class_id] = 1.0

    # Assemble feature vector
    features = np.array(
        [cx, cy, w, h, area, aspect_ratio, mean_intensity, confidence]
        + list(class_onehot),
        dtype=np.float32,
    )

    return features  # Shape: (22,)


def build_edges(
    detections: list,
    distance_threshold: float = 0.3,
    iou_threshold: float = 0.0,
) -> tuple:
    """
    Build edges between graph nodes based on spatial relationships.

    Edge criteria (any one connects two nodes):
      1. Euclidean distance between centers < distance_threshold
      2. IoU > iou_threshold (overlapping boxes)
      3. Both in same lung half (cx both < 0.5 or both > 0.5)

    Returns:
        edge_index: Tensor of shape (2, num_edges) — bidirectional.
        edge_attr:  Tensor of shape (num_edges, 3) — [distance, iou, same_lung].
    """
    n = len(detections)
    if n <= 1:
        # Single node: self-loop
        edge_index = torch.tensor([[0], [0]], dtype=torch.long) if n == 1 else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((edge_index.shape[1], 3), dtype=torch.float32)
        return edge_index, edge_attr

    sources = []
    targets = []
    attrs = []

    for i in range(n):
        for j in range(i + 1, n):
            box_i = detections[i]["bbox_norm"]
            box_j = detections[j]["bbox_norm"]

            # Euclidean distance between centers
            dist = np.sqrt((box_i[0] - box_j[0]) ** 2 + (box_i[1] - box_j[1]) ** 2)

            # IoU
            iou = compute_iou(box_i, box_j)

            # Same lung half heuristic (based on center x-coordinate)
            same_lung = 1.0 if (box_i[0] < 0.5 and box_j[0] < 0.5) or (box_i[0] >= 0.5 and box_j[0] >= 0.5) else 0.0

            # Connect if spatially close or overlapping (same_lung kept as edge feature only)
            connect = (dist < distance_threshold) or (iou > iou_threshold)

            if connect:
                # Add bidirectional edge
                sources.extend([i, j])
                targets.extend([j, i])
                edge_feat = [dist, iou, same_lung]
                attrs.extend([edge_feat, edge_feat])

    if len(sources) == 0:
        # No edges — add self-loops for all nodes
        sources = list(range(n))
        targets = list(range(n))
        attrs = [[0.0, 0.0, 1.0]] * n

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float32)

    return edge_index, edge_attr


def build_graph_from_detections(
    detection_json: dict,
    image: np.ndarray = None,
    distance_threshold: float = 0.3,
    iou_threshold: float = 0.0,
) -> Data:
    """
    Build a PyTorch Geometric Data object from YOLO detections.

    Args:
        detection_json: Dict from inference_yolo output.
        image:          Optional grayscale image for intensity features.
        distance_threshold: Max distance to connect nodes.
        iou_threshold:  Min IoU to connect nodes.

    Returns:
        PyG Data object with x, edge_index, edge_attr.
    """
    detections = detection_json["detections"]

    if len(detections) == 0:
        return None  # No graph for images with no detections

    # Build node features
    node_features = []
    for det in detections:
        feat = compute_node_features(det, image)
        node_features.append(feat)

    x = torch.tensor(np.array(node_features), dtype=torch.float32)

    # Build edges
    edge_index, edge_attr = build_edges(detections, distance_threshold, iou_threshold)

    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(detections),
    )

    # Store metadata
    data.image_id = detection_json["image_id"]
    data.num_detections = len(detections)
    data.class_ids = [d["class_id"] for d in detections]

    return data


def batch_build_graphs(
    detections_dir: str,
    images_dir: str,
    output_dir: str,
    distance_threshold: float = 0.3,
    iou_threshold: float = 0.0,
    limit: int = None,
) -> dict:
    """
    Build graph objects from all detection JSON files.

    Returns:
        Stats dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    det_path = Path(detections_dir)
    json_files = sorted(det_path.glob("*.json"))

    if limit is not None:
        json_files = json_files[:limit]

    stats = {
        "total": len(json_files),
        "graphs_created": 0,
        "skipped_no_detections": 0,
        "node_counts": [],
        "edge_counts": [],
    }

    for json_file in tqdm(json_files, desc="Building graphs"):
        with open(json_file) as f:
            det_data = json.load(f)

        if det_data["num_detections"] == 0:
            stats["skipped_no_detections"] += 1
            continue

        # Load corresponding image for intensity features
        image_id = det_data["image_id"]
        img_path = os.path.join(images_dir, f"{image_id}.png")
        image = None
        if os.path.exists(img_path):
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Build graph
        graph = build_graph_from_detections(
            det_data, image, distance_threshold, iou_threshold
        )

        if graph is not None:
            # Save graph
            graph_path = os.path.join(output_dir, f"{image_id}.pt")
            torch.save(graph, graph_path)
            stats["graphs_created"] += 1
            stats["node_counts"].append(graph.num_nodes)
            stats["edge_counts"].append(graph.edge_index.shape[1])

    print(f"\nGraph construction complete:")
    print(f"  Graphs created: {stats['graphs_created']}/{stats['total']}")
    print(f"  Skipped (no detections): {stats['skipped_no_detections']}")
    if stats["node_counts"]:
        print(f"  Avg nodes/graph: {np.mean(stats['node_counts']):.1f}")
        print(f"  Avg edges/graph: {np.mean(stats['edge_counts']):.1f}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build graphs from YOLO detections")
    parser.add_argument("--detections", required=True, help="Directory with detection JSONs")
    parser.add_argument("--images", required=True, help="Directory with processed PNGs")
    parser.add_argument("--output", required=True, help="Output directory for graph .pt files")
    parser.add_argument("--distance_threshold", type=float, default=0.3)
    parser.add_argument("--iou_threshold", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    batch_build_graphs(
        args.detections, args.images, args.output,
        args.distance_threshold, args.iou_threshold, args.limit,
    )
