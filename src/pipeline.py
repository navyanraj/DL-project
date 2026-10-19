"""
End-to-end inference pipeline.

Given a single chest X-ray image + trained model weights, runs the
full pipeline: preprocess → wavelet → YOLO → graph → GAT → explain.
"""

import os
import json
import numpy as np
import cv2
import torch
from pathlib import Path

from src.preprocessing.clahe import apply_clahe
from src.preprocessing.wavelet_enhance import wavelet_enhance
from src.graph.build_graph import build_graph_from_detections
from src.graph.gat_model import LungGAT, build_model
from src.explainability.visualize import create_full_visualization
from src.explainability.explain import generate_explanation, print_explanation


CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity",
    "Nodule/Mass", "Other lesion", "Pleural effusion", "Pleural thickening",
    "Pneumothorax", "Pulmonary fibrosis", "Normal",
]


def run_pipeline(
    image_path: str,
    yolo_weights: str,
    gat_weights: str,
    output_dir: str = "outputs/pipeline",
    config: dict = None,
) -> dict:
    """
    Run the full Wavelet-YOLO-GAT pipeline on a single image.

    Args:
        image_path:   Path to chest X-ray image (PNG).
        yolo_weights: Path to trained YOLO best.pt.
        gat_weights:  Path to trained GAT best_gat.pt.
        output_dir:   Directory to save outputs.
        config:       Pipeline config dict.

    Returns:
        Result dict with prediction, explanation, and output paths.
    """
    if config is None:
        config = {
            "image_size": 640,
            "clahe_clip_limit": 2.0,
            "wavelet_family": "db2",
            "detail_boost": 1.5,
            "hh_boost": 1.2,
            "yolo_confidence": 0.25,
            "yolo_iou": 0.45,
            "graph_distance_threshold": 0.3,
            "gat_config": {
                "input_dim": 22, "hidden_dim": 64, "num_heads": 8,
                "num_layers": 2, "num_classes": 15, "dropout": 0.3,
            },
        }

    os.makedirs(output_dir, exist_ok=True)
    image_id = Path(image_path).stem
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Processing: {image_id}")
    print(f"Device: {device}")

    # ── Step 1: Load and preprocess ──────────────────────────────
    print("  [1/6] Preprocessing...")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image = cv2.resize(image, (config["image_size"], config["image_size"]))
    image_clahe = apply_clahe(image, clip_limit=config["clahe_clip_limit"])

    # ── Step 2: Wavelet enhancement ──────────────────────────────
    print("  [2/6] Wavelet enhancement...")
    image_enhanced = wavelet_enhance(
        image_clahe,
        wavelet=config["wavelet_family"],
        detail_boost=config["detail_boost"],
        hh_boost=config["hh_boost"],
    )

    # Save enhanced image
    enhanced_path = os.path.join(output_dir, f"{image_id}_enhanced.png")
    cv2.imwrite(enhanced_path, image_enhanced)

    # ── Step 3: YOLO detection ───────────────────────────────────
    print("  [3/6] YOLO detection...")
    from ultralytics import YOLO

    yolo_model = YOLO(yolo_weights)
    results = yolo_model.predict(
        image_enhanced,
        conf=config["yolo_confidence"],
        iou=config["yolo_iou"],
        imgsz=config["image_size"],
        device=str(device),
        verbose=False,
    )

    result = results[0]
    boxes = result.boxes

    detections = []
    if len(boxes) > 0:
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].cpu().numpy()
            xywhn = boxes.xywhn[i].cpu().numpy()

            detections.append({
                "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "Unknown",
                "confidence": round(conf, 4),
                "bbox": [round(float(v), 2) for v in xyxy],
                "bbox_norm": [round(float(v), 6) for v in xywhn],
            })

    print(f"    Found {len(detections)} detections")

    if len(detections) == 0:
        # No detections → predict as Normal
        prediction = {
            "class_id": 14,
            "class_name": "Normal",
            "confidence": 1.0,
            "probabilities": {name: 0.0 for name in CLASS_NAMES},
        }
        prediction["probabilities"]["Normal"] = 1.0

        explanation = generate_explanation(detections, prediction)
        print_explanation(explanation)

        result_dict = {
            "image_id": image_id,
            "prediction": prediction,
            "explanation": explanation,
            "detections": detections,
        }

        # Save
        with open(os.path.join(output_dir, f"{image_id}_result.json"), "w") as f:
            json.dump(result_dict, f, indent=2)

        return result_dict

    # ── Step 4: Graph construction ───────────────────────────────
    print("  [4/6] Graph construction...")
    det_json = {"image_id": image_id, "detections": detections, "num_detections": len(detections)}
    graph = build_graph_from_detections(
        det_json, image_enhanced,
        distance_threshold=config["graph_distance_threshold"],
    )

    # ── Step 5: GAT reasoning ────────────────────────────────────
    print("  [5/6] GAT reasoning...")
    gat_config = config["gat_config"]
    gat_model = build_model(gat_config)
    gat_model.load_state_dict(torch.load(gat_weights, map_location=device, weights_only=True))
    gat_model = gat_model.to(device)
    gat_model.eval()

    graph = graph.to(device)
    # Add batch attribute for single graph
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)

    with torch.no_grad():
        logits, attention_weights = gat_model(graph)

    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_class_id = int(probs.argmax())
    pred_confidence = float(probs.max())

    prediction = {
        "class_id": pred_class_id,
        "class_name": CLASS_NAMES[pred_class_id] if pred_class_id < len(CLASS_NAMES) else "Unknown",
        "confidence": round(pred_confidence, 4),
        "probabilities": {
            CLASS_NAMES[i]: round(float(probs[i]), 4)
            for i in range(len(CLASS_NAMES))
            if i < len(probs)
        },
    }

    # ── Step 6: Explainability ───────────────────────────────────
    print("  [6/6] Generating explanations...")

    explanation = generate_explanation(detections, prediction, attention_weights)
    print_explanation(explanation)

    # Get edge index and attention for visualization
    edge_index_np = graph.edge_index.cpu().numpy()
    attn_np = None
    if attention_weights:
        last_attn = attention_weights[-1]
        if hasattr(last_attn, "cpu"):
            attn_np = last_attn.cpu().numpy()
            if len(attn_np.shape) > 1:
                attn_np = attn_np.mean(axis=1)

    # Create visualization
    viz_path = os.path.join(output_dir, f"{image_id}_visualization.png")
    create_full_visualization(
        image_enhanced, detections, edge_index_np, attn_np,
        prediction, viz_path, config["image_size"],
    )

    # Save result
    result_dict = {
        "image_id": image_id,
        "prediction": prediction,
        "explanation": explanation,
        "detections": detections,
        "num_graph_nodes": graph.num_nodes,
        "num_graph_edges": int(graph.edge_index.shape[1]),
        "output_files": {
            "enhanced_image": enhanced_path,
            "visualization": viz_path,
        },
    }

    with open(os.path.join(output_dir, f"{image_id}_result.json"), "w") as f:
        json.dump(result_dict, f, indent=2)

    print(f"\nPipeline complete for {image_id}")
    return result_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run full Wavelet-YOLO-GAT pipeline")
    parser.add_argument("--image", required=True, help="Path to chest X-ray image")
    parser.add_argument("--yolo_weights", required=True, help="Path to YOLO best.pt")
    parser.add_argument("--gat_weights", required=True, help="Path to GAT best_gat.pt")
    parser.add_argument("--output", default="outputs/pipeline", help="Output directory")
    args = parser.parse_args()

    run_pipeline(args.image, args.yolo_weights, args.gat_weights, args.output)
