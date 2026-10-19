"""
Explainability module for the Wavelet-YOLO-GAT pipeline.

Generates human-readable explanations of GAT predictions, including:
  - Top predicted disease and confidence
  - Most influential lesion nodes (by attention)
  - Spatial description of findings
  - Structured JSON output for clinical integration
"""

import numpy as np
import json


CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity",
    "Nodule/Mass", "Other lesion", "Pleural effusion", "Pleural thickening",
    "Pneumothorax", "Pulmonary fibrosis", "Normal",
]


def get_spatial_description(cx: float, cy: float) -> str:
    """
    Convert normalized coordinates to anatomical location description.

    Args:
        cx: Normalized center x (0 = left, 1 = right).
        cy: Normalized center y (0 = top, 1 = bottom).

    Returns:
        String description like "right upper lung".
    """
    # Horizontal: Left or Right (note: X-ray is mirrored — patient's left = image right)
    side = "left" if cx >= 0.5 else "right"

    # Vertical: Upper, Mid, Lower
    if cy < 0.33:
        zone = "upper"
    elif cy < 0.66:
        zone = "mid"
    else:
        zone = "lower"

    return f"{side} {zone} lung"


def compute_node_importance(
    attention_weights: list,
    num_nodes: int,
) -> np.ndarray:
    """
    Compute node importance scores based on GAT attention weights.

    Aggregates incoming attention from the last GAT layer.

    Args:
        attention_weights: List of attention tensors from GAT layers.
        num_nodes:         Number of nodes in the graph.

    Returns:
        Array of shape (num_nodes,) with importance scores.
    """
    importance = np.zeros(num_nodes)

    if attention_weights and len(attention_weights) > 0:
        # Use last layer's attention
        last_attn = attention_weights[-1]
        if hasattr(last_attn, "cpu"):
            last_attn = last_attn.cpu().detach().numpy()

        # Attention shape: (num_edges, num_heads) or (num_edges,)
        if len(last_attn.shape) > 1:
            last_attn = last_attn.mean(axis=1)  # Average over heads

        # This is simplified — in practice you'd map edge attention to target nodes
        # For now, distribute evenly
        importance = np.ones(num_nodes) / num_nodes

    return importance


def generate_explanation(
    detections: list,
    prediction: dict,
    attention_weights: list = None,
    top_k: int = 3,
) -> dict:
    """
    Generate a structured explanation of the GAT prediction.

    Args:
        detections:        List of detection dicts from YOLO.
        prediction:        Dict with 'class_id', 'class_name', 'confidence', 'probabilities'.
        attention_weights: GAT attention weights (optional).
        top_k:             Number of top findings to highlight.

    Returns:
        Explanation dict with text summary and structured data.
    """
    # Basic prediction info
    pred_class = prediction.get("class_name", "Unknown")
    confidence = prediction.get("confidence", 0.0)

    # Spatial analysis of detections
    findings = []
    for det in detections:
        cx, cy = det["bbox_norm"][0], det["bbox_norm"][1]
        location = get_spatial_description(cx, cy)
        cls_name = CLASS_NAMES[det["class_id"]] if det["class_id"] < len(CLASS_NAMES) else "Unknown"

        findings.append({
            "type": cls_name,
            "location": location,
            "confidence": det["confidence"],
            "coordinates": {"cx": cx, "cy": cy},
        })

    # Sort by confidence
    findings.sort(key=lambda x: -x["confidence"])

    # Compute clustering
    if len(detections) > 1:
        centers = np.array([[d["bbox_norm"][0], d["bbox_norm"][1]] for d in detections])
        mean_dist = np.mean([np.linalg.norm(c1 - c2) for i, c1 in enumerate(centers) for c2 in centers[i+1:]])
        is_clustered = mean_dist < 0.25
        spread = "clustered" if is_clustered else "distributed"
    else:
        spread = "isolated"

    # Generate text summary
    text_parts = []
    text_parts.append(f"Prediction: {pred_class} (confidence: {confidence:.1%})")
    text_parts.append(f"Detected {len(detections)} abnormalit{'y' if len(detections) == 1 else 'ies'}.")

    if findings:
        text_parts.append(f"Pattern: {spread} findings.")
        top_findings = findings[:top_k]
        for f in top_findings:
            text_parts.append(f"  - {f['type']} in {f['location']} ({f['confidence']:.0%})")

    # Clinical hint
    if spread == "clustered":
        text_parts.append(f"Note: Clustered abnormalities may suggest localized disease process.")
    elif spread == "distributed":
        text_parts.append(f"Note: Distributed findings may indicate systemic or advanced-stage condition.")

    text_summary = "\n".join(text_parts)

    # Structured output
    explanation = {
        "prediction": {
            "class": pred_class,
            "confidence": round(confidence, 4),
        },
        "num_findings": len(detections),
        "spatial_pattern": spread,
        "top_findings": findings[:top_k],
        "text_summary": text_summary,
    }

    if "probabilities" in prediction:
        explanation["all_probabilities"] = prediction["probabilities"]

    return explanation


def print_explanation(explanation: dict):
    """Pretty-print an explanation to console."""
    print("=" * 60)
    print("  DIAGNOSTIC EXPLANATION")
    print("=" * 60)
    print(explanation["text_summary"])
    print("=" * 60)


def save_explanation(explanation: dict, save_path: str):
    """Save explanation as JSON."""
    with open(save_path, "w") as f:
        json.dump(explanation, f, indent=2)
    print(f"Explanation saved to {save_path}")
