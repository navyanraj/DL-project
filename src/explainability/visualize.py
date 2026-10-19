"""
Visualization module for the Wavelet-YOLO-GAT pipeline.

Generates clinician-friendly visualizations:
  - X-ray with YOLO bounding boxes
  - Lesion interaction graph overlay
  - GAT attention weight heatmap
  - Side-by-side comparison panels
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity",
    "Nodule/Mass", "Other lesion", "Pleural effusion", "Pleural thickening",
    "Pneumothorax", "Pulmonary fibrosis",
]

# Color palette for classes (14 distinct colors)
CLASS_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
    "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
    "#BB8FCE", "#85C1E9", "#F8C471", "#82E0AA",
    "#F1948A", "#AED6F1",
]


def draw_detections(
    image: np.ndarray,
    detections: list,
    show_labels: bool = True,
    show_confidence: bool = True,
) -> np.ndarray:
    """
    Draw YOLO bounding boxes on an X-ray image.

    Args:
        image:           Grayscale or BGR image.
        detections:      List of detection dicts from inference_yolo.
        show_labels:     Show class name text.
        show_confidence: Show confidence score.

    Returns:
        Annotated image (BGR).
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    annotated = image.copy()

    for det in detections:
        cls_id = det["class_id"]
        conf = det["confidence"]
        bbox = det["bbox"]  # [x_min, y_min, x_max, y_max]

        # Convert hex color to BGR
        hex_color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        color = (b, g, r)

        x_min, y_min, x_max, y_max = [int(v) for v in bbox]

        # Draw box
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, 2)

        # Label text
        if show_labels or show_confidence:
            parts = []
            if show_labels and cls_id < len(CLASS_NAMES):
                parts.append(CLASS_NAMES[cls_id])
            if show_confidence:
                parts.append(f"{conf:.2f}")
            label = " | ".join(parts)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(annotated, (x_min, y_min - th - 6), (x_min + tw + 4, y_min), color, -1)
            cv2.putText(annotated, label, (x_min + 2, y_min - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return annotated


def draw_graph_overlay(
    image: np.ndarray,
    detections: list,
    edge_index: np.ndarray,
    attention_weights: np.ndarray = None,
    image_size: int = 640,
) -> np.ndarray:
    """
    Draw graph structure overlaid on the X-ray image.

    Nodes = lesion centers, Edges = spatial relationships.
    Edge thickness proportional to attention weight.

    Args:
        image:              Grayscale or BGR image.
        detections:         List of detection dicts.
        edge_index:         (2, num_edges) array.
        attention_weights:  Optional (num_edges,) array of attention values.
        image_size:         Image dimension for coordinate scaling.

    Returns:
        Image with graph overlay (BGR).
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = image.copy()

    # Get node centers (pixel coords)
    centers = []
    for det in detections:
        cx, cy = det["bbox_norm"][0], det["bbox_norm"][1]
        px, py = int(cx * image_size), int(cy * image_size)
        centers.append((px, py))

    # Draw edges
    if edge_index is not None and edge_index.shape[1] > 0:
        if attention_weights is None:
            attention_weights = np.ones(edge_index.shape[1])

        # Normalize attention for visualization
        attn_norm = attention_weights / max(attention_weights.max(), 1e-6)

        for k in range(edge_index.shape[1]):
            i, j = int(edge_index[0, k]), int(edge_index[1, k])
            if i == j:
                continue  # Skip self-loops

            thickness = max(1, int(attn_norm[k] * 5))
            alpha = max(0.3, float(attn_norm[k]))

            # Color: red (high attention) → blue (low attention)
            r = int(255 * attn_norm[k])
            b = int(255 * (1 - attn_norm[k]))
            color = (b, 50, r)

            cv2.line(overlay, centers[i], centers[j], color, thickness)

    # Draw nodes
    for idx, (px, py) in enumerate(centers):
        cls_id = detections[idx]["class_id"]
        hex_color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        cv2.circle(overlay, (px, py), 8, (b, g, r), -1)
        cv2.circle(overlay, (px, py), 8, (255, 255, 255), 2)

    return overlay


def create_full_visualization(
    image: np.ndarray,
    detections: list,
    edge_index: np.ndarray = None,
    attention_weights: np.ndarray = None,
    prediction: dict = None,
    save_path: str = None,
    image_size: int = 640,
):
    """
    Create a multi-panel visualization combining all outputs.

    Panels: Original | Detections | Graph Overlay | Prediction

    Args:
        image:              Grayscale image.
        detections:         List of detection dicts.
        edge_index:         (2, num_edges) array.
        attention_weights:  (num_edges,) array.
        prediction:         Dict with 'class', 'confidence', 'probabilities'.
        save_path:          Path to save the figure.
        image_size:         Image dimension.
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Panel 1: Original X-ray
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original X-ray", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: YOLO Detections
    det_img = draw_detections(image, detections)
    axes[1].imshow(cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"YOLO Detections ({len(detections)} found)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Graph Overlay
    if edge_index is not None:
        graph_img = draw_graph_overlay(image, detections, edge_index, attention_weights, image_size)
        axes[2].imshow(cv2.cvtColor(graph_img, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Lesion Interaction Graph", fontsize=12, fontweight="bold")
    else:
        axes[2].imshow(image, cmap="gray")
        axes[2].set_title("Graph (N/A)", fontsize=12)
    axes[2].axis("off")

    # Panel 4: Prediction
    axes[3].axis("off")
    if prediction:
        pred_label = prediction.get('class_name') or prediction.get('class', 'Unknown')
        pred_text = f"Prediction: {pred_label}\n"
        pred_text += f"Confidence: {prediction.get('confidence', 0):.1%}\n\n"

        if "probabilities" in prediction:
            pred_text += "Top Probabilities:\n"
            probs = prediction["probabilities"]
            sorted_probs = sorted(probs.items(), key=lambda x: -x[1])[:5]
            for name, prob in sorted_probs:
                bar = "█" * int(prob * 20)
                pred_text += f"  {name}: {prob:.1%} {bar}\n"

        axes[3].text(
            0.1, 0.5, pred_text,
            transform=axes[3].transAxes,
            fontsize=10, fontfamily="monospace",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="gray"),
        )
        axes[3].set_title("GAT Prediction", fontsize=12, fontweight="bold")
    else:
        axes[3].set_title("Prediction (N/A)", fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")
    plt.close()


def visualize_graph_networkx(
    detections: list,
    edge_index: np.ndarray,
    attention_weights: np.ndarray = None,
    save_path: str = None,
):
    """
    Visualize the lesion graph using NetworkX.

    Nodes are colored by class, edges weighted by attention.
    """
    G = nx.Graph()

    # Add nodes
    for i, det in enumerate(detections):
        cls_name = CLASS_NAMES[det["class_id"]] if det["class_id"] < len(CLASS_NAMES) else "Unknown"
        G.add_node(i, label=f"{cls_name}\n({det['confidence']:.2f})")

    # Add edges
    if edge_index is not None and edge_index.shape[1] > 0:
        for k in range(edge_index.shape[1]):
            i, j = int(edge_index[0, k]), int(edge_index[1, k])
            if i < j:  # Avoid duplicates
                weight = attention_weights[k] if attention_weights is not None else 1.0
                G.add_edge(i, j, weight=float(weight))

    fig, ax = plt.subplots(figsize=(10, 8))

    pos = nx.spring_layout(G, seed=42)
    node_colors = [CLASS_COLORS[detections[n]["class_id"] % len(CLASS_COLORS)] for n in G.nodes()]
    node_labels = nx.get_node_attributes(G, "label")

    # Draw edges
    if G.edges():
        edge_weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
        max_w = max(edge_weights) if edge_weights else 1.0
        edge_widths = [3 * w / max_w for w in edge_weights]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color="gray", ax=ax)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, edgecolors="black", linewidths=2, ax=ax)
    nx.draw_networkx_labels(G, pos, node_labels, font_size=7, ax=ax)

    ax.set_title("Lesion Interaction Graph", fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Graph visualization saved to {save_path}")
    plt.close()
