"""
GAT training script.

Loads graph .pt files, trains GAT model with early stopping,
class-weighted loss for imbalanced data, and tracks metrics
(loss, accuracy, F1, AUC-ROC).
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report
from pathlib import Path
from tqdm import tqdm

from src.graph.gat_model import LungGAT, build_model


def load_graph_dataset(
    graphs_dir: str,
    label_mapping: dict = None,
) -> list:
    """
    Load graph .pt files into a list of PyG Data objects.

    Args:
        graphs_dir:    Directory containing .pt graph files.
        label_mapping: Optional dict mapping image_id → disease label (int).
                       If None, uses the most frequent class_id in the graph.

    Returns:
        List of Data objects with `y` (label) attribute set.
    """
    graph_path = Path(graphs_dir)
    pt_files = sorted(graph_path.glob("*.pt"))

    dataset = []
    for pt_file in tqdm(pt_files, desc="Loading graphs"):
        graph = torch.load(pt_file, weights_only=False)

        # Assign label
        if label_mapping and hasattr(graph, "image_id") and graph.image_id in label_mapping:
            label = label_mapping[graph.image_id]
        elif hasattr(graph, "class_ids") and len(graph.class_ids) > 0:
            # Use most frequent class in detections as graph label
            from collections import Counter
            label = Counter(graph.class_ids).most_common(1)[0][0]
        else:
            continue  # Skip graphs without labels

        graph.y = torch.tensor([label], dtype=torch.long)
        dataset.append(graph)

    print(f"Loaded {len(dataset)} graphs.")
    return dataset


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits, _ = model(batch)
        loss = criterion(logits, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch.y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=15):
    """Evaluate on a dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch)
        loss = criterion(logits, batch.y)

        total_loss += loss.item() * batch.num_graphs
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(batch.y.cpu().numpy())
        all_probs.extend(probs)

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    # AUC-ROC (one-vs-rest, if enough classes present)
    auc = 0.0
    try:
        all_probs = np.array(all_probs)
        all_labels_arr = np.array(all_labels)
        unique_labels = np.unique(all_labels_arr)
        if len(unique_labels) > 1:
            # Pad probabilities to num_classes if needed
            if all_probs.shape[1] < num_classes:
                padded = np.zeros((all_probs.shape[0], num_classes))
                padded[:, :all_probs.shape[1]] = all_probs
                all_probs = padded
            # Only compute AUC for classes present in labels
            auc = roc_auc_score(
                all_labels_arr, all_probs, multi_class="ovr",
                average="weighted", labels=unique_labels,
            )
    except Exception:
        pass

    return avg_loss, acc, f1, auc, all_preds, all_labels


def train_gat(
    graphs_dir: str,
    output_dir: str,
    config: dict = None,
    label_mapping: dict = None,
) -> dict:
    """
    Full GAT training pipeline.

    Args:
        graphs_dir:    Directory with graph .pt files.
        output_dir:    Directory to save weights and results.
        config:        GAT config dict (from default.yaml).
        label_mapping: Optional image_id → label mapping.

    Returns:
        Dict with training history and best metrics.
    """
    if config is None:
        config = {
            "input_dim": 22, "hidden_dim": 64, "num_heads": 8,
            "num_layers": 2, "num_classes": 15, "dropout": 0.3,
            "learning_rate": 0.001, "weight_decay": 0.0005,
            "epochs": 100, "batch_size": 32, "patience": 15,
        }

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = load_graph_dataset(graphs_dir, label_mapping)

    if len(dataset) == 0:
        print("ERROR: No graphs loaded. Check graphs_dir and labels.")
        return {}

    # Train/val split (80/20)
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42,
        stratify=[d.y.item() for d in dataset],
    )
    train_dataset = [dataset[i] for i in train_idx]
    val_dataset = [dataset[i] for i in val_idx]

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Model
    model = build_model(config).to(device)

    # Compute class weights from training set (inverse frequency)
    from collections import Counter
    train_labels = [d.y.item() for d in train_dataset]
    label_counts = Counter(train_labels)
    num_samples = len(train_labels)
    num_present_classes = len(label_counts)
    class_weights = torch.ones(config["num_classes"], dtype=torch.float32)
    for cls_id, count in label_counts.items():
        if cls_id < config["num_classes"]:
            class_weights[cls_id] = num_samples / (num_present_classes * count)
    class_weights = class_weights.to(device)
    print(f"Class weights: {dict(sorted(label_counts.items()))}")

    # Optimizer, scheduler, loss
    optimizer = Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    best_val_f1 = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
               "train_f1": [], "val_f1": [], "val_auc": []}

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_f1, val_auc, _, _ = evaluate(
            model, val_loader, criterion, device, config["num_classes"]
        )

        scheduler.step(val_loss)

        # Log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_auc)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} AUC: {val_auc:.4f}"
            )

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_gat.pt"))
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final evaluation with best model
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_gat.pt"), weights_only=True))
    val_loss, val_acc, val_f1, val_auc, preds, labels = evaluate(
        model, val_loader, criterion, device, config["num_classes"]
    )

    print(f"\nBest model results:")
    print(f"  Val Accuracy: {val_acc:.4f}")
    print(f"  Val F1:       {val_f1:.4f}")
    print(f"  Val AUC-ROC:  {val_auc:.4f}")

    # Save results
    results = {
        "best_val_f1": best_val_f1,
        "best_val_acc": val_acc,
        "best_val_auc": val_auc,
        "epochs_trained": len(history["train_loss"]),
        "history": history,
    }

    with open(os.path.join(output_dir, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save training curves
    save_training_curves(history, os.path.join(output_dir, "training_curves.png"))

    return results


def save_training_curves(history: dict, save_path: str):
    """Save training/validation curves as a figure."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    # F1 Score
    axes[2].plot(history["train_f1"], label="Train F1")
    axes[2].plot(history["val_f1"], label="Val F1")
    axes[2].plot(history["val_auc"], label="Val AUC", linestyle="--")
    axes[2].set_title("F1 & AUC-ROC")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train GAT model")
    parser.add_argument("--graphs", required=True, help="Directory with graph .pt files")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    config = {
        "input_dim": 22, "hidden_dim": 64, "num_heads": 8,
        "num_layers": 2, "num_classes": 15, "dropout": 0.3,
        "learning_rate": args.lr, "weight_decay": 0.0005,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "patience": args.patience,
    }

    train_gat(args.graphs, args.output, config)
