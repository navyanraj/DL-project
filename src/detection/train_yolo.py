"""
YOLO training script for VinBigData chest X-ray lesion detection.

Fine-tunes a pretrained YOLOv8 model on wavelet-enhanced chest X-ray images.
Designed to run on Kaggle with P100/T4 GPU.
"""

from ultralytics import YOLO


def train_yolo(
    dataset_yaml: str,
    model_name: str = "yolov8s.pt",
    epochs: int = 100,
    batch_size: int = 16,
    image_size: int = 640,
    project: str = "outputs",
    name: str = "yolo_training",
    device: str = "0",
) -> str:
    """
    Train YOLOv8 on the prepared dataset.

    Args:
        dataset_yaml: Path to dataset.yaml (from prepare_yolo_data).
        model_name:   Pretrained model to fine-tune.
        epochs:       Number of training epochs.
        batch_size:   Batch size.
        image_size:   Input image size.
        project:      Output project directory.
        name:         Run name (subdirectory under project).
        device:       CUDA device ('0' for GPU, 'cpu' for CPU).

    Returns:
        Path to best weights file.
    """
    # Load pretrained model
    model = YOLO(model_name)

    # Train
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        project=project,
        name=name,
        device=device,
        # Augmentation settings suited for medical images
        hsv_h=0.0,         # No hue shift (grayscale images)
        hsv_s=0.0,         # No saturation shift
        hsv_v=0.2,         # Slight brightness variation
        flipud=0.5,        # Vertical flip (chest X-rays are symmetric-ish)
        fliplr=0.5,        # Horizontal flip
        mosaic=0.5,        # Reduced mosaic (medical images need spatial consistency)
        scale=0.5,         # Scale augmentation
        translate=0.1,     # Translation
        degrees=5.0,       # Slight rotation only
        cos_lr=True,       # Cosine LR schedule (smoother convergence)
        patience=25,       # Early stopping (higher for 100 epochs)
        save=True,
        save_period=10,    # Save checkpoint every 10 epochs
        verbose=True,
    )

    best_weights = f"{project}/{name}/weights/best.pt"
    print(f"\nTraining complete. Best weights: {best_weights}")

    return best_weights


def evaluate_yolo(
    weights_path: str,
    dataset_yaml: str,
    image_size: int = 640,
    device: str = "0",
) -> dict:
    """
    Evaluate trained YOLO model on the validation set.

    Returns:
        Dict with mAP metrics.
    """
    model = YOLO(weights_path)
    metrics = model.val(data=dataset_yaml, imgsz=image_size, device=device)

    results = {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

    print(f"\nEvaluation Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLOv8 on VinBigData")
    parser.add_argument("--dataset", required=True, help="Path to dataset.yaml")
    parser.add_argument("--model", default="yolov8s.pt", help="Pretrained model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    weights = train_yolo(args.dataset, args.model, args.epochs, args.batch, device=args.device)
    evaluate_yolo(weights, args.dataset, device=args.device)
