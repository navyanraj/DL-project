# TECHSPEC — Wavelet-Enhanced YOLO + GAT for Early Lung Disease Detection

> **Read this at the start of every session. This is the single source of truth for the project.**

---

## 1. Project Title

**Early Lung Disease Detection from Chest X-ray using Wavelet-Enhanced YOLO with Graph Neural Network Reasoning**

---

## 2. One-Line Summary

YOLO detects lung lesions on wavelet-enhanced X-rays; a Graph Attention Network reasons over their spatial/pathological relationships for early, explainable disease diagnosis.

---

## 3. Pipeline Overview

```
Chest X-ray (DICOM)
      ↓
[Module 1] Preprocessing (resize, normalize, CLAHE)
      ↓
[Module 2] Wavelet Enhancement (DWT sub-band fusion)
      ↓
[Module 3] YOLO Lesion Detection (bounding boxes + class + confidence)
      ↓
[Module 4] Graph Construction (lesions → nodes, spatial relations → edges)
      ↓
[Module 5] GAT Reasoning (inter-lesion attention, graph-level classification)
      ↓
[Module 6] Prediction + Explainability (disease label, early-stage score, attention maps)
```

---

## 4. Dataset

### Primary: VinBigData Chest X-ray Abnormalities Detection (Kaggle)

| Field | Value |
|-------|-------|
| Images | 18,000 (15k train / 3k test) |
| Format | DICOM |
| Annotations | Bounding boxes (`x_min, y_min, x_max, y_max`) |
| Classes | 14 abnormalities + "No finding" |
| Annotators | Multiple radiologists (`rad_id` per annotation) |

**14 Detection Classes:**
`Aortic enlargement`, `Atelectasis`, `Calcification`, `Cardiomegaly`, `Consolidation`, `ILD`, `Infiltration`, `Lung Opacity`, `Nodule/Mass`, `Other lesion`, `Pleural effusion`, `Pleural thickening`, `Pneumothorax`, `Pulmonary fibrosis`

**Annotation note:** Multiple radiologists annotated the same images — consensus strategy needed during preprocessing (majority vote or union).

---

## 5. Module Details

### Module 1: Preprocessing

- **Input:** Raw DICOM chest X-rays
- **Steps:**
  1. DICOM → PNG/numpy conversion
  2. Resize to `640×640` (YOLO input size)
  3. Normalize pixel values to `[0, 1]`
  4. Apply CLAHE for contrast enhancement
  5. Optional: lung-field segmentation mask (U-Net or thresholding)
- **Output:** Enhanced grayscale image (640×640)

### Module 2: Wavelet Enhancement

- **Purpose:** Amplify subtle, early-stage disease cues invisible in spatial domain
- **Method:** Discrete Wavelet Transform (DWT)
  1. Decompose image → `LL` (approx), `LH`, `HL`, `HH` (detail sub-bands)
  2. Enhance selected detail sub-bands (boost `LH`, `HL` for edge/texture features)
  3. Inverse DWT → reconstruct enhanced image
- **Wavelet family:** `db2` or `haar` (to be experimentally decided)
- **Output:** Wavelet-enhanced image (640×640)
- **Novelty:** Wavelet + YOLO combination is rare in medical imaging literature

### Module 3: YOLO Lesion Detection

- **Model:** YOLOv8-nano or YOLOv8-small (ultralytics)
- **Task:** Multi-class object detection
- **Training:** Fine-tune pretrained YOLO on VinBigData with wavelet-enhanced images
- **Output per image:**
  - List of bounding boxes
  - Class label per box (one of 14 abnormalities)
  - Confidence score per box
- **Annotation format:** Convert VinBigData CSV → YOLO txt format (`class cx cy w h`, normalized)

### Module 4: Graph Construction

- **Trigger:** Only for images where YOLO detects ≥ 1 lesion
- **Nodes:** Each detected bounding box = 1 node
- **Node feature vector:**
  - Bounding box center `(cx, cy)`
  - Box dimensions `(w, h)`
  - Area and aspect ratio
  - Mean pixel intensity within box
  - YOLO confidence score
  - Class label (one-hot encoded)
- **Edges:** Connect nodes based on:
  - Euclidean distance < threshold (spatially close lesions)
  - Same lung region (left/right heuristic based on cx)
  - Overlapping or adjacent boxes (IoU > 0)
- **Edge features (optional):** distance, IoU, same-class flag
- **Output:** PyTorch Geometric `Data` object per image
- **Single-lesion case:** Self-loop graph (1 node, 1 edge to itself)

### Module 5: GAT Reasoning

- **Model:** Graph Attention Network (2–3 GAT layers)
- **Input:** Graph per image (from Module 4)
- **Operations:**
  1. Node-level attention-weighted message passing
  2. Graph-level readout (mean/max pooling over nodes)
  3. MLP classifier head → disease class probabilities
- **Output:**
  - Disease classification (14 classes + normal)
  - Early-stage likelihood score
  - Per-node and per-edge attention weights
- **Framework:** PyTorch Geometric

### Module 6: Prediction & Explainability

- **Final outputs:**
  - Disease label (or "Normal")
  - Confidence / probability distribution
  - Early-stage risk score
- **Explainability artifacts:**
  - Original X-ray with YOLO bounding boxes overlaid
  - Graph visualization (nodes = lesions, edges = relationships, edge thickness = attention weight)
  - Attention heatmap highlighting most influential lesions
- **Clinician-facing:** Human-readable summary (e.g., "Clustered opacities in right lung suggest early pneumonia")

---

## 6. Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Deep Learning | PyTorch |
| Object Detection | Ultralytics YOLOv8 |
| Graph Neural Networks | PyTorch Geometric |
| Wavelet Transform | PyWavelets (`pywt`) |
| Image Processing | OpenCV, Pillow |
| DICOM Handling | `pydicom` |
| Data / Metrics | NumPy, Pandas, scikit-learn |
| Visualization | Matplotlib, NetworkX |
| Training Platform | Kaggle Notebooks (free GPU) or local GPU |

---

## 7. Project Structure (Target)

```
CPSD-YOLO_GAT/
├── techspec.md              ← THIS FILE
├── data/
│   ├── raw/                 ← Raw DICOM images (not committed)
│   ├── processed/           ← Preprocessed + wavelet-enhanced PNGs
│   ├── yolo_labels/         ← YOLO-format annotation txts
│   └── graphs/              ← Serialized graph objects
├── src/
│   ├── preprocessing/
│   │   ├── dicom_to_png.py
│   │   ├── clahe.py
│   │   └── wavelet_enhance.py
│   ├── detection/
│   │   ├── prepare_yolo_data.py
│   │   ├── train_yolo.py
│   │   └── inference_yolo.py
│   ├── graph/
│   │   ├── build_graph.py
│   │   ├── gat_model.py
│   │   └── train_gat.py
│   └── explainability/
│       ├── visualize.py
│       └── explain.py
├── configs/
│   └── default.yaml         ← Hyperparameters, paths, thresholds
├── notebooks/
│   └── exploration.ipynb
├── outputs/
│   ├── yolo_weights/
│   ├── gat_weights/
│   └── results/
└── requirements.txt
```

---

## 8. Evaluation Metrics

| Stage | Metrics |
|-------|---------|
| YOLO Detection | mAP@0.5, mAP@0.5:0.95, Precision, Recall |
| GAT Classification | Accuracy, F1-score (macro/weighted), AUC-ROC |
| Explainability | Qualitative — attention map alignment with radiologist annotations |
| Early Detection | Sensitivity on early-stage cases (if subset labeled) |

---

## 9. Key Design Decisions (Log)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Dataset | VinBigData | 14-class bbox annotations, ideal for multi-class YOLO + diverse graph nodes |
| YOLO variant | YOLOv8-nano/small | Lightweight, proven medical imaging performance |
| GNN type | GAT | Attention weights enable explainability |
| Wavelet | DWT (db2/haar) | Simple, effective for edge/texture enhancement |
| Graph edges | Distance + lung-region + IoU | Captures spatial and anatomical relationships |
| Annotation consensus | TBD | Need to decide: majority vote vs union across radiologists |

---

## 10. Risks & Open Questions

1. **Annotation noise:** VinBigData has multi-radiologist disagreements — need a consensus strategy
2. **Few-lesion images:** Many X-rays may have 0–1 detections → graphs are trivial → GAT adds no value on these
3. **Class imbalance:** Some classes (e.g., ILD, Calcification) may have very few bbox samples
4. **"Early-stage" definition:** Dataset doesn't explicitly label early vs advanced — may need proxy criteria
5. **Compute:** YOLO training feasible on Kaggle free GPU; GAT training is lightweight

---

## 11. Milestones

- [ ] **M1:** Data download + preprocessing pipeline (DICOM → enhanced PNG + YOLO labels)
- [ ] **M2:** YOLO training + evaluation on VinBigData
- [ ] **M3:** Graph construction from YOLO outputs
- [ ] **M4:** GAT model training + evaluation
- [ ] **M5:** Explainability visualizations
- [ ] **M6:** End-to-end inference pipeline
- [ ] **M7:** Results documentation + paper-ready figures
