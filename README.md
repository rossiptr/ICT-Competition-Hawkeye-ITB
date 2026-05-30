# ICT-Competition-Hawkeye-ITB — Reproducibility Guide

## Overview

This repository contains the implementation of **DermaLens**, an AI-based skin disease detection system designed for edge deployment using Huawei technologies.

DermaLens uses a **ResNet50 + MobileNetV2 ensemble** trained on 14 skin disease classes, designed to run on Huawei Ascend NPUs (ModelArts) and exported for edge deployment on Orange Pi.

This README focuses on **how to reproduce the model training and inference pipeline**.

---

## Repository Structure

```
ICT-Competition-Hawkeye-ITB/
├── Dataset/
├── Model Training and Implementation/
│   └── *.ipynb
├── Saved Model/
│   └── dermalens_ensemble.onnx
├── Training Logs/
├── requirements.txt
└── README.md
```

---

## Dataset

The dataset contains **14 skin disease classes** and is hosted externally:

- Roboflow: https://universe.roboflow.com/wr1-hkft0/dermalens_test-2
- Huawei OBS: `obs://dermalens/dermalens_dataset` (CN-Hong Kong region)

Expected local structure after download:
```
dermalens_data_v1/
├── train/
├── valid/
└── test/
```

---

## Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/rossiptr/ICT-Competition-Hawkeye-ITB.git
cd ICT-Competition-Hawkeye-ITB
```

### 2. Hardware & Platform

Training was conducted on **Huawei ModelArts** with an **Ascend NPU** (`device_target="Ascend"`). Full reproduction requires this environment.

For local experimentation (CPU/GPU), change the context in the notebook:
```python
# For CPU/GPU (local)
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")  # or "GPU"

# Original Ascend setting
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=0)
```

### 3. Install Dependencies

Recommended: **Python 3.8+**

```bash
pip install -r requirements.txt
```

Key dependencies:

| Package | Purpose |
|---|---|
| `mindspore` | Core ML framework (Ascend NPU) |
| `mindcv` | Pretrained ResNet50 & MobileNetV2 backbones |
| `numpy` | Numerical operations |
| `matplotlib` | Training visualization |
| `seaborn` | Confusion matrix plots |
| `scikit-learn` | Evaluation metrics (accuracy, F1, confusion matrix) |
| `moxing-framework` | Huawei OBS file transfer (ModelArts only) |

> **Note:** `moxing-framework` is pre-installed on Huawei ModelArts and is not available on PyPI. Remove OBS copy steps if running outside of Huawei Cloud and handle data transfer manually.

---

## Download Model Weights

This repository uses **Git LFS** for large files.

```bash
git lfs pull
```

Model location:
```
Saved Model/dermalens_ensemble.onnx
```

---

## Reproducing Model Training

### Training Configuration

| Parameter | Value |
|---|---|
| Batch size | 32 |
| Num classes | 14 |
| Learning rate | 0.001 (piecewise decay) |
| Epochs | 20 |
| Image size | 224 × 224 |
| Optimizer | Adam |
| Loss | SoftmaxCrossEntropyWithLogits (sparse) |

Learning rate schedule:
- Epochs 1–7: `0.001`
- Epochs 8–14: `0.0001`
- Epochs 15–20: `0.00001`

### Model Architecture

The ensemble combines two pretrained backbones from `mindcv`:

- **ResNet50** — feature output: `[B, 2048]`
- **MobileNetV2 (width=1.0)** — feature output: `[B, 1280]`

Both are concatenated and passed through a fusion classifier:
```
Dense(3328 → 512) → ReLU → Dropout(p=0.3) → Dense(512 → 14)
```

### Steps

1. Navigate to the notebook directory:
```
Model Training and Implementation/
```

2. Open the training notebook:
```bash
jupyter notebook
```

3. If running on ModelArts, the dataset will be copied automatically from OBS. Otherwise, update the data path:
```python
LOCAL_DATA_PATH = '/your/local/path/dermalens_data_v1'
```

4. Run cells sequentially: data pipeline → model definition → training → evaluation.

Checkpoints are saved every epoch to:
```
/home/ma-user/work/checkpoints/
```

---

## Evaluation

After training, the evaluation notebook produces:

- **Top-1, Top-3, Top-5 accuracy**
- **Per-class precision, recall, F1** (via `classification_report`)
- **Confusion matrix** (raw counts + normalized) saved as `confusion_matrix.png`
- **Per-class accuracy bar chart** saved as `per_class_accuracy.png`

---

## Model Export

### 1. Export to MindIR / ONNX

The export notebook handles checkpoint loading and ONNX export in two stages:

```python
# Stage 1: Load checkpoint in PYNATIVE_MODE
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
ms.load_checkpoint(CKPT_PATH, net=network)

# Stage 2: Switch to GRAPH_MODE for export
ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
ms.export(network, dummy_input, file_name='dermalens_ensemble', file_format='ONNX')
```

> **Note:** GRAPH_MODE is required for ONNX export. PYNATIVE_MODE is required for checkpoint loading. Both steps must be run in sequence within the same session.

### 2. Convert to RKNN (Edge Deployment)

After exporting the ONNX model, convert to RKNN format on an x86 PC:
```bash
python convert_to_rknn.py
```
Then deploy to the Orange Pi target device.

---

## Running Inference (ONNX)

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("Saved Model/dermalens_ensemble.onnx")

# Input: batch of images, normalized with ImageNet mean/std, shape [B, 3, 224, 224]
input_tensor = np.random.rand(1, 3, 224, 224).astype(np.float32)
outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
print(outputs)
```

Input preprocessing (must match training):
```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
# Normalize, then convert HWC → CHW
```

---

## Reproducibility Notes

- Dataset structure must match: `train/`, `valid/`, `test/` subfolders with one folder per class
- Full training reproduction requires Huawei ModelArts + Ascend NPU
- Results may vary slightly depending on hardware, MindSpore version, and random initialization
- `mindcv` model names are version-sensitive: use `"resnet50"` and `"mobilenet_v2_100"` exactly

---

## Limitations

- Edge deployment (Orange Pi + RKNN) is not fully reproducible in this repository — hardware-specific setup required
- `moxing` OBS integration is only available inside Huawei ModelArts environments
- Mixed precision (`amp_level="O1"`) is broken in Jupyter notebooks due to `inspect.getsource()` limitations; training uses `amp_level="O0"` instead
