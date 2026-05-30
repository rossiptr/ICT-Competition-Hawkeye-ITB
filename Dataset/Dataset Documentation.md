# Dataset

## Overview

The DermaLens dataset contains images across **14 skin disease classes**, used for training, validation, and testing of the ensemble model.

| # | Class |
|---|---|
| 1 | Basal Cell Carcinoma |
| 2 | Dermatofibroma |
| 3 | Eczema |
| 4 | Healthy Skin |
| 5 | Herpes |
| 6 | Impetigo |
| 7 | Keratosis |
| 8 | Leprosy |
| 9 | Melanocytic Nevi |
| 10 | Melanoma |
| 11 | Psoriasis |
| 12 | Scabies |
| 13 | Tinea |
| 14 | Warts |

---

## Source & Access

The dataset is managed on **Roboflow** and publicly accessible:

- **Roboflow:** https://universe.roboflow.com/wr1-hkft0/dermalens_test-2

---

## Storage

The dataset is uploaded to Roboflow first, then mirrored to **Huawei Object Storage Service (OBS)** for use during training on ModelArts.

| Storage | Location |
|---|---|
| Roboflow | https://universe.roboflow.com/wr1-hkft0/dermalens_test-2 |
| Huawei OBS | `obs://dermalens/dermalens_dataset` (CN-Hong Kong region) |

---

## Structure

After downloading or copying from OBS, the dataset should follow this structure:

```
dermalens_data_v1/
├── train/
│   ├── Basal Cell Carcinoma/
│   ├── Dermatofibroma/
│   ├── Eczema/
│   ├── Healthy Skin/
│   ├── Herpes/
│   ├── Impetigo/
│   ├── Keratosis/
│   ├── Leprosy/
│   ├── Melanocytic Nevi/
│   ├── Melanoma/
│   ├── Psoriasis/
│   ├── Scabies/
│   ├── Tinea/
│   └── Warts/
├── valid/
│   └── (same 14 subfolders)
└── test/
    └── (same 14 subfolders)
```

Each subfolder name is the class label and is used directly by `mindspore.dataset.ImageFolderDataset`.

---

## Loading on ModelArts

The training notebook copies the dataset from OBS to local storage automatically:

```python
import moxing as mox

OBS_DATA_PATH   = 'obs://dermalens/dermalens_dataset'
LOCAL_DATA_PATH = '/home/ma-user/work/dermalens_data_v1'

if not os.path.exists(LOCAL_DATA_PATH):
    mox.file.copy_parallel(OBS_DATA_PATH, LOCAL_DATA_PATH)
```

> `moxing` is only available inside Huawei ModelArts. For local runs, download the dataset from Roboflow and update `LOCAL_DATA_PATH` in the notebook accordingly.
