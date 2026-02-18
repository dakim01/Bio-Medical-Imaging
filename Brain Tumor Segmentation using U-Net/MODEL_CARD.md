# Model Card — Brain Tumor Segmentation U-Net

## Model Overview

| Field | Details |
|---|---|
| **Model Name** | Brain Tumor Segmentation U-Net |
| **Task** | Multi-class medical image segmentation |
| **Architecture** | U-Net (2D, applied slice-by-slice to 3D MRI volumes) |
| **Framework** | TensorFlow / Keras |
| **Input** | 2-channel 2D slices: FLAIR + T1ce, shape (128, 128, 2) |
| **Output** | 4-class softmax segmentation map, shape (128, 128, 4) |
| **Training Dataset** | BraTS 2020 — 369 glioma patients (~250 used for training) |

---

## Intended Use

This model is intended for **research and educational purposes** in the domain of medical image analysis. It demonstrates how deep learning can assist in automating the segmentation of brain tumors from multimodal MRI data.

**This model is NOT intended for clinical use or diagnosis.**

---

## Architecture Details

The U-Net consists of a symmetric encoder-decoder with skip connections:

### Encoder (Contracting Path)
| Block | Filters | Operation |
|---|---|---|
| Block 1 | 32 | Conv2D × 2 → MaxPool |
| Block 2 | 64 | Conv2D × 2 → MaxPool |
| Block 3 | 128 | Conv2D × 2 → MaxPool |
| Block 4 | 256 | Conv2D × 2 → MaxPool |
| Bottleneck | 512 | Conv2D × 2 + Dropout(0.2) |

### Decoder (Expanding Path)
| Block | Filters | Operation |
|---|---|---|
| Block 4 | 256 | UpSampling + Concat + Conv2D × 2 |
| Block 3 | 128 | UpSampling + Concat + Conv2D × 2 |
| Block 2 | 64 | UpSampling + Concat + Conv2D × 2 |
| Block 1 | 32 | UpSampling + Concat + Conv2D × 2 |
| Output | 4 | Conv2D (1×1) + Softmax |

- **Total Parameters:** ~7.8M (approximate)
- **Activation:** ReLU (all hidden layers), Softmax (output)
- **Kernel Initializer:** He Normal

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Loss | Categorical Cross-Entropy |
| Optimizer | Adam |
| Learning Rate | 0.001 (initial) |
| LR Schedule | ReduceLROnPlateau (factor=0.2, patience=2, min=1e-6) |
| Batch Size | 1 |
| Epochs | 35 |
| Best Checkpoint | Epoch 19 (val_loss = 0.021449) |
| Input Size | 128 × 128 × 2 |
| Volume Slices | 100 (starting at slice 22) |

---

## Segmentation Classes

| Label | Class | Description |
|---|---|---|
| 0 | Background / Healthy Tissue | Non-tumor voxels |
| 1 | Necrotic Core | Dead tissue / Non-enhancing tumor core |
| 2 | Edema | Peritumoral swelling |
| 3 | Enhancing Tumor | Actively growing tumor (original label 4) |

---

## Evaluation Results (Test Set)

| Metric | Score |
|---|---|
| Loss (Categorical Cross-Entropy) | 0.0214 |
| Accuracy | > 99% |
| Precision | > 99% |
| Mean IoU | 0.8426 |
| Dice Coefficient (Overall) | 0.6480 |
| Dice — Necrotic Core | (see training.log) |
| Dice — Edema | (see training.log) |
| Dice — Enhancing Tumor | (see training.log) |
| Sensitivity (Recall) | 0.9916 |
| Specificity | 0.9978 |

---

## Limitations

- The model operates on **2D slices** rather than full 3D volumes, which means it does not exploit inter-slice contextual information.
- Trained exclusively on **glioma** cases — performance on other tumor types is unknown.
- Input is limited to **FLAIR and T1ce** modalities; T1 and T2 are excluded.
- Image resolution is downsampled to **128×128**, which may lose fine-grained spatial detail.
- Performance may degrade on data from different scanners or acquisition protocols (domain shift).

---

## How to Load and Use

```python
import tensorflow as tf
from tensorflow.keras.layers import Input
import keras

# Rebuild model architecture
IMG_SIZE = 128
input_layer = Input((IMG_SIZE, IMG_SIZE, 2))
model = build_unet(input_layer, 'he_normal', 0.2)

# Compile with custom metrics
model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=4),
             dice_coef, precision, sensitivity, specificity,
             dice_coef_necrotic, dice_coef_edema, dice_coef_enhancing]
)

# Load best weights
model.load_weights('model_.19-0.021449.m5')

# Or load full saved model
model = keras.models.load_model('my_model.keras', custom_objects={...})
```

---

## Citation

If you use this work, please cite:

```
Ronneberger, O., Fischer, P., & Brox, T. (2015).
U-Net: Convolutional Networks for Biomedical Image Segmentation.
Medical Image Computing and Computer-Assisted Intervention (MICCAI).
https://arxiv.org/abs/1505.04597
```

```
Menze, B.H. et al. (2015).
The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS).
IEEE Transactions on Medical Imaging, 34(10), 1993–2024.
```
