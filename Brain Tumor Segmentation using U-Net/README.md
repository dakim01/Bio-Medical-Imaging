# 🧠 Brain Tumor Segmentation using U-Net

> Automated segmentation of brain tumors from multimodal MRI scans using a U-Net convolutional neural network, trained on the BraTS 2020 dataset.

---

## 📌 Overview

Brain tumor segmentation is a critical step in clinical diagnosis and treatment planning. Manual segmentation by radiologists is time-consuming and subject to variability. This project leverages deep learning — specifically the **U-Net architecture** — to automatically delineate tumor sub-regions from multimodal MRI volumes.

The model identifies and segments three tumor regions:
- **Necrotic / Non-enhancing Tumor Core** (Label 1)
- **Peritumoral Edema** (Label 2)
- **Enhancing Tumor** (Label 4 → remapped to 3)

---

## 📂 Dataset

**BraTS 2020 — Brain Tumor Segmentation Challenge**

| Property | Details |
|---|---|
| Source | [Kaggle: BraTS2020 Dataset](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation) |
| Patients | 369 training cases |
| Modalities | T1, T1ce (contrast-enhanced), T2, T2-FLAIR |
| Format | NIfTI (.nii) — 3D volumes of shape (240, 240, 155) |
| Annotations | Expert-labeled segmentation masks (4 classes) |

Each patient folder contains four MRI modality files and one segmentation mask. All volumes share identical spatial dimensions, allowing direct multi-channel stacking.

**Selected Modalities:** T1ce and FLAIR were chosen as the two input channels — T1ce highlights the enhancing tumor boundary via gadolinium contrast, while FLAIR best reveals peritumoral edema.

---

## 🏗️ Project Structure

```
├── brain-tumor-segmentation-using-u-net.ipynb   # Main notebook
├── model_.19-0.021449.m5                         # Best model weights (epoch 19)
├── my_model.keras                                # Final saved model
├── training.log                                  # Per-epoch training metrics (CSV)
└── README.md
```

---

## ⚙️ Methodology

### 1. Data Preprocessing

- **Normalization:** MinMaxScaler applied per-modality per-patient to bring all pixel values into [0, 1].
- **Slice Selection:** 100 axial slices per patient (from index 22 onward), skipping uninformative boundary slices.
- **Resizing:** Each 2D slice resized from 240×240 to **128×128** pixels.
- **Label Remapping:** Original label 4 (Enhancing Tumor) remapped to 3 for contiguous one-hot encoding.
- **One-Hot Encoding:** Ground truth masks encoded into 4-channel binary maps.

### 2. Data Generator

A custom `DataGenerator` (subclassing `keras.utils.Sequence`) feeds data slice-by-slice to avoid memory overflow when working with 3D NIfTI volumes. Each batch yields:

- **X:** shape `(batch, 128, 128, 2)` — stacked FLAIR and T1ce slices
- **y:** shape `(batch, 128, 128, 4)` — one-hot segmentation mask

### 3. Dataset Split

| Split | Patients | Approximate % |
|---|---|---|
| Training | ~250 | 68% |
| Validation | ~74 | 20% |
| Test | ~45 | 12% |

### 4. U-Net Architecture

The U-Net is an encoder-decoder CNN with skip connections that preserve spatial context lost during downsampling.

```
Input (128×128×2)
    │
    ├─ Encoder
    │   ├─ Conv Block (32 filters) → MaxPool
    │   ├─ Conv Block (64 filters) → MaxPool
    │   ├─ Conv Block (128 filters) → MaxPool
    │   ├─ Conv Block (256 filters) → MaxPool
    │   └─ Bottleneck (512 filters)
    │
    └─ Decoder
        ├─ UpSample + Skip → Conv Block (256 filters)
        ├─ UpSample + Skip → Conv Block (128 filters)
        ├─ UpSample + Skip → Conv Block (64 filters)
        ├─ UpSample + Skip → Conv Block (32 filters)
        └─ Output Conv (4 filters, softmax)
```

- **Activation:** ReLU (hidden), Softmax (output)
- **Kernel Initializer:** He Normal
- **Dropout:** 0.2 (applied in the bottleneck)

### 5. Training Configuration

| Parameter | Value |
|---|---|
| Loss Function | Categorical Cross-Entropy + Dice Loss |
| Optimizer | Adam (lr = 0.001) |
| Epochs | 35 |
| Batch Size | 1 |
| LR Scheduler | ReduceLROnPlateau (factor=0.2, patience=2) |
| Best Epoch | ~19 |

---

## 📊 Evaluation Metrics

Multiple metrics are tracked to assess performance across the highly imbalanced label distribution:

| Metric | Description |
|---|---|
| **Accuracy** | Pixel-wise classification accuracy |
| **Mean IoU** | Average intersection-over-union across 4 classes |
| **Dice Coefficient** | Overlap measure averaged across all 4 classes |
| **Dice (Necrotic)** | Dice for Label 1 — Necrotic Core |
| **Dice (Edema)** | Dice for Label 2 — Peritumoral Edema |
| **Dice (Enhancing)** | Dice for Label 3 — Enhancing Tumor |
| **Sensitivity** | True Positive Rate (recall) |
| **Specificity** | True Negative Rate |
| **Precision** | Positive Predictive Value |

### Test Set Results

| Metric | Score |
|---|---|
| Accuracy | > 99% |
| Precision | > 99% |
| Mean IoU | 0.8426 |
| Dice Coefficient | 0.6480 |
| Sensitivity | 0.9916 |
| Specificity | 0.9978 |

---

## 🔧 How to Run

### Requirements

```bash
pip install tensorflow keras nibabel scikit-image scikit-learn opencv-python matplotlib pandas numpy
```

### Steps

1. **Download the dataset** via Kaggle API:
   ```bash
   kaggle datasets download awsaf49/brats20-dataset-training-validation
   unzip brats20-dataset-training-validation.zip
   ```

2. **Fix the misnamed file** in patient folder `BraTS20_Training_355` (the notebook handles this automatically).

3. **Run the notebook** end-to-end:
   ```
   brain-tumor-segmentation-using-u-net.ipynb
   ```

4. **Inference on new patients:** Use `predictByPath()` or `predict_segmentation()` with a path to a patient folder.

---

## 📈 Training Curves

Training was monitored across 35 epochs for accuracy, loss, Dice coefficient, and Mean IoU. Key observations:

- Both training and validation **accuracy plateau** at high values, indicating good generalization.
- **Loss decreases steadily** with no significant divergence between train and validation curves.
- **Dice coefficient improves consistently** across epochs.
- Best checkpoint saved at **epoch 19** based on minimum validation loss.

---

## 🔍 Qualitative Results

The model produces segmentation predictions for each of the three tumor sub-regions. Visual comparisons of ground truth vs. predicted masks on test patients demonstrate strong spatial alignment, particularly for the enhancing tumor and edema regions.

---

## 📚 References

- Ronneberger, O., Fischer, P., & Brox, T. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). *MICCAI 2015*.
- Menze, B.H. et al. (2015). [The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)](https://ieeexplore.ieee.org/document/6975210). *IEEE Transactions on Medical Imaging*.
- BraTS2020 Dataset on Kaggle by [awsaf49](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation).

---

## 📝 License

This project is released for educational and research purposes. The BraTS2020 dataset is subject to its own usage terms — please refer to the [official challenge page](https://www.med.upenn.edu/cbica/brats2020/) for details.
