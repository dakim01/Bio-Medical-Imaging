# Diabetic Retinopathy Detection

A deep learning system for automated grading of diabetic retinopathy severity from retinal fundus photographs, using a two-stage transfer learning approach with EfficientNet-B7.

---

## Overview

Diabetic retinopathy (DR) is one of the leading causes of preventable blindness worldwide, affecting over 25% of people living with diabetes. Early detection is critical — but manual screening requires trained ophthalmologists to examine each retinal photograph individually, creating bottlenecks that delay diagnosis, especially in underserved regions.

This project builds a convolutional neural network that automatically classifies the severity of DR from fundus images, assigning each image one of five clinical grades:

| Grade | Severity |
|-------|----------|
| 0 | No diabetic retinopathy |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

---

## Datasets

The project leverages two publicly available Kaggle datasets:

**2015 Diabetic Retinopathy Detection Dataset**
- 35,126 labeled retinal JPEG images
- Used for pre-training the model
- Source: [Kaggle 2015 Competition](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)

**2019 APTOS Blindness Detection Dataset**
- 3,662 labeled retinal PNG images collected from Indian clinical sites
- Primary target dataset used for fine-tuning and evaluation
- Source: [Kaggle APTOS 2019 Competition](https://www.kaggle.com/c/aptos2019-blindness-detection/data)

---

## Methodology

### Image Preprocessing

Raw fundus photographs undergo a multi-step preprocessing pipeline before being fed into the model:

1. **Black border removal** — Fundus cameras produce circular images surrounded by black borders. A pixel intensity threshold masks and crops these uninformative regions.
2. **Resize** — All images are resized to a uniform 256×256 resolution.
3. **Gaussian illumination correction (Ben Graham's method)** — Subtracts a blurred version of the image from the original to remove low-frequency lighting variation, sharpening local contrast and making subtle lesions (microaneurysms, exudates, hemorrhages) more visible.
4. **Circular masking** — A circular crop zeroes out pixels outside the retinal disc, eliminating remaining camera edge artifacts.

### Data Augmentation

During training, the following augmentations are applied to improve model generalization:

- **Random rotation (±360°)** — Retinal images have no canonical orientation; full-range rotation is clinically valid.
- **Random horizontal and vertical flips** — DR lesion patterns are not laterally asymmetric, so flipping does not alter the diagnosis.

No augmentation is applied during validation or inference, ensuring consistent and reproducible evaluation.

### Model Architecture

The backbone is **EfficientNet-B7**, the largest variant of the EfficientNet family (Tan & Le, 2019). EfficientNet uses compound scaling to simultaneously increase network depth, width, and input resolution using a principled set of coefficients, achieving state-of-the-art accuracy with fewer parameters than comparably performing architectures.

The final 1000-class ImageNet classification head is replaced with a 5-class output layer matching the DR grading scale.

### Two-Stage Training Strategy

**Phase 1 — Pre-training on 2015 data:**
All model parameters are trained on the large 35K-image 2015 dataset. The 2019 APTOS dataset is held out as a cross-domain validation set. This phase teaches the network to recognize retinal pathology features broadly.

**Phase 2 — Fine-tuning on 2019 data:**
The pre-trained weights from Phase 1 are loaded. Early layers are frozen to preserve the learned retinal representations, and only the final classification layers are fine-tuned on the smaller 3.6K-image 2019 dataset using Stratified K-Fold cross-validation (4 folds). This approach:
- Prevents catastrophic forgetting of features learned during pre-training
- Reduces overfitting on the limited 2019 training data
- Produces out-of-fold (OOF) predictions covering the full training set without data leakage

### Optimization

- **Optimizer:** Adam with an initial learning rate of `1e-3`
- **LR Scheduler:** StepLR — learning rate halved every 5 epochs
- **Loss Function:** Cross-Entropy Loss
- **Early Stopping:** Training halts if validation performance does not improve for 5 consecutive epochs

---

## Evaluation

The primary metric is **Cohen's Quadratic Weighted Kappa (QWK)**, the standard metric for ordinal classification tasks. Unlike accuracy, QWK penalizes predictions proportionally to how far they are from the true grade — predicting grade 0 when the true grade is 4 is penalized far more than predicting grade 3.

- **Kappa = 1.0** → Perfect agreement with ground truth
- **Kappa = 0.0** → Agreement no better than random chance
- **Kappa < 0.0** → Worse than random

Model performance is visualized through:
- Training and validation loss curves (to diagnose overfitting)
- Validation kappa curves across epochs
- Normalized confusion matrices (to identify which grades are most commonly confused)
- Per-class precision, recall, and F1-score reports

---

## Project Structure

```
├── diabetic-retinopathy-detection.ipynb   # Main notebook
├── models/                                # Saved model checkpoints
│   └── model_enet_b7_fold{1..4}.bin
└── README.md
```

---

## Requirements

```
torch
torchvision
efficientnet_pytorch
opencv-python
Pillow
numpy
pandas
scikit-learn
matplotlib
seaborn
scipy
tqdm
```

---

## References

- Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML 2019.
- APTOS 2019 Blindness Detection — Kaggle Competition
- 2015 Diabetic Retinopathy Detection — Kaggle Competition
- Graham, B. (2015). *Kaggle Diabetic Retinopathy Detection Competition Report.*
