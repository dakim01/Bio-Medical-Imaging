# Notebook Description — Kaggle Publish Card

> Use this text in the **"Description"** field when publishing your Kaggle notebook.

---

## Short Description (for the subtitle / tagline field)

```
End-to-end brain tumor segmentation pipeline using U-Net on the BraTS 2020 multimodal MRI dataset — with custom data generators, Dice loss, and per-class evaluation.
```

---

## Full Description (paste into the notebook description box)

This notebook walks through the complete pipeline for **automated brain tumor segmentation** from multimodal MRI scans using a **U-Net** deep learning architecture, trained on the publicly available **BraTS 2020** dataset.

### What you'll learn

- How to work with **3D NIfTI medical images** using `nibabel`
- How to build a **custom Keras DataGenerator** for memory-efficient 3D volume handling
- How to implement the **U-Net architecture** from scratch with TensorFlow/Keras
- How to define and use **Dice loss** and **per-class Dice coefficients** as custom metrics
- How to visualize and interpret **multi-class segmentation masks** across axial, coronal, and sagittal planes
- How to evaluate segmentation performance using **Mean IoU, sensitivity, specificity, precision**, and more

### Dataset

**BraTS 2020** — 369 glioma patients with four MRI modalities (T1, T1ce, T2, FLAIR) and expert-annotated segmentation masks covering:
- Label 0: Healthy tissue
- Label 1: Necrotic / Non-enhancing Tumor Core
- Label 2: Peritumoral Edema
- Label 4: Enhancing Tumor

### Key Results (Test Set)

| Metric | Score |
|---|---|
| Accuracy | > 99% |
| Mean IoU | 0.8426 |
| Dice Coefficient | 0.6480 |
| Sensitivity | 0.9916 |
| Specificity | 0.9978 |

### Notebook Sections

1. Introduction to image segmentation and medical imaging
2. Dataset download and setup
3. Dataset exploration (modalities, NIfTI format, slice planes)
4. Data splitting (train / val / test)
5. Custom DataGenerator with preprocessing
6. Loss functions and evaluation metrics
7. U-Net model definition
8. Model training with callbacks
9. Metrics analysis and learning curves
10. Inference and visual predictions
11. Final model evaluation

---

## Tags to add on Kaggle

```
deep learning, image segmentation, medical imaging, U-Net, brain tumor, BraTS, MRI, TensorFlow, Keras, computer vision
```

---

## Recommended Cover Image Description

A side-by-side comparison showing:
- Left: Original FLAIR MRI slice (grayscale)
- Middle: Ground truth segmentation mask (colorized by class)
- Right: Model prediction mask

(Export one of the `showPredictsById()` outputs from the notebook as a PNG and use it as the notebook thumbnail.)
