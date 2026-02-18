# 🔍 Enhanced MRI Brain Tumor Classification

![Cover Image](https://github.com/FarzadNekouee/Enhanced_MRI_Tumor_Classification_Web_App/blob/master/cover_image/cover_image.png?raw=true)

## 📋 Project Overview
This project focuses on creating a machine learning model that reliably identifies normal and various tumor-affected brain images, such as pituitary, meningioma, and glioma tumors. A core objective of this work is ensuring high performance on low-quality images to promote utility in diverse healthcare settings where medical imaging quality may be limited.

## 🎯 Key Objectives
* **Dataset Exploration:** Examine class balances and image dimensions to understand characteristics.
* **Robust Training:** Enhance model resilience to low-quality images using controlled degradation techniques.
* **Transfer Learning:** Use the **ResNet50V2** pre-trained model for high accuracy without overfitting.
* **Performance Validation:** Evaluate the model thoroughly on a validation set to ensure dependable performance.

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras (ResNet50V2)
* **Computer Vision:** OpenCV
* **Data Science:** Pandas, NumPy, Scikit-learn
* **Visualization:** Seaborn, Matplotlib

## 📉 Methodology
### Image Quality Augmentation
To simulate real-world scenarios in under-resourced settings, the following modifications were applied to the dataset:
1.  **Gaussian Noise:** Replicating common imaging artifacts.
2.  **Gaussian Blur:** Mimicking blurring from patient movement or hardware constraints.
3.  **Downsampling:** Representing capabilities of less advanced MRI machinery.

### Model Architecture
The project utilizes a custom-head ResNet50V2 model:
* **GlobalAveragePooling2D** for dimensionality reduction.
* **Dense Layer (1024 units)** with ReLU activation.
* **Dropout (0.5)** to prevent overfitting.
* **Softmax Output Layer** for 4-class classification.



## 📊 Performance
The model exhibits high accuracy and well-balanced F1-scores across all categories. 
* **Learning Curves:** Training and validation loss show effective convergence.
* **Confusion Matrix:** High precision in identifying 'normal' cases and distinguishing between tumor types.

## 🚀 How to Run
1. **Clone the repo:** `git clone https://github.com/YourUsername/your-repo-name.git`
2. **Install requirements:** `pip install -r requirements.txt`
3. **Run the Notebook:** Open `enhanced-mri-brain-tumor-classification.ipynb` in any Jupyter environment.

---
*Created by [Your Name]*
