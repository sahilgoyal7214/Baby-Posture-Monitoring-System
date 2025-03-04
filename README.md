# Infant Pose Estimation and Classification using Multiple Modalities

## 📌 Project Overview

This project implements **Infant Pose Classification** using a custom dataset containing images and joint data across three modalities:
- **Uncovered** (infant without covers)
- **Cover1** (infant with partial cover 1)
- **Cover2** (infant with partial cover 2)

### Objectives
- Extract **HOG features** from images across all three modalities.
- Combine **pose information (joint coordinates)** with image features.
- Train **SVM, Random Forest, and k-NN classifiers** to classify infant poses.
- Perform **hyperparameter tuning** to optimize model performance.
- Evaluate models on **train, validation, and test sets**.

---

## 📂 Dataset Structure

The dataset is organized into 5 **folds** for cross-validation.  
Each fold contains 3 splits:
- **train** (Training Set)
- **val** (Validation Set)
- **test** (Test Set)

Each split contains:
- `uncovered.npy` - Images without covers
- `cover1.npy` - Images with cover type 1
- `cover2.npy` - Images with cover type 2
- `joints.npy` - Joint coordinate data for each sample
- `labels.npy` - Class labels for each sample

---

## 🔹 Key Features

- **Custom Dataset Class (`InfantPoseCustomDataset`)**
    - Handles loading data from the directory structure.
    - Supports accessing each sample directly via `__getitem__`.

- **Feature Extraction**
    - **HOG (Histogram of Oriented Gradients)** features extracted from images.
    - **Joint coordinates** are flattened and standardized for use with classifiers.

- **Model Training**
    - Three models are trained and compared:
        - **Support Vector Machine (SVM)**
        - **Random Forest (RF)**
        - **k-Nearest Neighbors (k-NN)**

- **Hyperparameter Tuning**
    - Performed using **GridSearchCV** for all three models.
    - Optimal hyperparameters are reported for each fold and modality.

- **Validation Curves**
    - For k-NN, a **validation curve** is plotted to show the effect of `n_neighbors` on accuracy.

- **Evaluation**
    - Models are evaluated on:
        - Training Set
        - Validation Set
        - Test Set

---

## 📊 Methodology

### 1. **Data Loading**
- Data is loaded from all folds and splits using `InfantPoseCustomDataset`.
- Each batch contains images, joints, and labels.

### 2. **Feature Engineering**
- Each image is converted to **HOG features**.
- Joint data is **flattened** and **standardized**.

### 3. **Train-Validation-Test Split**
- Each modality (uncovered, cover1, cover2) is split into 60% train, 20% validation, and 20% test sets.

### 4. **Model Training and Validation**
- Models (SVM, RF, k-NN) are trained on each modality’s train set.
- Validation accuracy is computed for each.

### 5. **Hyperparameter Tuning**
- Grid Search is applied to:
    - **SVM** (C, kernel)
    - **Random Forest** (n_estimators, max_depth, min_samples_split)
    - **k-NN** (n_neighbors, weights)

### 6. **Test Evaluation**
- Each tuned model is evaluated on the test set to compute final accuracy.

---


