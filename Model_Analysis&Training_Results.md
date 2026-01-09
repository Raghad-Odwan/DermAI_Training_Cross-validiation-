# Model Analysis and Cross-Validation Results — DermAI Training Pipeline

## 1. Overview

This document presents an academic and structured analysis of the **stratified cross-validation results** for the DermAI binary skin lesion classification system (Benign vs. Malignant). The analysis reflects the finalized experimental outcomes obtained from a 3-fold stratified cross-validation strategy and is intended for public release as part of the project documentation on GitHub.

The DermAI training pipeline emphasizes **clinical reliability**, **robust generalization**, and **transparent evaluation**, with a specific focus on maximizing malignant lesion sensitivity while maintaining stable overall performance.

**Repository:** [https://github.com/Raghad-Odwan/DermAI_Training](https://github.com/Raghad-Odwan/DermAI_Training)

---

## 2. Dataset and Cross-Validation Strategy

### Dataset Characteristics

* **Total Images:** 19,505 dermoscopic images
* **Classes:** Binary (Benign / Malignant)
* **Source:** ISIC Dataset
* **Key Challenge:** Natural class imbalance favoring benign samples

### Stratified 3-Fold Cross-Validation

A stratified 3-fold cross-validation approach was adopted to ensure consistent class distribution across training and validation splits.

| Fold   | Training Samples | Validation Samples |
| ------ | ---------------- | ------------------ |
| Fold 1 | 13,000           | 6,501              |
| Fold 2 | 13,001           | 6,500              |
| Fold 3 | 13,001           | 6,500              |

This strategy provides a reliable estimate of generalization performance while minimizing variance caused by data imbalance.

---

## 3. Model Architecture

### Backbone: ResNet50

* Pretrained on ImageNet
* Fine-tuning applied to the last **40 layers**

### Custom Classification Head

* Global Average Pooling
* Fully Connected Layer + Dropout (0.4)
* Fully Connected Layer + Dropout (0.3)
* Softmax Output (2 classes)

### Training Configuration

* **Loss Function:** Binary Cross-Entropy
* **Optimizer:** AdamW
* **Initial Learning Rate:** 1e-4 with ReduceLROnPlateau
* **Regularization:** Dropout + Class Weights
* **Early Stopping:** Enabled

Class weighting was applied to mitigate imbalance:

```
{0: ~0.73 (Benign), 1: ~1.57 (Malignant)}
```

---

## 4. Training Pipeline

### Data Augmentation

* Rotation (±40°)
* Width/Height Shift (±10%)
* Zoom (≤20%)
* Horizontal Flip
* Brightness Adjustment (±20%)
* Pixel Scaling (1/255)

### Training Parameters

* **Batch Size:** 32
* **Epochs:** Up to 30 per fold
* **Input Resolution:** 224 × 224

### Monitoring and Callbacks

* EarlyStopping
* ReduceLROnPlateau
* ModelCheckpoint (best validation loss)

---

## 5. Fold-Level Performance Analysis

The following metrics were computed on each fold’s validation set using the restored best checkpoint.

### Quantitative Results

| Fold   | Accuracy | Precision (Malignant) | Recall (Malignant) | F1-Score |
| ------ | -------- | --------------------- | ------------------ | -------- |
| Fold 1 | 0.8296   | 0.7839                | 0.6415             | 0.7056   |
| Fold 2 | 0.8182   | 0.7913                | 0.5826             | 0.6711   |
| Fold 3 | 0.8148   | 0.6862                | 0.7710             | 0.7261   |

### Observations

* **Fold 1:** Highest overall accuracy, but relatively lower malignant recall.
* **Fold 2:** Best precision stability and cleanest validation loss behavior.
* **Fold 3:** Highest malignant recall, at the expense of lower precision.

---

## 6. Cross-Fold Summary Statistics

| Metric             | Mean  | Range       |
| ------------------ | ----- | ----------- |
| Accuracy           | 0.821 | 0.815–0.830 |
| Malignant Recall   | 0.665 | 0.583–0.771 |
| Malignant F1-Score | 0.701 | 0.671–0.726 |
| Benign Recall      | ~0.90 | Stable      |

These results demonstrate **consistent generalization** with expected trade-offs between recall and precision across folds.

---

## 7. Model Selection for Final Training

Although Fold 3 achieved the highest malignant recall, **Fold 2 was selected as the initialization point for final full-dataset training**, based on the following criteria:

* More stable validation loss trajectory
* Higher precision for malignant class
* Reduced overfitting behavior
* Better calibration consistency

**Final Training Strategy:**

* Initialize with **Fold 2 best weights**
* Retrain on the full dataset (19,505 images)
* Apply optimized threshold and augmentations

---

## 8. Ensemble Considerations

An ensemble model was constructed by averaging probabilistic outputs from all three folds.

### Benefits

* Reduced prediction variance
* Improved robustness
* Smoother probability calibration

### Ensemble Performance (Validation-Level)

| Metric           | Value |
| ---------------- | ----- |
| Accuracy         | ~0.83 |
| Malignant Recall | ~0.73 |
| Stability        | High  |

---

## 9. Threshold Optimization

Default probability threshold (0.5) was found to be overly conservative for malignant detection.

| Threshold | Recall      | Precision | F1       | Clinical Suitability |
| --------- | ----------- | --------- | -------- | -------------------- |
| 0.35      | High        | Moderate  | Balanced | Screening-focused    |
| **0.40**  | **Optimal** | Good      | **Best** | **Recommended**      |
| 0.45      | Moderate    | Better    | Good     | Balanced             |
| 0.50      | Lower       | Highest   | Moderate | Conservative         |

**Recommended Deployment Threshold:** **0.40**

---

## 10. Strengths of the Pipeline

### Technical

* Reproducible and modular design
* Stable cross-fold behavior
* Efficient training workflow

### Methodological

* Stratified cross-validation
* Class-weight handling
* Ensemble learning
* Threshold optimization
* Explainability via Grad-CAM

### Documentation

* Complete training logs
* Confusion matrices and ROC curves
* Metric summaries per fold

---

## 11. Limitations and Future Work

### Current Limitations

* Malignant recall remains below ideal clinical target (75–80%)
* Residual class imbalance
* Limited representation of rare subtypes
* Fixed input resolution

### Planned Improvements

**Short-Term**

* Increase fine-tuned layers
* Higher input resolution
* Stronger augmentation

**Medium-Term**

* Focal Loss
* Attention mechanisms
* EfficientNetV2 / ConvNeXt backbones

**Long-Term**

* Multi-class classification
* Metadata integration
* Mobile-friendly optimization

---

## 12. Conclusion

The DermAI cross-validation study demonstrates a **research-grade, clinically aware training pipeline**. The results confirm reliable generalization, interpretable behavior, and a well-justified model selection strategy.

The selection of **Fold 2 as the foundation for final training** ensures balanced performance and stable convergence, positioning DermAI as a strong candidate for deployment in clinical decision-support settings.

---

## 13. Artifacts and Outputs

* `best_resnet50_fold1.keras`
* `best_resnet50_fold2.keras` (Selected)
* `best_resnet50_fold3.keras`
* Training logs
* Metrics (JSON)
* ROC curves
* Confusion matrices
* Grad-CAM visualizations

---

