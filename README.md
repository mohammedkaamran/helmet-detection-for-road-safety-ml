# CNN-Based Helmet Detection for Road Safety Compliance

> Machine Learning Course — Phase 2: Proposal & Code Implementation  
> Framework: TensorFlow / Keras | Environment: Kaggle Notebook (GPU)

---

## Project Overview

This project builds a **CNN-based binary image classifier** to detect whether riders on roads are wearing safety helmets. Using road surveillance imagery, the model can automatically identify helmet violations — enabling real-time enforcement without manual monitoring.

**Classes:** `Helmet` | `No Helmet`

**Mapped from XML labels:** `helmet` → Helmet | `head` / `person` (no helmet) → No Helmet

---

## Repository Structure

```
├── helmet_detection_cnn.ipynb              # Main Kaggle notebook (complete code)
├── Helmet_Detection_Road_Safety_Proposal.docx  # Project proposal document
├── README.md                               # This file
└── output_figures/                         # Generated figures (after running notebook)
    ├── fig_sample_images.png
    ├── fig_class_distribution.png
    ├── cnn_curves.png
    ├── mobilenet_curves.png
    ├── custom_cnn_confusion_matrix.png
    ├── mobilenetv2_confusion_matrix.png
    ├── custom_cnn_roc_curve.png
    ├── mobilenetv2_roc_curve.png
    ├── fig_model_comparison.png
    └── fig_gradcam.png
```

---

## Dataset

- **Name:** Helmet Detection
- **Source:** Kaggle — andrewmvd
- **Link:** https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection
- **Format:** Images + Pascal VOC XML annotations
- **Classes:** 2 (Helmet / No Helmet)
- **Split:** 70% Train / 15% Validation / 15% Test

---

## Models Implemented

| Model | Description |
|---|---|
| Custom CNN | 3 Conv blocks + BatchNorm + Dropout + Dense head |
| MobileNetV2 | Pretrained ImageNet weights + 2-phase fine-tuning |

---

## How to Run on Kaggle (Recommended)

1. Go to [kaggle.com](https://www.kaggle.com) and create a new notebook
2. Click **"Add Data"** → search **"Hard Hat Detection"** by andrewmvd → add it
3. Upload `helmet_detection_cnn.ipynb` via File → Import Notebook
4. Enable **GPU**: Settings → Accelerator → GPU T4 x1
5. Click **Run All** (takes ~20–30 minutes)
6. Click **Share** → set to **Public** → copy the link for submission

### Run Locally

```bash
pip install tensorflow scikit-learn matplotlib seaborn pandas numpy
jupyter notebook helmet_detection_cnn.ipynb
```

---

## Code Implementation Includes

- XML annotation parsing (Pascal VOC format)
- Dataset loading and label mapping
- Image resizing (224×224) and normalization
- Data augmentation (flip, rotation, zoom, shift)
- Stratified Train/Val/Test split (70/15/15)
- Custom CNN with 3 convolutional blocks
- Transfer learning with MobileNetV2 (frozen base + fine-tuning)
- Confusion matrix and classification report
- Accuracy/loss training curves
- ROC-AUC score and ROC curve
- Model comparison table
- Grad-CAM visualization (bonus)

---

## Expected Results

| Metric | Custom CNN | MobileNetV2 |
|---|---|---|
| Accuracy | ~82–88% | ~90–95% |
| F1-Score | ~0.83 | ~0.91 |
| ROC-AUC | ~0.88 | ~0.95 |

*Actual results will vary based on training run.*

---

## References

1. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521, 436–444.
2. Sandler, M., et al. (2018). MobileNetV2. CVPR 2018.
3. Dataset: https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection
4. WHO (2023). Road Traffic Injuries Fact Sheet. World Health Organization.
5. Chollet, F. (2021). *Deep Learning with Python*, 2nd Ed. Manning.
