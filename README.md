# Histopathology Image Classification: Lung, Colon and Breast

A deep learning project for **tissue identification** and **histopathological image classification**, developed as a data science learning project.

- **Notebook 1 — Histopathology Image Classification:** prepares the datasets and develops specialist classifiers for lung, colon and breast tissue.
- **Notebook 2 — Tissue Type Classification:** trains a MobileNetV2 model to identify whether an image comes from lung, colon or breast tissue.

The intended application is a two-stage system: first identify the tissue, then route the image to the corresponding specialist model. 
## 1. Dataset and tools

The original datasets contained **3,400 images**. After removing **25 duplicate images**, **3,375 images** remained.

| Tissue | Original images | Images after duplicate removal | Histopathological classes |
|---|---:|---:|---|
| Lung | 1,500 | 1,487 | Adenocarcinoma (`lung_aca`), benign (`lung_n`), squamous cell carcinoma (`lung_scc`) |
| Colon | 1,500 | 1,488 | Adenocarcinoma (`colon_aca`), benign (`colon_n`) |
| Breast | 400 | 400 | Benign (`breast_ben`), carcinoma in situ (`breast_insi`), invasive carcinoma (`breast_inv`), normal (`breast_n`) |
| **Total** | **3,400** | **3,375** | **9 classes across 3 tissues** |

**Dataset Source:** 

[ Grand Challenge on Breast Cancer Histology images (Zenodo)](https://zenodo.org/records/3632035).

[Lung and Colon Cancer Histopathological Images (kaggle)](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images?)

**Tools:** Python, Jupyter Notebook, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, TensorFlow/Keras.

## 2. Notebook 1 — Specialist histopathology classifiers

### Image preprocessing

- Check of image validity, dimensions and color channels, then resize and preprocess inputs for the models.

| Property | Lung | Colon | Breast |
|---|---|---|---|
| Original image dimensions | 768 × 768 | 768 × 768 | 2048 × 1536 |
| Original image format | JPEG | JPEG | TIFF |
| Model input size | 224 × 224 × 3 | 224 × 224 × 3 | 224 × 224 × 3 |
| Color | RGB | RGB | RGB |

- Data split: **70% training / 15% validation / 15% test** split. Lung and colon were modeled using custom CNNs (Convolutional Neural Network). The breast experiments compared a CNN against MobileNetV2 transfer learning.

| Lung image resizing example |

<img width="970" height="505" alt="image" src="https://github.com/user-attachments/assets/7d38ee52-ae6d-4de9-87b6-3770dfe870be" />

### Lung and Colon Models Training and Validation Results:

Both specialist CNNs were initially trained for 10 epochs with a batch size of 32. The colon model was trained further with early stopping and checkpointing.

| Lung CNN — training and validation accuracy | Colon CNN — training and validation accuracy |

<img width="1051" height="366" alt="image" src="https://github.com/user-attachments/assets/fce3f426-67fb-4c73-abd6-404d1ddcb29c" />


Both CNN models were trained and evaluated using validation accuracy and loss to monitor their performance. The training curves show how each model's performance changed across epochs.

### Lung and colon: test results

| Metric | Lung CNN | Colon CNN |
|---|---:|---:|
| Test images | 224 | 224 |
| Test accuracy | **86.16%** | **86.61%** |
| Test loss | 0.3321 | 0.3091 |
| Number of classes | 3 | 2 |

| Lung CNN — confusion matrix | Colon CNN — confusion matrix |
|:---:|:---:|
| ![Lung CNN confusion matrix](attachment:lung-confusion-matrix.png) | ![Colon CNN confusion matrix](attachment:colon-confusion-matrix.png) |

- Adenocarcinoma (lung_aca): 58 of 74 images were correctly classified. The model confused 14 adenocarcinoma images with squamous cell carcinoma.
Normal (lung_n): 70 of 75 images were correctly classified: This is the best-performing class. Squamous Cell Carcinoma (lung_scc): 65 of 75 images were correctly classified. 
- As regards the Colon model, TRUE POSITIVE (TP) 92 cancer correctly classified as cancer, FALSE NEGATIVE (FN) 20 cancer incorrectly classified as benign, FALSE POSITIVE (FP) 10 benign incorrectly classified as cancer, TRUE NEGATIVE (TN) 102 benign correctly classified as benign tissue.

**Lung classification report**

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| `lung_aca` | 0.7945 | 0.7838 | 0.7891 | 74 |
| `lung_n` | 0.9722 | 0.9333 | 0.9524 | 75 |
| `lung_scc` | 0.8228 | 0.8667 | 0.8442 | 75 |
| **Accuracy** | | | **0.8616** | **224** |
| Macro avg | 0.8632 | 0.8613 | 0.8619 | 224 |
| Weighted avg | 0.8635 | 0.8616 | 0.8622 | 224 |

**Colon classification report**

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| `colon_aca` | 0.9020 | 0.8214 | 0.8598 | 112 |
| `colon_n` | 0.8361 | 0.9107 | 0.8718 | 112 |
| **Accuracy** | | | **0.8661** | **224** |
| Macro avg | 0.8690 | 0.8661 | 0.8658 | 224 |
| Weighted avg | 0.8690 | 0.8661 | 0.8658 | 224 |

### Breast classification: baseline CNN versus transfer learning

The breast dataset contains **400 TIFF images**, evenly divided across four classes (100 per class). The custom CNN reached a maximum reported validation accuracy of **50%**. Transfer learning with **MobileNetV2** improved the best reported validation accuracy to **76.67%**, a gain of **26.67 percentage points**.

| Breast model | Best reported validation accuracy | Test accuracy | Test loss |
|---|---:|---:|---:|
| Custom CNN baseline | 50.00% | - | - |
| MobileNetV2 | 76.67% | **73.33%** | 0.6665 |

<img width="921" height="640" alt="image" src="https://github.com/user-attachments/assets/76349e79-7168-4d82-862a-3f8a692a9015" />

- Invasive carcinoma is the best-recognized class: The model correctly identifies 12 out of 15 invasive carcinoma images.
- Normal tissue is the most difficult class: The model correctly identifies only 10 out of 15 normal tissue images.
- The model sometimes confuses benign tissue with carcinoma in situ: Four benign images were incorrectly classified as carcinoma in situ.
- Some cancer images are classified as normal.

**Breast MobileNetV2 classification report**

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| `breast_ben` | 0.7333 | 0.7333 | 0.7333 | 15 |
| `breast_insi` | 0.5789 | 0.7333 | 0.6471 | 15 |
| `breast_inv` | 0.9231 | 0.8000 | 0.8571 | 15 |
| `breast_n` | 0.7692 | 0.6667 | 0.7143 | 15 |
| **Accuracy** | | | **0.7333** | **60** |
| Macro avg | 0.7511 | 0.7333 | 0.7380 | 60 |
| Weighted avg | 0.7511 | 0.7333 | 0.7380 | 60 |
