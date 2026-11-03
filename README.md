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
 ![Lung histology image before and after resizing](attachment:lung-image-resizing.png)


