# Detection of Pneumonia from Chest X-Ray Images

## Description

This project implements a deep learning-based approach to detect **Pneumonia** from chest X-ray images using a **Custom Convolutional Neural Network (CNN)** and **Transfer Learning with InceptionV3**. The model is trained on 5,856 labeled chest X-ray images (Normal vs. Pneumonia) from the publicly available Kaggle dataset. The custom CNN achieved a testing accuracy of **89.53%** with a recall of **95.48%** for pneumonia detection, making it effective for assisting in early screening of pneumonia cases.

---

## Features

- Custom deep CNN architecture built from scratch using Keras
- Transfer learning support using **InceptionV3** (pretrained on ImageNet)
- Image preprocessing with data augmentation (shear, zoom, horizontal flip)
- Class weight balancing to handle dataset imbalance
- Training callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
- Evaluation with Precision, Recall, F1-Score, and Confusion Matrix
- Visual comparison of predicted vs. actual labels on test images

---

## Tech Stack

| Component         | Details                                      |
|-------------------|----------------------------------------------|
| Language          | Python 3.6+                                  |
| Deep Learning     | Keras, TensorFlow                            |
| Pretrained Model  | InceptionV3 (ImageNet weights)               |
| Image Processing  | OpenCV, Pillow, Keras ImageDataGenerator     |
| Visualization     | Matplotlib, Seaborn, mlxtend                 |
| Evaluation        | scikit-learn (Precision, Recall, F1, CM)     |
| Environment       | Jupyter Notebook / Anaconda                  |

---

## Project Architecture

```
Input (Chest X-Ray Images)
        |
        v
  Image Preprocessing
  (Resize to 150x150, Rescale, Augmentation)
        |
        v
  Model Training
  +-----------------------------+
  | Option A: Custom CNN        |
  |   5 Conv blocks + Dense     |
  | Option B: InceptionV3       |
  |   Fine-tuned top layers     |
  +-----------------------------+
        |
        v
  Evaluation & Metrics
  (Accuracy, Loss, Precision, Recall, F1, Confusion Matrix)
        |
        v
  Prediction Visualization
  (True vs. Predicted labels on test images)
```

---

## Dataset

| Property           | Value                                        |
|--------------------|----------------------------------------------|
| Dataset Name       | Chest X-Ray Images (Pneumonia)               |
| Source             | [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) |
| Classes            | 2 (Normal, Pneumonia)                        |
| Total Images       | 5,856 (1.15 GB)                              |
| Training           | 5,216 images                                 |
| Validation         | 320 images                                   |
| Testing            | 320 images                                   |

> **Note:** The dataset is not included in this repository due to its size. Download it from the Kaggle link above and place it under `code/data/input/` with `train/`, `val/`, and `test/` subdirectories.

---

## Results (Custom CNN)

| Metric                | Value   |
|-----------------------|---------|
| Testing Accuracy      | 89.53%  |
| Testing Loss          | 0.41    |
| Precision             | 88.37%  |
| Recall (Pneumonia)    | 95.48%  |
| F1-Score              | 91.79%  |

### Confusion Matrix

<img src="demo/report/CM.png" alt="Confusion Matrix" width="500">

### Sample Predictions

<img src="demo/sample/sample.png" alt="Sample Output" width="700">

---

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Vignan2659/Detection-of-Pneumonia.git
   cd Detection-of-Pneumonia
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/Mac
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - Download from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
   - Extract and place the `train/`, `val/`, `test/` folders inside `code/data/input/`

---

## How to Run

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook "code/Detection of Pneumonia from Chest X-Ray Images 1.0.0.3.ipynb"
   ```
2. Run cells sequentially from top to bottom.
3. The notebook will:
   - Load and preprocess chest X-ray images
   - Build and train the CNN model (or load a saved model)
   - Evaluate on the test set and display metrics
   - Visualize predictions with true vs. predicted labels

---

## Repository Structure

```
Detection-of-Pneumonia/
|
+-- README.md                  # Project documentation
+-- LICENSE                    # MIT License
+-- requirements.txt           # Python dependencies
+-- .gitignore                 # Git ignore rules
|
+-- code/
|   +-- Detection of Pneumonia from Chest X-Ray Images 1.0.0.3.ipynb
|   |                          # Main notebook (latest version)
|   +-- obsolete/              # Earlier notebook versions (for reference)
|
+-- demo/
    +-- images/                # Result images
    +-- report/                # Confusion matrix and classification reports
    +-- sample/                # Sample prediction outputs
```

---

## Contributors

- **Vignan Karthikeya** -- [GitHub Profile](https://github.com/Vignan2659)
- **Contributor 2** -- Project Collaborator

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## References

- Kermany, D. S., et al. *"Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning."* Cell, 2018. [Link](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)
- Dataset: [Chest X-Ray Images (Pneumonia) -- Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
