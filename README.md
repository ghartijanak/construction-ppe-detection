# Construction-Site PPE Detection System

A deep learning and computer vision project for detecting Personal Protective Equipment (PPE) in construction-site environments.

This project explores three deep learning approaches:

* **Custom Convolutional Neural Network (CNN)** for PPE classification
* **EfficientNetB0** using transfer learning and fine-tuning for PPE classification
* **YOLO11n** for object detection in full construction-site images

The project investigates how different deep learning approaches can be used to support automated construction-site safety monitoring.

## Project Overview

Construction sites contain multiple workers, safety equipment and complex environments. Manually monitoring whether workers are wearing the required Personal Protective Equipment can be difficult, particularly when many workers are present.

This project uses computer vision and deep learning to identify PPE-related classes from construction-site images.

The project follows two main computer vision tasks:

### Image Classification

A custom CNN and EfficientNetB0 were trained to classify individual PPE object crops into five classes.

### Object Detection

YOLO11n was trained to detect and localise PPE-related objects directly in full construction-site images using bounding boxes.

The overall workflow is:

```text
Construction-Site Images
          │
          ▼
   Data Exploration
          │
          ▼
   Data Preprocessing
          │
     ┌────┴─────┐
     ▼          ▼
Object Crops   Full Images
     │             │
     ▼             ▼
 Custom CNN      YOLO11n
     │             │
     ▼             ▼
EfficientNetB0   Detection
     │
     ▼
Classification
     │
     └──────────┬──────────┘
                ▼
        Model Evaluation
```

# Objectives

The main objectives of this project are:

1. Develop a custom CNN for PPE classification.
2. Apply transfer learning using EfficientNetB0.
3. Fine-tune EfficientNetB0 for the PPE classification task.
4. Develop an object detection model using YOLO11n.
5. Evaluate classification models using appropriate classification metrics.
6. Evaluate YOLO11n using object-detection metrics.
7. Compare the different approaches.
8. Test the trained models on unseen images.
9. Demonstrate a computer vision approach for construction-site PPE monitoring.

# Dataset

The **Personal Protective Equipment dataset** used in this project was obtained from **Kaggle**.

The dataset contains workplace images with five different classes:

* **Helmet**
* **No Helmet**
* **Safety Vest**
* **No Vest**
* **Person**

The images contain a variety of real-world conditions. Workers can appear under different lighting conditions, in different locations, at different distances, from different camera angles, and against different backgrounds.

Some images contain a single worker, while others contain multiple people and PPE items.

The dataset is divided into **training, validation and test** folders. Each folder contains separate `images` and `labels` subfolders.

The `data.yaml` file stores the dataset path and class names, while the TXT label files contain the bounding-box annotations used for object detection.

### Dataset Structure

```text
dataset/
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
├── test/
│   ├── images/
│   └── labels/
│
└── data.yaml
```

### Dataset Statistics

The dataset used in this project contains:

* **4,060 images**
* **30,064 annotated objects**
* **5 classes**

The dataset is **not included in this repository**. Users should obtain the dataset from its original Kaggle source and follow the applicable dataset terms and licensing conditions.

# Models Used

## 1. Custom CNN

A custom Convolutional Neural Network was developed as a baseline classification model.

The CNN learns visual features from cropped PPE images and predicts the corresponding PPE class.

### CNN workflow

```text
Input PPE Crop
      ↓
Convolution
      ↓
Pooling
      ↓
Feature Extraction
      ↓
Fully Connected Layers
      ↓
Classification
      ↓
PPE Class
```

The custom CNN provides a baseline against which the transfer-learning approach can be evaluated.

# 2. EfficientNetB0 Transfer Learning

EfficientNetB0 was used to investigate whether a pretrained deep-learning model could provide better classification performance than the custom CNN.

The pretrained model was adapted to the five PPE classes.

The process included:

1. Loading the pretrained EfficientNetB0 model.
2. Adapting the classification layer.
3. Training the model on the PPE dataset.
4. Fine-tuning selected layers.
5. Evaluating the final model on unseen test data.

### EfficientNetB0 workflow

```text
Pretrained EfficientNetB0
          ↓
Adapt Classification Layer
          ↓
Train on PPE Dataset
          ↓
Fine-Tuning
          ↓
Evaluation
          ↓
PPE Classification
```

# 3. YOLO11n Object Detection

YOLO11n was used for object detection.

Unlike the CNN and EfficientNetB0 classification models, YOLO11n can identify and localise multiple objects within a complete construction-site image.

The model predicts:

* Object class
* Bounding-box location
* Confidence score

### YOLO11n workflow

```text
Full Construction Image
          ↓
        YOLO11n
          ↓
    Object Detection
          ↓
 ┌─────────────────────┐
 │ Bounding Boxes      │
 │ Class Labels        │
 │ Confidence Scores   │
 └─────────────────────┘
```

The YOLO11n model was trained using the bounding-box annotations provided with the dataset.

# Methodology

The project followed the following workflow:

### 1. Dataset Exploration

The dataset was explored to understand:

* Image distribution
* Class distribution
* Image dimensions
* Object annotations
* PPE categories

### 2. Data Validation

Images and annotations were checked to identify potential data or annotation issues.

### 3. Data Preprocessing

The images were prepared for the classification and object-detection tasks.

### 4. Object-Crop Generation

Bounding-box annotations were used to extract individual PPE objects for the classification models.

This allowed the CNN and EfficientNetB0 models to perform classification on individual objects.

### 5. Custom CNN Training

A CNN model was trained as a baseline classification approach.

### 6. EfficientNetB0 Transfer Learning

A pretrained EfficientNetB0 model was adapted to the PPE classification task.

### 7. EfficientNetB0 Fine-Tuning

Selected layers were fine-tuned to further adapt the pretrained model to the dataset.

### 8. Classification Evaluation

The classification models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Classification report

### 9. YOLO11n Training

YOLO11n was trained using the original image-level bounding-box annotations.

### 10. Object Detection Evaluation

YOLO11n was evaluated using:

* Precision
* Recall
* mAP@50
* mAP@50-95

### 11. Unseen Image Testing

The trained models were tested on images that were not used during model training to demonstrate their practical predictions.

# Results

## Classification Results

The custom CNN and EfficientNetB0 models were evaluated on the PPE classification task.

| Model          |   Accuracy | Macro Precision | Macro Recall | Macro F1 |
| -------------- | ---------: | --------------: | -----------: | -------: |
| Custom CNN     | **89.44%** |          87.69% |       88.97% |   88.27% |
| EfficientNetB0 | **93.63%** |          92.46% |       93.34% |   92.88% |

The results show the performance obtained by the two classification approaches on the evaluated test data.

## YOLO11n Detection Results

YOLO11n was evaluated using object-detection metrics.

| Metric    |     Result |
| --------- | ---------: |
| Precision | **90.68%** |
| Recall    | **82.43%** |
| mAP@50    | **86.98%** |
| mAP@50-95 | **58.54%** |

YOLO11n was also tested on unseen images to demonstrate object detection and localisation in construction-site scenes.

# Evaluation Metrics

## Classification

### Accuracy

The proportion of correctly classified samples.

### Precision

The proportion of predicted samples for a class that were correctly classified.

### Recall

The proportion of actual samples belonging to a class that were correctly identified.

### F1-Score

The harmonic mean of precision and recall.

### Confusion Matrix

A visual representation of correct and incorrect predictions for each class.

## Object Detection

### Precision

Measures the proportion of predicted detections that are correct.

### Recall

Measures the proportion of actual objects that are successfully detected.

### mAP@50

Mean Average Precision calculated at an IoU threshold of 0.50.

### mAP@50-95

Mean Average Precision calculated across IoU thresholds from 0.50 to 0.95.

# Demo

The `demo/` directory contains example predictions produced by the trained detection system.

Example workflow:

```text
Input Construction Image
          ↓
        YOLO11n
          ↓
   Detect PPE Objects
          ↓
┌─────────────────────────┐
│ Person                  │
│ Helmet                  │
│ No Helmet               │
│ Safety Vest             │
│ No Vest                 │
└─────────────────────────┘
```

Example detection results can be found in:

```text
demo/
```

# Repository Structure

```text
construction-ppe-detection/
│
├── README.md
│
├── notebook/
│   └── construction_ppe_detection.ipynb
│
├── results/
│   ├── cnn/
│   ├── efficientnet/
│   ├── yolo/
│   └── ...
│
└── demo/
    ├── ...
    └── ...
```

The repository intentionally contains the main **notebook, experimental results and demonstration outputs** rather than the original dataset.

# Technologies Used

### Programming Language

* Python

### Deep Learning

* TensorFlow / Keras
* Convolutional Neural Networks
* EfficientNetB0
* YOLO11

### Computer Vision

* OpenCV
* Pillow

### Machine Learning & Data Processing

* NumPy
* Pandas
* Scikit-learn
* Matplotlib

### Development

* Jupyter Notebook
* Git
* GitHub

# How to Use

Clone the repository:

```bash
git clone git@github.com:janakgharti1/construction-ppe-detection.git
```

Navigate into the repository:

```bash
cd construction-ppe-detection
```

Open the notebook:

```text
notebook/construction_ppe_detection.ipynb
```

The notebook contains the project workflow, including:

```text
Dataset Exploration
        ↓
Data Preparation
        ↓
CNN Training
        ↓
EfficientNetB0 Training
        ↓
EfficientNetB0 Fine-Tuning
        ↓
YOLO11n Training
        ↓
Model Evaluation
        ↓
Unseen Image Testing
```

> **Note:** The original dataset is not included in this repository. The dataset must be obtained separately from its original Kaggle source before reproducing the experiments.

# Limitations

The project has several limitations:

* Performance depends on the quality and diversity of the dataset.
* Small or partially occluded PPE objects can be challenging to detect.
* Different lighting conditions and camera angles can affect model performance.
* Multiple workers in the same image can increase detection complexity.
* The CNN and EfficientNetB0 perform classification on cropped objects, whereas YOLO11n performs object detection on full images.
* Therefore, classification metrics and YOLO detection metrics should not be directly treated as the same evaluation task.

# Future Improvements

Potential future improvements include:

* Real-time webcam PPE detection
* Video-based PPE monitoring
* Real-time safety alerts
* FastAPI deployment
* Web-based PPE monitoring dashboard
* Testing on additional construction-site datasets
* Improved detection of small and partially occluded PPE
* Deployment on edge devices
* Integration with a real-time construction-site monitoring system

A possible future architecture would be:

```text
Camera / Image
      ↓
    YOLO11
      ↓
PPE Detection
      ↓
Safety Compliance Analysis
      ↓
   FastAPI
      ↓
Web Dashboard
      ↓
Safety Alert
```

# Key Learning Outcomes

Through this project, practical experience was gained in:

* Convolutional Neural Networks
* Transfer Learning
* EfficientNetB0
* Fine-Tuning
* Object Detection
* YOLO11
* Image preprocessing
* Data augmentation
* Bounding-box annotations
* Model evaluation
* Confusion matrices
* Classification reports
* Precision and Recall
* mAP
* Computer Vision
* Deep Learning experimentation

# Author

**Janak Gharti**

MSc Data Science and Computational Intelligence
Coventry University, UK

### Interests

* Data Science
* Machine Learning
* Deep Learning
* Computer Vision
* Artificial Intelligence
* Generative AI
