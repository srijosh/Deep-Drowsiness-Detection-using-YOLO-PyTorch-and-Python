# Deep Drowsiness Detection using YOLO, PyTorch, and Python

This repository contains a project for detecting drowsiness using deep learning with the YOLOv5 model. The project is designed to identify whether a person is awake or drowsy by analyzing facial features in real-time, with the aim of improving safety in scenarios like driving.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)

## Introduction

Drowsiness detection is a crucial task for enhancing safety and reducing accidents caused by fatigue. This project employs YOLOv5, a state-of-the-art object detection model, to classify a person's state as awake or drowsy based on labeled image data. The system can process real-time video input, making it practical for deployment in applications such as driver assistance systems.

## Dataset

The dataset for this project was created using images captured via OpenCV. The images were manually labeled as awake or drowsy using LabelImg, a graphical image annotation tool. The labeled data was saved in YOLO format and included in the dataset.yml configuration file for training.

## Installation

To set up and run the project, follow these steps:

### Prerequisites

Ensure you have Python 3.x and the following libraries installed:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install numpy matplotlib opencv-python
```

### Clone the Repository

```
git clone https://github.com/srijosh/Deep-Drowsiness-Detection-using-YOLO-PyTorch-and-Python.git
cd Deep-Drowsiness-Detection-using-YOLO-PyTorch-and-Python

```

## Usage

### Training the Model

1. Prepare the Dataset:

- Use LabelImg to annotate your images and save them in YOLO format.
- Update the dataset.yml file in the YOLOv5 folder with the dataset paths.

2. Train the YOLOv5 Model: Navigate to the YOLOv5 folder and run the following command:

```
cd yolov5
python train.py --img 320 --batch 16 --epochs 500 --data dataset.yml --weights yolov5s.pt --workers 2
```

### Loading the Trained Model

After training, load the best weights into the model:

```
import torch
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp3/weights/best.pt', force_reload=True)
```

## Model

The project uses YOLOv5 for real-time object detection, which is fine-tuned to classify images as awake or drowsy.

### Training Details:

- Base Model: YOLOv5s
- Image Size: 320x320
- Epochs: 500
- Batch Size: 16
- Pretrained Weights: yolov5s.pt

### Evaluation Metrics:

- Precision
- Recall
- mAP (mean Average Precision)
