# -*- coding: utf-8 -*-
"""Untitled78.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cleel3oKN84TueNrXgwL5bbOuiKcxFsX
"""

!pip install ultralytics roboflow

api_key="Us0BLBPgpxYZg4HZvpUO"

from roboflow import Roboflow
rf = Roboflow(api_key=api_key)
project = rf.workspace("robox-iumhb").project("gun_detction")
version = project.version(1)
dataset = version.download("yolov11")

from ultralytics import YOLO

# Load a pretrained YOLOv11 model
model = YOLO("yolo11n.pt")

# Train the model with your custom dataset
results = model.train(data="/content/gun_detction-1/data.yaml", epochs=50, batch=64, imgsz=256)

results

import os

train_path = "/content/gun_detction-1/train"
val_path = "/content/gun_detction-1/valid"
test_path = "/content/gun_detction-1/test"

print("Train folder exists:", os.path.exists(train_path))
print("Validation folder exists:", os.path.exists(val_path))
print("Test folder exists:", os.path.exists(test_path))

model.export(format="onnx")

gun_detect=YOLO("/content/runs/detect/train/weights/best.pt")

res=gun_detect.predict(
    r'/content/th (79).jpeg'
)

import matplotlib.pyplot as plt
import cv2
import numpy as np
cv2.cvtColor(res[0].plot(),cv2.COLOR_BGR2RGB)

!export LC_ALL=C.UTF-8
!export LANG=C.UTF-8
!pip install gradio

!locale

!apt-get update && apt-get install -y locales
!locale-gen en_US.UTF-8
!update-locale LANG=en_US.UTF-8
!export LANG=en_US.UTF-8
!export LC_ALL=en_US.UTF-8

