#!/bin/bash

# Navigate to the YOLOv5 directory
cd yolov5

# Run the YOLO training command
python train.py --img 416 --batch 32 --epochs 50 --data ../dataset.yaml --weights yolov5s.pt
