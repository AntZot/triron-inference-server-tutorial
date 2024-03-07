#!/bin/bash

mkdir -p model_repository/detection/3/

python utils/yolo8_to_onnx.py

mv ./yolov8n.onnx ./model_repository/detection/3/model.onnx