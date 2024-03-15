#!/bin/bash

python utils/yolo8_to_onnx.py

local_var=$( (cd ./model_repository/detection/; ls -d */) | wc -l)
new_dir=$(( $local_var + 1 ))

mkdir -p ./model_repository/detection/$new_dir/

#mv ./yolov8n.onnx ./model_repository/detection/$new_dir/model.onnx
mv ./yolov8n.onnx ./model_repository/detection/$new_dir/model.onnx