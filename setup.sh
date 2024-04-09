#!/usr/bin/bash

BASEDIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd "$BASEDIR/triton-tutorial"
# Deps
sudo which ubuntu-drivers || (echo "Run 'sudo apt install ubuntu-drivers-common' to install ubuntu-drivers first"; exit "1")
nvidia-smi || (echo "Run 'sudo ubuntu-drivers install --gpgpu nvidia:535-server; sudo apt install -y nvidia-utils-535-server; sudo reboot' to install the NVIDIA drivers first"; exit "1")
sudo apt update -y && sudo apt upgrade -y
which pip || (wget https://bootstrap.pypa.io/get-pip.py && sudo python get-pip.py; rm get-pip.py)
pip install -U tf2onnx tensorflow torch
# Text detection: OpenCV's EAST model
ls frozen_east_text_detection.pb || (wget https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz; tar -xvf frozen_east_text_detection.tar.gz; rm frozen_east_text_detection.tar.gz)
ls detection.onnx || python -m tf2onnx.convert --input frozen_east_text_detection.pb --inputs "input_images:0" --outputs "feature_fusion/Conv_7/Sigmoid:0","feature_fusion/concat_3:0" --output detection.onnx
# Text recognition: ResNet
ls resnet.pth || wget https://www.dropbox.com/sh/j3xmli4di1zuv3s/AABzCC1KGbIRe2wRwa3diWKwa/None-ResNet-None-CTC.pth -O resnet.pth
ls resnet.onnx || (python convert_resnet.py; rm resnet.pth)
# Setup model repository
mkdir -p model_repository/text_detection/1
mv detection.onnx model_repository/text_detection/1/model.onnx
mkdir -p model_repository/text_recognition/1
mv resnet.onnx model_repository/text_recognition/1/model.onnx
# Launch the server
# Replace the yy.mm in the image name with the release year and month
# of the Triton version needed, eg. 22.08
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v "$BASEDIR/triton-tutorial/model_repository:/models nvcr.io/nvidia/tritonserver:24.03-py3"