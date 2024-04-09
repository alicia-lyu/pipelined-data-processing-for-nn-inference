#!/usr/bin/bash

# Followinhg https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_1-model_deployment#deploying-multiple-models
BASEDIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd "$BASEDIR/triton-tutorial"
# Deps
sudo which ubuntu-drivers || (echo "Run 'sudo apt install ubuntu-drivers-common' to install ubuntu-drivers first"; exit "1")
nvidia-smi || (echo "Run 'sudo ubuntu-drivers autoinstall; sudo apt -y install nvidia-cuda-toolkit' to install the NVIDIA drivers first"; exit "1")
sudo apt update -y && sudo apt upgrade -y
docker run hello-world || source ./install_docker.sh
which pip || (wget https://bootstrap.pypa.io/get-pip.py && sudo python get-pip.py; rm get-pip.py)
pip install -U tf2onnx tensorflow torch
# Text detection: OpenCV's EAST model
ls $BASEDIR/triton-tutorial/model_repository/text_detection/1/model.onnx || (wget https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz; tar -xvf frozen_east_text_detection.tar.gz; rm frozen_east_text_detection.tar.gz; python -m tf2onnx.convert --input frozen_east_text_detection.pb --inputs "input_images:0" --outputs "feature_fusion/Conv_7/Sigmoid:0","feature_fusion/concat_3:0" --output detection.onnx; mkdir -p model_repository/text_detection/1; mv detection.onnx model_repository/text_detection/1/model.onnx)
# Text recognition: ResNet
ls $BASEDIR/triton-tutorial/model_repository/text_recognition/1/model.onnx || (wget https://www.dropbox.com/sh/j3xmli4di1zuv3s/AABzCC1KGbIRe2wRwa3diWKwa/None-ResNet-None-CTC.pth -O resnet.pth; python convert_resnet.py; rm resnet.pth; mkdir -p model_repository/text_recognition/1; mv resnet.onnx model_repository/text_recognition/1/model.onnx)
# Launch the server
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v "$BASEDIR/triton-tutorial/model_repository:/models" nvcr.io/nvidia/tritonserver:24.03-py3