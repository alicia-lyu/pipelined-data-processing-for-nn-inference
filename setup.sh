#!/usr/bin/bash

BASEDIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd "$BASEDIR/triton-tutorial"
# Deps
# sudo which ubuntu-drivers || (echo "Run 'sudo apt install ubuntu-drivers-common' to install ubuntu-drivers first"; exit "1") using GCP deep learing VM, which does not use ubuntu-drivers
nvidia-smi || (echo "Run 'sudo ubuntu-drivers autoinstall; sudo apt -y install nvidia-cuda-toolkit' to install the NVIDIA drivers first"; exit "1")
sudo apt update -y && sudo apt upgrade -y
docker run hello-world || source ./install_docker.sh
which pip || (wget https://bootstrap.pypa.io/get-pip.py && sudo python get-pip.py; rm get-pip.py)
pip install -U tf2onnx tensorflow torch tritonclient[http] opencv-python-headless
