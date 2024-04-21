docker run --gpus=all -it --shm-size=256m -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:24.03-py3
pip install torchvision opencv-python-headless
tritonserver --model-repository=/models
