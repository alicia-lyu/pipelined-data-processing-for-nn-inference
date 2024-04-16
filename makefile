BASEDIR=$(shell cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)/triton-tutorial
MODEL_REPO=$(BASEDIR)/model_repository
OPEN_CV=$(MODEL_REPO)/text_detection/1/model.onnx
RESNET=$(MODEL_REPO)/text_recognition/1/model.onnx
IMAGE_CLIENT1=$(BASEDIR)/client.py

$(OPEN_CV):
	wget https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz
	tar -xvf frozen_east_text_detection.tar.gz
	rm frozen_east_text_detection.tar.gz
	python -m tf2onnx.convert --input frozen_east_text_detection.pb --inputs "input_images:0" --outputs "feature_fusion/Conv_7/Sigmoid:0","feature_fusion/concat_3:0" --output detection.onnx
	mkdir -p $(MODEL_REPO)/text_detection/1
	mv detection.onnx $(OPEN_CV)

$(RESNET)/text_recognition/1/model.onnx:
	wget https://www.dropbox.com/sh/j3xmli4di1zuv3s/AABzCC1KGbIRe2wRwa3diWKwa/None-ResNet-None-CTC.pth -O resnet.pth
	python convert_resnet.py
	rm resnet.pth
	mkdir -p $(MODEL_REPO)/text_recognition/1
	mv resnet.onnx $(RESNET)

restart-triton-server:
	docker stop $(shell docker ps -q --filter ancestor=triton-image) || echo "No Triton server running"
	make triton-server

triton-server: $(OPEN_CV) $(RESNET)
	docker build -t triton-image .
	docker run -d --gpus=all --shm-size=256m --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v "$(MODEL_REPO):/models" triton-image || echo "Triton server already running"

test-image: $(IMAGE_CLIENT1) triton-server
	cd $(BASEDIR)
	python ./client.py