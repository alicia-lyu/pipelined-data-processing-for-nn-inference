BASEDIR=$(shell cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)
MODEL_REPO=$(BASEDIR)/model_repository
OPEN_CV=$(MODEL_REPO)/text_detection/1/model.onnx
RESNET=$(MODEL_REPO)/text_recognition/1/model.onnx
WAV2VEC=$(MODEL_REPO)/speech_recognition/1/model.onnx
IMAGE_CLIENT1=$(BASEDIR)/clients/image_client.py
MIN_INTERVAL := 1
MAX_INTERVAL := 5
BATCH_SIZE := 2

$(OPEN_CV):
	wget https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz
	tar -xvf frozen_east_text_detection.tar.gz
	rm frozen_east_text_detection.tar.gz
	python -m tf2onnx.convert --input frozen_east_text_detection.pb --inputs "input_images:0" --outputs "feature_fusion/Conv_7/Sigmoid:0","feature_fusion/concat_3:0" --output detection.onnx
	mkdir -p $(MODEL_REPO)/text_detection/1
	mv detection.onnx $(OPEN_CV)
	rm frozen_east_text_detection.pb

$(RESNET):
	wget https://www.dropbox.com/sh/j3xmli4di1zuv3s/AABzCC1KGbIRe2wRwa3diWKwa/None-ResNet-None-CTC.pth -O resnet.pth
	python utils/convert_resnet.py; rm resnet.pth
	mkdir -p $(MODEL_REPO)/text_recognition/1
	mv resnet.onnx $(RESNET)

$(WAV2VEC):
	python utils/convert_wav2vec.py
	mkdir -p $(MODEL_REPO)/speech_recognition/1
	mv wav2vec2.onnx $(WAV2VEC)

./datasets:
	mkdir -p ./datasets

ic03.zip:
	wget http://www.iapr-tc11.org/dataset/ICDAR2003_RobustReading/TrialTrain/scene.zip -o ic03.zip

./datasets/ic03: ic03.zip
	mkdir -p ./datasets/ic03
	unzip ic03.zip -d ./datasets/ic03
	rm ic03.zip

restart-triton-server:
	docker stop $(shell docker ps -q --filter ancestor=triton-image) || echo "No Triton server running"
	make triton-server

triton-server: $(OPEN_CV) $(RESNET) $(WAV2VEC)
	docker build -t triton-image .
	docker run --gpus=all --shm-size=256m --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v "$(MODEL_REPO):/models" triton-image || echo "Triton server already running"

test-image: $(IMAGE_CLIENT1) triton-server
	cd $(BASEDIR)/clients && python ./image_client.py

test-image-subprocesses: $(IMAGE_CLIENT1) triton-server
	cd $(BASEDIR)/clients && python ./image_subprocesses.py --min=$(MIN_INTERVAL) --max=$(MAX_INTERVAL) --batch_size=$(BATCH_SIZE)

test-audio: triton-server
	cd $(BASEDIR)/clients && python ./audio_client.py