BASEDIR=$(shell cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)
MODEL_REPO=$(BASEDIR)/triton-tutorial/model_repository
OPEN_CV=$(MODEL_REPO)/text_detection/1/model.onnx
RESNET=$(MODEL_REPO)/text_recognition/1/model.onnx

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

run-triton: $(OPEN_CV) $(RESNET)
	echo "Running triton server with models $(OPEN_CV) and $(RESNET)"
	docker-compose build
	docker-compose up
