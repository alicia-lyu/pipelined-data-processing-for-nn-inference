FROM nvcr.io/nvidia/tritonserver:24.03-py3

CMD ["tritonserver", "--model-repository=/models", "--model-control-mode=poll"]