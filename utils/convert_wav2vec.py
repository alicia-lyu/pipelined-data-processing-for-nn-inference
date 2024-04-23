import torch
from transformers import Wav2Vec2ForCTC

model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

audio_len = 250000
dummy_input = torch.randn(1, audio_len, requires_grad=True)

torch.onnx.export(model,  # model being run
                  dummy_input,  # model input (or a tuple for multiple inputs)
                  "wav2vec2.onnx",  # where to save the model
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=14,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': [0],'output': [0]}) 