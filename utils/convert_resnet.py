# https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_1-model_deployment#model-2-text-recognition
import torch
from model import STRModel

# Create PyTorch Model Object
model = STRModel(input_channels=1, output_channels=512, num_classes=37)

# Load model weights from external file
state = torch.load("resnet.pth")
state = {key.replace("module.", ""): value for key, value in state.items()}
model.load_state_dict(state)

# Create ONNX file by tracing model
trace_input = torch.randn(1, 1, 32, 100)
torch.onnx.export(model, trace_input, "resnet.onnx", verbose=True)