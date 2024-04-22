import tritonclient.http as httpclient
import numpy as np

#Setup client
client = httpclient.InferenceServerClient(url="localhost:8000")

sample_rate = 16000
audio_data = np.random.randn(sample_rate).astype(np.float32)

inputs = [
    httpclient.InferInput("input.1", audio_data.shape, "FP32")
]
inputs[0].set_data_from_numpy(audio_data)

# outputs = [
#     httpclient.InferRequestedOutput("audio_output")
# ] 

results = client.infer(model_name="speech_recognition", inputs=inputs)
output_data = results.as_numpy("308")
print("Inference output:", output_data)

