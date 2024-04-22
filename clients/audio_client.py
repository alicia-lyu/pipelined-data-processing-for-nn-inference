import tritonclient.http as httpclient
import numpy as np

#Setup client
client = httpclient.InferenceServerClient(url="localhost:8000")

sample_rate = 8
audio_data = np.random.randn(sample_rate).astype(np.float32)

inputs = [
    httpclient.InferInput("input", audio_data.shape, "FP32")
]
inputs[0].set_data_from_numpy(audio_data)

outputs = [
    httpclient.InferRequestedOutput("output")
] 

results = client.infer(model_name="speech_recognition", inputs=inputs, outputs=outputs)
output_data = results.as_numpy("output")
print("Inference output:", output_data)

