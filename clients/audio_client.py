import tritonclient.http as httpclient
import numpy as np
import soundfile as sf

#Setup client
client = httpclient.InferenceServerClient(url="localhost:8000")

# sample_rate = 16000
audio_data, sample_rate = sf.read("1.wav")

# np.random.randn(sample_rate).astype(np.float32)

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

