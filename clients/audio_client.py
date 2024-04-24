import tritonclient.http as httpclient
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2Processor
from datasets import load_dataset
# import torch

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

librispeech_samples_ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

# load audio
audio_input, sample_rate = sf.read(librispeech_samples_ds[0]["file"])

# pad input values and return pt tensor
input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values

#Setup client
client = httpclient.InferenceServerClient(url="localhost:8000")

# audio_data, sample_rate = sf.read("1.wav", dtype='float32')

inputs = [
    httpclient.InferInput("input", audio_input.shape, "FP32").set_data_from_numpy(audio_input)
]
# inputs[0].set_data_from_numpy(audio_data)

#outputs = [
#    httpclient.InferRequestedOutput("output")
#] 

results = client.infer(model_name="speech_recognition", inputs=inputs) #, outputs=outputs)
print(results)

# retrieve logits & take argmax
# logits = results.logits
# predicted_ids = torch.argmax(logits, dim=-1)

# transcribe
# transcription = processor.decode(predicted_ids[0])

# print(transcription)

# output_data = results.as_numpy("output")
# print("Inference output:", output_data)

