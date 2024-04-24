import tritonclient.http as httpclient
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2Processor
from datasets import load_dataset
import torch

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# to be updated
librispeech_samples_ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

# load audio - to be updated
audio_input, sample_rate = sf.read(librispeech_samples_ds[0]["file"], dtype='float32')

# pad input values and return pt tensor
input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values

# setup client
client = httpclient.InferenceServerClient(url="localhost:8000")

inputs = [
    httpclient.InferInput("input", input_values.shape, "FP32")
]
inputs[0].set_data_from_numpy(input_values.numpy())

results = client.infer(model_name="speech_recognition", inputs=inputs)
print(results.get_response())
print(results.get_output("output"))
print(results.as_numpy("output"))
# retrieve logits & take argmax
# logits = results.logits
predicted_ids = torch.argmax(torch.tensor(results.as_numpy("output")), dim=-1)

# transcribe
transcription = processor.decode(predicted_ids[0])

print(transcription)

