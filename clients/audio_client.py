import tritonclient.http as httpclient
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2Processor
from datasets import load_dataset
# import torch

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# librispeech_samples_ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation") # TODO: pending download from http://www.openslr.org/12

# load audio
audio_input, sample_rate = sf.read("1.wav", dtype='float32') # TODO: this audio file won't work because it is not sampled at 16kHz. Download from the link above

# pad input values and return pt tensor
input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
print("input values:", input_values.shape)

#Setup client
client = httpclient.InferenceServerClient(url="localhost:8000")

# audio_data, sample_rate = sf.read("1.wav", dtype='float32')

audio_input = [
    httpclient.InferInput("input", input_values.shape, "FP32")
]
audio_input.set_data_from_numpy(input_values)

#outputs = [
#    httpclient.InferRequestedOutput("output")
#] 

results = client.infer(model_name="speech_recognition", inputs=[audio_input]) #, outputs=outputs)
print(results.get_response())
print(results.get_output("output"))
print(results.as_numpy("output"))

# retrieve logits & take argmax
# logits = results.logits
# predicted_ids = torch.argmax(logits, dim=-1)

# transcribe
# transcription = processor.decode(predicted_ids[0])

# print(transcription)

# output_data = results.as_numpy("output")
# print("Inference output:", output_data)

