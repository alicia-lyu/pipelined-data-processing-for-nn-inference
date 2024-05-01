import tritonclient.http as httpclient
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2Processor
import torch
import sys
from multiprocessing.connection import Connection
from enum import Enum
import time
from datasets import load_dataset

class Message(Enum):
    CPU_AVAILABLE = "CPU_AVAILABLE"
    WAITING_FOR_CPU = "WAITING_FOR_CPU"
    CREATE_PROCESS = "CREATE_PROCESS"

class Stage(Enum):
    NOT_START = "NOT_START"
    PREPROCESSING = "PREPROCESSING"
    RECOGNITION_INFERENCE = "RECOGNITION_INFERENCE"
    POSTPROCESSING = "POSTPROCESSING"

class CPUState(Enum):
    CPU = "CPU"
    GPU = "GPU"
    WAITING_FOR_CPU = "WAITING_FOR_CPU"

def wait_signal(process_id: int, signal_awaited: str, signal_pipe: Connection) -> None:
    if signal_pipe == None: # Not coordinating multiple processes
        return
    start = time.time()
    # send a signal that it is waiting
    while True:
        receiver_id, signal_type = signal_pipe.recv()
        if receiver_id == process_id and signal_type == signal_awaited:
            break
    end = time.time()

def send_signal(process_id, signal_to_send, signal_pipe: Connection):
    if signal_pipe == None: # Not coordinating multiple processes
        return
    signal_pipe.send((process_id, signal_to_send))

def audio_preprocess(audio_path, processor: Wav2Vec2Processor):
    audio_input, sample_rate = sf.read(audio_path, dtype='float32')
    # pad input values and return pt tensor
    input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
    return input_values

def audio_postprocess(results, processor: Wav2Vec2Processor):
    predicted_ids = torch.argmax(torch.tensor(results.as_numpy("output")), dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

def main(audio_paths, process_id, signal_pipe: Connection = None):

    t1 = time.time()
    print(f"t1: {t1}")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    # librispeech_samples_ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    # load audio - to be updated
    # audio_input, sample_rate = sf.read(librispeech_samples_ds[0]["file"], dtype='float32')
    audio_input, sample_rate = sf.read(audio_paths[0], dtype='float32')
    # pad input values and return pt tensor
    preprocessed_audio = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values


    # preprocessed_audios = []
    # for path in audio_paths:
    #     preprocessed_audios.append(audio_preprocess(path, processor))

    # preprocessed_audio = audio_preprocess(librispeech_samples_ds[0]["file"], processor)

    t2 = time.time()
    print(f"t2: {t2}")


    # setup client
    client = httpclient.InferenceServerClient(url="localhost:8000")
   
    infer_inputs = [
        httpclient.InferInput("input", preprocessed_audio.shape, datatype="FP32")
    ]
    infer_inputs[0].set_data_from_numpy(preprocessed_audio.numpy())

    
    t3 = time.time()
    print(f"t3: {t3}")
    # query server
    results = client.infer(model_name="speech_recognition", inputs=infer_inputs)

    t4 = time.time()
    print(f"t4: {t4}")

    transcription = audio_postprocess(results, processor)

    t5 = time.time()
    print(f"t5: {t5}")

    print(transcription)

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Not pipelined!")
        audio_paths = [
            "../../datasets/audio_data/wav16_data_1/common_voice_en_12192.wav",
            "../../datasets/audio_data/wav16_data_1/common_voice_en_12194.wav",
            "../../datasets/audio_data/wav16_data_1/common_voice_en_12195.wav"
        ]
        process_id = 0
    else:
        process_id = sys.argv[1]
        audio_paths = sys.argv[2:]
        print("Pipeline batch size: "+str(len(audio_paths))+"!")

    final_text = main(audio_paths, process_id)