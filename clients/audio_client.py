import tritonclient.http as httpclient
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2Processor
import torch
import sys
from multiprocessing.connection import Connection
from enum import Enum
import time

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

def audio_preprocess(audio_paths, processor: Wav2Vec2Processor):

    audios = []
    for path in audio_paths:
        audio_input, sample_rate = sf.read(path, dtype='float32')
        input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
        audios.append(input_values)

    max_cols = max(tensor[0].size(0) for tensor in audios)
  
    # Padding to make sizes compatible
    padded_data = [torch.nn.functional.pad(tensor, (0, max_cols - tensor.size(-1))) for tensor in audios]

    reduced_dim = []
    for tensor in padded_data:
        reduced_dim.append(tensor[0])

    # Stack padded tensors
    stacked_data = torch.stack(reduced_dim, dim=0)
    return stacked_data

def audio_postprocess(results, processor: Wav2Vec2Processor):
    transcriptions = []
    predicted_ids = torch.argmax(torch.tensor(results.as_numpy("output")), dim=-1)
    for prediction in predicted_ids:
        transcriptions.append(processor.decode(prediction))
    return transcriptions

def main(log_dir_name:str, audio_paths, process_id, signal_pipe: Connection = None):
    
    t1 = time.time()
    # print(f"t1: {t1}")

    # --- wait for CPU ---

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    preprocessed_audios = audio_preprocess(audio_paths, processor)
    t2 = time.time()
    # print(f"t2: {t2}")

    # --- relinquish CPU ---

    # setup client
    client = httpclient.InferenceServerClient(url="localhost:8000")
   
    infer_inputs = [
        httpclient.InferInput("input", preprocessed_audios.shape, datatype="FP32")
    ]
    infer_inputs[0].set_data_from_numpy(preprocessed_audios.numpy())

    t3 = time.time()
    # print(f"t3: {t3}")

    # query server
    results = client.infer(model_name="speech_recognition", inputs=infer_inputs)

    t4 = time.time()
    # print(f"t4: {t4}")

    transcriptions = audio_postprocess(results, processor)

    t5 = time.time()
    # print(f"t5: {t5}")

    for t in transcriptions:
        print(t)

    return transcriptions

if __name__ == "__main__":

    audio_paths = [
        "../../datasets/audio_data/mp3_16_data_2/common_voice_en_100229_16kHz.mp3",
        "../../datasets/audio_data/mp3_16_data_2/common_voice_en_137150_16kHz.mp3"
    ]

    if len(sys.argv) < 2:
        print("Not pipelined!")
        process_id = 0
    else:
        log_path = sys.argv[1]
        process_id = sys.argv[2]
        audio_paths = sys.argv[3:]
        print("Pipeline batch size: "+str(len(audio_paths))+"!")

    final_text = main(log_path, audio_paths, process_id)