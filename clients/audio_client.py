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
    print(f"max cols: {max_cols}")
  
    # Padding to make sizes compatible
    padded_data = [torch.nn.functional.pad(tensor, (0, max_cols - tensor.size(-1))) for tensor in audios]
    print(padded_data)

    # Stack padded tensors
    stacked_data = torch.stack((tensor[0] for tensor in padded_data), dim=0)
    print(stacked_data)
    print(stacked_data.shape)

    return stacked_data

def audio_postprocess(results, processor: Wav2Vec2Processor):
    predicted_ids = torch.argmax(torch.tensor(results.as_numpy("output")), dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

def main(audio_paths, process_id, signal_pipe: Connection = None):
    print(audio_paths)
    t1 = time.time()
    print(f"t1: {t1}")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    # preprocessed_audios = []
    # for path in audio_paths:
    #     preprocessed_audios.append(audio_preprocess(path, processor))
    # preprocessed_audios = torch.stack(preprocessed_audios)
    # preprocessed_audio = audio_preprocess(audio_paths[0], processor)

    preprocessed_audios = audio_preprocess(audio_paths, processor)
    print(preprocessed_audios)
    print(len(preprocessed_audios))
    t2 = time.time()
    print(f"t2: {t2}")


    # setup client
    client = httpclient.InferenceServerClient(url="localhost:8000")
   
    infer_inputs = [
        httpclient.InferInput("input", preprocessed_audios.shape, datatype="FP32")
    ]
    infer_inputs[0].set_data_from_numpy(preprocessed_audios.numpy())

    
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
            "../../datasets/audio_data/mp3_16_data_2/common_voice_en_137150_16kHz.mp3",
            "../../datasets/audio_data/mp3_16_data_2/common_voice_en_100229_16kHz.mp3"
        ]
        process_id = 0
    else:
        process_id = sys.argv[1]
        audio_paths = sys.argv[2:]
        print("Pipeline batch size: "+str(len(audio_paths))+"!")

    final_text = main(audio_paths, process_id)