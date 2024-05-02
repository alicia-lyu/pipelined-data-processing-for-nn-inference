import tritonclient.http as httpclient
import soundfile as sf
from transformers import Wav2Vec2Processor
import torch
import sys, os
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

class AudioRecognitionClient:
    def __init__(self, log_dir_name, audio_paths, process_id, signal_pipe: Connection = None, t0: float = None):
        self.log_dir_name = log_dir_name
        self.audio_paths = audio_paths
        self.process_id = process_id
        self.signal_pipe = signal_pipe
        if t0 is None:
            self.t0 = time.time()
        else:
            self.t0 = t0
        self.t1 = None
        self.t2 = None
        self.t3 = None
        self.t4 = None
        self.t5 = None

    def wait_signal(self, signal_awaited: str) -> None:
        if self.signal_pipe is None:  # Not coordinating multiple processes
            return
        while True:
            receiver_id, signal_type = self.signal_pipe.recv()
            assert(receiver_id == self.process_id)
            if signal_type == signal_awaited:
                break

    def send_signal(self, signal_to_send):
        if self.signal_pipe is None:  # Not coordinating multiple processes
            return
        self.signal_pipe.send((self.process_id, signal_to_send))

    def audio_preprocess(self, processor: Wav2Vec2Processor):
        self.t1 = time.time()
        audios = []
        for path in self.audio_paths:
            audio_input, sample_rate = sf.read(path, dtype='float32')
            input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
            audios.append(input_values)

        max_cols = max(tensor[0].size(0) for tensor in audios)
        padded_data = [torch.nn.functional.pad(tensor, (0, max_cols - tensor.size(-1))) for tensor in audios]

        reduced_dim = [tensor[0] for tensor in padded_data]
        stacked_data = torch.stack(reduced_dim, dim=0)
        self.t2 = time.time()
        return stacked_data

    def audio_postprocess(self, results, processor: Wav2Vec2Processor):
        self.t4 = time.time()
        transcriptions = []
        predicted_ids = torch.argmax(torch.tensor(results.as_numpy("output")), dim=-1)
        for prediction in predicted_ids:
            transcriptions.append(processor.decode(prediction))
        self.t5 = time.time()
        return transcriptions

    def run(self):
        self.wait_signal("ALLOCATE_CPU")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        preprocessed_audios = self.audio_preprocess(processor)
        self.send_signal("RELINQUISH_CPU")

        triton_client = httpclient.InferenceServerClient(url="localhost:8000")
        infer_inputs = [httpclient.InferInput("input", preprocessed_audios.shape, datatype="FP32")]
        infer_inputs[0].set_data_from_numpy(preprocessed_audios.numpy())

        self.wait_signal("ALLOCATE_CPU")
        self.t3 = time.time()
        results = triton_client.infer(model_name="speech_recognition", inputs=infer_inputs)
        self.send_signal("RELINQUISH_CPU")

        transcriptions = self.audio_postprocess(results, processor)
        return transcriptions

    def log(self):
        with open(os.path.join(self.log_dir_name, f"{self.process_id:03}.txt"), "w") as f:
            f.write(f"{self.t0} process created\n")
            f.write(f"{self.t1} preprocessing started\n")
            f.write(f"{self.t2} preprocessing ended, inference started\n")
            f.write(f"{self.t3} inference ended\n")
            f.write(f"{self.t4} postprocessing started\n")
            f.write(f"{self.t5} postprocessing ended\n")
            f.write(f"\n")
            f.write(f"{self.t5 - self.t0} process length\n")
            f.write(f"{self.t2 - self.t1} preprocessing length\n")
            f.write(f"{self.t3 - self.t2} inference length\n")
            f.write(f"{self.t5 - self.t4} postprocessing length\n")
            f.write("\n")
            f.write(f"{self.t1 - self.t0} waiting for preprocessing time\n")
            f.write(f"{self.t4 - self.t3} waiting for postprocessing time\n")

if __name__ == "__main__":

    audio_paths = [
        "../../datasets/audio_data/mp3_16_data_2/common_voice_en_100229_16kHz.mp3",
        "../../datasets/audio_data/mp3_16_data_2/common_voice_en_137150_16kHz.mp3"
    ]

    if len(sys.argv) < 2:
        print("Not pipelined!")
        start_time = time.time()
        log_path = "../log_audio/test_"+str(start_time)+"/"
        process_id = 0
        os.makedirs(log_path, exist_ok=True)
    else:
        log_path = sys.argv[1]
        process_id = sys.argv[2]
        audio_paths = sys.argv[3:]
        print("Pipeline batch size: "+str(len(audio_paths))+"!")
    
    client = AudioRecognitionClient(log_path, audio_paths, process_id)
    transcriptions = client.run()
    print(transcriptions)