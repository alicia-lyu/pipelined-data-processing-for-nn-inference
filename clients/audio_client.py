import tritonclient.http as httpclient # type: ignore
import soundfile as sf # type: ignore
from transformers import Wav2Vec2Processor # type: ignore
import torch # type: ignore
import sys, os
from multiprocessing.connection import Connection
import time
from utils import trace
from Scheduler import Message
from Client import Client
from typing import Dict

class AudioRecognitionClient(Client):
    def __init__(self, log_dir_name, batch, process_id, t0: float, stats: Dict = {}, signal_pipe: Connection = None) -> None:
        self.t1 = None
        self.t2 = None
        self.t3 = None
        self.t4 = None
        self.t5 = None
        super().__init__(log_dir_name, batch, process_id, t0, stats, signal_pipe)

    @trace(__file__)
    def audio_preprocess(self, processor: Wav2Vec2Processor):
        # print(self.audio_preprocess.trace_prefix(), f"Process {self.process_id}: PREPROCESSING start at {time.strftime('%H:%M:%S.')}")
        self.t1 = time.time()
        audios = []
        for path in self.batch:
            audio_input, sample_rate = sf.read(path, dtype='float32')
            input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
            audios.append(input_values)

        max_cols = max(tensor[0].size(0) for tensor in audios)
        padded_data = [torch.nn.functional.pad(tensor, (0, max_cols - tensor.size(-1))) for tensor in audios]

        reduced_dim = [tensor[0] for tensor in padded_data]
        stacked_data = torch.stack(reduced_dim, dim=0)
        self.t2 = time.time()
        # print(self.audio_preprocess.trace_prefix(), f"Process {self.process_id}: PREPROCESSING finish at {time.strftime('%H:%M:%S.')}")
        return stacked_data

    @trace(__file__)
    def audio_postprocess(self, results, processor: Wav2Vec2Processor):
        # print(self.audio_postprocess.trace_prefix(), f"Process {self.process_id}: POSTPROCESSING start at {time.strftime('%H:%M:%S.')}")
        self.t4 = time.time()
        transcriptions = []
        predicted_ids = torch.argmax(torch.tensor(results.as_numpy("output")), dim=-1)
        for prediction in predicted_ids:
            transcriptions.append(processor.decode(prediction))
        self.t5 = time.time()
        # print(self.audio_postprocess.trace_prefix(), f"Process {self.process_id}: POSTPROCESSING finish at {time.strftime('%H:%M:%S.')}")
        return transcriptions

    @trace(__file__)
    def run(self):
        self.wait_signal(Message.ALLOCATE_CPU)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        preprocessed_audios = self.audio_preprocess(processor)
        self.send_signal(Message.RELINQUISH_CPU)

        # print(self.run.trace_prefix(), f"Process {self.process_id}: INFERENCE start at {time.strftime('%H:%M:%S.')}")
        infer_inputs = [httpclient.InferInput("input", preprocessed_audios.shape, datatype="FP32")]
        infer_inputs[0].set_data_from_numpy(preprocessed_audios.numpy())
        self.t3 = time.time()
        results = self.triton_client.infer(model_name="speech_recognition", inputs=infer_inputs)
        # print(self.run.trace_prefix(), f"Process {self.process_id}: INFERENCE finish at {time.strftime('%H:%M:%S.')}")

        self.wait_signal(Message.ALLOCATE_CPU)
        transcriptions = self.audio_postprocess(results, processor)
        self.send_signal(Message.FINISHED)

        self.log()
        print(self.process_id, transcriptions)
        super().run()
        return self.stats

    def log(self):
        self.stats["preprocess_start"] = self.t1
        self.stats["preprocess_end"] = self.t2
        self.stats["inference_start"] = self.t2
        self.stats["inference_end"] = self.t3
        self.stats["postprocess_start"] = self.t4
        self.stats["postprocess_end"] = self.t5
        with open(self.filename, "w") as f:
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
            f.close()

if __name__ == "__main__":

    batch = [
        "../../datasets/audio_data/mp3_16_data_2/common_voice_en_100229_16kHz.mp3",
        "../../datasets/audio_data/mp3_16_data_2/common_voice_en_137150_16kHz.mp3"
    ]

    if len(sys.argv) < 2:
        # print("Not pipelined!")
        start_time = time.time()
        log_path = "../log_audio/test_"+str(start_time)+"/"
        process_id = 0
        os.makedirs(log_path, exist_ok=True)
    else:
        log_path = sys.argv[1]
        process_id = sys.argv[2]
        batch = sys.argv[3:]
        # print("Pipeline batch size: "+str(len(batch))+"!")
    
    client = AudioRecognitionClient(log_path, batch, process_id)
    transcriptions = client.run()
    # print(transcriptions)