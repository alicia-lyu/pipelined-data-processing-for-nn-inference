import os, cv2 # type: ignore
import numpy as np # type: ignore
import tritonclient.http as httpclient # type: ignore
import time
import sys
from multiprocessing.connection import Connection
from Scheduler import Message
from utils import trace
from image_processing import detection_preprocessing, detection_postprocessing, recognition_postprocessing
from Client import Client
from typing import Dict

SAVE_INTERMEDIATE_IMAGES = False

class TextRecognitionClient(Client):
    def __init__(self, log_dir_name, batch, process_id, t0: float, stats: Dict = {}, signal_pipe: Connection = None) -> None:
        self.t1 = None
        self.t2 = None
        self.t3 = None
        self.t4 = None
        self.t5 = None
        self.t6 = None
        self.t7 = None
        self.t8 = None
        super().__init__(log_dir_name, batch, process_id, t0, stats, signal_pipe)
    
    @trace(__file__)
    def preprocess(self):
        self.t1 = time.time()
        # print(self.preprocess.trace_prefix(), f"Process {self.process_id}: PREPROCESSING start at {time.strftime('%H:%M:%S.')}")
        raw_images = []
        for path in self.batch:
            raw_images.append(cv2.imread(path))

        preprocessed_images = []
        for raw_image in raw_images:
            preprocessed_images.append(detection_preprocessing(raw_image)[0]) # (1, 480, 640, 3), 1 being batch size
        preprocessed_images = np.stack(preprocessed_images,axis=0) # matching dimension: (batch_size, 480, 640, 3)
        self.t2 = time.time()
        # print(self.preprocess.trace_prefix(), f"Process {self.process_id}: PREPROCESSING finish, DETECTION INFERENCE start at {time.strftime('%H:%M:%S')}")
        return preprocessed_images

    @trace(__file__)
    def detect(self, preprocessed_images):
        detection_input = httpclient.InferInput(
            "input_images:0", preprocessed_images.shape, datatype="FP32" 
        )
        detection_input.set_data_from_numpy(preprocessed_images, binary_data=True)
        detection_response = self.triton_client.infer(
            model_name="text_detection", inputs=[detection_input]
        )
        self.t3 = time.time()
        # print(self.detect.trace_prefix(), f"Process {self.process_id}: DETECTION INFERENCE finish at {time.strftime('%H:%M:%S.')}")
        return detection_response

    @trace(__file__)
    def crop(self, detection_response, preprocessed_images):
        self.t4 = time.time()
        # print(self.crop.trace_prefix(), f"Process {self.process_id}: CROPPING start at {time.strftime('%H:%M:%S.')}")

        # Depending on parent to schedule the first process, i.e., the process that has waited the longest
        # FIFO policy. Assuming there is no priority among processes
        cropped_images = detection_postprocessing(detection_response,preprocessed_images)
        cropped_images = np.array(cropped_images, dtype=np.single)
        if cropped_images.shape[0] == 0:
            self.t5 = time.time()
            self.t6 = time.time()
            self.t7 = time.time()
            self.t8 = time.time()
            # print(self.crop.trace_prefix(), f"Process {self.process_id}: CROPPING returns no image, end early at {time.strftime('%H:%M:%S.')}")
            return
        self.t5 = time.time()
        # print(self.crop.trace_prefix(), f"Process {self.process_id}: CROPPING finish, RECOGNITION INFERENCE start at {time.strftime('%H:%M:%S.%f')}")
        return cropped_images

    @trace(__file__)
    def recognize(self, cropped_images):
        recognition_input = httpclient.InferInput(
            "input.1", cropped_images.shape, datatype="FP32"
        )
        recognition_input.set_data_from_numpy(cropped_images, binary_data=True)
        
        recognition_response = self.triton_client.infer(
            model_name="text_recognition", inputs=[recognition_input]
        )
        self.t6 = time.time()
        # print(self.recognize.trace_prefix(), f"Process {self.process_id}: RECOGNITION INFERENCE finish at {time.strftime('%H:%M:%S.')}")
        return recognition_response

    @trace(__file__)
    def postprocess(self, recognition_response):
        # print(self.postprocess.trace_prefix(), f"Process {self.process_id}: POSTPROCESSING start at {time.strftime('%H:%M:%S.')}")
        self.t7 = time.time()
        final_text = recognition_postprocessing(recognition_response.as_numpy("308"))
        # print(self.postprocess.trace_prefix(), f"Process {self.process_id}: POSTPROCESSING finish at {time.strftime('%H:%M:%S.')}")
        self.t8 = time.time()
        return final_text
    
    def run(self):
        #### PREPROCESSING (CPU)
        self.wait_signal(Message.ALLOCATE_CPU)
        preprocessed_images = self.preprocess()
        self.send_signal(Message.RELINQUISH_CPU) # Parent can now schedule another CPU task

        #### DETECTION INFERENCE (GPU)
        detection_response = self.detect(preprocessed_images)
        # Potentially multiple processes running on the CPU before this process gets blocked by wait_signal
        # This should be fine, as the task here is not compute-intensive

        #### CROPPING (CPU)
        self.wait_signal(Message.ALLOCATE_CPU) 
        cropped_images = self.crop(detection_response, preprocessed_images)
        if cropped_images is None:
            self.log()
            self.send_signal(Message.FINISHED)
            return self.stats
        self.send_signal(Message.RELINQUISH_CPU)

        #### RECOGNITION INFERENCE (GPU)
        recognition_response = self.recognize(cropped_images)

        #### POSTPROCESSING (CPU)
        self.wait_signal(Message.ALLOCATE_CPU)
        final_text = self.postprocess(recognition_response)
        print(self.process_id, final_text)
        self.send_signal(Message.FINISHED)

        self.log()

        return self.stats

    def log(self) -> bool:
        self.stats["preprocess_start"] = self.t1
        self.stats["preprocess_end"] = self.t2
        self.stats["inference_start"] = self.t2
        self.stats["inference_end"] = self.t3
        self.stats["midprocessing_start"] = self.t4
        self.stats["midprocessing_end"] = self.t5
        self.stats["inference2_start"] = self.t5
        self.stats["inference2_end"] = self.t6
        self.stats["postprocess_start"] = self.t7
        self.stats["postprocess_end"] = self.t8
        with open(self.filename, "w") as f:
            f.write(str(self.t0) + " process created\n")
            f.write(str(self.t1)+" preprocessing started\n")
            f.write(str(self.t2)+" preprocessing ended, detection inference started\n")
            f.write(str(self.t3)+" detection inference ended\n")
            f.write(str(self.t4)+" cropping started\n")
            f.write(str(self.t5)+" cropping ended, recognition inference started\n")
            f.write(str(self.t6)+" recognition inference ended\n")
            f.write(str(self.t7)+" postprrocessing started\n")
            f.write(str(self.t8)+" postprrocessing ended\n")
            f.write("\n")
            f.write(str(self.t8-self.t0)+" process length\n")
            f.write(str(self.t2-self.t1)+" preprocessing length\n")
            f.write(str(self.t3-self.t2)+" detection inference length\n")
            f.write(str(self.t5-self.t4)+" cropping length\n")
            f.write(str(self.t6-self.t5)+" recognition inference length\n")
            f.write(str(self.t8-self.t7)+" postprrocessing length\n")
            f.write("\n")
            f.write(str(self.t1-self.t0)+" waiting for preprocessing time\n")
            f.write(str(self.t4-self.t3)+" waiting for cropping time\n")
            f.write(str(self.t7-self.t6)+" waiting for postprrocessing time\n")
            f.close()
        return True

if __name__ == "__main__":

    if len(sys.argv) < 2:
        # print("Not pipelined!")
        batch = [
            "../../datasets/SceneTrialTrain/lfsosa_12.08.2002/IMG_2617.JPG",
            "../../datasets/SceneTrialTrain/lfsosa_12.08.2002/IMG_2618.JPG"
        ]
        process_id = 0
        start_time = time.time()
        log_path = "../log_image/test_"+str(start_time)+"/"
        os.makedirs(log_path, exist_ok=True)
    else:
        log_path = sys.argv[1]
        process_id = sys.argv[2]
        batch = sys.argv[3:]
        # print("Pipeline batch size: "+str(len(batch))+"!")

    client = TextRecognitionClient(log_path, batch, process_id)
    final_text = client.run()
    
    