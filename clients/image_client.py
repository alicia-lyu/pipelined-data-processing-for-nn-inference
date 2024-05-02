import os, cv2
import numpy as np
import tritonclient.http as httpclient
import time
import sys
from multiprocessing.connection import Connection
from Scheduler import Message
from utils import trace
from image_processing import detection_preprocessing, detection_postprocessing, recognition_postprocessing

SAVE_INTERMEDIATE_IMAGES = False

class TextRecognitionClient:
    def __init__(self, log_dir_name, image_paths, process_id, signal_pipe: Connection = None, t0: float = None) -> None:
        self.filename = log_dir_name + str(process_id).zfill(3) + ".txt"
        self.image_paths = image_paths
        self.process_id = process_id
        self.triton_client = httpclient.InferenceServerClient(url="localhost:8000")
        self.pipe = signal_pipe
        if t0 is None:
            self.t0 = time.time()
        else:
            self.t0 = t0
        self.t1 = None
        self.t2 = None
        self.t3 = None
        self.t4 = None
        self.t5 = None
        self.t6 = None
        self.t7 = None
        self.t8 = None
    
    @trace(__file__)
    def preprocess(self):
        self.t1 = time.time()
        print(self.preprocess.trace_prefix(), f"Process {self.process_id}: PREPROCESSING start at {time.strftime('%H:%M:%S.')}")
        raw_images = []
        for path in self.image_paths:
            raw_images.append(cv2.imread(path))

        preprocessed_images = []
        for raw_image in raw_images:
            preprocessed_images.append(detection_preprocessing(raw_image)[0]) # (1, 480, 640, 3), 1 being batch size
        preprocessed_images = np.stack(preprocessed_images,axis=0) # matching dimension: (batch_size, 480, 640, 3)
        self.t2 = time.time()
        print(self.preprocess.trace_prefix(), f"Process {self.process_id}: PREPROCESSING finish, DETECTION INFERENCE start at {time.strftime('%H:%M:%S')}")
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
        print(self.detect.trace_prefix(), f"Process {self.process_id}: DETECTION INFERENCE finish at {time.strftime('%H:%M:%S.')}")
        return detection_response

    @trace(__file__)
    def crop(self, detection_response, preprocessed_images):
        self.t4 = time.time()
        print(self.crop.trace_prefix(), f"Process {self.process_id}: CROPPING start at {time.strftime('%H:%M:%S.')}")

        # Depending on parent to schedule the first process, i.e., the process that has waited the longest
        # FIFO policy. Assuming there is no priority among processes
        cropped_images = detection_postprocessing(detection_response,preprocessed_images)
        cropped_images = np.array(cropped_images, dtype=np.single)
        if cropped_images.shape[0] == 0:
            self.t5 = time.time()
            self.t6 = time.time()
            self.t7 = time.time()
            self.t8 = time.time()
            print(self.crop.trace_prefix(), f"Process {self.process_id}: CROPPING returns no image, end early at {time.strftime('%H:%M:%S.')}")
            return
        self.t5 = time.time()
        print(self.crop.trace_prefix(), f"Process {self.process_id}: CROPPING finish, RECOGNITION INFERENCE start at {time.strftime('%H:%M:%S.%f')}")
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
        print(self.recognize.trace_prefix(), f"Process {self.process_id}: RECOGNITION INFERENCE finish at {time.strftime('%H:%M:%S.')}")
        return recognition_response

    @trace(__file__)
    def postprocess(self, recognition_response):
        print(self.postprocess.trace_prefix(), f"Process {self.process_id}: POSTPROCESSING start at {time.strftime('%H:%M:%S.')}")
        self.t7 = time.time()
        final_text = recognition_postprocessing(recognition_response.as_numpy("308"))
        print(self.postprocess.trace_prefix(), f"Process {self.process_id}: POSTPROCESSING finish at {time.strftime('%H:%M:%S.')}")
        self.t8 = time.time()
        print(final_text)
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
            return []
        self.send_signal(Message.RELINQUISH_CPU)

        #### RECOGNITION INFERENCE (GPU)
        recognition_response = self.recognize(cropped_images)

        #### POSTPROCESSING (CPU)
        self.wait_signal(Message.ALLOCATE_CPU)
        final_text = self.postprocess(recognition_response)
        self.send_signal(Message.FINISHED)

        self.log()

        return final_text

    def log(self) -> None:
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

    @trace(__file__)
    def wait_signal(self, signal_awaited: str) -> None:
        if self.pipe == None: # Not coordinating multiple processes
            return
        start = time.time()
        self.send_signal(Message.WAITING_FOR_CPU) # tell scheduler that the process is waiting for CPU
        while True:
            receiver_id, signal_type = self.pipe.recv()
            if receiver_id == self.process_id and signal_type == signal_awaited:
                break
        end = time.time()
        print(self.wait_signal.trace_prefix(), f"Process {self.process_id} waited for signal {signal_awaited} for {end - start: .5f}.")

    @trace(__file__)
    def send_signal(self, signal_to_send):
        if self.pipe == None: # Not coordinating multiple processes
            return
        print(self.send_signal.trace_prefix(), "Process %d sent signal %s." % (self.process_id, signal_to_send))
        self.pipe.send((self.process_id, signal_to_send))

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Not pipelined!")
        image_paths = [
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
        image_paths = sys.argv[3:]
        print("Pipeline batch size: "+str(len(image_paths))+"!")

    client = TextRecognitionClient(log_path, image_paths, process_id)
    final_text = client.run()
    
    