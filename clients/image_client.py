# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import numpy as np
import tritonclient.http as httpclient
import time
import sys
from multiprocessing.connection import Connection
from enum import Enum
from utils import trace
from image_processing import detection_preprocessing, detection_postprocessing, recognition_postprocessing

SAVE_INTERMEDIATE_IMAGES = False

class Message(Enum):
    RELINQUISH_CPU = "RELINQUISH_CPU"
    ALLOCATE_CPU = "ALLOCATE_CPU"
    WAITING_FOR_CPU = "WAITING_FOR_CPU"
    FINISHED = "FINISHED"

class CPUState(Enum):
    CPU = "CPU"
    GPU = "GPU"
    WAITING_FOR_CPU = "WAITING_FOR_CPU"

@trace(__file__)
def wait_signal(process_id: int, signal_awaited: str, signal_pipe: Connection) -> None:
    if signal_pipe == None: # Not coordinating multiple processes
        return
    start = time.time()
    send_signal(process_id, Message.WAITING_FOR_CPU, signal_pipe) # tell scheduler that the process is waiting for CPU
    while True:
        receiver_id, signal_type = signal_pipe.recv()
        if receiver_id == process_id and signal_type == signal_awaited:
            break
    end = time.time()
    print(wait_signal.trace_prefix(), f"Process {process_id} waited for signal {signal_awaited} for {end - start: .5f}.")

@trace(__file__)
def send_signal(process_id, signal_to_send, signal_pipe: Connection):
    if signal_pipe == None: # Not coordinating multiple processes
        return
    print(send_signal.trace_prefix(), "Process %d sent signal %s." % (process_id, signal_to_send))
    signal_pipe.send((process_id, signal_to_send))

@trace(__file__)
def main(log_dir_name:str, image_paths, process_id, signal_pipe: Connection = None, t0: float = None):
    with open(log_dir_name+str(process_id).zfill(3)+".txt","a") as f:
        f.write(str(t0)+" process created\n")
        f.close()
    #### PREPROCESSING (CPU)
    wait_signal(process_id, Message.ALLOCATE_CPU, signal_pipe)

    t1 = time.time()
    with open(log_dir_name+str(process_id).zfill(3)+".txt","a") as f:
        f.write(str(t1)+" preprocessing started\n")
        f.close()
    print(main.trace_prefix(), f"Process {process_id}: PREPROCESSING start at {time.strftime('%H:%M:%S.')}")


    raw_images = []
    for path in image_paths:
        raw_images.append(cv2.imread(path))

    preprocessed_images = []
    for raw_image in raw_images:
        preprocessed_images.append(detection_preprocessing(raw_image)[0]) # (1, 480, 640, 3), 1 being batch size
    preprocessed_images = np.stack(preprocessed_images,axis=0) # matching dimension: (batch_size, 480, 640, 3)

    t2 = time.time()
    with open(log_dir_name+str(process_id).zfill(3)+".txt","a") as f:
        f.write(str(t2)+" preprocessing ended, detection inference started\n")
        f.close()
    # print("Detection preprocessing succeeded, took %.5f ms." % (t2 - t1))
    send_signal(process_id, Message.RELINQUISH_CPU, signal_pipe) # Parent can now schedule another CPU task
    print(main.trace_prefix(), f"Process {process_id}: PREPROCESSING finish, DETECTION INFERENCE start at {time.strftime('%H:%M:%S')}")

    #### DETECTION INFERENCE (GPU)
    client = httpclient.InferenceServerClient(url="localhost:8000")
    detection_input = httpclient.InferInput(
        "input_images:0", preprocessed_images.shape, datatype="FP32" 
    )
    detection_input.set_data_from_numpy(preprocessed_images, binary_data=True)

    
    detection_response = client.infer(
        model_name="text_detection", inputs=[detection_input]
    )
    # Potentially multiple processes running on the CPU before this process gets blocked by wait_signal
    # This should be fine, as the task here is not compute-intensive
    t3 = time.time()
    with open(log_dir_name+str(process_id).zfill(3)+".txt","a") as f:
        f.write(str(t3)+" detection inference ended\n")
        f.close()
    # print("Text detection succeeded, took %.5f ms." % (t2 - t1))

    #### CROPPING (CPU)
    print(main.trace_prefix(), f"Process {process_id}: DETECTION INFERENCE finish at {time.strftime('%H:%M:%S.')}")

    wait_signal(process_id, Message.ALLOCATE_CPU, signal_pipe) 
    t4 = time.time()
    with open(log_dir_name+str(process_id).zfill(3)+".txt","a") as f:
        f.write(str(t4)+" cropping started\n")
        f.close()
    print(main.trace_prefix(), f"Process {process_id}: CROPPING start at {time.strftime('%H:%M:%S.')}")

    # Depending on parent to schedule the first process, i.e., the process that has waited the longest
    # FIFO policy. Assuming there is no priority among processes
    cropped_images = detection_postprocessing(detection_response,preprocessed_images)
    cropped_images = np.array(cropped_images, dtype=np.single)
    if cropped_images.shape[0] == 0:
        send_signal(process_id, Message.RELINQUISH_CPU, signal_pipe)
        end_time = time.time()
        print(main.trace_prefix(), f"Process {process_id}: CROPPING wrong, end early at {time.strftime('%H:%M:%S.')}")
        with open(log_dir_name+str(process_id).zfill(3)+".txt","w") as f:
            f.write("\n")
            f.write(str(end_time-t0)+" process length\n")
            f.write(str(t2-t1)+" preprocessing length\n")
            f.write(str(t3-t2)+" detection inference length\n")
            f.write("\n")
            f.write(str(t1-t0)+" waiting for preprocessing time\n")
            f.write(str(t4-t3)+" waiting for cropping time\n")
            f.close()
        return None
    
    t5 = time.time()
    with open(log_dir_name+str(process_id).zfill(3)+".txt","a") as f:
        f.write(str(t5)+" cropping ended, recognition inference started\n")
        f.close()

    print(main.trace_prefix(), f"Process {process_id}: CROPPING finish, RECOGNITION INFERENCE start at {time.strftime('%H:%M:%S.%f')}")
    send_signal(process_id, Message.RELINQUISH_CPU, signal_pipe)

    #### RECOGNITION INFERENCE (GPU)
    recognition_input = httpclient.InferInput(
        "input.1", cropped_images.shape, datatype="FP32"
    )
    recognition_input.set_data_from_numpy(cropped_images, binary_data=True)
    
    recognition_response = client.infer(
        model_name="text_recognition", inputs=[recognition_input]
    )
    t6 = time.time()
    with open(log_dir_name+str(process_id).zfill(3)+".txt","a") as f:
        f.write(str(t6)+" recognition inference ended\n")
        f.close()
    # print("Text recognition succeeded, took %.5f ms." % (t4 - t3))

    #### POSTPROCESSING (CPU)
    print(main.trace_prefix(), f"Process {process_id}: RECOGNITION INFERENCE finish at {time.strftime('%H:%M:%S.')}")
    wait_signal(process_id, Message.ALLOCATE_CPU, signal_pipe)
    print(main.trace_prefix(), f"Process {process_id}: POSTPROCESSING start at {time.strftime('%H:%M:%S.')}")
    t7 = time.time()
    with open(log_dir_name+str(process_id).zfill(3)+".txt","a") as f:
        f.write(str(t7)+" postprrocessing started\n")
        f.close()
    final_text = recognition_postprocessing(recognition_response.as_numpy("308"))
    send_signal(process_id, Message.FINISHED, signal_pipe)
    print(main.trace_prefix(), f"Process {process_id}: POSTPROCESSING finish at {time.strftime('%H:%M:%S.')}")
    t8 = time.time()
    print(final_text)

    with open(log_dir_name+str(process_id).zfill(3)+".txt","a") as f:
        f.write(str(t8)+" postprrocessing ended\n")
        f.write("\n")
        f.write(str(t8-t0)+" process length\n")
        f.write(str(t2-t1)+" preprocessing length\n")
        f.write(str(t3-t2)+" detection inference length\n")
        f.write(str(t5-t4)+" cropping length\n")
        f.write(str(t6-t5)+" recognition inference length\n")
        f.write(str(t8-t7)+" postprrocessing length\n")
        f.write("\n")
        f.write(str(t1-t0)+" waiting for preprocessing time\n")
        f.write(str(t4-t3)+" waiting for cropping time\n")
        f.write(str(t7-t6)+" waiting for postprrocessing time\n")
        f.close()
    

    return final_text

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

    final_text = main(log_path,image_paths, process_id)
    
    