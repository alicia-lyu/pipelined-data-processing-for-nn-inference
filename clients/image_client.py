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

import math

import cv2
import numpy as np
import tritonclient.http as httpclient
import time
import sys
from multiprocessing.connection import Connection
from enum import Enum
from utils import trace

SAVE_INTERMEDIATE_IMAGES = False

class Message(Enum):
    CPU_AVAILABLE = 0
    WAITING_FOR_CPU = 1
    CREATE_PROCESS = 2

class Stage(Enum):
    NOT_START = 0
    PREPROCESSING = 1
    DETECTION_INFERENCE = 2
    CROPPING = 3
    RECOGNITION_INFERENCE = 4
    POSTPROCESSING = 5

class CPUState(Enum):
    CPU = 0
    GPU = 1
    WAITING_FOR_CPU = 2

def detection_preprocessing(image: cv2.Mat) -> np.ndarray:
    inpWidth = 640
    inpHeight = 480

    # pre-process image
    blob = cv2.dnn.blobFromImage(
        image, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False
    )
    blob = np.transpose(blob, (0, 2, 3, 1))
    return blob

def detection_postprocessing(detection_response,preprocessed_images):
    def fourPointsTransform(frame, vertices):
        vertices = np.asarray(vertices)
        outputSize = (100, 32)
        targetVertices = np.array(
            [
                [0, outputSize[1] - 1],
                [0, 0],
                [outputSize[0] - 1, 0],
                [outputSize[0] - 1, outputSize[1] - 1],
            ],
            dtype="float32",
        )

        rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
        result = cv2.warpPerspective(frame, rotationMatrix, outputSize)
        return result

    def decodeBoundingBoxes(scores, geometry, scoreThresh=0.5):
        detections = []
        confidences = []

        ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ########
        assert len(scores.shape) == 4, "Incorrect dimensions of scores"
        assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
        assert scores.shape[0] == 1, "Invalid dimensions of scores"
        assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
        assert scores.shape[1] == 1, "Invalid dimensions of scores"
        assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
        assert (
            scores.shape[2] == geometry.shape[2]
        ), "Invalid dimensions of scores and geometry"
        assert (
            scores.shape[3] == geometry.shape[3]
        ), "Invalid dimensions of scores and geometry"
        height = scores.shape[2]
        width = scores.shape[3]
        for y in range(0, height):
            # Extract data from scores
            scoresData = scores[0][0][y]
            x0_data = geometry[0][0][y]
            x1_data = geometry[0][1][y]
            x2_data = geometry[0][2][y]
            x3_data = geometry[0][3][y]
            anglesData = geometry[0][4][y]
            for x in range(0, width):
                score = scoresData[x]

                # If score is lower than threshold score, move to next x
                if score < scoreThresh:
                    continue

                # Calculate offset
                offsetX = x * 4.0
                offsetY = y * 4.0
                angle = anglesData[x]

                # Calculate cos and sin of angle
                cosA = math.cos(angle)
                sinA = math.sin(angle)
                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]

                # Calculate offset
                offset = [
                    offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                    offsetY - sinA * x1_data[x] + cosA * x2_data[x],
                ]

                # Find points for rectangle
                p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
                p3 = (-cosA * w + offset[0], sinA * w + offset[1])
                center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
                detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
                confidences.append(float(score))

        # Return detections and confidences
        return [detections, confidences]

    # Process responses from detection model
    scores = detection_response.as_numpy("feature_fusion/Conv_7/Sigmoid:0")
    geometry = detection_response.as_numpy("feature_fusion/concat_3:0")
    cropped_images = []
    for i in range(preprocessed_images.shape[0]): # crop image one by one
        # matching dimension
        preprocessed_image = np.array([preprocessed_images[i]]) # (1, 480, 640, 3)
        scores_each = np.array([scores[i]]).transpose(0, 3, 1, 2)
        geometry_each = np.array([geometry[i]]).transpose(0, 3, 1, 2)

        frame = np.squeeze(preprocessed_image, axis=0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        [boxes, confidences] = decodeBoundingBoxes(scores_each, geometry_each)
        indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, 0.5, 0.4)

        cropped_list = []
        cv2.imwrite("../intermediates/frame.png", frame)
        count = 0
        for i in indices:
            # get 4 corners of the rotated rect
            count += 1
            vertices = cv2.boxPoints(boxes[i])
            cropped = fourPointsTransform(frame, vertices)
            cv2.imwrite("../intermediates/"+str(count) + ".png", cropped)
            cropped = np.expand_dims(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), axis=0)
            cropped_list.append(((cropped / 255.0) - 0.5) * 2)
        if(len(cropped_list)>1):
            cropped_arr = np.stack(cropped_list, axis=0)
        else:
            cropped_arr = np.array(cropped_list)
        cropped_images.extend(cropped_arr)

    return cropped_images

def recognition_postprocessing(scores: np.ndarray) -> str:
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"

    text_list = []
    for i in range(scores.shape[0]):
        text = ""
        for j in range(scores.shape[1]):
            c = np.argmax(scores[i][j])
            if c != 0:
                text += alphabet[c - 1]
            else:
                text += "-"
        text_list.append(text)
    # adjacent same letters as well as background text must be removed
    # to get the final output
    final_text_list = []
    for text in text_list:
        char_list = []
        for i, char in enumerate(text):
            if char != "-" and (not (i > 0 and char == text[i - 1])):
                char_list.append(char)
        final_text = "".join(char_list)
        final_text_list.append(final_text)
    return final_text_list

@trace(__file__)
def wait_signal(process_id, signal_awaited, signal_pipe: Connection):
    if signal_pipe == None: # Not coordinating multiple processes
        return
    start = time.time()
    # send a signal that it is waiting
    while True:
        receiver_id, signal_type = signal_pipe.recv()
        if receiver_id == process_id and signal_type == signal_awaited:
            break
    end = time.time()
    print("Process %d waited for signal for %.5f." % (process_id, end-start), signal_awaited)

@trace(__file__)
def send_signal(process_id, signal_to_send, signal_pipe: Connection):
    if signal_pipe == None: # Not coordinating multiple processes
        return
    signal_pipe.send((process_id, signal_to_send))

@trace(__file__)
def main(image_paths, process_id = 0, signal_pipe: Connection = None):
    #### PREPROCESSING (CPU)
    wait_signal(process_id, Message.CPU_AVAILABLE, signal_pipe)
    # Only one process occupies CPU to ensure meeting latency SLO
    # TODO: Enable a certain number of processes to run together, as we have multi-core CPU? Use semaphore or multiprocess.Queue?
    print(process_id,"PREPROCESSING start")

    t0 = time.time()
    client = httpclient.InferenceServerClient(url="localhost:8000")

    raw_images = []
    for path in image_paths:
        raw_images.append(cv2.imread(path))

    preprocessed_images = []
    for raw_image in raw_images:
        preprocessed_images.append(detection_preprocessing(raw_image)[0]) # (1, 480, 640, 3), 1 being batch size
    preprocessed_images = np.stack(preprocessed_images,axis=0) # matching dimension: (batch_size, 480, 640, 3)
    # print("Stacked images dimensions:", preprocessed_images.shape)

    t1 = time.time()
    # print("Detection preprocessing succeeded, took %.5f ms." % (t1 - t0))

    #### DETECTION INFERENCE (GPU)
    detection_input = httpclient.InferInput(
        "input_images:0", preprocessed_images.shape, datatype="FP32" 
    )
    detection_input.set_data_from_numpy(preprocessed_images, binary_data=True)

    send_signal(process_id, Message.CPU_AVAILABLE, signal_pipe) # Parent can now schedule another CPU task
    print(process_id,"PREPROCESSING finish, DETECTION INFERENCE start")

    detection_response = client.infer(
        model_name="text_detection", inputs=[detection_input]
    )
    # Potentially multiple processes running on the CPU before this process gets blocked by wait_signal
    # This should be fine, as the task here is not compute-intensive
    t2 = time.time()
    # print("Text detection succeeded, took %.5f ms." % (t2 - t1))

    #### CROPPING (CPU)
    print(process_id,"DETECTION INFERENCE finish")
    send_signal(process_id, Message.WAITING_FOR_CPU, signal_pipe) # tell scheduler that the process is waiting for CPU
    wait_signal(process_id, Message.CPU_AVAILABLE, signal_pipe) 
    print(process_id,"CROPPING start")
    # Depending on parent to schedule the first process, i.e., the process that has waited the longest
    # FIFO policy. Assuming there is no priority among processes
    cropped_images = detection_postprocessing(detection_response,preprocessed_images)
    cropped_images = np.array(cropped_images, dtype=np.single)

    if cropped_images.shape[0] == 0:
        exit(0)
    t3 = time.time()
    # print("Cropped image", cropped_images.shape,"based on detection, got %d sub images, took %.5f ms." % (len(cropped_images), (t3 - t2)))

    #### RECOGNITION INFERENCE (GPU)
    recognition_input = httpclient.InferInput(
        "input.1", cropped_images.shape, datatype="FP32"
    )
    recognition_input.set_data_from_numpy(cropped_images, binary_data=True)
    print(process_id,"CROPPING finish, RECOGNITION INFERENCE start")
    send_signal(process_id, Message.CPU_AVAILABLE, signal_pipe)
    recognition_response = client.infer(
        model_name="text_recognition", inputs=[recognition_input]
    )
    t4 = time.time()
    # print("Text recognition succeeded, took %.5f ms." % (t4 - t3))

    #### POSTPROCESSING (CPU)
    print(process_id,"RECOGNITION INFERENCE finish")
    send_signal(process_id, Message.WAITING_FOR_CPU, signal_pipe) # tell scheduler that the process is waiting for CPU
    print(process_id,"POSTPROCESSING start")
    wait_signal(process_id, Message.CPU_AVAILABLE, signal_pipe)
    final_text = recognition_postprocessing(recognition_response.as_numpy("308"))
    send_signal(process_id, Message.CPU_AVAILABLE, signal_pipe)
    print(process_id,"POSTPROCESSING finish")
    
    return final_text

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Not pipeline!")
        image_paths = [
            "../../datasets/SceneTrialTrain/lfsosa_12.08.2002/IMG_2617.JPG",
            "../../datasets/SceneTrialTrain/lfsosa_12.08.2002/IMG_2618.JPG"
        ]
    else:
        image_paths = sys.argv[1:]
        print("Pipeline batch size: "+str(len(image_paths))+"!")

    final_text = main(image_paths)

    print(final_text)
    
    