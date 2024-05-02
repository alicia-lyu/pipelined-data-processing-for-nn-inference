from typing import Callable, List
import os
import time
import random
import argparse
import multiprocessing
from multiprocessing import Event,Process

PROCESS_CAP = 50

def trace(path: str):
    file_name = os.path.basename(path)
    def decorator(func: Callable):
        def trace_prefix():
            return f"*** {file_name}, {func.__name__} ***"
        setattr(func, "trace_prefix", trace_prefix)
        return func
    return decorator

@trace(__file__)
def batch_arrival(min_interval: int, max_interval: int, batch_size: int, system_type:str, 
                  data_paths: List[str], create_client_func: Callable, stop_flag: Event = None,
                  processes: List[Process] = None) -> int:
    start_time = time.time()
    log_path = "../log_image/"+system_type+"_"+str(start_time)+"/"
    os.makedirs(log_path, exist_ok=True)
    blocked_time = 0
    for i in range(0, len(data_paths), batch_size):
    # for i in range(0, 10, batch_size):
        batch = data_paths[i: i + batch_size]
        client_id = i // batch_size
        if stop_flag!=None:
            if stop_flag.is_set():
                print(batch_arrival.trace_prefix(), f"Ends earlier, sent {client_id} clients in total.")
                return client_id
        t0 = time.time()
        # Calibrate t0 for naive sequential to include the blocked time by execution of previous processes
        # In other systems, blocked_time should be close to 0, as it only involves a non-blocking behavior of starting the processes
        while len(multiprocessing.active_children()) > PROCESS_CAP:
            time.sleep(0.001)
        p = create_client_func(log_path, batch, client_id, t0 - blocked_time) # blocked time: should've started earlier
        blocked_time += time.time() - t0
        print(batch_arrival.trace_prefix(), f"Total blocked time: {blocked_time: .5f}")
        if processes != None:
            processes.append(p)
        print(batch_arrival.trace_prefix(), f"Client {client_id} processes {len(batch)} images.")
        interval = random.uniform(min_interval, max_interval)
        time.sleep(interval)
    
    client_num = len(data_paths) // batch_size
    print(batch_arrival.trace_prefix(), f"Sent {client_num} clients in total.")
    return client_num

@trace(__file__)
def get_batch_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set pipeline data arrival interval")

    parser.add_argument("--min", type=float, help="Minimum data arrival interval")
    parser.add_argument("--max", type=float, help="Maximum data arrival interval")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--timeout", type=float, help="Scheduler timeout threhold")
    parser.add_argument("--type", type=str,default="pipeline", help="System type, naive-sequential/non-coordinate-batch/pipeline")

    args = parser.parse_args()

    print(get_batch_args.trace_prefix(), args)

    return args

IMAGE_FOLDER = "../../datasets/SceneTrialTrain"
AUDIO_FOLDER = "../../datasets/audio_data/mp3_16_data_2"

@trace(__file__)
def read_images_from_folder(root_folder: str) -> List[str]:
    image_paths = []

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".jpg"):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)

    print(read_images_from_folder.trace_prefix(), f"Found {len(image_paths)} images.")

    return image_paths

def read_audios_from_folder(root_folder: str) -> List[str]:
    audio_paths = []

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".mp3"):
                audio_path = os.path.join(root, file)
                audio_paths.append(audio_path)

    return audio_paths

@trace(__file__)
def batch_arrival_audio(min_interval: int, max_interval: int, batch_size: int, system_type:str, 
                  data_paths: List[str], create_client_func: Callable, stop_flag:Event = None) -> int:
    start_time = time.time()
    log_path = "../log_audio/"+system_type+"_"+str(start_time)+"/"
    os.makedirs(log_path, exist_ok=True)
    for i in range(0, len(data_paths), batch_size):
        batch = data_paths[i: i + batch_size]
        client_id = i // batch_size
        if stop_flag!=None:
            if stop_flag.is_set():
                print(batch_arrival.trace_prefix(), f"Ends earlier, sent {client_id} clients in total.")
                return client_id
        # TODO: request arrival time
        create_client_func(log_path,batch, client_id)
        # TODO: response time for naive sequential
        print(batch_arrival.trace_prefix(), f"Client {client_id} processes {len(batch)} audios.")
        interval = random.uniform(min_interval, max_interval)
        time.sleep(interval)
    
    client_num = len(data_paths) // batch_size
    print(batch_arrival.trace_prefix(), f"Sent {client_num} clients in total.")
    return client_num