from typing import Callable, List
import os
import time
import random
import argparse
import multiprocessing
from enum import Enum
from multiprocessing import Event, Process
import numpy as np

PROCESS_CAP = 20

class RandomPattern(Enum):
    UNIFORM = "UNIFORM"
    EXP = "EXP"

def trace(path: str):
    file_name = os.path.basename(path)
    def decorator(func: Callable):
        def trace_prefix():
            return f"*** {file_name}, {func.__name__} ***"
        setattr(func, "trace_prefix", trace_prefix)
        return func
    return decorator

def exp_random(min_val,max_val,lambda_val=1):
    exponential_random_number = np.random.exponential(scale=1/lambda_val)
    return min_val + (max_val - min_val) * (exponential_random_number / (1/lambda_val))

@trace(__file__)
def batch_arrival(min_interval: int, max_interval: int, batch_size: int,  
                  data_paths: List[str], create_client_func: Callable, 
                  log_path: str, stop_flag: Event = None, # type: ignore
                  random_patten:RandomPattern=RandomPattern.UNIFORM) -> int:
    
    processes: List[Process] = []
    blocked_time = 0
    for i in range(0, len(data_paths), batch_size):
    # for i in range(0, 10, batch_size):
        batch = data_paths[i: i + batch_size]
        client_id = i // batch_size
        if stop_flag is not None and stop_flag.is_set():
            print(batch_arrival.trace_prefix(), f"Ends earlier, sent {client_id} clients in total.")
            return client_id
        t0 = time.time()
        # Calibrate t0 for naive sequential to include the blocked time by execution of previous processes
        # In other systems, blocked_time should be close to 0, as it only involves a non-blocking behavior of starting the processes
        while len(multiprocessing.active_children()) > PROCESS_CAP:
            time.sleep(0.001)
        # TODO: Generate a random SLO latency goal for each process.
        # A good number should be a little more than the median latency that we profiled,
        # which dependends on the min_interval and max_interval.
        # --- come out with a good formula to calculate the max_slo and min_slo based on min_interval and max_interval?
        
        p = create_client_func(log_path, batch, client_id, t0 - blocked_time) # blocked time: should've started earlier
        blocked_time += time.time() - t0
        print(batch_arrival.trace_prefix(), f"Total blocked time: {blocked_time: .5f}")
        if p != None:
            processes.append(p)
        print(batch_arrival.trace_prefix(), f"Client {client_id} processes {len(batch)} data in its batch.")
        if random_patten == RandomPattern.UNIFORM:
            interval = random.uniform(min_interval, max_interval) # data_arrival_pattern(min_interval, max_interval, pattern: str)
        elif random_patten == RandomPattern.EXP:
            interval = exp_random(min_interval, max_interval)
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
    # parser.add_argument("--data-distribution", type=str, default="uniform", help="Data arrival distribution pattern, now support uniform or exponential") # TODO: Add poisson distribution
    # # The following are only for pipeline system
    # parser.add_argument("--cpu-parallelism", type=int, default=4, help="Number of CPU tasks can be run in parallel")
    # parser.add_argument("--policy", type=str, default="SLO", help="Policy to schedule the tasks, now support FIFO or SLO-oriented")

    args = parser.parse_args()

    print(get_batch_args.trace_prefix(), args)

    return args

IMAGE_FOLDER = "../../datasets/SceneTrialTrain"
AUDIO_FOLDER = "../../datasets/audio_data/mp3_16_data_2"

@trace(__file__)
def read_data_from_folder(root_folder: str, extension: str) -> List[str]:
    image_paths = []

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(extension):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)

    print(read_data_from_folder.trace_prefix(), f"Found {len(image_paths)} data.")

    return image_paths