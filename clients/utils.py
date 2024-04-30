from typing import Callable, List
import os
import time
import random
import argparse
from multiprocessing import Event

def trace(path: str):
    file_name = os.path.basename(path)
    def decorator(func: Callable):
        def trace_prefix():
            return f"*** {file_name}, {func.__name__} ***"
        setattr(func, "trace_prefix", trace_prefix)
        return func
    return decorator

@trace(__file__)
def batch_arrival(min_interval: int, max_interval: int, batch_size: int, stop_flag:Event,
                  data_paths: List[str], create_client_func: Callable) -> int:

    for i in range(0, len(data_paths), batch_size):
        batch = data_paths[i: i + batch_size]
        client_id = i // batch_size
        if stop_flag.is_set():
            print(batch_arrival.trace_prefix(), f"Ends earlier, sent {client_id} clients in total.")
            return client_id
        create_client_func(batch, client_id)
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

    args = parser.parse_args()

    print(get_batch_args.trace_prefix(), args)

    return args

IMAGE_FOLDER = "../../datasets/SceneTrialTrain"

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