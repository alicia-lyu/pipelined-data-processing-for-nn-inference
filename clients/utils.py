from typing import Callable, List
import os
import time
import argparse
from enum import Enum

class ModelType(Enum):
    IMAGE = "IMAGE"
    AUDIO = "AUDIO"

def trace(path: str):
    file_name = os.path.basename(path)
    def decorator(func: Callable):
        def trace_prefix():
            return f"*** {file_name}, {func.__name__} ***"
        setattr(func, "trace_prefix", trace_prefix)
        return func
    return decorator

def get_log_dir(type: str) -> str:
    start_time = time.time()
    assert(isinstance(type, ModelType))
    if type == ModelType.IMAGE:
        parent_dir = "../log_image/"
    else:
        parent_dir = "../log_audio/"
    return parent_dir + type.value.lower() + "_" + str(start_time) + "/"

@trace(__file__)
def get_batch_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set pipeline data arrival interval")

    parser.add_argument("--min", type=float, help="Minimum data arrival interval")
    parser.add_argument("--max", type=float, help="Maximum data arrival interval")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--timeout", type=float, help="Scheduler timeout threhold")
    parser.add_argument("--data_type", type=str, help="Data type, image or audio", required=True)
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
def read_data_from_folder(extension: str) -> List[str]:
    image_paths = []
    
    if extension == ".jpg":
        root_folder = IMAGE_FOLDER
    elif extension == ".mp3":
        root_folder = AUDIO_FOLDER
    else:
        raise ValueError("Invalid extension")

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(extension):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)

    print(read_data_from_folder.trace_prefix(), f"Found {len(image_paths)} data.")

    return image_paths