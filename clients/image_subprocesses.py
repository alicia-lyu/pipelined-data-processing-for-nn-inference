from utils import trace, batch_arrival, get_batch_args, read_data_from_folder, IMAGE_FOLDER, get_log_dir, ModelType
from typing import List, Callable
from image_client import TextRecognitionClient
from multiprocessing import Process
import time, os

CLIENT = "image_client.py"

@trace(__file__)
def run_subprocess(log_dir_name:str, image_paths: List[str], process_id: int, t0: float = None) -> None:
    client = TextRecognitionClient(log_dir_name,image_paths, process_id, None, t0)
    p = Process(target=client.run)
    p.start()
    return p

@trace(__file__)
def naive_sequential(log_dir_name:str, image_paths: List[str], process_id: int, t0: float = None) -> None:
    client = TextRecognitionClient(log_dir_name,image_paths, process_id, None, t0)
    client.run()
    return None

def batch_subprocesses(min_interval: int, max_interval: int, batch_size: int, create_client_func: Callable):
    image_paths = read_data_from_folder(IMAGE_FOLDER, ".jpg")
    log_path = get_log_dir(ModelType.IMAGE)
    os.makedirs(log_path, exist_ok=True)
    batch_arrival(min_interval, max_interval, batch_size, image_paths, create_client_func, log_path)

def batch_sequential(min_interval: int, max_interval: int, batch_size: int, create_client_func: Callable):
    image_paths = read_data_from_folder(IMAGE_FOLDER, ".jpg")
    log_path = get_log_dir(ModelType.IMAGE)
    os.makedirs(log_path, exist_ok=True)
    batch_arrival(min_interval, max_interval, batch_size, image_paths, create_client_func, log_path)

if __name__ == "__main__":

    args = get_batch_args()

    if args.type == "non-coordinate-batch":
        batch_subprocesses(args.min, args.max, args.batch_size, run_subprocess)
    elif args.type == "naive-sequential":
        batch_sequential(args.min, args.max, args.batch_size, naive_sequential)