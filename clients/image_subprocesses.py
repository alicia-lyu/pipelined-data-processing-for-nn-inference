from utils import trace, get_batch_args, read_data_from_folder, get_log_dir, ModelType
from batch_arrive import batch_arrive
from typing import List, Callable
from image_client import TextRecognitionClient
from audio_client import AudioRecognitionClient
from multiprocessing import Process
import os
from Client import Client

PRIORITY_TO_LATENCY_IMAGE = {
    1: 3.0,
    2: 4.0,
    3: 5.0,
    4: 6.0
}

PRIORITY_TO_LATENCY_AUDIO = {
    1: 7.0,
    2: 10.0,
    3: 13.0,
    4: 16.0
}

@trace(__file__)
def run_subprocess(data_paths: List[str], process_id: int, t0: float,
                   log_dir_name:str, client_class: Client) -> None:
    client = client_class(log_dir_name,data_paths, process_id, None, t0)
    p = Process(target=client.run)
    p.start()
    return p

@trace(__file__)
def naive_sequential(data_paths: List[str], process_id: int, t0: float,
                   log_dir_name:str, client_class: Client) -> None:
    client = client_class(log_dir_name,data_paths, process_id, None, t0)
    client.run()
    return None

def get_params(data_type: str):
    if data_type == "image":
        client_class = TextRecognitionClient
        extension = ".jpg"
        model_type = ModelType.IMAGE
        priority_map = PRIORITY_TO_LATENCY_IMAGE
    elif data_type == "audio":
        client_class = AudioRecognitionClient
        extension = ".mp3"
        model_type = ModelType.AUDIO
        priority_map = PRIORITY_TO_LATENCY_AUDIO
    else:
        raise ValueError("Invalid data type")
    data_paths = read_data_from_folder(extension)
    log_path = get_log_dir(model_type)
    os.makedirs(log_path, exist_ok=True)
    return client_class, data_paths, log_path, priority_map

def batch_sequential(min_interval: int, max_interval: int, batch_size: int, data_type: int, create_client_func: Callable):
    client_class, data_paths, log_path = get_params(data_type)
    batch_arrive(min_interval, max_interval, batch_size, data_paths,
                 lambda batch, id, t0: create_client_func(batch, id, t0, log_path, client_class))

if __name__ == "__main__":

    args = get_batch_args()

    if args.type == "non-coordinate-batch":
        create_client_func = run_subprocess
    elif args.type == "naive-sequential":
        create_client_func = naive_sequential

    else:
        raise ValueError("Invalid data type")
    
    batch_sequential(args.min, args.max, args.batch_size, args.data_type, naive_sequential)