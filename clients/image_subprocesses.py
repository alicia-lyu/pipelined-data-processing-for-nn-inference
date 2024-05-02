from utils import trace, batch_arrival, get_batch_args, read_data_from_folder, IMAGE_FOLDER
from typing import List
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

if __name__ == "__main__":

    args = get_batch_args()

    image_paths = read_data_from_folder(IMAGE_FOLDER, ".jpg")
    start_time = time.time()
    log_path = "../log_image/"+args.type+"_"+str(start_time)+"/"
    os.makedirs(log_path, exist_ok=True)

    if args.type == "non-coordinate-batch":
        batch_arrival(args.min, args.max, args.batch_size, image_paths, run_subprocess, log_path)
    elif args.type == "naive-sequential":
        batch_arrival(args.min, args.max, args.batch_size, image_paths,naive_sequential, log_path)