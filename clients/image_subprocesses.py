import subprocess
from utils import trace, batch_arrival, get_batch_args, read_images_from_folder, IMAGE_FOLDER
from typing import List
from image_client import main as client
from multiprocessing import Process

CLIENT = "image_client.py"

# TODO: I am not sure how to record the response time of a subprocess, i.e., how to know when the process finishes 
# Maintain a list of all subprocesses?

@trace(__file__)
def run_subprocess(log_dir_name:str,image_paths: List[str], process_id: int) -> None:
    p = Process(target=client, args=(log_dir_name,image_paths, process_id))
    p.start()
    return p

@trace(__file__)
def naive_sequential(log_dir_name:str,image_paths: List[str], process_id: int) -> None:
    client(log_dir_name,image_paths, process_id)
    return None

if __name__ == "__main__":

    args = get_batch_args()

    image_paths = read_images_from_folder(IMAGE_FOLDER)

    if args.type == "non-coordinate-batch":
        batch_arrival(args.min, args.max, args.batch_size, args.type, image_paths, run_subprocess)
    elif args.type == "naive-sequential":
        batch_arrival(args.min, args.max, args.batch_size, args.type, image_paths, naive_sequential)

    # TODO: call subprocess.return_code() or some other API to get the response time stamp?