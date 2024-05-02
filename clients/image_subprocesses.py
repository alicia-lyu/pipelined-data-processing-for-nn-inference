from utils import trace, batch_arrival, get_batch_args, read_images_from_folder, IMAGE_FOLDER
from typing import List
from image_client import main as client
from multiprocessing import Process

CLIENT = "image_client.py"

@trace(__file__)
def run_subprocess(log_dir_name:str, image_paths: List[str], process_id: int, t0: float = None) -> None:
    p = Process(target=client, args=(log_dir_name,image_paths, process_id, None, t0))
    p.start()
    return p

@trace(__file__)
def naive_sequential(log_dir_name:str,image_paths: List[str], process_id: int, t0: float = None) -> None:
    client(log_dir_name, image_paths, process_id, None, t0)
    return None

if __name__ == "__main__":

    args = get_batch_args()

    image_paths = read_images_from_folder(IMAGE_FOLDER)

    if args.type == "non-coordinate-batch":
        batch_arrival(args.min, args.max, args.batch_size, args.type, image_paths, run_subprocess)
    elif args.type == "naive-sequential":
        batch_arrival(args.min, args.max, args.batch_size, args.type, image_paths, naive_sequential)