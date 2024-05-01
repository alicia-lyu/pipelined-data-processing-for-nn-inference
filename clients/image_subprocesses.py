import subprocess
from utils import trace, batch_arrival, get_batch_args, read_images_from_folder, IMAGE_FOLDER
from typing import List
from image_client import main as client

CLIENT = "image_client.py"

# TODO: I am not sure how to record the response time of a subprocess, i.e., how to know when the process finishes 
# Maintain a list of all subprocesses?

@trace(__file__)
def run_subprocess(image_paths: List[str], process_id: int) -> None:
    subprocess.run(["python", CLIENT, str(process_id)] + image_paths )

@trace(__file__)
def naive_sequential(image_paths: List[str], process_id: int) -> None:
    client(image_paths, process_id)
    # TODO: Plug this to batch arrival to enable a sequential system
    # when target is naive_sequential, for loop in `batch_arrival` will be blocked

if __name__ == "__main__":

    args = get_batch_args()

    image_paths = read_images_from_folder(IMAGE_FOLDER)

    batch_arrival(args.min, args.max, args.batch_size, image_paths, run_subprocess)

    # TODO: call subprocess.return_code() or some other API to get the response time stamp?