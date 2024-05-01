import os
import subprocess
from utils import trace, batch_arrival, get_batch_args, read_audios_from_folder, AUDIO_FOLDER
from typing import List
from image_client import main as client

CLIENT = "audio_client.py"

@trace(__file__)
def run_subprocess(audio_paths: List[str], process_id: int) -> None:
    subprocess.run(["python", CLIENT, str(process_id)] + audio_paths )

@trace(__file__)
def naive_sequential(image_paths: List[str], process_id: int) -> None:
    client(image_paths, process_id)
    # TODO: Plug this to batch arrival to enable a sequential system
    # when target is naive_sequential, for loop in `batch_arrival` will be blocked

if __name__ == "__main__":

    args = get_batch_args()

    audio_paths = read_audios_from_folder(AUDIO_FOLDER)

    batch_arrival(args.min, args.max, args.batch_size, audio_paths, run_subprocess)