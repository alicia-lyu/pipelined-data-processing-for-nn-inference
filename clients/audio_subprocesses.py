import subprocess
from utils import trace, batch_arrival_audio, get_batch_args, read_audios_from_folder, AUDIO_FOLDER
from typing import List
from image_client import main as client

CLIENT = "audio_client.py"

@trace(__file__)
def run_subprocess(log_dir_name:str, audio_paths: List[str], process_id: int) -> None:
    subprocess.run(["python", CLIENT, log_dir_name, str(process_id)] + audio_paths )

@trace(__file__)
def naive_sequential(log_dir_name:str, audio_paths: List[str], process_id: int) -> None:
    client(log_dir_name, audio_paths, process_id)

if __name__ == "__main__":

    args = get_batch_args()

    audio_paths = read_audios_from_folder(AUDIO_FOLDER)

    if args.type == "non-coordinate-batch":
        batch_arrival_audio(args.min, args.max, args.batch_size, args.type, audio_paths, run_subprocess)
    elif args.type == "naive-sequential":
        batch_arrival_audio(args.min, args.max, args.batch_size, args.type, audio_paths, naive_sequential)