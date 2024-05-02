from multiprocessing import Process
from utils import trace, batch_arrival, get_batch_args, read_data_from_folder, AUDIO_FOLDER
from typing import List
from audio_client import AudioRecognitionClient
import time, os

CLIENT = "audio_client.py"

@trace(__file__)
def run_subprocess(log_dir_name:str, audio_paths: List[str], process_id: int, t0: float = None) -> None:
    client = AudioRecognitionClient(log_dir_name, audio_paths, process_id, None, t0)
    p = Process(target=client.run)
    p.start()
    return p

@trace(__file__)
def naive_sequential(log_dir_name:str, audio_paths: List[str], process_id: int, t0: float = None) -> None:
    client = AudioRecognitionClient(log_dir_name, audio_paths, process_id, None, t0)
    client.run()
    return None

if __name__ == "__main__":

    args = get_batch_args()

    audio_paths = read_data_from_folder(AUDIO_FOLDER, ".mp3")

    start_time = time.time()
    log_path = "../log_audio/"+args.type+"_"+str(start_time)+"/"
    os.makedirs(log_path, exist_ok=True)

    if args.type == "non-coordinate-batch":
        batch_arrival(args.min, args.max, args.batch_size, audio_paths, run_subprocess, log_path)
    elif args.type == "naive-sequential":
        batch_arrival(args.min, args.max, args.batch_size, audio_paths, naive_sequential, log_path)