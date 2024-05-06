from multiprocessing.connection import Connection
from multiprocessing import Process, Event, Pipe
from audio_client import AudioRecognitionClient
from typing import List
from utils import get_batch_args, trace, read_data_from_folder, AUDIO_FOLDER, batch_arrival
from Scheduler import Scheduler, Policy
import time, os

PRIORITY_TO_LATENCY_GOAL = {
    1: 7.0,
    2: 10.0,
    3: 13.0,
    4: 16.0
}

@trace(__file__)
def create_client(log_dir_name:str, audio_paths: List[str], process_id: int, child_pipe: Connection, t0: float = None) -> None:
    client = AudioRecognitionClient(log_dir_name, audio_paths, process_id, child_pipe, t0)
    p = Process(target=client.run)
    p.start()
    return p

if __name__ == "__main__":
    stop_flag = Event() # stop the batch arrival when the scheduler stops

    args = get_batch_args()
    audio_paths = read_data_from_folder(AUDIO_FOLDER, ".mp3")
    
    parent_pipes: List[Connection] = []
    child_pipes: List[Connection] = []


    for i in range(len(audio_paths) // args.batch_size):
        parent_pipe,child_pipe = Pipe()
        parent_pipes.append(parent_pipe)
        child_pipes.append(child_pipe)

    start_time = time.time()
    log_path = "../log_audio/"+args.type+"_"+str(start_time)+"/"
    os.makedirs(log_path, exist_ok=True)
    # Non-blocking (run at the same time with the scheduler): images arrive in batch
    batch_arrival_process = Process(target=batch_arrival, \
                                args=(args.min, args.max, args.batch_size, audio_paths,
                                        lambda log_dir_name, batch, id, t0: create_client(log_dir_name, batch, id, child_pipes[id], t0), 
                                        log_path, stop_flag))
    batch_arrival_process.start()

    scheduler = Scheduler(parent_pipes, args.timeout, Policy.FIFO)
    ret = scheduler.run()

    if ret is True:
        batch_arrival_process.terminate()