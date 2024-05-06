from image_client import TextRecognitionClient
from multiprocessing import Pipe, Process, Event
from multiprocessing.connection import Connection
from typing import List
from utils import trace, batch_arrival, get_batch_args, read_data_from_folder, IMAGE_FOLDER
import time, os
from Scheduler import Scheduler, Policy

PRIORITY_TO_LATENCY_GOAL = {
    1: 3.0,
    2: 4.0,
    3: 5.0,
    4: 6.0
}

@trace(__file__)
def create_client(log_dir_name:str,image_paths: List[str], process_id: int, child_pipe: Connection, t0: float = None) -> None:
    client = TextRecognitionClient(log_dir_name, image_paths, process_id, child_pipe, t0)
    p = Process(target=client.run)
    p.start()
    return p

if __name__ == "__main__":
    stop_flag = Event() # stop the batch arrival when the scheduler stops

    args = get_batch_args()
    image_paths = read_data_from_folder(IMAGE_FOLDER, ".jpg")
    
    parent_pipes: List[Connection] = []
    child_pipes: List[Connection] = []

    start_time = time.time()
    log_path = "../log_image/"+args.type+"_"+str(start_time)+"/"
    os.makedirs(log_path, exist_ok=True)


    for i in range(len(image_paths) // args.batch_size):
        parent_pipe,child_pipe = Pipe()
        parent_pipes.append(parent_pipe)
        child_pipes.append(child_pipe)

    # Non-blocking (run at the same time with the scheduler): images arrive in batch
    batch_arrival_process = Process(target=batch_arrival, 
                                    args=(args.min, args.max, args.batch_size, image_paths,
                                    lambda log_dir_name, batch, id, t0: create_client(log_dir_name, batch, id, child_pipes[id], t0), 
                                    log_path, stop_flag))
    batch_arrival_process.start()

    scheduler = Scheduler(parent_pipes, args.timeout, Policy.FIFO, cpu_tasks_cap=4, priority_to_latency_map=PRIORITY_TO_LATENCY_GOAL)
    ret = scheduler.run()

    if ret is True:
        batch_arrival_process.terminate()