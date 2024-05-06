from image_client import TextRecognitionClient
from multiprocessing import Pipe, Process, Event
from multiprocessing.connection import Connection
from typing import List
from utils import trace, get_batch_args
from batch_arrive import batch_arrive
import random
from Scheduler import Scheduler, Policy
from Client import Client
from audio_client import AudioRecognitionClient
from image_subprocesses import get_params

@trace(__file__)
def create_client(data_paths: List[str], process_id: int, t0: float, # provided in batch_arrive
                  log_dir_name: str, client_class: Client, # shared by all systems
                  child_pipe: Connection, include_priority = True # specific to pipeline
                  ) -> None:
    if include_priority:
        priority = random.randint(1, 4)
    else:
        priority = None
    client = client_class(log_dir_name, data_paths, process_id, child_pipe, t0, priority)
    p = Process(target=client.run)
    p.start()
    return p

def pipeline(min_interval: int, max_interval: int, batch_size: int, timeout: int, data_type: str,
             policy: Policy, cpu_tasks_cap: int):
    
    client_class, data_paths, log_path, priority_map = get_params(data_type)
    assert((policy == Policy.SLO_ORIENTED) == (priority_map is not None))
    
    parent_pipes: List[Connection] = []
    child_pipes: List[Connection] = []

    for i in range(len(data_paths) // args.batch_size):
        parent_pipe,child_pipe = Pipe()
        parent_pipes.append(parent_pipe)
        child_pipes.append(child_pipe)

    batch_arrive_process = Process(
        target = batch_arrive, 
        args = (
            min_interval, max_interval, batch_size, data_paths,
            lambda batch, id, t0: create_client(
                batch, id, t0, log_path, client_class, 
                child_pipes[id], include_priority=(policy == Policy.SLO_ORIENTED)
            ),
            log_path, stop_flag
        )
    )
    batch_arrive_process.start()
    scheduler = Scheduler(parent_pipes, timeout, policy, cpu_tasks_cap, priority_map)
    scheduler.run()

if __name__ == "__main__":
    stop_flag = Event() # stop the batch arrival when the scheduler stops
    args = get_batch_args()
    pipeline(args.min, args.max, args.batch_size, args.timeout, args.data_type,
             args.policy, args.cpu_tasks_cap)