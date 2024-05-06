from image_client import TextRecognitionClient
from multiprocessing import Pipe, Process, Event
from multiprocessing.connection import Connection
from typing import List
from utils import trace, get_batch_args, read_data_from_folder, IMAGE_FOLDER, get_log_dir, ModelType
from batch_arrive import batch_arrive
import random
from Scheduler import Scheduler, Policy

PRIORITY_TO_LATENCY_GOAL = {
    1: 3.0,
    2: 4.0,
    3: 5.0,
    4: 6.0
}

@trace(__file__)
def create_client(log_dir_name: str, image_paths: List[str], process_id: int, 
                  child_pipe: Connection, t0: float = None, include_priority = True) -> None:
    if include_priority:
        priority = random.choice(list(PRIORITY_TO_LATENCY_GOAL.keys()))
    else:
        priority = None
    client = TextRecognitionClient(log_dir_name, image_paths, process_id, child_pipe, t0, priority)
    p = Process(target=client.run)
    p.start()
    return p

def pipeline(min_interval: int, max_interval: int, batch_size: int, timeout: int,
             policy: Policy, cpu_tasks_cap: int, priority_to_latency_map: dict[int, float] = None):
    assert((policy == Policy.SLO_ORIENTED) == (priority_to_latency_map is not None))
    
    image_paths = read_data_from_folder(IMAGE_FOLDER, ".jpg")
    log_path = get_log_dir(ModelType.IMAGE)
    parent_pipes: List[Connection] = []
    child_pipes: List[Connection] = []


    for i in range(len(image_paths) // args.batch_size):
        parent_pipe,child_pipe = Pipe()
        parent_pipes.append(parent_pipe)
        child_pipes.append(child_pipe)

    batch_arrive_process = Process(
        target = batch_arrive, 
        args = (
            min_interval, max_interval, batch_size, image_paths,
            lambda log_dir_name, batch, id, t0: create_client(
                log_dir_name, batch, id, child_pipes[id], 
                t0, policy == Policy.SLO_ORIENTED), 
            log_path, stop_flag
        )
    )
    batch_arrive_process.start()
    scheduler = Scheduler(parent_pipes, timeout, policy, cpu_tasks_cap, priority_to_latency_map)
    scheduler.run()

if __name__ == "__main__":
    stop_flag = Event() # stop the batch arrival when the scheduler stops

    args = get_batch_args()
    pipeline(args.min, args.max, args.batch_size, args.timeout, args.policy, args.cpu_tasks_cap, PRIORITY_TO_LATENCY_GOAL)