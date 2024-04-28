from image_client import main as client, Message, Stage, CPUState
from multiprocessing import Pipe
from multiprocessing.connection import Connection
import multiprocessing
from typing import Callable, Dict, List
from image_subprocesses import batch_arrival, get_batch_args
from utils import trace
from policies import non_sharing_pipeline

# grant CPU to a child process and update its stage and state accordingly
@trace(__file__)
def grant_cpu(process_id: int, hashmap_stage: Dict[int, Stage], hashmap_state: Dict[int, CPUState]) -> None:
    print(grant_cpu.trace_prefix(), f"Granted CPU to {process_id}, originally {hashmap_stage[process_id]}, {hashmap_state[process_id]}")
    if hashmap_stage[process_id] == Stage.NOT_START:
        hashmap_stage[process_id] = Stage.PREPROCESSING
        hashmap_state[process_id] = CPUState.CPU
    if hashmap_stage[process_id] == Stage.DETECTION_INFERENCE:
        hashmap_stage[process_id] = Stage.CROPPING
        hashmap_state[process_id] = CPUState.CPU
    elif hashmap_stage[process_id] == Stage.RECOGNITION_INFERENCE:
        hashmap_stage[process_id] = Stage.POSTPROCESSING
        hashmap_state[process_id] = CPUState.CPU

# Update the stage and state of a child client when it relinquishes CPU
@trace(__file__)
def relinquish_cpu(process_id: int, hashmap_stage: Dict[int, Stage], hashmap_state: Dict[int, CPUState]) -> None:
    print(relinquish_cpu.trace_prefix(), f"{process_id} relinquished CPU, originally {hashmap_stage[process_id]}, {hashmap_state[process_id]}")
    if hashmap_stage[process_id] == Stage.POSTPROCESSING: # Finished all stages
        del hashmap_stage[process_id]
        del hashmap_state[process_id]
    elif hashmap_stage[process_id] == Stage.PREPROCESSING:
        hashmap_stage[process_id] = Stage.DETECTION_INFERENCE
        hashmap_state[process_id] = CPUState.GPU
    elif hashmap_stage[process_id] == Stage.CROPPING:
        hashmap_stage[process_id] = Stage.RECOGNITION_INFERENCE
        hashmap_state[process_id] = CPUState.GPU

@trace(__file__)
def schedule(parent_pipe: Connection, 
             grant_cpu_func: Callable[[int, Dict[int, Stage], Dict[int, CPUState]], None], 
             relinquish_cpu_func: Callable[[int, Dict[int, Stage], Dict[int, CPUState]], None]
             ) -> None:
    hashmap_stage: Dict[int, Stage] = {}
    hashmap_state: Dict[int, CPUState] = {}
    cpu_using: bool = False # Only one process occupies CPU to ensure meeting latency SLO
    
    # Act on received signal from a child
    while True:
        client_id, signal_type = parent_pipe.recv()
        client_id = int(client_id)
        print(schedule.trace_prefix(), f"Received signal {signal_type} from {client_id}", hashmap_stage, hashmap_state)
        # A child client is first created (in create_client)
        if signal_type == Message.CREATE_PROCESS:
            hashmap_stage[client_id] = Stage.NOT_START
            hashmap_state[client_id] = CPUState.WAITING_FOR_CPU
        # A child client relinquishes CPU
        elif signal_type == Message.CPU_AVAILABLE:
            relinquish_cpu_func(client_id, hashmap_stage, hashmap_state)
            cpu_using = False
         # A child client finishes GPU tasks and is now waiting for CPU
        elif signal_type == Message.WAITING_FOR_CPU:
            hashmap_state[client_id] = CPUState.WAITING_FOR_CPU

        if not cpu_using:
            non_sharing_pipeline(parent_pipe, hashmap_stage, hashmap_state, cpu_using, grant_cpu)

@trace(__file__)
def create_client(image_paths: List[str], process_id: int, child_pipe: Connection) -> None:
    child_pipe.send((process_id, Message.CREATE_PROCESS))
    p = multiprocessing.Process(target=client, args=(image_paths, process_id, child_pipe))
    p.start()

if __name__ == "__main__":
    args = get_batch_args()
    parent_pipe, child_pipe = Pipe()
    # Non-blocking (run at the same time with the scheduler): images arrive in batch
    p = multiprocessing.Process(target=batch_arrival, \
                                args=(args.min, args.max, args.batch_size, \
                                      lambda batch, id: create_client(batch, id, child_pipe)))
    p.start()
    schedule(parent_pipe, grant_cpu, relinquish_cpu)