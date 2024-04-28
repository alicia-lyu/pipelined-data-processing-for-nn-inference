from typing import Dict, Callable
from image_client import Stage, CPUState, Message
from utils import trace
from multiprocessing.connection import Connection

# Allocate CPU to the eligible client with the smallest id (FIFO)
@trace(__file__)
def non_sharing_pipeline(parent_pipe: Connection, hashmap_stage: Dict[int, Stage], hashmap_state: Dict[int, CPUState], cpu_using: bool, grant_cpu_func: Callable) -> None:
    assert(cpu_using == False)
    try:
        min_process_id = min(key for key, value in hashmap_state.items() if value == CPUState.WAITING_FOR_CPU)
        min_process_id = int(min_process_id)
        grant_cpu_func(min_process_id, hashmap_stage, hashmap_state)
        parent_pipe.send((min_process_id, Message.CPU_AVAILABLE))
        cpu_using = True
    except ValueError:
        print(non_sharing_pipeline.trace_prefix(), "No client to allocate the spare CPU. Hashmap_stage:", hashmap_stage)
        cpu_using = False
    else:
        print(non_sharing_pipeline.trace_prefix(), f"CPU is allocated to {min_process_id}.")