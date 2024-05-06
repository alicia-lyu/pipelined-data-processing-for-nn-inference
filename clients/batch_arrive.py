import time, random, multiprocessing, numpy as np
from typing import Callable, List
from multiprocessing import Process, Event
from enum import Enum
from utils import trace

PROCESS_CAP = 20

class RandomPattern(Enum):
    UNIFORM = "UNIFORM"
    EXP = "EXP"
    POISSON = "POISSON"

@trace(__file__)
def batch_arrive(min_interval: int, max_interval: int, batch_size: int,  
                  data_paths: List[str], create_client_func: Callable, 
                  stop_flag: Event = None, # type: ignore
                  random_patten:RandomPattern=RandomPattern.UNIFORM) -> int:
    
    processes: List[Process] = []
    blocked_time = 0
    
    for i in range(0, len(data_paths), batch_size):
        
        batch = data_paths[i: i + batch_size]
        client_id = i // batch_size
        if stop_flag is not None and stop_flag.is_set():
            print(batch_arrive.trace_prefix(), f"Ends earlier, sent {client_id} clients in total.")
            return client_id
        
        # ----- Avoid overcrowding the system
        while len(multiprocessing.active_children()) > PROCESS_CAP:
            time.sleep(0.001)
        
        # ----- Start the process
        t0 = time.time()
        # Calibrate t0 for naive sequential to include the blocked time by execution of previous processes
        # In other systems, blocked_time should be close to 0
        p = create_client_func(batch, client_id, t0 - blocked_time) # blocked time: should've started earlier
        blocked_time += time.time() - t0
        if p is not None:
            processes.append(p)
        
        # ----- Data arrival pattern determines interval
        if random_patten == RandomPattern.UNIFORM:
            interval = random.uniform(min_interval, max_interval) # data_arrival_pattern(min_interval, max_interval, pattern: str)
        elif random_patten == RandomPattern.EXP:
            interval = exp_random(min_interval, max_interval)
        elif random_patten == RandomPattern.POISSON:
            interval = poisson_random(min_interval, max_interval)
        else:
            raise ValueError("Invalid random pattern")
        time.sleep(interval)

    client_num = len(data_paths) // batch_size
    print(batch_arrive.trace_prefix(), f"Sent {client_num} clients in total.")
    return client_num

def exp_random(min_val,max_val,lambda_val=1):
    exponential_random_number = np.random.exponential(scale=1/lambda_val)
    return min_val + (max_val - min_val) * (exponential_random_number / (1/lambda_val))

def poisson_random(min_val,max_val,lambda_val=1):
    poisson_random_number = np.random.poisson(lam=lambda_val)
    return min_val + (max_val - min_val) * (poisson_random_number / lambda_val)