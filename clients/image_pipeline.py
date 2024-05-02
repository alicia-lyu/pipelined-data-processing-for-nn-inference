from image_client import main as client, Message, CPUState
from multiprocessing import Pipe, Process, Event
from multiprocessing.connection import Connection
from typing import Callable, Dict, List
from utils import trace, batch_arrival, get_batch_args, read_images_from_folder, IMAGE_FOLDER
import select, sys
from enum import Enum

class Policy(Enum):
    FIFO = "FIFO"
    SLO_ORIENTED = "SLO_ORIENTED"

class Scheduler:
    def __init__(self, parent_pipes: List[Connection], timeout_in_seconds: float, policy: Policy, cpu_tasks_cap: int = 1) -> None:
        self.parent_pipes: List[Connection] = parent_pipes
        self.timeout: float = timeout_in_seconds
        if policy == Policy.FIFO:
            self.policy = self.fifo
        elif policy == Policy.SLO_ORIENTED:
            self.policy: Callable[..., int] = self.slo_oriented

        self.children_states: Dict[int, CPUState] = {}
        self.active_cpus = 0
        self.CPU_TASKS_CAP = cpu_tasks_cap # TODO: Tune this variable. I think 2-4 is a good number, since our machine has 2 cores

    @trace(__file__)
    def run(self) -> None:
        # Act on received signal from a child
        while True:
            # Allocate all available CPUs
            while self.active_cpus < self.CPU_TASKS_CAP:
                allocated = self.policy()
                if allocated:
                    self.active_cpus += 1
                else:
                    break
            # Wait for data on the parent_pipe or until the timeout expires
            ready = select.select(self.parent_pipes, [], [], self.timeout)
            if ready[0]:
                # Process the received data
                for ready_pipe in ready[0]:
                    client_id, signal_type = ready_pipe.recv()
                    client_id = int(client_id)
                    print(self.run.trace_prefix(), f"Received signal {signal_type} from {client_id}", "Active CPUs: "+str(self.active_cpus))#, hashmap_stage, hashmap_state)
                    # A child client relinquishes CPU
                    if signal_type == Message.RELINQUISH_CPU:
                        self.children_states[client_id] = CPUState.GPU
                        self.active_cpus -= 1
                    # A child client finishes GPU tasks and is now waiting for CPU
                    elif signal_type == Message.WAITING_FOR_CPU:
                        self.children_states[client_id] = CPUState.WAITING_FOR_CPU
                    elif signal_type == Message.FINISHED:
                        del self.children_states[client_id]
                        self.active_cpus -= 1
            else:
                # Handle timeout
                if len(self.children_states) != 0:
                    for process_id, child_state in self.children_states.items():
                        print(f"Client {process_id} stuck at {child_state}!", file=sys.stderr)
                print("No data received within the timeout period.")
                break  # Break out of the loop

    @trace(__file__)
    def fifo(self) -> int:
        if len(self.children_states) == 0:
            return False
        try:
            min_process_id = min(key for key, value in self.children_states.items() if value == CPUState.WAITING_FOR_CPU)
            min_process_id = int(min_process_id)
            self.parent_pipes[min_process_id].send((min_process_id, Message.ALLOCATE_CPU))
        except ValueError:
            print(self.fifo.trace_prefix(), "No client to allocate the spare CPU. States:", self.children_states)
            return False
        else:
            self.children_states[min_process_id] = CPUState.CPU
            print(self.fifo.trace_prefix(), f"CPU is allocated to {min_process_id}.")
            return True

    def slo_oriented():
        pass

@trace(__file__)
def create_client(log_dir_name:str,image_paths: List[str], process_id: int, child_pipe: Connection, t0: float = None) -> None:
    p = Process(target=client, args=(log_dir_name,image_paths, process_id, child_pipe, t0))
    p.start()
    return p

if __name__ == "__main__":
    stop_flag = Event() # stop the batch arrival when the scheduler stops

    args = get_batch_args()
    image_paths = read_images_from_folder(IMAGE_FOLDER)
    
    parent_pipes: List[Connection] = []
    child_pipes: List[Connection] = []


    for i in range(len(image_paths) // args.batch_size):
        parent_pipe,child_pipe = Pipe()
        parent_pipes.append(parent_pipe)
        child_pipes.append(child_pipe)

    # Non-blocking (run at the same time with the scheduler): images arrive in batch
    batch_arrival_process = Process(target=batch_arrival, \
                                args=(args.min, args.max, args.batch_size,args.type, \
                                        image_paths, \
                                        lambda log_dir_name, batch, id, t0: create_client(log_dir_name, batch, id, child_pipes[id], t0), 
                                        stop_flag))
    batch_arrival_process.start()

    scheduler = Scheduler(parent_pipes, args.timeout, Policy.FIFO)
    scheduler.run()

    stop_flag.set()