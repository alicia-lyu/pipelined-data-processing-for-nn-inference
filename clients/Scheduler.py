from enum import Enum
from typing import Callable, List, Dict
from multiprocessing.connection import Connection
from utils import trace
import select, sys, time

class Policy(Enum):
    FIFO = "FIFO"
    SLO_ORIENTED = "SLO_ORIENTED"

class Message(Enum):
    RELINQUISH_CPU = "RELINQUISH_CPU"
    ALLOCATE_CPU = "ALLOCATE_CPU"
    WAITING_FOR_CPU = "WAITING_FOR_CPU"
    FINISHED = "FINISHED"

class CPUState(Enum):
    CPU = "CPU"
    GPU = "GPU"
    WAITING_FOR_CPU = "WAITING_FOR_CPU"

class Scheduler:
    def __init__(self, 
                 parent_pipes: List[Connection], 
                 timeout_in_seconds: float, 
                 policy: Policy,
                 cpu_tasks_cap: int,
                 deadlines: List[float],
                 lost_cause_threshold: int,
     ) -> None:
        self.parent_pipes: List[Connection] = parent_pipes
        self.timeout: float = timeout_in_seconds
        if policy == Policy.FIFO:
            self.policy = self.fifo
        elif policy == Policy.SLO_ORIENTED:
            self.policy: Callable[..., int] = self.slo_oriented

        self.children_states: Dict[int, CPUState] = {}
        self.active_cpus = 0
        self.CPU_TASKS_CAP = cpu_tasks_cap
        self.deadlines = deadlines
        self.lost_cause_threshold = lost_cause_threshold
        self.trace_prefix = f"*** {self.__class__.__name__}: "
        print(self.trace_prefix, f"Policy: {policy}, CPU tasks cap: {cpu_tasks_cap}, Lost cause threshold: {lost_cause_threshold}")
        self.start_time = time.time()
    
    def run(self) -> bool:
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
                    # print(self.trace_prefix, f"Received signal {signal_type} from {client_id}", "Active CPUs: "+str(self.active_cpus))#, hashmap_stage, hashmap_state)
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
                        raise ValueError(f"Invalid signal type {signal_type}")
            else:
                # Handle timeout
                if len(self.children_states) != 0:
                    for process_id, child_state in self.children_states.items():
                        print(f"Client {process_id} stuck at {child_state}!", file=sys.stderr)
                print("No data received within the timeout period.")
                break  # Break out of the loop
        self.cleanup()
        return True
    
    def cleanup(self):
        for pipe in self.parent_pipes:
            pipe.close()

    def fifo(self) -> bool:
        candidates = [key for key, value in self.children_states.items() if value == CPUState.WAITING_FOR_CPU]
        if len(candidates) == 0:
            return False
        min_process_id = int(min(candidates))
        try:
            self.parent_pipes[min_process_id].send((min_process_id, Message.ALLOCATE_CPU))
        except ValueError:
            print(self.trace_prefix, "No client to allocate the spare CPU. States:", self.children_states)
            return False
        else:
            self.children_states[min_process_id] = CPUState.CPU
            # print(self.trace_prefix, f"CPU is allocated to {min_process_id}.")
            return True

    def slo_oriented(self):
        
        def get_remaining_time(deadline):
            return self.start_time + deadline - time.time()
        
        candidates = [child_id
            for child_id, deadline in enumerate(self.deadlines)
            if - get_remaining_time(deadline) < self.lost_cause_threshold and 
            child_id in self.children_states and self.children_states[child_id] == CPUState.WAITING_FOR_CPU
        ]
        if len(candidates) == 0:
            candidates = [key for key, value in self.children_states.items() if value == CPUState.WAITING_FOR_CPU]
            if len(candidates) == 0:
                return False
            else:
                print(self.trace_prefix, f"No other candidates but {len(candidates)} lost causes.")
        # Prioritize the client with the earliest deadline
        # But consider a client as failed cause if it misses its deadline by max(PRIORITY_TO_LATENCY_GOAL)
        process_chosen = int(min(candidates, key = lambda candidate: get_remaining_time(self.deadlines[candidate])))
        try:
            self.parent_pipes[process_chosen].send((process_chosen, Message.ALLOCATE_CPU))
        except ValueError:
            print(self.trace_prefix, "No client to allocate the spare CPU. States:", self.children_states)
            return False
        else:
            self.children_states[process_chosen] = CPUState.CPU
            print(self.trace_prefix, f"CPU is allocated to {process_chosen}, trying to satisfy its deadline in {get_remaining_time(self.deadlines[process_chosen]):5f} s.")
            return True