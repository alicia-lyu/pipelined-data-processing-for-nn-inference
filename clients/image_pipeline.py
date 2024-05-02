from image_client import main as client, Message, Stage, CPUState
from multiprocessing import Pipe, Process, Event
from multiprocessing.connection import Connection
from typing import Callable, Dict, List
from utils import trace, batch_arrival, get_batch_args, read_images_from_folder, IMAGE_FOLDER
from policies import non_sharing_pipeline
import select

CPU_TASKS_CAP = 10 # LATER: Tune this variable

# grant CPU to a child process and update its stage and state accordingly
@trace(__file__)
def grant_cpu(process_id: int, hashmap_stage: Dict[int, Stage], hashmap_state: Dict[int, CPUState]) -> None:
    print(grant_cpu.trace_prefix(), f"Granted CPU to {process_id}, originally {hashmap_stage[process_id]}, {hashmap_state[process_id]}")
    if hashmap_stage[process_id] == Stage.NOT_START:
        hashmap_stage[process_id] = Stage.PREPROCESSING
        hashmap_state[process_id] = CPUState.CPU
    elif hashmap_stage[process_id] == Stage.DETECTION_INFERENCE:
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
        # TODO: response time for pipeline
    elif hashmap_stage[process_id] == Stage.PREPROCESSING:
        hashmap_stage[process_id] = Stage.DETECTION_INFERENCE
        hashmap_state[process_id] = CPUState.GPU
    elif hashmap_stage[process_id] == Stage.CROPPING:
        hashmap_stage[process_id] = Stage.RECOGNITION_INFERENCE
        hashmap_state[process_id] = CPUState.GPU

@trace(__file__)
def schedule(parent_pipes: List[Connection],child_pipes: List[Connection],timeout_in_seconds:float,
             grant_cpu_func: Callable[[int, Dict[int, Stage], Dict[int, CPUState]], None],
             relinquish_cpu_func: Callable[[int, Dict[int, Stage], Dict[int, CPUState]], None],
             policy_func: Callable[[Connection, Dict[int, Stage], Dict[int, CPUState], bool, Callable[[int], None]], None]
             ) -> None:
    hashmap_stage: Dict[int, Stage] = {}
    hashmap_state: Dict[int, CPUState] = {}
    cpu_using: bool = False # Only one process occupies CPU to ensure meeting latency SLO
    # LATER: I think to allow multiple processes to work together but cap the number of processes working simultaneously
    # We can keep all code here, simply change the cpu_using from a bool to a int, if it is smaller than CPU_TASKS_CAP,
    # we can grant resource to a new task
    
    # Act on received signal from a child
    while True: 
        # Wait for data on the parent_pipe or until the timeout expires
        # LATER: Only wait if cpu reaches cap
        ready = select.select(parent_pipes, [], [], timeout_in_seconds)
        if ready[0]:
            # Process the received data
            for ready_pipe in ready[0]:
                client_id, signal_type = ready_pipe.recv()
                client_id = int(client_id)
                print(schedule.trace_prefix(), f"Received signal {signal_type} from {client_id}", "CPU using: "+str(cpu_using))#, hashmap_stage, hashmap_state)
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
                    cpu_using = policy_func(parent_pipes, hashmap_stage, hashmap_state, cpu_using, grant_cpu_func)
        else:
            # Handle timeout
            print("No data received within the timeout period.")
            break  # Break out of the loop

@trace(__file__)
def create_client(log_dir_name:str,image_paths: List[str], process_id: int, child_pipe: Connection, t0: float = None) -> None:
    child_pipe.send((process_id, Message.CREATE_PROCESS))
    p = Process(target=client, args=(log_dir_name,image_paths, process_id, child_pipe, t0))
    p.start()
    return p

if __name__ == "__main__":
    stop_flag = Event() # stop the batch arrival when the scheduler stops

    args = get_batch_args()
    image_paths = read_images_from_folder(IMAGE_FOLDER)
    
    parent_pipes: List[Connection] = []
    child_pipes: List[Connection] = []
    processes: List[Process] = []

    for i in range(len(image_paths) // args.batch_size):
        parent_pipe,child_pipe = Pipe()
        parent_pipes.append(parent_pipe)
        child_pipes.append(child_pipe)

    # Non-blocking (run at the same time with the scheduler): images arrive in batch
    batch_arrival_process = Process(target=batch_arrival, \
                                args=(args.min, args.max, args.batch_size,args.type, \
                                    image_paths, \
                                    lambda log_dir_name,batch, id,t0: create_client(log_dir_name,batch, id,child_pipes[id],t0),stop_flag))
    batch_arrival_process.start()

    schedule(parent_pipes,child_pipes,args.timeout, grant_cpu, relinquish_cpu, non_sharing_pipeline)
    stop_flag.set()
    for p in processes:
        p.terminate()