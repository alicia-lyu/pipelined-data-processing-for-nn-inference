from image_client import main as client, Message, Stage, CPUState
from multiprocessing import Pipe,Manager
import multiprocessing
from multiprocessing.connection import Connection
from image_subprocesses import batch_arrival, get_batch_args

def grant_cpu(process_id, hashmap_stage, hashmap_state): # grant CPU to a child process and update its stage and state accordingly
    if hashmap_stage[process_id] == Stage.NOT_START:
        hashmap_stage[process_id] = Stage.PREPROCESSING
        hashmap_state[process_id] = CPUState.CPU
    if hashmap_stage[process_id] == Stage.DETECTION_INFERENCE:
        hashmap_stage[process_id] = Stage.CROPPING
        hashmap_state[process_id] = CPUState.CPU
    elif hashmap_stage[process_id] == Stage.RECOGNITION_INFERENCE:
        hashmap_stage[process_id] = Stage.POSTPROCESSING
        hashmap_state[process_id] = CPUState.CPU

def relinquish_cpu(process_id, hashmap_stage, hashmap_state): # Update the stage and state of a child client when it relinquishes CPU
    if hashmap_stage[process_id] == Stage.POSTPROCESSING: # Finished all stages
        del hashmap_stage[process_id]
        del hashmap_state[process_id]
    elif hashmap_stage[process_id] == Stage.PREPROCESSING:
        hashmap_stage[process_id] = Stage.DETECTION_INFERENCE
        hashmap_state[process_id] = CPUState.GPU
    elif hashmap_stage[process_id] == Stage.CROPPING:
        hashmap_stage[process_id] = Stage.RECOGNITION_INFERENCE
        hashmap_state[process_id] = CPUState.GPU

def schedule(parent_pipe, grant_cpu_func, relinquish_cpu_func):
    hashmap_stage = {}
    hashmap_state = {}
    cpu_using = False

    def try_run_cpu():
        nonlocal parent_pipe, hashmap_stage, hashmap_state, cpu_using

        try:
            min_process_id = min(key \
                                 for key, value in hashmap_state.items() if value == CPUState.WAITING_FOR_CPU) # Allocate CPU to the eligible client with the smallest id (FIFO)
            min_process_id = int(min_process_id)
            grant_cpu_func(min_process_id, hashmap_stage, hashmap_state)
            parent_pipe.send((min_process_id, Message.CPU_AVAILABLE))
            # print("CPU working",min_process_id,str(hashmap_stage[min_process_id]))
            cpu_using = True
        except ValueError:
            print("No client to allocate the spare CPU. Hashmap_stage:", hashmap_stage)
            cpu_using = False
    
    while True:
        client_id, signal_type = parent_pipe.recv()
        # Act on received signal from a child
        client_id = int(client_id)
        print("pipeline received:",client_id, signal_type,cpu_using)
        print(hashmap_stage)
        print(hashmap_state)
        if signal_type == Message.CREATE_PROCESS: # A child client is first created (in create_client)
            hashmap_stage[client_id] = Stage.NOT_START
            hashmap_state[client_id] = CPUState.WAITING_FOR_CPU
            if not cpu_using:
                try_run_cpu()

        elif signal_type == Message.CPU_AVAILABLE: # A child client relinquishes CPU
            relinquish_cpu_func(client_id, hashmap_stage, hashmap_state)
            try_run_cpu() # Allocate the available CPU according to the policy coded in try_run_cpu

        elif signal_type == Message.WAITING_FOR_CPU: # A child client finishes GPU tasks and is now waiting for CPU
            hashmap_state[client_id] = CPUState.WAITING_FOR_CPU
            if not cpu_using: # When the CPU is not allocated when it first became available
                try_run_cpu()

def create_client(image_paths, process_id, child_pipe):
    child_pipe.send((process_id, Message.CREATE_PROCESS))
    p = multiprocessing.Process(target=client, args=(image_paths,process_id,child_pipe))
    p.start()

if __name__ == "__main__":
    args = get_batch_args()
    parent_pipe, child_pipe = Pipe()
    # Non-blocking (run at the same time with the scheduler): images arrive in batch
    p = multiprocessing.Process(target = batch_arrival, args=(args.min, args.max, args.batch_size, \
            lambda batch, id : create_client(batch, id, child_pipe)))
    p.start()
    schedule(parent_pipe, grant_cpu, relinquish_cpu)