from image_client import main as client, Message, Stage, CPUState
from multiprocessing import Pipe,Manager
import multiprocessing
from multiprocessing.connection import Connection
from image_subprocesses import batch_arrival, get_batch_args

hashmap_stage = {}
hashmap_state = {}
CPU_USING = True

def schedule(min_interval, max_interval, batch_size):
    def try_run_cpu(parent_pipe):
        global hashmap_stage,hashmap_state,CPU_USING
        try:
            min_process_id = min(key for key, value in hashmap_state.items() if value == CPUState.WAITING_FOR_CPU)
            min_process_id = int(min_process_id)
            if hashmap_stage[min_process_id] == Stage.NOT_START:
                hashmap_stage[min_process_id] = Stage.PREPROCESSING
                hashmap_state[min_process_id] = CPUState.CPU
            if hashmap_stage[min_process_id] == Stage.DETECTION_INFERENCE:
                hashmap_stage[min_process_id] = Stage.CROPPING
                hashmap_state[min_process_id] = CPUState.CPU
            elif hashmap_stage[min_process_id] == Stage.RECOGNITION_INFERENCE:
                hashmap_stage[min_process_id] = Stage.POSTPROCESSING
                hashmap_state[min_process_id] = CPUState.CPU
            parent_pipe.send((min_process_id, Message.CPU_AVAILABLE))
            # print("CPU working",min_process_id,str(hashmap_stage[min_process_id]))
            CPU_USING = True
        except ValueError:
            print("spare CPU, hashmap_stage:", hashmap_stage)
            CPU_USING = False

    
    global hashmap_stage,hashmap_state,CPU_USING
    parent_pipe, child_pipe = Pipe()

    p = multiprocessing.Process(target = batch_arrival,args=(min_interval, max_interval, batch_size, \
            lambda batch, id : create_client(batch, id, child_pipe, parent_pipe)))
    p.start()
    

    while True:
        client_id, signal_type = parent_pipe.recv()
        client_id = int(client_id)
        print("pipeline received:",client_id, signal_type,CPU_USING)
        print(hashmap_stage)
        print(hashmap_state)
        if signal_type == Message.CREATE_PROCESS:
            if client_id == 0:
                hashmap_stage[client_id] = Stage.PREPROCESSING
                hashmap_state[client_id] = CPUState.CPU
            else: 
                hashmap_stage[client_id] = Stage.NOT_START
                hashmap_state[client_id] = CPUState.WAITING_FOR_CPU
                if not CPU_USING:
                    try_run_cpu(parent_pipe)
        elif signal_type == Message.CPU_AVAILABLE:
            # TODO: Schedule the incomplete process with the smallest id in the hashmap √
            # TODO: Update hashmap: Add the process that is first scheduled OR update the stage of an active process √
            if hashmap_stage[client_id] == Stage.POSTPROCESSING:
                del hashmap_stage[client_id]
                del hashmap_state[client_id]
            elif hashmap_stage[client_id] == Stage.PREPROCESSING:
                hashmap_stage[client_id] = Stage.DETECTION_INFERENCE
                hashmap_state[client_id] = CPUState.GPU
            elif hashmap_stage[client_id] == Stage.CROPPING:
                hashmap_stage[client_id] = Stage.RECOGNITION_INFERENCE
                hashmap_state[client_id] = CPUState.GPU
            try_run_cpu(parent_pipe)
        elif signal_type == Message.WAITING_FOR_CPU:
            hashmap_state[client_id] = CPUState.WAITING_FOR_CPU
            if not CPU_USING:
                try_run_cpu(parent_pipe)

def create_client(image_paths, process_id, child_pipe, parent_pipe):
    child_pipe.send((process_id, Message.CREATE_PROCESS))
    p = multiprocessing.Process(target=client, args=(image_paths,process_id,child_pipe))
    p.start()
    if process_id == 0: # start the first process upon creation
        parent_pipe.send((0, Message.CPU_AVAILABLE))

if __name__ == "__main__":
    args = get_batch_args()
    schedule(args.min, args.max, args.batch_size)