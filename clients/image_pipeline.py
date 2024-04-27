from image_client import main as client, Message
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from image_subprocesses import batch_arrival, get_batch_args

def schedule(min_interval, max_interval, batch_size):
    parent_pipe, child_pipe = Pipe()
    batch_arrival(min_interval, max_interval, batch_size, \
            lambda batch, id : create_client(batch, id, child_pipe, parent_pipe))
    # TODO: Make batch_arrival non-blocking, so that the while loop can happen in the same time as batch_arrival (using thread?)
    
    while True:
        client_id, signal_type = parent_pipe.recv()
        # TODO: Maintain a hashmap of all active process (started but not finished) (id -> stage)
        # TODO: Update hashmap: Update the stage of the process when receiving its message; delete it if it will complete, i.e. won't be blocked by wait_signal again
        
        if signal_type == Message.CPU_AVAILABLE:
            # TODO: Schedule the incomplete process with the smallest id in the hashmap
            # TODO: Update hashmap: Add the process that is first scheduled OR update the stage of an active process
            pass

def create_client(image_paths, process_id, child_pipe, parent_pipe):
    # multiprocessing.??(target=client, args=[])
    if process_id == 0: # start the first process upon creation
        parent_pipe.send((0, Message.CPU_AVAILABLE))
        # TODO initiate hashmap and add client 0
    

if __name__ == "__main__":
    args = get_batch_args()
    schedule(args.min, args.max, args.batch_size)