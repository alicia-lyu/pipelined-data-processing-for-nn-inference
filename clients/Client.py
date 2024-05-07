from multiprocessing.connection import Connection
import time
from abc import ABCMeta, abstractmethod
import tritonclient.http as httpclient # type: ignore
from utils import trace
from Scheduler import Message
from typing import Dict

class Client(metaclass=ABCMeta):
    def __init__(self, log_dir_name, batch, process_id, t0: float, stats: Dict = None, signal_pipe: Connection = None):
        self.filename = log_dir_name + str(process_id).zfill(3) + ".txt"
        self.batch = batch
        self.process_id = process_id
        self.triton_client = httpclient.InferenceServerClient(url="localhost:8000")
        self.pipe = signal_pipe
        if t0 is None:
            self.t0 = time.time()
        else:
            self.t0 = t0
        self.stats = stats
        self.stats["created"] = t0
        self.trace_prefix = f"*** {self.__class__.__name__}: "
    
    @abstractmethod
    def run(self):
        pass
    
    @abstractmethod
    def log(self):
        pass
    
    @trace(__file__)
    def wait_signal(self, signal_awaited: str) -> None:
        if self.pipe == None: # Not coordinating multiple processes
            return
        start = time.time()
        self.send_signal(Message.WAITING_FOR_CPU) # tell scheduler that the process is waiting for CPU
        while True:
            receiver_id, signal_type = self.pipe.recv()
            if receiver_id == self.process_id and signal_type == signal_awaited:
                break
        end = time.time()
        print(self.trace_prefix(), f"Process {self.process_id} waited for signal {signal_awaited} for {end - start: .5f}.")
        
    @trace(__file__)
    def send_signal(self, signal_to_send):
        if self.pipe == None: # Not coordinating multiple processes
            return
        print(self.trace_prefix(), "Process %d sent signal %s." % (self.process_id, signal_to_send))
        self.pipe.send((self.process_id, signal_to_send))