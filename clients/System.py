
import time, os, multiprocessing
from typing import List, Dict
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from Client import Client
from Scheduler import Scheduler, Policy
from Comparison import DataType, SystemType, Comparison

PROCESS_CAP = 20

class System:
    def __init__(self, comparison: Comparison, system_type: SystemType, 
                 policy: Policy = None, cpu_tasks_cap: int = 4, lost_cause_threshold: int = 5,  # for pipeline
                 ) -> None:
        self.comparison = comparison
        self.trace_prefix = f"*** {self.__class__.__name__}: "
        self.log_path = System.get_log_dir(comparison.data_type)
        os.makedirs(self.log_path, exist_ok = True)
        self.clients: List[Client] = []
        
        # ----- System specific parameters
        if system_type == SystemType.NAIVE_SEQUENTIAL:
            self.create_client_func = self.naive_sequential
        elif system_type == SystemType.NON_COORDINATED_BATCH:
            self.create_client_func = self.non_coordinated_batch
        elif system_type == SystemType.PIPELINE:
            self.create_client_func = self.pipeline_client
            assert(policy is not None)
            self.policy = policy
            self.cpu_tasks_cap = cpu_tasks_cap
            self.lost_cause_threshold = lost_cause_threshold
            self.parent_pipes: List[Connection] = []
            self.child_pipes: List[Connection] = []
            for i in range(comparison.client_num):
                parent_pipe,child_pipe = Pipe()
                self.parent_pipes.append(parent_pipe)
                self.child_pipes.append(child_pipe)
        else:
            raise ValueError("Invalid system type")
    
    def run(self) -> Dict[str, float]:
        batch_arrive_process = Process(target = self.batch_arrive)
        batch_arrive_process.start()
        if self.create_client_func == self.pipeline_client:
            scheduler = Scheduler(self.parent_pipes, self.comparison.batch_size, self.policy, self.cpu_tasks_cap, self.comparison.deadlines, self.lost_cause_threshold)
            scheduler.run()
            
    
    def naive_sequential(self, batch, client_id, t0) -> None:
        client = self.client_class(self.log_path, batch, client_id, None, t0)
        self.clients.append(client)
        client.run()
        return None
    
    def non_coordinated_batch(self, batch, client_id, t0) -> None:
        client = self.client_class(self.log_path, batch, client_id, None, t0)
        self.clients.append(client)
        p = Process(target=client.run)
        p.start()
        return None
    
    def pipeline_client(self, batch, client_id, t0) -> None:
        client = self.client_class(self.log_path, batch, client_id, self.child_pipes[client_id], t0)
        self.clients.append(client)
        p = Process(target=client.run)
        p.start()
        return None
    
    def batch_arrive(self) -> None:
        data_paths = self.comparison.data_paths
        client_num = self.comparison.client_num
        batch_size = self.comparison.batch_size
        blocked_time = 0
        
        for client_id in range(client_num):
            t0 = time.time()
            batch = data_paths[client_id * batch_size : (client_id + 1) * batch_size]
            
            # ----- Avoid overcrowding the system
            while len(multiprocessing.active_children()) > PROCESS_CAP:
                time.sleep(0.001)
            
            # ----- Start the process
            # In other systems, blocked_time should be close to 0
            p = self.create_client_func(batch, client_id, t0 - blocked_time) # blocked time: should've started earlier
            
            blocked_time += time.time() - t0
            # Calibrate t0 for
            # 1. blocking behavior of naive sequential 
            # 2. blocking time from avoiding overcrowding
            # Requests keep coming in, regardless of those behaviors

            time.sleep(self.comparison.intervals[client_id])
            # Interval should solely contribute to the difference in time between requests

        print(self.trace_prefix, f"Sent {client_num} clients in total.")
    
    @staticmethod
    def get_log_dir(type: str) -> str:
        start_time = time.time()
        assert(isinstance(type, DataType))
        if type == DataType.IMAGE:
            parent_dir = "../log_image/"
        else:
            parent_dir = "../log_audio/"
        return parent_dir + type.value.lower() + "_" + str(start_time) + "/"