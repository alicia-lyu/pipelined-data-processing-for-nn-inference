
import time, os, multiprocessing
from typing import List, Dict
from multiprocessing import Process, Pipe, Manager
from multiprocessing.connection import Connection
from Scheduler import Scheduler
from Comparison import SystemType, Comparison, SystemArgs, DataType
from audio_client import AudioRecognitionClient
from image_client import TextRecognitionClient
from StatsProcessor import Stats, ImageStats

PROCESS_CAP = 10

class System:
    def __init__(self, comparison: Comparison, system_args: SystemArgs) -> None:
        self.comparison = comparison
        self.system_args = system_args
        self.trace_prefix = f"*** {self.__class__.__name__}: "
        
        system_type, policy, cpu_tasks_cap, lost_cause_threshold = system_args
        
        self.log_path = self.get_log_dir()
        print(self.trace_prefix, f"System type: {system_type}, Policy: {policy}, CPU tasks cap: {cpu_tasks_cap}, Lost cause threshold: {lost_cause_threshold}, log_path: {self.log_path}")
        os.makedirs(self.log_path, exist_ok = True)
        self.clients: List[Process] = []
        self.clients_stats: Dict[int, Dict[str, float]] = {}
        
        if comparison.data_type == DataType.IMAGE:
            self.client_class = TextRecognitionClient
        else:
            self.client_class = AudioRecognitionClient
        
        # ----- System specific parameters
        if system_type == SystemType.NAIVE_SEQUENTIAL:
            self.create_client_func = self.naive_sequential
        elif system_type == SystemType.NON_COORDINATED_BATCH:
            self.create_client_func = self.non_coordinated_batch
            self.manager = Manager()
        elif system_type == SystemType.PIPELINE:
            self.create_client_func = self.pipeline_client
            self.manager = Manager() # For creating shared memory objects
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
    
    def run(self) -> List[Stats]:
        if self.create_client_func == self.pipeline_client:
            scheduler = Scheduler(self.parent_pipes, max(self.comparison.priority_map.values()) * 2, self.policy, self.cpu_tasks_cap, self.comparison.deadlines, self.lost_cause_threshold)
            scheduler_process = Process(target = scheduler.run)
            scheduler_process.start()
        else:
            scheduler_process = None
        self.batch_arrive()
        if scheduler_process is not None:
            scheduler_process.join()
        for client in self.clients:
            client.join()
        if self.create_client_func == self.pipeline_client:
            for pipe in self.child_pipes:
                pipe.close()
        print(self.trace_prefix, f"All clients have finished. Got stats from {len(self.clients_stats.items())} clients.")
        return self.convert_stats()
    
    def naive_sequential(self, batch, client_id, t0) -> None:
        client = self.client_class(self.log_path, batch, client_id, t0)
        stats = client.run()
        self.clients_stats[client_id] = stats
        return None
    
    def non_coordinated_batch(self, batch, client_id, t0) -> None:
        stats = self.manager.dict()
        self.clients_stats[client_id] = stats
        client = self.client_class(self.log_path, batch, client_id, t0, stats)
        p = Process(target=client.run)
        p.start()
        self.clients.append(p)
        return None
    
    def pipeline_client(self, batch, client_id, t0) -> None:
        stats = self.manager.dict()
        self.clients_stats[client_id] = stats
        client = self.client_class(self.log_path, batch, client_id, t0, stats, self.child_pipes[client_id])
        p = Process(target=client.run)
        p.start()
        self.clients.append(p)
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
            self.create_client_func(batch, client_id, t0 - blocked_time) # blocked time: should've started earlier
            
            blocked_time += time.time() - t0
            # Calibrate t0 for
            # 1. blocking behavior of naive sequential 
            # 2. blocking time from avoiding overcrowding
            # Requests keep coming in, regardless of those behaviors

            time.sleep(self.comparison.intervals[client_id])
            # Interval should solely contribute to the difference in time between requests

        print(self.trace_prefix, f"Sent {client_num} clients in total.")
    
    def convert_stats(self) -> List[Stats]:
        final_stats_list = []
        assert(len(self.clients_stats) == self.comparison.client_num)
        for client_id, stats in sorted(self.clients_stats.items(), key=lambda x: x[0]):
            if len(stats) == 7:
                try:
                    final_stats = Stats(
                        created=stats["created"],
                        preprocess_start=stats["preprocess_start"],
                        preprocess_end=stats["preprocess_end"],
                        inference_start=stats["inference_start"],
                        inference_end=stats["inference_end"],
                        postprocess_start=stats["postprocess_start"],
                        postprocess_end=stats["postprocess_end"]
                    )
                except KeyError:
                    raise ValueError("Invalid stats keys")    
            elif len(stats) == 11:
                try:
                    final_stats = ImageStats(
                        created=stats["created"],
                        preprocess_start=stats["preprocess_start"],
                        preprocess_end=stats["preprocess_end"],
                        inference_start=stats["inference_start"],
                        inference_end=stats["inference_end"],
                        midprocessing_start=stats["midprocessing_start"],
                        midprocessing_end=stats["midprocessing_end"],
                        inference2_start=stats["inference2_start"],
                        inference2_end=stats["inference2_end"],
                        postprocess_start=stats["postprocess_start"],
                        postprocess_end=stats["postprocess_end"]
                    )
                except KeyError:
                    raise ValueError("Invalid stats keys")
            else:
                raise ValueError(f"Invalid stats length {len(stats)} at {client_id}: {stats}")
            final_stats_list.append(final_stats)
        assert(client_id == self.comparison.client_num - 1)
        return final_stats_list
    
    def get_log_dir(self) -> str:
        return os.path.join(self.comparison.dir_name, str(self.system_args))