from audio_client import AudioRecognitionClient
from image_client import TextRecognitionClient
from enum import Enum
import time, os, random, multiprocessing
from typing import List, Callable
from multiprocessing import Process, Event, Pipe
from multiprocessing.connection import Connection
import numpy as np
from Scheduler import Scheduler, Policy

class SystemType(Enum):
    NAIVE_SEQUENTIAL = "naive-sequential"
    NON_COORDINATED_BATCH = "non-coordinated-batch"
    PIPELINE = "pipeline"
    
class DataType(Enum):
    IMAGE = "image"
    AUDIO = "audio"
    
PRIORITY_TO_LATENCY_IMAGE = {
    1: 3.0,
    2: 4.0,
    3: 5.0,
    4: 6.0
}

PRIORITY_TO_LATENCY_AUDIO = {
    1: 7.0,
    2: 10.0,
    3: 13.0,
    4: 16.0
}

class ModelType(Enum):
    IMAGE = "IMAGE"
    AUDIO = "AUDIO"

IMAGE_FOLDER = "../../datasets/SceneTrialTrain"
AUDIO_FOLDER = "../../datasets/audio_data/mp3_16_data_2"

PROCESS_CAP = 20

class RandomPattern(Enum):
    UNIFORM = "UNIFORM"
    EXP = "EXP"
    POISSON = "POISSON"

class System:
    def __init__(self, min_interval: int, max_interval: int, batch_size: int, data_type: DataType, system_type: SystemType, random_pattern: RandomPattern,
                 policy: Policy = None, cpu_tasks_cap: int = 4, priority_map_index: int = None, lost_cause_threshold: int = 5,  # for pipeline
                 ) -> None:
        
        self.batch_params = (min_interval, max_interval, batch_size)
        self.trace_prefix = f"*** {self.__class__.__name__}: "
        self.random_pattern = random_pattern
        self.processes: List[Process] = []
        
        # ----- Data specific parameters
        if data_type == DataType.IMAGE:
            self.client_class = TextRecognitionClient
            extension = ".jpg"
            model_type = ModelType.IMAGE
            self.priority_map = PRIORITY_TO_LATENCY_IMAGE # TODO: Provide more priority map choices
        elif data_type == DataType.AUDIO:
            self.client_class = AudioRecognitionClient
            extension = ".mp3"
            model_type = ModelType.AUDIO
            self.priority_map = PRIORITY_TO_LATENCY_AUDIO
        else:
            raise ValueError("Invalid data type")
        self.data_paths = System.read_data_from_folder(extension)
        self.log_path = System.get_log_dir(model_type)
        os.makedirs(self.log_path, exist_ok = True)
        
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
            self.parent_pipes: List[Connection] = []
            self.child_pipes: List[Connection] = []
            for i in range(len(self.data_paths) // batch_size):
                parent_pipe,child_pipe = Pipe()
                self.parent_pipes.append(parent_pipe)
                self.child_pipes.append(child_pipe)
        else:
            raise ValueError("Invalid system type")
    
    def run(self) -> None:
        batch_arrive_process = Process(target = self.batch_arrive)
        batch_arrive_process.start()
        if self.create_client_func == self.pipeline_client:
            scheduler = Scheduler(self.parent_pipes, self.batch_params[2], self.policy, self.cpu_tasks_cap, self.priority_map)
            scheduler.run()
        if len(self.processes) > 0:
            for p in self.processes:
                # TODO collect stats?
                pass
    
    def naive_sequential(self, batch, client_id, t0) -> None:
        client = self.client_class(self.log_path, batch, client_id, None, t0)
        client.run()
        return None # TODO stats?
    
    def non_coordinated_batch(self, batch, client_id, t0) -> Process:
        client = self.client_class(self.log_path, batch, client_id, None, t0)
        p = Process(target=client.run)
        p.start()
        return p
    
    def pipeline_client(self, batch, client_id, t0) -> Process:
        if self.policy == Policy.SLO_ORIENTED:
            priority = random.randint(1, 4)
        else:
            priority = None
        deadline = t0 + self.priority_map[priority] # TODO: All systems should have a deadline, whether honored or not?
        client = self.client_class(self.log_path, batch, client_id, self.child_pipes[client_id], t0)
        self.child_pipes[client_id].send(deadline)
        p = Process(target=client.run)
        p.start()
        return p
    
    def batch_arrive(self) -> int:
        
        def exp_random(min_val, max_val, lambda_val=1):
            exponential_random_number = np.random.exponential(scale=1/lambda_val)
            return min_val + (max_val - min_val) * (exponential_random_number / (1/lambda_val))

        def poisson_random(min_val, max_val, lambda_val=1):
            poisson_random_number = np.random.poisson(lam=lambda_val)
            return min_val + (max_val - min_val) * (poisson_random_number / lambda_val)

        min_interval, max_interval, batch_size = self.batch_params
        blocked_time = 0
        
        for i in range(0, len(self.data_paths), batch_size):
            
            batch = self.data_paths[i: i + batch_size]
            client_id = i // batch_size
            
            # ----- Avoid overcrowding the system
            while len(multiprocessing.active_children()) > PROCESS_CAP:
                time.sleep(0.001)
            
            # ----- Start the process
            t0 = time.time()
            # Calibrate t0 for naive sequential to include the blocked time by execution of previous processes
            # In other systems, blocked_time should be close to 0
            p = self.create_client_func(batch, client_id, t0 - blocked_time) # blocked time: should've started earlier
            blocked_time += time.time() - t0
            if p is not None:
                self.processes.append(p)
            
            # ----- Data arrival pattern determines interval
            if self.random_patten == RandomPattern.UNIFORM:
                interval = random.uniform(min_interval, max_interval) # data_arrival_pattern(min_interval, max_interval, pattern: str)
            elif self.random_patten == RandomPattern.EXP:
                interval = exp_random(min_interval, max_interval)
            elif self.random_patten == RandomPattern.POISSON:
                interval = poisson_random(min_interval, max_interval)
            else:
                raise ValueError("Invalid random pattern")
            time.sleep(interval)

        client_num = len(self.data_paths) // batch_size
        print(self.trace_prefix, f"Sent {client_num} clients in total.")
        return client_num
    
    @staticmethod
    def trace_prefix():
        return f"*** {System.__class__.__name__} static: "    
    
    @staticmethod
    def get_log_dir(type: str) -> str:
        start_time = time.time()
        assert(isinstance(type, ModelType))
        if type == ModelType.IMAGE:
            parent_dir = "../log_image/"
        else:
            parent_dir = "../log_audio/"
        return parent_dir + type.value.lower() + "_" + str(start_time) + "/"

    @staticmethod
    def read_data_from_folder(extension: str) -> List[str]:
        data_paths = []
        
        if extension == ".jpg":
            root_folder = IMAGE_FOLDER
        elif extension == ".mp3":
            root_folder = AUDIO_FOLDER
        else:
            raise ValueError("Invalid extension")

        for root, dirs, files in os.walk(root_folder):
            for file in files:
                if file.lower().endswith(extension):
                    image_path = os.path.join(root, file)
                    data_paths.append(image_path)

        print(System.trace_prefix(), f"Found {len(data_paths)} data.")

        return data_paths
    
    @staticmethod
    def map_args_to_enum(data_type_arg: str, system_type_arg: str, random_pattern_arg) -> tuple[DataType, SystemType]:
        try:
            data_type = DataType(data_type_arg)
            system_type = SystemType(system_type_arg)
            random_pattern = RandomPattern(random_pattern_arg)
        except ValueError:
            raise ValueError("Invalid data type, system type, or random pattern")
        return data_type, system_type, random_pattern