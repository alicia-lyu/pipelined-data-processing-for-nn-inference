from audio_client import AudioRecognitionClient
from image_client import TextRecognitionClient
from enum import Enum
from typing import List, NamedTuple, Optional, Dict
import os, math, random, numpy as np
from Scheduler import Policy
from System import System
from Client import Stats

class DataType(Enum):
    IMAGE = "image"
    AUDIO = "audio"

class RandomPattern(Enum):
    UNIFORM = "UNIFORM"
    EXP = "EXP"
    POISSON = "POISSON"
    

class SystemType(Enum):
    NAIVE_SEQUENTIAL = "naive-sequential"
    NON_COORDINATED_BATCH = "non-coordinated-batch"
    PIPELINE = "pipeline"

class SystemArgs(NamedTuple):
    system_type: SystemType
    policy: Optional[Policy]
    cpu_tasks_cap: Optional[int]
    lost_cause_threshold: Optional[int]
    def __str__(self):
        system_name = self.system_type.value
        system_name += f"_{self.policy.value}" if self.policy is not None else ""
        system_name += f"_cpu_tasks_cap={self.cpu_tasks_cap}" if self.cpu_tasks_cap is not None else ""
        system_name += f"_lost_cause_threshold={self.lost_cause_threshold}" if self.lost_cause_threshold is not None else ""
        return system_name

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

IMAGE_FOLDER = "../../datasets/SceneTrialTrain"
AUDIO_FOLDER = "../../datasets/audio_data/mp3_16_data_2"

class Comparison:
    def __init__(self, min_interval: int, max_interval: int, batch_size: int, data_type: DataType, random_pattern: RandomPattern) -> None:
        # ----- Batch specific parameters
        self.batch_size = batch_size
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.random_pattern = random_pattern
        self.trace_prefix = f"*** {self.__class__.__name__}: "
        # ----- Data type specific parameters
        if data_type == DataType.IMAGE:
            self.client_class = TextRecognitionClient
            extension = ".jpg"
            priority_map = PRIORITY_TO_LATENCY_IMAGE # TODO: Provide more priority map choices
        elif data_type == DataType.AUDIO:
            self.client_class = AudioRecognitionClient
            extension = ".mp3"
            priority_map = PRIORITY_TO_LATENCY_AUDIO
        else:
            raise ValueError("Invalid data type")
        self.data_type = data_type
        self.data_paths = Comparison.read_data_from_folder(extension)
        
        self.client_num = math.ceil(len(self.data_paths) / batch_size)
        self.intervals = self.generate_random_intervals()
        self.deadlines = self.generate_deadlines(priority_map)
        self.stats: Dict[str, List[Stats]] = {}

    def compare(self, system_args_list: List[SystemArgs]) -> None:
        for system_args in system_args_list:
            system = System(self, system_args.system_type, system_args.policy, system_args.cpu_tasks_cap, system_args.lost_cause_threshold)
            system_stats = system.run()
            self.stats[str(system_args)] = system_stats
        self.plot()
        
    def plot(self) -> None:
        pass # TODO
    
    def generate_random_intervals(self):
        intervals = []
        for i in range(self.client_num):
            if self.random_pattern == RandomPattern.UNIFORM:
                interval = random.uniform(self.min_interval, self.max_interval)
            elif self.random_pattern == RandomPattern.EXP:
                interval = Comparison.exp_random(self.min_interval, self.max_interval)
            elif self.random_pattern == RandomPattern.POISSON:
                interval = Comparison.poisson_random(self.min_interval, self.max_interval)
            intervals.append(interval)
        return intervals
    
    def generate_deadlines(self, priority_map: dict):
        assert(len(self.intervals) == self.client_num)
        deadlines = []
        interval_accumulator = 0
        for i in range(self.client_num):
            priority = random.randint(1, len(priority_map))
            latency_goal = priority_map[priority]
            deadlines[i] = interval_accumulator + latency_goal
            interval_accumulator += self.intervals[i]
        return deadlines
    
    @staticmethod 
    def exp_random(min_val, max_val, lambda_val=1):
        exponential_random_number = np.random.exponential(scale=1/lambda_val)
        return min_val + (max_val - min_val) * (exponential_random_number / (1/lambda_val))

    @staticmethod
    def poisson_random(min_val, max_val, lambda_val=1):
        poisson_random_number = np.random.poisson(lam=lambda_val)
        return min_val + (max_val - min_val) * (poisson_random_number / lambda_val)
    
    @staticmethod
    def trace_prefix():
        return f"*** {Comparison.__class__.__name__} static: "
    
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

        print(Comparison.trace_prefix(), f"Found {len(data_paths)} data.")

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