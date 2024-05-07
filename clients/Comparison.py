from audio_client import AudioRecognitionClient
from image_client import TextRecognitionClient
from enum import Enum
from typing import List, NamedTuple, Optional, Dict
import os, math, random, numpy as np # type: ignore
from Scheduler import Policy
import time
from StatsProcessor import StatsProcessor, Stats, ImageStats

class DataType(Enum):
    IMAGE = "image"
    AUDIO = "audio"

class RandomPattern(Enum):
    UNIFORM = "uniform"
    POISSON = "poisson"
    

class SystemType(Enum):
    NAIVE_SEQUENTIAL = "naive-sequential"
    NON_COORDINATED_BATCH = "non-coordinated-batch"
    PIPELINE = "pipeline"

class SystemArgs(NamedTuple):
    system_type: SystemType
    policy: Optional[Policy] = None
    cpu_tasks_cap: Optional[int] = None
    lost_cause_threshold: Optional[int] = None
    def __str__(self):
        system_name = self.system_type.value
        system_name += f"_{self.policy.value}" if self.policy is not None else ""
        system_name += f"_cpu_tasks_cap={self.cpu_tasks_cap}" if self.cpu_tasks_cap is not None and self.cpu_tasks_cap > 1 else ""
        return system_name

PRIORITY_TO_LATENCY_IMAGE = {
    1: 3.0,
    2: 5.0,
    3: 7.0,
    4: 9.0
}

PRIORITY_TO_LATENCY_AUDIO = {
    1: 6.0,
    2: 8.0,
    3: 10.0,
    4: 12.0
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
        self.dir_name = f"../logs/{data_type.value}-{random_pattern.value.lower()}-{min_interval}-{max_interval}-{batch_size}-{time.strftime('%H:%M:%S')}"
        os.makedirs(self.dir_name, exist_ok=True)
        # ----- Data type specific parameters
        if data_type == DataType.IMAGE:
            self.client_class = TextRecognitionClient
            extension = ".jpg"
            priority_map = PRIORITY_TO_LATENCY_IMAGE
        elif data_type == DataType.AUDIO:
            self.client_class = AudioRecognitionClient
            extension = ".mp3"
            priority_map = PRIORITY_TO_LATENCY_AUDIO
        else:
            raise ValueError("Invalid data type")
        self.data_type = data_type
        self.data_paths = Comparison.read_data_from_folder(extension)[:120]
        self.priority_map = priority_map
        
        self.client_num = math.ceil(len(self.data_paths) / batch_size)
        self.intervals = self.generate_random_intervals()
        self.priorities, self.deadlines = self.generate_deadlines(priority_map) # Deadlines are relative to the start of a system
        # print(self.intervals)
        # print(self.deadlines)
        self.stats: Dict[str, List[Stats]] = {}
        
        print(self.trace_prefix, f"batch_size={batch_size}, client_num={self.client_num}, \
min_interval={min_interval}, max_interval={max_interval}, random_pattern={random_pattern}, \
data_type={data_type}, priority_map={priority_map}")

    def compare(self, system_args_list: List[SystemArgs]) -> bool:
        from System import System
        print(self.trace_prefix, f"Comparing {len(system_args_list)} systems. Created directory {self.dir_name}")
        for system_args in system_args_list:
            system = System(self, system_args)
            system_stats = system.run()
            self.stats[str(system_args)] = system_stats
            if len(system_stats) < self.client_num:
                print(self.trace_prefix, f"{str(system_args)} failed to collect stats from all clients!")
                return False
        for system_name, system_stats in self.stats.items():
            print(self.trace_prefix, f"{system_name} took {system_stats[-1].postprocess_end - system_stats[0].created} s in total")
        self.save_stats()
        self.plot()
        return True
        
    def plot(self) -> None:
        stats_processor = StatsProcessor(self.dir_name, self.stats, self.deadlines, self.priority_map, self.priorities)
        stats_processor.plot_batches()
        stats_processor.plot_stages()
    
    def save_stats(self) -> None:
        # Save deadlines
        with open(os.path.join(self.dir_name, "deadlines.csv"), "w") as f:
            for i in range(self.client_num):
                f.write(f"{i},{self.priorities[i]},{self.deadlines[i]}\n")
            f.close()
        # Save priority map
        with open(os.path.join(self.dir_name, "priority_map.csv"), "w") as f:
            for priority, latency in self.priority_map.items():
                f.write(f"{priority},{latency}\n")
            f.close()
        # Save stats
        for system_args, system_stats in self.stats.items():
            with open(os.path.join(self.dir_name, f"{str(system_args)}.csv"), "w") as f:
                for client_id, client_stats in enumerate(system_stats):
                    if isinstance(client_stats, ImageStats):
                        f.write(f"{client_id},{client_stats.created},{client_stats.preprocess_start},{client_stats.preprocess_end},\
{client_stats.inference_start},{client_stats.inference_end},{client_stats.midprocessing_start},{client_stats.midprocessing_end},\
{client_stats.inference2_start},{client_stats.inference2_end},{client_stats.postprocess_start},{client_stats.postprocess_end}\n")
                    else:
                        f.write(f"{client_id},{client_stats.created},{client_stats.preprocess_start},{client_stats.preprocess_end},\
{client_stats.inference_start},{client_stats.inference_end},{client_stats.postprocess_start},{client_stats.postprocess_end}\n")
                f.close()
    
    def generate_random_intervals(self):
        intervals = []
        for i in range(self.client_num):
            if self.random_pattern == RandomPattern.UNIFORM:
                interval = random.uniform(self.min_interval, self.max_interval)
            elif self.random_pattern == RandomPattern.POISSON:
                interval = Comparison.exp_random(self.min_interval, self.max_interval)
            intervals.append(interval)
        return intervals
    
    def generate_deadlines(self, priority_map: dict):
        assert(len(self.intervals) == self.client_num)
        deadlines = []
        priorities = []
        interval_accumulator = 0
        for i in range(self.client_num):
            priority = random.randint(1, len(priority_map))
            priorities.append(priority)
            latency_goal = priority_map[priority]
            deadlines.append(interval_accumulator + latency_goal)
            interval_accumulator += self.intervals[i]
        return priorities, deadlines
    
    @staticmethod
    def exp_random(min_interval, max_interval):
        mean_interval = (min_interval + max_interval) / 2
        return np.random.exponential(mean_interval)
    
    @staticmethod
    def trace_prefix():
        return f"*** Comparison static: "
    
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
    def map_args_to_enum(data_type_arg: str, random_pattern_arg: str) -> tuple[DataType, SystemType]:
        try:
            data_type = DataType(data_type_arg)
            random_pattern = RandomPattern(random_pattern_arg)
        except ValueError:
            raise ValueError("Invalid data type, system type, or random pattern")
        return data_type, random_pattern