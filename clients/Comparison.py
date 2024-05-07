from audio_client import AudioRecognitionClient
from image_client import TextRecognitionClient
from enum import Enum
from typing import List, NamedTuple, Optional, Dict
import os, math, random, numpy as np # type: ignore
from Scheduler import Policy
from dataclasses import dataclass, field
import matplotlib.pyplot as plt # type: ignore

class DataType(Enum):
    IMAGE = "image"
    AUDIO = "audio"

class RandomPattern(Enum):
    UNIFORM = "uniform"
    EXP = "exponential"
    POISSON = "poisson"
    

class SystemType(Enum):
    NAIVE_SEQUENTIAL = "naive-sequential"
    NON_COORDINATED_BATCH = "non-coordinated-batch"
    PIPELINE = "pipeline"

@dataclass
class Stats:
    created: float = field(default=None)
    preprocess_start: float = field(default=None)
    preprocess_end: float = field(default=None)
    inference_start: float = field(default=None)
    inference_end: float = field(default=None)
    postprocess_start: float = field(default=None)
    postprocess_end: float = field(default=None)

@dataclass
class ImageStats(Stats):
    midprocessing_start: float = field(default=None)
    midprocessing_end: float = field(default=None)
    inference2_start: float = field(default=None)
    inference2_end: float = field(default=None)

class SystemArgs(NamedTuple):
    system_type: SystemType
    policy: Optional[Policy] = None
    cpu_tasks_cap: Optional[int] = None
    lost_cause_threshold: Optional[int] = None
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
        self.priority_map = priority_map
        
        self.client_num = math.ceil(len(self.data_paths) / batch_size)
        self.intervals = self.generate_random_intervals()
        self.priorities, self.deadlines = self.generate_deadlines(priority_map)
        self.stats: Dict[str, List[Stats]] = {}
        
        print(self.trace_prefix, f"batch_size={batch_size}, client_num={self.client_num}, \
min_interval={min_interval}, max_interval={max_interval}, random_pattern={random_pattern}, \
data_type={data_type}, priority_map={priority_map}")

    def compare(self, system_args_list: List[SystemArgs]) -> None:
        from System import System
        self.dir_name = os.path.join(f"../log_{self.data_type.value}", "__".join([str(system_args) for system_args in system_args_list]))
        os.makedirs(self.dir_name, exist_ok=True)
        print(self.trace_prefix, f"Comparing {len(system_args_list)} systems. Created directory {self.dir_name}")
        for system_args in system_args_list:
            system = System(self, system_args)
            system_stats = system.run()
            self.stats[str(system_args)] = system_stats
        self.plot()
        
    def plot(self) -> None:
        # ----- Median time taken for each stage, grouped by priority, for all system types
        fig, axes = plt.subplots(1, 3, figsize=(30, 15))
        for i, (system_name, system_stats) in enumerate(self.stats.items()):
            times: Dict = {}
            if isinstance(system_stats, ImageStats):
                stages = [
                    'Preprocess Wait',
                    'Preprocess',
                    'Inference',
                    'Midprocessing Wait',
                    'Midprocessing',
                    'Inference2',
                    'Postprocess Wait',
                    'Postprocess'
                ]
                colors = [
                    'red',
                    'yellow',
                    'blue',
                    'red',
                    'yellow',
                    'blue',
                    'red',
                    'yellow'
                ]
            else:
                stages = [
                    'Preprocess Wait',
                    'Preprocess',
                    'Inference',
                    'Postprocess Wait',
                    'Postprocess'
                ]
                colors = [
                    'red',
                    'yellow',
                    'blue',
                    'red',
                    'yellow'
                ]
            for priority in range(1, len(self.priority_map) + 1):
                stage_times = {}
                for stage in stages:
                    stage_times[stage] = []
                for client_id, client_stats in enumerate(system_stats):
                    if self.priorities[client_id] == priority:
                        if isinstance(system_stats, ImageStats):
                            stage_times['Preprocess Wait'].append(client_stats.preprocess_start - client_stats.created)
                            stage_times['Preprocess'].append(client_stats.preprocess_end - client_stats.preprocess_start)
                            stage_times['Inference'].append(client_stats.inference_end - client_stats.inference_start)
                            stage_times['Midprocessing Wait'].append(client_stats.midprocessing_start - client_stats.inference_end)
                            stage_times['Midprocessing'].append(client_stats.midprocessing_end - client_stats.midprocessing_start)
                            stage_times['Inference2'].append(client_stats.inference2_end - client_stats.inference2_start)
                            stage_times['Postprocess Wait'].append(client_stats.postprocess_start - client_stats.midprocessing_end)
                            stage_times['Postprocess'].append(client_stats.postprocess_end - client_stats.postprocess_start)
                        else:
                            stage_times['Preprocess Wait'].append(client_stats.preprocess_start - client_stats.created)
                            stage_times['Preprocess'].append(client_stats.preprocess_end - client_stats.preprocess_start)
                            stage_times['Inference'].append(client_stats.inference_end - client_stats.inference_start)
                            stage_times['Postprocess Wait'].append(client_stats.postprocess_start - client_stats.inference_end)
                            stage_times['Postprocess'].append(client_stats.postprocess_end - client_stats.postprocess_start)
                for stage in stages:
                    one_stage_times = stage_times[stage]
                    times[stage].append(np.median(one_stage_times))
            # ----- Plot all priorities
            priorities = list(range(1, len(self.priority_map) + 1))
            bottom = np.zero(len(stages))
            for i, (stage, time_all_priorities) in enumerate(times.items()):
                time_all_priorities = np.array(time_all_priorities)
                p = axes[i].bar(priorities, time_all_priorities, bottom=bottom, color=colors[i], label=stage)
                bottom += time_all_priorities
            axes[i].set_title(system_name)
            axes[i].legend(loc="upper right")
                
        plt.savefig(os.path.join(self.dir_name, "stages.png"))
    
    def save_stats(self) -> None:
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
            elif self.random_pattern == RandomPattern.EXP:
                interval = Comparison.exp_random(self.min_interval, self.max_interval)
            elif self.random_pattern == RandomPattern.POISSON:
                interval = Comparison.poisson_random(self.min_interval, self.max_interval)
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
    def exp_random(min_val, max_val, lambda_val=1):
        exponential_random_number = np.random.exponential(scale=1/lambda_val)
        return min_val + (max_val - min_val) * (exponential_random_number / (1/lambda_val))

    @staticmethod
    def poisson_random(min_val, max_val, lambda_val=1):
        poisson_random_number = np.random.poisson(lam=lambda_val)
        return min_val + (max_val - min_val) * (poisson_random_number / lambda_val)
    
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