from typing import List, Dict
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import os
from dataclasses import dataclass, field

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

class StatsProcessor:
    def __init__(self, base_dir, stats, deadlines, priority_map, priorities = None):
        if isinstance(stats, Dict):
            self.stats: Dict[str, List[Stats]] = stats
        elif isinstance(stats, List): # list of filenames
            for filename in stats:
                with open(filename, 'r') as f:
                    for line in f.readlines():
                        client_id = line.split()[0]
                        client_stats = line.split()[1:]
                        if len(client_stats) == 7:
                            stats = Stats(
                                process_created = float(client_stats[0]),
                                preprocessing_start = float(client_stats[1]),
                                preprocessing_end = float(client_stats[2]),
                                inference_start = float(client_stats[3]),
                                inference_end = float(client_stats[4]),
                                postprocessing_start = float(client_stats[5]),
                                postprocessing_end = float(client_stats[6])
                            )
                            self.stats[filename].append(stats)
                        elif len(client_stats) == 11:
                            stats = Stats(
                                process_created = float(client_stats[0]),
                                preprocessing_start = float(client_stats[1]),
                                preprocessing_end = float(client_stats[2]),
                                inference_start = float(client_stats[3]),
                                inference_end = float(client_stats[4]),
                                midprocessing_start = float(client_stats[5]),
                                midprocessing_end = float(client_stats[6]),
                                inference2_start = float(client_stats[7]),
                                inference2_end = float(client_stats[8]),
                                postprocessing_start = float(client_stats[9]),
                                postprocessing_end = float(client_stats[10])
                            )
                            self.stats[filename].append(stats)
                        else:
                            raise ValueError(f"Invalid stats length {len(client_stats)} at {filename}: {client_stats}")
                    f.close()
        else:
            raise ValueError(f"Stats {stats} not supported")
        self.dir_name = base_dir
        os.makedirs(self.dir_name, exist_ok=True)
        
        if isinstance(deadlines, List):
            self.deadlines = deadlines
        elif isinstance(deadlines, str):
            self.deadlines = [0 for _ in range(len(self.stats.values()[0]))]
            self.priorities = [0 for _ in range(len(self.stats.values()[0]))]
            with open(deadlines, 'r') as f:
                for line in f.readlines():
                    client_id, priority, deadline = line.split()
                    self.deadlines[int(client_id)] = float(deadline)
                    self.priorities[int(client_id)] = int(priority)
                f.close()
                
        if isinstance(priority_map, Dict):
            self.priority_map = priority_map
        elif isinstance(priority_map, str):
            self.priority_map = {}
            with open(priority_map, 'r') as f:
                for line in f.readlines():
                    priority, latency = line.split()
                    self.priority_map[int(priority)] = float(latency)
                f.close()
        else:
            raise ValueError(f"Priority map {priority_map} not supported")
        
        if priorities is not None:
            self.priorities = priorities
        
        self.client_num = len(self.priorities)
        
        self.latency_goals = [self.priority_map[
                self.priorities[client_id]
            ] for client_id in range(len(self.priorities))]
        
        print(f"Plotting in dir {self.dir_name}.")
        # print(self.stats)
        # print(self.deadlines)
        # print(self.priorities)
        # print(self.priority_map)
        
    def plot_batches(self):
        latencies = {}
        for system, system_stats in self.stats.items():
            latencies[system] = []
            for stats in system_stats:
                latencies[system].append(stats.postprocess_end - stats.created)
        plt.rcParams.update({'font.size': 12})
        fig, ax = plt.subplots(figsize=(10, 8))
        batches = range(len(latencies[system]))
        colors = ['r', 'g', 'b', 'y', 'm']
        failed_goal = {system: 0 for system in latencies.keys()}
        for client_id, latency_goal in enumerate(self.latency_goals):
            for system, latency in latencies.items():
                if latency[client_id] > latency_goal * 1.1:
                    failed_goal[system] += 1
        for i, (system, latency) in enumerate(latencies.items()):
            ax.plot(batches, latency, label=f"{system}: Failing {failed_goal[system]}/{self.client_num}", color=colors[i])
            ax.axhline(np.mean(latency), linestyle='--', color=colors[i])
            
        if sum(failed_goal.values()) > self.client_num * 0.02: # Don't plot if only few didn't achieve goal
            print(f"Failed goal: {failed_goal} / {len(latencies.items()) * self.client_num}")
            ax.plot(batches, self.latency_goals, label='Deadline', color='k')
        
        ax.set_title('Latency Comparison')
        ax.set_xlabel('# Request')
        ax.set_ylabel('Latency (s)')
        ax.grid()
        ax.legend()
        fig.savefig(os.path.join(self.dir_name, "latency.png"))
        
    def plot_stages(self) -> None:
        # ----- Median time taken for each stage, grouped by priority, for all system types
        plt.rcParams.update({'font.size': 12})
        fig, axes = plt.subplots(1, len(self.stats.items()), figsize=(6 * len(self.stats.items()), 8))
        for i, (system_name, system_stats) in enumerate(self.stats.items()):
            times: Dict = {}
            if isinstance(system_stats[0], ImageStats):
                stages = [
                    'Preprocess Wait', 'Preprocess', 'Inference',
                    'Midprocessing Wait', 'Midprocessing', 'Inference2',
                    'Postprocess Wait', 'Postprocess'
                ]
                colors = ['red', 'yellow', 'blue', 'red', 'yellow', 'blue', 'red', 'yellow']
            else:
                stages = [
                    'Preprocess Wait', 'Preprocess', 'Inference', 'Postprocess Wait', 'Postprocess'
                ]
                colors = ['red', 'yellow', 'blue', 'red', 'yellow']

            # Initialize times dictionary to store median times for each stage
            times = {stage: [] for stage in stages}
            for priority in range(1, len(self.priority_map) + 1):
                stage_times = {stage: [] for stage in stages}
                for client_id, client_stats in enumerate(system_stats):
                    if self.priorities[client_id] == priority:
                        if isinstance(client_stats, ImageStats):
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
                    times[stage].append(np.median(stage_times[stage]))
            # ----- Plot all priorities
            print(times)
            priorities = [f"Priority {i}" for i in range(1, len(self.priority_map) + 1)]
            bottom = np.zeros(len(priorities))
            for j, (stage, time_all_priorities) in enumerate(times.items()):
                time_all_priorities = np.array(time_all_priorities)
                p = axes[i].bar(priorities, time_all_priorities, bottom=bottom, color=colors[j], label=stage)
                bottom += time_all_priorities
            axes[i].set_title(system_name)
            axes[i].grid()
            axes[i].legend(loc="upper right")
            
        fig.savefig(os.path.join(self.dir_name, "stages.png"))