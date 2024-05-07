from typing import List, Dict
from Comparison import Stats, ImageStats
import matplotlib.pyplot as plt
import numpy as np
import os

class StatsProcessor:
    def __init__(self, stats, deadlines):
        self.stats: Dict[str, List[Stats]] = stats
        self.dir_name = "../stats_" + "__".join(stats.keys())
        self.deadlines = deadlines
        
    def __init__(self, stats_file_names: List[str], deadline_file_name: str):
        for filename in stats_file_names:
            with open(filename, 'r') as f:
                for line in f.readlines():
                    client_stats = line.split()
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
        self.deadlines = [0 for _ in range(len(self.stats[filename]))]
        with open(deadline_file_name, 'r') as f:
            for line in f.readlines():
                client_id, priority, deadline = line.split()
                self.deadlines[int(client_id)] = float(deadline)
            f.close()

    def plot_batches(self):
        latencies = {}
        for system, system_stats in self.stats:
            latencies[system] = []
            for stats in system_stats:
                latencies[system].append(stats.postprocessing_end - stats.process_created)
        fig, ax = plt.subplots()
        batches = range(len(latencies[system]))
        colors = ['r', 'g', 'b', 'y', 'm']
        for i, (system, latency) in enumerate(latencies.items()):
            ax.plot(batches, latency, label=system, color=colors[i])
            ax.hline(np.mean(latency), linestyle='--', color=colors[i])
        ax.plot(batches, self.deadlines, label='Deadline', color='k')
        
        ax.title('Latency Comparison')
        ax.xlabel('# Request')
        ax.ylabel('Latency (s)')
        fig.savefig(self.dir_name + "/latency.png")
        
    def plot_stages(self) -> None:
        # ----- Median time taken for each stage, grouped by priority, for all system types
        fig, axes = plt.subplots(1, 3, figsize=(30, 15))
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
            priorities = list(range(1, len(self.priority_map) + 1))
            bottom = np.zeros(len(priorities))
            for j, (stage, time_all_priorities) in enumerate(times.items()):
                time_all_priorities = np.array(time_all_priorities)
                p = axes[i].bar(priorities, time_all_priorities, bottom=bottom, color=colors[j], label=stage)
                bottom += time_all_priorities
            axes[i].set_title(system_name)
            axes[i].legend(loc="upper right")
                
        plt.savefig(os.path.join(self.dir_name, "stages.png"))