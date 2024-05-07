from typing import List, Dict
from Comparison import Stats
import matplotlib.pyplot as plt
import numpy as np

class StatsProcessor:
    def __init__(self, stats):
        self.stats: Dict[str, List[Stats]] = stats
        self.dir_name = "../stats_" + "__".join(stats.keys())
        
    def __init__(self, stats_file_names: List[str]):
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
        
        ax.title('Latency Comparison')
        ax.xlabel('# Request')
        ax.ylabel('Latency (s)')
        fig.savefig(self.dir_name + "/latency.png")
            