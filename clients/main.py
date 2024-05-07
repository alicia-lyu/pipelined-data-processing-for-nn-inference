from Comparison import Comparison, SystemArgs, SystemType, Policy
from utils import get_batch_args

comparison_args_combinations = [
    (0.15, 0.3, 2, "image", "poisson"),
    (0.2, 0.4, 2, "image", "poisson"),
    (0.3, 0.5, 2, "image", "poisson"),
    (0.2, 0.4, 2, "image", "uniform"),
    (1.5, 3, 2, "audio", "poisson"),
    (3, 5, 2, "audio", "poisson"),
    (5, 8, 2, "audio", "poisson"),
    (3, 5, 2, "audio", "uniform")
]

system_args_combinations = [
    [
        SystemArgs(SystemType.NON_COORDINATED_BATCH),
        SystemArgs(SystemType.NAIVE_SEQUENTIAL),
        SystemArgs(SystemType.PIPELINE, Policy.SLO_ORIENTED, 1, 10),
        SystemArgs(SystemType.PIPELINE, Policy.FIFO, 1, 10)
    ],
    [
        SystemArgs(SystemType.PIPELINE, Policy.SLO_ORIENTED, 1, 10),
        SystemArgs(SystemType.PIPELINE, Policy.SLO_ORIENTED, 2, 10),
        SystemArgs(SystemType.PIPELINE, Policy.SLO_ORIENTED, 4, 10),
        SystemArgs(SystemType.PIPELINE, Policy.SLO_ORIENTED, 8, 10)
    ],
    [
        SystemArgs(SystemType.NON_COORDINATED_BATCH),
        SystemArgs(SystemType.PIPELINE, Policy.SLO_ORIENTED, 8, 10),
        SystemArgs(SystemType.PIPELINE, Policy.FIFO, 8, 10)
    ]
]

def compare_systems(comparison_args, system_args):
    comparison = Comparison(*comparison_args)
    comparison.compare(system_args)

if __name__ == "__main__":
    for comparison_args in comparison_args_combinations:
        for system_args in system_args_combinations:
            compare_systems(comparison_args, system_args)