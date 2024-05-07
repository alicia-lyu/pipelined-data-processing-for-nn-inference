from Comparison import Comparison, SystemArgs, SystemType, DataType, Policy, RandomPattern
import time

comparison_args_combinations = [
    (1, 1, 2, DataType.IMAGE, RandomPattern.POISSON),
    (0.5, 0.5, 2, DataType.IMAGE, RandomPattern.POISSON),
    (0.15, 0.3, 2, DataType.IMAGE, RandomPattern.UNIFORM),
    (3, 3, 2,  DataType.AUDIO, RandomPattern.POISSON),
    (1.5, 1.5, 2,  DataType.AUDIO, RandomPattern.POISSON),
    (3, 5, 2, DataType.AUDIO, RandomPattern.UNIFORM)
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
            ret = False
            retries = 0
            while ret == False and retries < 10:
                ret = compare_systems(comparison_args, system_args)
                retries += 1
            if ret == False:
                print(f"Failed to compare with {comparison_args} and {[str(system_arg) for system_arg in system_args]}")
            time.sleep(20)