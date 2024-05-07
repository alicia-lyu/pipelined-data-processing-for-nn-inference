from Comparison import Comparison, SystemArgs, SystemType, DataType, Policy, RandomPattern
import argparse

comparison_args_combinations = [
    (0.25, 0.25, 2, DataType.IMAGE, RandomPattern.POISSON),
    (0.3, 0.3, 2, DataType.IMAGE, RandomPattern.POISSON),
    (0.15, 0.35, 2, DataType.IMAGE, RandomPattern.UNIFORM),
    (5, 5, 2,  DataType.AUDIO, RandomPattern.POISSON),
    (3.5, 3.5, 2,  DataType.AUDIO, RandomPattern.POISSON),
    (2, 4.5, 2, DataType.AUDIO, RandomPattern.UNIFORM)
]

system_args_combinations = [
    [
        SystemArgs(SystemType.NON_COORDINATED_BATCH),
        SystemArgs(SystemType.PIPELINE, Policy.SLO_ORIENTED, 1, 10),
        SystemArgs(SystemType.PIPELINE, Policy.FIFO, 1, 10)
    ],
    [
        SystemArgs(SystemType.PIPELINE, Policy.SLO_ORIENTED, 1, 10),
        SystemArgs(SystemType.PIPELINE, Policy.SLO_ORIENTED, 2, 10),
        SystemArgs(SystemType.PIPELINE, Policy.SLO_ORIENTED, 4, 10),
        SystemArgs(SystemType.PIPELINE, Policy.SLO_ORIENTED, 6, 10)
    ]
]

def compare_systems(comparison_args, system_args):
    comparison = Comparison(*comparison_args)
    comparison.compare(system_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set parameters for comparison")
    parser.add_argument("--comparison_args", type=int, help="Comparison arguments index")
    parser.add_argument("--system_args", type=int, help="System arguments index")
    args = parser.parse_args()
    if args.comparison_args is not None and args.system_args is not None:
        comparison_args = comparison_args_combinations[args.comparison_args]
        system_args = system_args_combinations[args.system_args]
        ret = False
        retries = 0
        while ret == False and retries < 10:
            ret = compare_systems(comparison_args, system_args)
            retries += 1
        if ret == False:
            print(f"Failed to compare with {comparison_args} and {[str(system_arg) for system_arg in system_args]}")