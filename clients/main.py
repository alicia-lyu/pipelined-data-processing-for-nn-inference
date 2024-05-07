from Comparison import Comparison, SystemArgs, SystemType, Policy
from utils import get_batch_args

if __name__ == "main":
    args = get_batch_args()
    min, max, batch_size, data_type, random_pattern = args.min, args.max, args.batch_size, args.data_type, args.random_pattern
    comparison = Comparison(min, max, batch_size, data_type, random_pattern)
    system1 = SystemArgs(SystemType.NAIVE_SEQUENTIAL)
    system2 = SystemArgs(SystemType.NON_COORDINATED_BATCH)
    system3 = SystemArgs(SystemType.PIPELINE, Policy.SLO_ORIENTED, 4, max(comparison.priority_map) * 2)
    comparison.compare([system1, system2, system3])
    