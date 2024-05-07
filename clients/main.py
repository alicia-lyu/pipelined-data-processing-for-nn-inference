from Comparison import Comparison, SystemArgs, SystemType, Policy
from utils import get_batch_args

args = get_batch_args()
min_interval, max_interval, batch_size = int(args.min), int(args.max), int(args.batch_size)
data_type, random_pattern = Comparison.map_args_to_enum(args.data_type, args.random_pattern)
comparison = Comparison(min_interval, max_interval, batch_size, data_type, random_pattern)
system1 = SystemArgs(SystemType.NAIVE_SEQUENTIAL)
system2 = SystemArgs(SystemType.NON_COORDINATED_BATCH)
system3 = SystemArgs(SystemType.PIPELINE, Policy.SLO_ORIENTED, 4, 
                     int(max(comparison.priority_map.values()) * 2))
comparison.compare([system3])