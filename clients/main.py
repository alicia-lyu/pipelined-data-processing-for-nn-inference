from Comparison import Comparison, SystemArgs, SystemType, Policy
from utils import get_batch_args

args = get_batch_args()
min_interval, max_interval, batch_size = float(args.min), float(args.max), int(args.batch_size)
data_type, random_pattern = Comparison.map_args_to_enum(args.data_type, args.random_pattern)
comparison = Comparison(min_interval, max_interval, batch_size, data_type, random_pattern)
system1 = SystemArgs(SystemType.NON_COORDINATED_BATCH) 
# Non_coordinated_batch should always be the first if you are running it, otherwise GPU will explode.
system2 = SystemArgs(SystemType.NAIVE_SEQUENTIAL)
system3 = SystemArgs(SystemType.PIPELINE, Policy.SLO_ORIENTED, 1, 
                     int(max(comparison.priority_map.values()) * 2))
system4 = SystemArgs(SystemType.PIPELINE, Policy.FIFO, 1, 
                     int(max(comparison.priority_map.values()) * 2))
comparison.compare([system1, system3, system4])

# TODO: Run more experiments: A cartesian product of A and B (8 * 3 = 24)

# A.1 min_interval = 0.15, max_interval = 0.3, batch_size = 2, data_type = "image", random_pattern = "poisson"
# A.2 min_interval = 0.2, max_interval = 0.4, batch_size = 2, data_type = "image", random_pattern = "poisson"
# A.3 min_interval = 0.3, max_interval = 0.5, batch_size = 2, data_type = "image", random_pattern = "poisson"

# A.4 min_interval = 0.2, max_interval = 0.4, batch_size = 2, data_type = "image", random_pattern = "uniform"

# A.5 min_interval = 1.5, max_interval = 3, batch_size = 2, data_type = "audio", random_pattern = "poisson"
# A.6 min_interval = 3, max_interval = 5, batch_size = 2, data_type = "audio", random_pattern = "poisson"
# A.7 min_interval = 5, max_interval = 8, batch_size = 2, data_type = "audio", random_pattern = "poisson"

# A.8 min_interval = 3, max_interval = 5, batch_size = 2, data_type = "audio", random_pattern = "uniform"

# B.1 (same as above)
### system1 = SystemArgs(SystemType.NAIVE_SEQUENTIAL)
### system2 = SystemArgs(SystemType.NON_COORDINATED_BATCH)
### system3 = SystemArgs(SystemType.PIPELINE, Policy.SLO_ORIENTED, 1, 
###                      int(max(comparison.priority_map.values()) * 2))
### system4 = SystemArgs(SystemType.PIPELINE, Policy.FIFO, 1, 
###                      int(max(comparison.priority_map.values()) * 2))

# B.2: Tune cpu_tasks_cap
### system1 = SystemArgs(SystemType.PIPELINE, Policy.SLO_ORIENTED, 1, 
###                      int(max(comparison.priority_map.values()) * 2))
### system2 = SystemArgs(SystemType.PIPELINE, Policy.SLO_ORIENTED, 2, 
###                      int(max(comparison.priority_map.values()) * 2))
### system3 = SystemArgs(SystemType.PIPELINE, Policy.SLO_ORIENTED, 4, 
###                      int(max(comparison.priority_map.values()) * 2))
### system4 = SystemArgs(SystemType.PIPELINE, Policy.SLO_ORIENTED, 8, 
###                      int(max(comparison.priority_map.values()) * 2))

# B.3: Different policies --- change cpu_tasks_cap below to the best performing from B.2
### system2 = SystemArgs(SystemType.NON_COORDINATED_BATCH)
### system3 = SystemArgs(SystemType.PIPELINE, Policy.SLO_ORIENTED, 1, 
###                      int(max(comparison.priority_map.values()) * 2))
### system4 = SystemArgs(SystemType.PIPELINE, Policy.FIFO, 1, 
###                      int(max(comparison.priority_map.values()) * 2))