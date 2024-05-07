from typing import Callable
import os
import argparse

def trace(path: str):
    file_name = os.path.basename(path)
    def decorator(func: Callable):
        def trace_prefix():
            return f"*** {file_name}, {func.__name__} ***"
        setattr(func, "trace_prefix", trace_prefix)
        return func
    return decorator

@trace(__file__)
def get_batch_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set pipeline data arrival interval")

    parser.add_argument("--min", type=float, help="Minimum data arrival interval")
    parser.add_argument("--max", type=float, help="Maximum data arrival interval")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--data_type", type=str, help="Data type, image or audio", required=True)
    parser.add_argument("--random_pattern", type=str, help="Random pattern, uniform, exponential, or poisson", default="uniform")

    args = parser.parse_args()

    print(get_batch_args.trace_prefix(), args)

    return args