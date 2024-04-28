def trace(file_name):
    def decorator(func):
        def newfunc(*args, **kwargs):
            ret = func(*args, **kwargs)

            def trace_prefix():
                return f"***{file_name}, {func.__name__}***"

            func.trace_prefix = trace_prefix
            return ret

        return newfunc
    return decorator
