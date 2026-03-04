import time
import traceback
import functools
import io
import cProfile
import pstats


def safe_return_false(error_msg=None):
    """Decorator that catches exceptions, prints traceback, and returns False."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                if error_msg:
                    print(f"[ERROR] {error_msg}")
                traceback.print_exc()
                return False
        return wrapper
    return decorator

def time_fn(logger, func, *args, **kwargs):
    """Decorator factory to time handle function execution.
    
    Args:
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start_time
    logger.info(f"[TIMING] {func.__name__} took {elapsed:.5f}s")
    return result

def v_time_fn(msg, func, *args, **kwargs):
    """Decorator to time handle function execution and print verbose timing info."""
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start_time
    print(f"[TIMING] {msg} - {func.__name__} took {elapsed:.5f}s")
    return result

def profile_function(func):
    """Decorator to profile a function and print timing info."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"🔍 Profiling {func.__name__}...")

        # Create profiler
        profiler = cProfile.Profile()
        profiler.enable()

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        profiler.disable()

        # Print timing
        print(f"⏱️ {func.__name__} took {end_time - start_time:.3f} seconds")

        # Print top functions
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(10)
        print(f"🔍 Top 10 functions in {func.__name__}:")
        print(s.getvalue())

        return result
    return wrapper
