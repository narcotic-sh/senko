import os
import sys
import time
import numpy as np
from termcolor import colored
from contextlib import contextmanager

def time_method(stats_key, stage_name=None, last=False):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Print stage description before starting
            if not self.quiet and stage_name:
                # Add a space before the dots
                dots = "." * (29 - len(stage_name))
                pipe_char = "└──" if last else "├──"
                print(f'      {colored(pipe_char, "dark_grey")} {stage_name} {dots}', end="", flush=True)

            start = time.time()
            result = func(self, *args, **kwargs)
            duration = round(time.time() - start, 2)

            if not hasattr(self, '_timing_stats'):
                self._timing_stats = {}
            self._timing_stats[stats_key] = duration

            # Print timing if not in quiet mode with colors
            if not self.quiet and stage_name:
                print(colored(' done ', 'green'), end="", flush=True)
                print(colored(f'[{duration:.2f}s]', 'dark_grey'))

            return result
        return wrapper
    return decorator

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

@contextmanager
def timed_operation(description, quiet=False):
    if not quiet:
        print(f"{description}", end="", flush=True)
    start_time = time.time()
    try:
        yield
    finally:
        if not quiet:
            elapsed = time.time() - start_time
            print(colored(" done", 'green'), colored(f"[{elapsed:.1f}s]", 'dark_grey'))

def speaker_similarity(centroid1, centroid2):
    # Ensure inputs are numpy arrays
    centroid1 = np.asarray(centroid1)
    centroid2 = np.asarray(centroid2)

    # Calculate cosine similarity: (a · b) / (||a|| * ||b||)
    dot_product = np.dot(centroid1, centroid2)
    norm1 = np.linalg.norm(centroid1)
    norm2 = np.linalg.norm(centroid2)

    # Avoid division by zero
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    similarity = dot_product / (norm1 * norm2)

    # Clamp to [-1, 1] to handle floating point errors
    return np.clip(similarity, -1.0, 1.0)