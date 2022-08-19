import time
import numpy as np
import fdlib
import gc
import matplotlib.pyplot as plt

def format_duration(seconds):
    """ Returns either seconds, milliseconds or microseconds depending on the value of seconds.
        Displays the value to two decimal places.
    """
    if seconds < 1e-3:
        return "{:.2f} microseconds".format(seconds*1e6)
    elif seconds < 1:
        return "{:.2f} milliseconds".format(seconds*1e3)
    else:
        return "{:.2f} seconds".format(seconds)

test_worker = fdlib.create_worker("tcp://127.0.0.1:3042")

start_time = time.time()
prev_params = test_worker.get_parameters()
end_time = time.time()
print("Time for first call: ", format_duration(end_time - start_time))
print("Buffer length: ", len(test_worker.get_parameters()))
print("I'm going to fetch 3 billion values")

# Buffer size = 10_000_000
start_time = time.time()
for _ in range(3_000_000_000 // 10_000_000):
    test_worker.get_parameters()
end_time = time.time()
print("Time for 3 billion values: ", format_duration(end_time - start_time))