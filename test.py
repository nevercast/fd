import time
import fdlib
import gc

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
test_worker.get_parameters()
end_time = time.time()
print("Time for first call: ", format_duration(end_time - start_time))
print("Buffer length: ", len(test_worker.get_parameters()))

gc.collect()
gc.disable()
start_time = time.time()
for _ in range(1000):
    test_worker.get_parameters()
end_time = time.time()
print("Rust time per call: {}".format(format_duration((end_time - start_time) / 1000)))
gc.enable()
gc.collect()


import numpy as np
gc.collect()
gc.disable()
start_time = time.time()
for _ in range(1000):
    x = np.random.randn(1_000_000)
end_time = time.time()
gc.enable()
gc.collect()
print("Numpy time per call: {}".format(format_duration((end_time - start_time) / 1000)))

print(test_worker.get_parameters())
# worker = fdlib.create_worker("tcp://127.0.0.1:3042")
# worker = fdlib.create_worker("wss://127.0.0.1:3044/some/job")

# while True:
#     worker.get_parameters()

#     # Run environment here
#     reward = 1

#     worker.send_returns(reward)

# parameters = worker.get_parameters()
# print("Parameters:", parameters)

# while True:
#     signal = worker.get_signal()
#     if signal == "No signal":
#         continue
#     print(signal)
#     time.sleep(0.01)



