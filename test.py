import time
import numpy as np
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
prev_params = test_worker.get_parameters()
end_time = time.time()
print("Time for first call: ", format_duration(end_time - start_time))
print("Buffer length: ", len(test_worker.get_parameters()))

gc.collect()
gc.disable()
call_accumulator = 0
for _ in range(1000):
    start_time = time.time()
    params = test_worker.get_parameters()
    end_time = time.time()
    call_accumulator += end_time - start_time
    if not np.equal(params, prev_params).all():
        print("Parameters diverged")
    prev_params = params
print("Rust time per call: {}".format(format_duration(call_accumulator / 1000)))
gc.enable()
gc.collect()

# Display histogram of params with np and matplotlib
import matplotlib.pyplot as plt

parameters = test_worker.get_parameters()

# Remove all 0.0 values from parameters
# Testing fold back distribution
parameters = parameters[parameters != 0.0]

print('Median', np.median(parameters))
print('Min, Max', np.min(parameters), np.max(parameters))

fig, ax = plt.subplots()
ax.hist(parameters, bins=100)
plt.show()

# gc.collect()
# gc.disable()
# start_time = time.time()
# for _ in range(1000):
#     x = np.random.randn(1_000_000)
# end_time = time.time()
# gc.enable()
# gc.collect()
# print("Numpy time per call: {}".format(format_duration((end_time - start_time) / 1000)))

print(test_worker.get_parameters())
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



