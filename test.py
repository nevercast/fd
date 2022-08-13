import time
import fdlib

start_time = time.time()
for _ in range(1000):
    fdlib.give_me_a_fat_buffer()
end_time = time.time()
print("Time per call: {}".format((end_time - start_time) / 1000))


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



