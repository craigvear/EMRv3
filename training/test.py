# echo-client.py

import sys
import trio

# arbitrary, but:
# - must be in between 1024 and 65535
# - can't be in use by some other program on your computer
# - must match what we set in our echo server
PORT = 12345

async def sender(client_stream):
    print("sender: started!")
    while True:
        data = b"async can sometimes be confusing, but I believe in you!"
        print("sender: sending {!r}".format(data))
        await client_stream.send_all(data)
        await trio.sleep(1)

async def receiver(client_stream):
    print("receiver: started!")
    async for data in client_stream:
        print("receiver: got data {!r}".format(data))
    print("receiver: connection closed")
    sys.exit()

async def parent():
    print("parent: connecting to 127.0.0.1:{}".format(PORT))
    client_stream = await trio.open_tcp_stream("127.0.0.1", PORT)
    async with client_stream:
        async with trio.open_nursery() as nursery:
            print("parent: spawning sender...")
            nursery.start_soon(sender, client_stream)

            print("parent: spawning receiver...")
            nursery.start_soon(receiver, client_stream)

trio.run(parent)




#
# # import tensorflow as tf
# from tensorflow.keras.models import load_model
# import numpy as np
#
# class Net:
#     def __init__(self):
#         print('net init')
#
#     def model(self):
#         self.model = load_model('models/EMR-3_RNN_skeleton_data.nose.x.h5')
#
#     def predict(self, inval):
#         pred = self.model.predict(inval)
#         return pred
#
# net = Net()
# net.model()
# for x in range(100):
#     x /= 100
#     inval = np.reshape(x, (1, 1, 1))
#     pred = net.predict(inval)
#     print(pred)

