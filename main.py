"""main client script
controls microphone stream and organise all audio responses"""

import socket
import pickle
import pyaudio
import numpy as np
import sys
import concurrent.futures
from time import sleep
import trio


class Client:
    def __init__(self):
        self.HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
        self.PORT = 65432
        # Port to listen on (non-privileged ports are > 1023)
        self.CHUNK = 2 ** 11
        self.RATE = 44100
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=self.RATE, input=True,
                            frames_per_buffer=self.CHUNK)
        self.peak = 0



    def snd_listen(self):
        print("mic listener: started!")
        while True:
            data = np.frombuffer(self.stream.read(self.CHUNK), dtype=np.int16)
            self.peak = np.average(np.abs(data)) * 2
            bars = "#" * int(50 * self.peak / 2 ** 16)
            if self.peak > 2000:
                print("%05d %s" % (self.peak, bars))
            # await trio.sleep(1 / self.RATE)

        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def mincer(self):
        while True:
            print('WAITING FOR AUDIO')
            sleep(1)

    async def sender(self, client_stream):
        print("sender: started!")
        while True:
            # data = peak
            send_data = pickle.dumps(self.peak, -1)
            # print(f"sender: sending {send_data}")
            # client_stream.sendall(send_data)
            # await client_stream.send_all(send_data)
            await client_stream.sendall(send_data)
            await trio.sleep(0.1)

    async def receiver(self, client_stream):
        print("receiver: started!")
        while True:
            async for data in client_stream.recv(1024):
                data_loaded = pickle.loads(data)
                print(f"receiver: got data {data_loaded!r}")

        print("receiver: connection closed")
        sys.exit()

    async def parent(self):
        print("parent: started!")
        while True:
            print(f"parent: connecting to {self.HOST}:{self.PORT}")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((self.HOST, self.PORT))
                s.listen()
                client_stream, addr = s.accept()
                async with client_stream:
                    print('Connected by', addr)
                    # with client_stream:
                    async with trio.open_nursery() as nursery:

                        # spawning listening port for user input
                        print("parent: spawning receiver...")
                        nursery.start_soon(self.receiver, client_stream)

                        # spawning sending port for output data
                        print("parent: spawning sender...")
                        nursery.start_soon(self.sender, client_stream)


    def pre_parent(self):
        trio.run(self.parent)

    def threader(self):
        tasks = [self.pre_parent]
        # tasks = [self.mincer, self.snd_listen, self.pre_parent] #, self.sender, self.receiver]
        # print(f"parent: connecting to {self.HOST}:{self.PORT}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(task): task for task in tasks}



        #
        #
        # with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        #     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        #         sock.bind((self.HOST, self.PORT))
        #         sock.listen(10)
        #         print("Server listening on port", self.PORT)
        #         while True:
        #             self.client_stream, addr = sock.accept()
        #             print('Connected by', addr)
        #             # futures = {executor.submit(task): task for task in tasks}
        #             futures = executor.submit(self.snd_listen)
        #
        #         # conn.close()
        #         # print("Server shutting down")


if __name__ == '__main__':
    cl = Client()
    cl.threader()


