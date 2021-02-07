"""main client script
controls microphone stream and organise all audio responses"""

import socket
import pickle
from random import random
import pyaudio
import numpy as np


def snd_listen(self):
    CHUNK = 2 ** 11
    RATE = 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    while self.running:
        data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        self.peak = np.average(np.abs(data)) * 2
        # bars = "#" * int(50 * self.peak / 2 ** 16)
        # print("%05d %s" % (self.peak, bars))
    stream.stop_stream()
    stream.close()
    p.terminate()


HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432
# Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            data = conn.recv(1024)
            data_loaded = pickle.loads(data)
            print(f'incoming data = {data_loaded}')

            rnd = random()
            send_rnd = pickle.dumps(rnd, -1)
            # if not data:
            #     break
            conn.sendall(send_rnd)


