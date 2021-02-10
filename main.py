"""main client script
controls microphone stream and organise all audio responses"""

import socket
import pickle
import pyaudio
import numpy as np
import concurrent.futures
from sound import SoundBot
from random import random


class Client:
    def __init__(self):
        self.HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
        self.PORT = 65432
        # Port to listen on (non-privileged ports are > 1023)
        self.CHUNK = 2 ** 11
        self.RATE = 44100
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.RATE,
                                  input=True,
                                  frames_per_buffer=self.CHUNK)

        # instantiate sound object
        self.snd = SoundBot()

        # build send data dict
        self.send_data_dict = {'mic_level': 0,
                               'speed': 1,
                               'tempo': 0.1
                               }

    def snd_listen(self):
        print("mic listener: started!")
        while True:
            data = np.frombuffer(self.stream.read(self.CHUNK),
                                 dtype=np.int16)
            peak = np.average(np.abs(data)) * 2
            if peak > 2000:
                bars = "#" * int(50 * peak / 2 ** 16)
                print("%05d %s" % (peak, bars))
            self.send_data_dict['mic_level'] = peak / 30000

        # self.stream.stop_stream()
        # self.stream.close()
        # self.p.terminate()

    def mincer(self, got_AI_data, rhythm_rate):
        print(f'in the mincer ===== {got_AI_data, rhythm_rate}')
        # tic = time()
        rnd_dur = random()
        duration = rnd_dur + rhythm_rate

        # make a sound at calc duration
        self.snd.play_sound(got_AI_data, duration)
        # toc = time()

    def client(self):
        print("client: started!")
        while True:
            print(f"client: connecting to {self.HOST}:{self.PORT}")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((self.HOST, self.PORT))
                s.listen()
                client_stream, addr = s.accept()
                with client_stream:
                    print('Connected by', addr)
                    while True:
                        # get data from stream
                        data = client_stream.recv(1024)
                        data_loaded = pickle.loads(data)
                        print(f"receiver: got data {data_loaded}")

                        # send it to the mincer for soundBot control
                        # NB play_with_simpleaudio does not hold thread
                        master_data = data_loaded['master_output']
                        rhythm_rate = data_loaded['rhythm_rate']
                        self.mincer(master_data, rhythm_rate)

                        # send out-going data to server
                        send_data = pickle.dumps(self.send_data_dict, -1)
                        client_stream.sendall(send_data)

    def main(self):
        tasks = [self.snd_listen, self.client]

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(task): task for task in tasks}


if __name__ == '__main__':
    cl = Client()
    cl.main()


