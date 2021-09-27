"""main client script
controls microphone stream and organise all audio responses"""

import socket
import pickle
import pyaudio
import numpy as np
import concurrent.futures
from sound import SoundBot
from random import random
import trio
from time import sleep


class Client:
    def __init__(self):
        self.running = True
        self.connected = False
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
        self.bot_left = SoundBot()
        # self.bot_right = SoundBot()

        # build send data dict
        self.send_data_dict = {'mic_level': 0,
                               'speed': 1,
                               'tempo': 0.1
                               }

        # init got dict
        self.got_dict = {}

    def snd_listen(self):
        print("mic listener: started!")
        while self.running:
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

    # def mincer(self, got_AI_data, rhythm_rate):
    #     print(f'in the mincer ===== {got_AI_data, rhythm_rate}')
    #     # tic = time()
    #     rnd_dur = random()
    #     duration = rnd_dur + rhythm_rate
    #
    #     # make a sound at calc duration
    #     self.snd.play_sound(got_AI_data, duration)
    #     # toc = time()

    async def left(self):
        print('left bot: started')
        while self.connected:
            # get latest data
            left_master_data = self.got_dict['master_output']
            left_rhythm_rate = self.got_dict['rhythm_rate']

            # calc temp timing
            rnd_dur = random()
            left_duration = rnd_dur + left_rhythm_rate * 10

            # make a sound at calc duration
            self.bot_left.play_sound(left_master_data, left_duration)

    async def right(self):
        print('right bot: started')
        while self.connected:
            # get latest data
            right_master_data = self.got_dict['master_output']
            right_rhythm_rate = self.got_dict['rhythm_rate']

            # calc temp timing
            rnd_dur = random()
            right_duration = rnd_dur + right_rhythm_rate

            # make a sound at calc duration
            self.bot_right.play_sound(right_master_data, right_duration)

    def client(self):
        print("client: started!")
        while self.running:
            print(f"client: connecting to {self.HOST}:{self.PORT}")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((self.HOST, self.PORT))
                s.listen()
                client_stream, addr = s.accept()
                with client_stream:
                    print('Connected by', addr)
                    self.connected = True
                    while self.connected:
                        # get data from stream
                        data = client_stream.recv(1024)
                        self.got_dict = pickle.loads(data)
                        print(f"receiver: got data {self.got_dict}")

                        # send it to the mincer for soundBot control
                        # NB play_with_simpleaudio does not hold thread
                        # master_data = self.got_dict['master_output']
                        # rhythm_rate = self.got_dict['rhythm_rate']
                        # self.mincer(master_data, rhythm_rate)

                        # send out-going data to server
                        send_data = pickle.dumps(self.send_data_dict, -1)
                        client_stream.sendall(send_data)

    async def parent(self):
        print("parent: started!")
        while self.connected:
            async with trio.open_nursery() as nursery:
                # spawning left independent soundbot
                print("parent: spawning left bot ...")
                nursery.start_soon(self.left)

                # # spawning right independent soundbot
                # print("parent: spawning right bot ...")
                # nursery.start_soon(self.right)

    def parent_go(self):
        while self.running:
            if not self.connected:
                sleep(1)
            else:
                trio.run(self.parent)

    def main(self):
        # snd_listen and client need dependent threads.
        # All other IO is ok as a single Trio thread inside self.client
        tasks = [self.client, self.snd_listen]#, self.left]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(task): task for task in tasks}


if __name__ == '__main__':
    cl = Client()
    cl.main()
