
import sys
import trio
from random import randrange, random
from time import sleep, time

class AiDataEngine:
    def __init__(self, speed=1):
        self.global_speed = speed
        self.interrupt_bang = False
        self.running = False
        self.routing = False

        # make a default dict for the engine
        self.datadict = {'master_move_output': 0,
                         'move_rnn': 0,
                         'affect_rnn': 0,
                         'move-affect_conv2': 0,
                         'affect-move_conv2': 0,
                         'user_in': 0,
                         'rnd_poetry': 0,
                         'rhythm_rnn': 0,
                         'affect_net': 0}

        # fill with random values
        self.dict_fill()
        print(self.datadict)

        # todo: build models and put into list (replace strings with models)
        self.netlist = ['move_rnn',
                        'affect_rnn',
                        'move-affect_conv2',
                        'affect-move_conv2']

        self.rhythm_rate = 0.1

    def dict_fill(self):
        rnd = random()
        for key in self.datadict.keys():
            self.datadict[key] = rnd

    async def nets(self, net):
        self.net = net
        self.net_name = self.netlist[net]
        while self.interrupt_bang:
            # get the current value of the net ready for input for prediction
            localval = self.datadict.get(self.net_name)

            # will replace this with predictions and input with localval
            rnd = random() * self.global_speed
            print(f"  {self.net_name} in: {localval} predicted {rnd}")

            # save to data dict and master move out if
            self.datadict[self.net_name] = rnd
            self.datadict['master_move_output'] = rnd

            if self.net == 1:
                self.datadict['affect-move_conv2'] = rnd

            await trio.sleep(rnd)
            print(f"  {net}: looping!")

    # controls master scheduling
    async def master_clock(self):
        loop_dur = randrange(6, 26) * self.global_speed
        print(f"                 interrupt_listener: started! sleeping now for {loop_dur}...")
        await trio.sleep(loop_dur)

        # sends a bang that retsrats the process ~ refilling the datadict
        self.interrupt_bang = False
        print(" ###################   restarting ######################")

    async def streamer(self):
        # responds to affect streamer
        # when restarts after interrupt bang fill dict with rnd
        while self.interrupt_bang:
            # if > 60 trigger interrupt bang, break and restart all processes
            if self.affect_listen > 60:
                self.dict_fill()
                self.interrupt_bang = False
            # if 30 <> 60 fill dict with random, all processes norm
            elif 30 < self.affect_listen < 59:
                self.dict_fill()
                self.routing = False
            # else slow the loop down
            else:
                await trio.sleep(self.rhythm_rate)

    # define which feed to listen to, and duration
    # and a course of affect response
    async def affect(self):
        while self.interrupt_bang:
            rnd_stream = randrange(3)
            if rnd_stream == 0:
                self.affect_listen = self.datadict['user_in']
            elif rnd_stream == 1:
                self.affect_listen = self.datadict['rnd_poetry']
            else:
                self.affect_listen = self.datadict['affect_net']

            # hold this stream for 1-4 secs, unless interrupt bang
            end_time = time() + (randrange(1000, 4000) / 1000)
            while time() < end_time:
                self.datadict['master_move_output'] = self.affect_listen

                # hold until end of loop, major affect_bang, or medium routing change
                if not self.interrupt_bang or not self.routing:
                    break
                else:
                    await trio.sleep(self.rhythm_rate)

    async def random_poetry(self):
        # outputs a stream of random poetry
        while self.interrupt_bang:
            self.datadict['rnd_poetry'] = random()
            await trio.sleep(self.rhythm_rate)

    async def flywheel(self):
        print("parent: started!")
        while self.running:
            self.interrupt_bang = True
            async with trio.open_nursery() as nursery:
                # spawning all the nets
                print("parent: spawning net1...")
                nursery.start_soon(self.nets, 0)

                print("parent: spawning net2...")
                nursery.start_soon(self.nets, 1)

                print("parent: spawning net3...")
                nursery.start_soon(self.nets, 2)

                print("parent: spawning net4...")
                nursery.start_soon(self.nets, 3)

                # spawning scheduling methods
                print("parent: spawning master cog...")
                nursery.start_soon(self.master_clock)

                # spawning rhythm gen
                print("parent: spawning rhythm generator...")
                nursery.start_soon(self.streamer)

                # spawning affect listener
                print("parent: spawning affect listener...")
                nursery.start_soon(self.affect)

                # spawning poetry gen
                print("parent: spawning rhythm generator live input...")
                nursery.start_soon(self.random_poetry)

    # user accessible methods
    # returns the live output from the class to user
    def grab(self):
        # todo - need to implement the affect/ intensity RNN & output

        return {'e-AI output': self.datadict.get('master_move_output'),
                'intensity': 0,
                'individual NN outs':
                    {'move RNN': self.datadict.get('move_rnn'),
                     'affect RNN': self.datadict.get('affect_rnn'),
                         'move-affect_conv2': self.datadict.get('move-affect_conv2'),
                         'affect-move_conv2': self.datadict.get('affect-move_conv2')
                     }
                }

    # live input of user-data into class (0.0-1.0)
    # called and scheduled by user class
    def put(self, user_data):
        self.datadict['user_in'] = user_data

    def speed(self, user_speed):
        self.global_speed = user_speed

    def tempo(self, user_tempo):
        self.rhythm_rate = user_tempo

    # stop start methods
    def go(self):
        self.running = True
        trio.run(self.flywheel)

    def quit(self):
        self.running = False

if __name__ == '__main__':
    engine = AiDataEngine()
    engine.go()
