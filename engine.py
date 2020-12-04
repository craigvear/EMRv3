# --------------------------------------------------
#
# Embodied AI Engine Prototype v0.9
# 2020/12/03
#
# © Craig Vear 2020
# cvear@dmu.ac.uk
#
# --------------------------------------------------

import trio
from random import randrange, random
from time import time
from tensorflow.keras.models import load_model
import numpy as np

# --------------------------------------------------
#
# instantiate an object for each neural net
#
# --------------------------------------------------

class MoveRNN:
    def __init__(self):
        print('MoveRNN initialization')
        self.move_rnn = load_model('training/models/EMR-3_RNN_skeleton_data.nose.x.h5')

    def predict(self, in_val):
        return self.move_rnn.predict(in_val)

class AffectRNN:
    def __init__(self):
        print('AffectRNN initialization')
        self.affect_rnn = load_model('training/models/EMR-3_RNN_bitalino.h5')

    def predict(self, in_val):
        return self.affect_rnn.predict(in_val)

class MoveAffectCONV2:
    def __init__(self):
        print('MoveAffectCONV2 initialization')
        self.move_affect_conv2 = load_model('training/models/EMR-3_conv2D_move-affect.h5')

    def predict(self, in_val):
        return self.move_affect_conv2.predict(in_val)

class AffectMoveCONV2:
    def __init__(self):
        print('AffectMoveCONV2 initialization')
        self.affect_move_conv2 = load_model('training/models/EMR-3_conv2D_affect-move.h5')

    def predict(self, in_val):
        return self.affect_move_conv2.predict(in_val)


# --------------------------------------------------
#
# controls all thought-trains and affect responses
#
# --------------------------------------------------

class AiDataEngine:
    def __init__(self, speed=1):
        self.global_speed = speed
        self.interrupt_bang = False
        self.running = False
        self.routing = False

        # make a default dict for the engine
        self.datadict = {'move_rnn': 0,
                         'affect_rnn': 0,
                         'move_affect_conv2': 0,
                         'affect_move_conv2': 0,
                         'master_move_output': 0,
                         'user_in': 0,
                         'rnd_poetry': 0,
                         'rhythm_rnn': 0,
                         'affect_net': 0}

        # fill with random values
        self.dict_fill()
        print(self.datadict)

        # instantiate nets as objects and make  models
        self.move_rnn = MoveRNN()
        self.affect_rnn = AffectRNN()
        self.move_affect_conv2 = MoveAffectCONV2()
        self.affect_move_conv2 = AffectMoveCONV2()

        # set out variables
        self.netnames = ['move_rnn',
                         'affect_rnn',
                         'move_affect_conv2',
                         'affect_move_conv2']

        self.netlist = [self.move_rnn,
                        self.affect_rnn,
                        self.move_affect_conv2,
                        self.affect_move_conv2]

        self.rhythm_rate = 0.1
        self.affect_listen = 0

    # fills the dictionary with rnd values for each key
    def dict_fill(self):
        for key in self.datadict.keys():
            rnd = random()
            self.datadict[key] = rnd

    # --------------------------------------------------
    #
    # prediction and rnd num gen zone
    #
    # --------------------------------------------------

    # makes a prediction for a given net and defined input var
    def prediction(self, net, in_val):
        print(net, in_val)
        self.pred_in_val = np.reshape(in_val, (1, 1, 1))
        print(self.localval)
        if self.net_name == 0:
            pred = self.move_rnn.predict(self.pred_in_val)
        elif self.net_name == 1:
            pred = self.affect_rnn.predict(self.pred_in_val)
        elif self.net_name == 2:
            pred = self.move_affect_conv2.predict(self.pred_in_val)
        else:
            pred = self.affect_move_conv2.predict(self.pred_in_val)
        # todo returning only 1st position. 4 could be used in a randomiser
        return pred[0][0]

    # control method for all net predictions
    async def nets(self, net, in_dict):
        self.net = net
        self.in_dict = self.netnames[in_dict]
        self.net_name = self.netnames[net]
        print(self.in_dict, self.net_name)

        while self.interrupt_bang:
            # get the current value of the net ready for input for prediction
            self.localval = self.datadict.get(self.in_dict)

            # predictions and input with localval
            self.pred = self.prediction(self.net, self.localval)
            print(f"  {self.net_name} in: {self.localval} predicted {self.pred}")

            # save to data dict and master move out if
            self.datadict[self.net_name] = self.pred
            self.datadict['master_move_output'] = self.pred

            await trio.sleep(self.rhythm_rate)
            print(f"  {self.net}: looping!")

    async def random_poetry(self):
        # outputs a stream of random poetry
        while self.interrupt_bang:
            self.datadict['rnd_poetry'] = random()
            await trio.sleep(self.rhythm_rate)

    # --------------------------------------------------
    #
    # affect and streaming methods
    #
    # --------------------------------------------------

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
        self.routing = True
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
                # print(self.datadict['master_move_output'])
                # hold until end of loop, major affect_bang, or medium routing change
                if not self.interrupt_bang or not self.routing:
                    break
                await trio.sleep(self.rhythm_rate)

    # --------------------------------------------------
    #
    # parent threading solution
    #
    # --------------------------------------------------

    async def flywheel(self):
        print("parent: started!")
        while self.running:
            self.interrupt_bang = True
            async with trio.open_nursery() as nursery:
                # spawning all the nets
                print("parent: spawning net1...")
                nursery.start_soon(self.nets, 0, 0)

                print("parent: spawning net2...")
                nursery.start_soon(self.nets, 1, 1)

                print("parent: spawning net3...")
                nursery.start_soon(self.nets, 2, 2)

                print("parent: spawning net4...")
                nursery.start_soon(self.nets, 3, 1)

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

    # --------------------------------------------------
    #
    # user accessible methods
    #
    # --------------------------------------------------

    # returns the live output from the class to user
    def grab(self):
        return {'e-AI output': self.datadict.get('master_move_output'),
                'intensity': 0,
                'individual NN outs':
                    {'move RNN': self.datadict.get('move_rnn'),
                     'affect RNN': self.datadict.get('affect_rnn'),
                         'move_affect_conv2': self.datadict.get('move_affect_conv2'),
                         'affect_move_conv2': self.datadict.get('affect_move_conv2')
                     }
                }

    # live input of user-data into class (0.0-1.0)
    # called and scheduled by user class
    def put(self, user_data):
        self.datadict['user_in'] = user_data

    # user change the overall speed of the engine
    def speed(self, user_speed):
        self.global_speed = user_speed

    # user change tempo of outputs and parsing
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
