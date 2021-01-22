# --------------------------------------------------
#
# Embodied AI Engine Prototype v0.10
# 2021/01/22
#
# © Craig Vear 2020
# cvear@dmu.ac.uk
#
# Dedicated to Fabrizio Poltronieri
#
# --------------------------------------------------

import trio
from random import randrange, random
from time import time
from tensorflow.keras.models import load_model
import numpy as np
import sys
import pickle

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
        # predictions and input with localval
        self.pred = self.move_rnn.predict(in_val)
        return self.pred

class AffectRNN:
    def __init__(self):
        print('AffectRNN initialization')
        self.affect_rnn = load_model('training/models/EMR-3_RNN_bitalino.h5')

    def predict(self, in_val):
        # predictions and input with localval
        self.pred = self.affect_rnn.predict(in_val)
        return self.pred

class MoveAffectCONV2:
    def __init__(self):
        print('MoveAffectCONV2 initialization')
        self.move_affect_conv2 = load_model('training/models/EMR-3_conv2D_move-affect.h5')

    def predict(self, in_val):
        # predictions and input with localval
        self.pred = self.move_affect_conv2.predict(in_val)
        return self.pred

class AffectMoveCONV2:
    def __init__(self):
        print('AffectMoveCONV2 initialization')
        self.affect_move_conv2 = load_model('training/models/EMR-3_conv2D_affect-move.h5')

    def predict(self, in_val):
        # predictions and input with localval
        self.pred = self.affect_move_conv2.predict(in_val)
        return self.pred

# --------------------------------------------------
#
# controls all thought-trains and affect responses
#
# --------------------------------------------------

class AiDataEngine():
    def __init__(self, speed=1):
        self.interrupt_bang = False
        self.running = False
        self.routing = False
        self.PORT = 65432
        self.IP_ADDR = "127.0.0.1"

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

        # name list for nets
        self.netnames = ['move_rnn',
                         'affect_rnn',
                         'move_affect_conv2',
                         'affect_move_conv2']

        self.rhythm_rate = 0.1
        self.affect_listen = 0
        self.global_speed = speed

        # fill with random values
        self.dict_fill()
        print(self.datadict)

        # instantiate nets as objects and make  models
        self.move_net = MoveRNN()
        self.affect_net = AffectRNN()
        self.move_affect_net = MoveAffectCONV2()
        self.affect_move_net = AffectMoveCONV2()

        # logging on/off switches
        self.net_logging = True
        self.master_logging = True
        self.streaming_logging = True

    # --------------------------------------------------
    #
    # prediction and rnd num gen zone
    #
    # --------------------------------------------------

    # makes a prediction for a given net and defined input var
    async def make_data(self):
        while self.interrupt_bang:
            # get input vars from dict (NB not always self)
            in_val1 = self.get_in_val(0)
            in_val2 = self.get_in_val(1)
            in_val3 = self.get_in_val(2)
            in_val4 = self.get_in_val(1)

            # send in vals to net object for prediction
            pred1 = self.move_net.predict(in_val1)
            pred2 = self.affect_net.predict(in_val2)
            pred3 = self.move_affect_net.predict(in_val3)
            pred4 = self.affect_move_net.predict(in_val4)

            if self.net_logging:
                print(f"  'move_rnn' in: {in_val1} predicted {pred1}")
                print(f"  'affect_rnn' in: {in_val2} predicted {pred2}")
                print(f"  move_affect_conv2' in: {in_val3} predicted {pred3}")
                print(f"  'affect_move_conv2' in: {in_val4} predicted {pred4}")

            # put predictions back into the dicts and master
            self.put_pred(0, pred1)
            self.put_pred(0, pred2)
            self.put_pred(0, pred3)
            self.put_pred(0, pred4)

            # outputs a stream of random poetry
            rnd_poetry = random()
            self.datadict['rnd_poetry'] = rnd_poetry
            if self.streaming_logging:
                print(f'random poetry = {rnd_poetry}')

            await trio.sleep(self.rhythm_rate)


    def get_in_val(self, which_dict):
        # get the current value and reshape ready for input for prediction
        input_val = self.datadict.get(self.netnames[which_dict])
        input_val = np.reshape(input_val, (1, 1, 1))
        return input_val

    def put_pred(self, which_dict, pred):
        out_pred_val = pred[0][randrange(4)]
        if self.master_logging:
            print(f"out pred val == {out_pred_val},   master move output == {self.datadict['master_move_output']}")
        # save to data dict and master move out ONLY 1st data
        self.datadict[self.netnames[which_dict]] = out_pred_val
        self.datadict['master_move_output'] = out_pred_val

    # fills the dictionary with rnd values for each key
    def dict_fill(self):
        for key in self.datadict.keys():
            rnd = random()
            self.datadict[key] = rnd

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
            if self.affect_listen > 0.60:
                self.dict_fill()
                self.interrupt_bang = False
            # if 30 <> 60 fill dict with random, all processes norm
            elif 0.30 < self.affect_listen < 0.59:
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

            # hold this stream for 1-4 secs, unless interrupt bang
            end_time = time() + (randrange(1000, 4000) / 1000)
            while time() < end_time:

                if rnd_stream == 0:
                    self.affect_listen = self.datadict['user_in']
                elif rnd_stream == 1:
                    self.affect_listen = self.datadict['rnd_poetry']
                else:
                    self.affect_listen = self.datadict['affect_net']

            # # hold this stream for 1-4 secs, unless interrupt bang
            # end_time = time() + (randrange(1000, 4000) / 1000)
            # while time() < end_time:
                self.datadict['master_move_output'] = self.affect_listen
                if self.streaming_logging:
                    print(f'master move output = {self.affect_listen}')
                # hold until end of loop, major affect_bang, or medium routing change
                if not self.interrupt_bang or not self.routing:
                    await trio.sleep(self.rhythm_rate)
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
            print("parent: connecting to 127.0.0.1:{}".format(self.PORT))
            client_stream = await trio.open_tcp_stream(self.IP_ADDR, self.PORT)
            async with client_stream:
                self.interrupt_bang = True
                async with trio.open_nursery() as nursery:
                    # spawning all the nets
                    print("parent: spawning making data ...")
                    nursery.start_soon(self.make_data)

                    # spawning scheduling methods
                    print("parent: spawning master cog...")
                    nursery.start_soon(self.master_clock)

                    # spawning rhythm gen
                    print("parent: spawning rhythm generator...")
                    nursery.start_soon(self.streamer)

                    # spawning affect listener
                    print("parent: spawning affect listener...")
                    nursery.start_soon(self.affect)

                    # # spawning poetry gen
                    # print("parent: spawning rhythm generator live input...")
                    # nursery.start_soon(self.random_poetry)

                    # spawning listening port for user input
                    print("parent: spawning receiver...")
                    nursery.start_soon(self.receiver, client_stream)

                    # spawning sending port for output data
                    print("parent: spawning sender...")
                    nursery.start_soon(self.sender, client_stream)

# todo - self affect percpetion!!!

    # --------------------------------------------------
    #
    # user accessible methods
    #
    # --------------------------------------------------

    # returns the live output from the class to user
    async def sender(self, client_stream):
        print("sender: started!")
        while self.running:
            data = {'e-AI output': self.datadict.get('master_move_output'),
                    'intensity': 0,
                    'individual NN outs':
                        {'move RNN': self.datadict.get('move_rnn'),
                         'affect RNN': self.datadict.get('affect_rnn'),
                             'move_affect_conv2': self.datadict.get('move_affect_conv2'),
                             'affect_move_conv2': self.datadict.get('affect_move_conv2')
                         }
                    }
            # serialise dict into json for Tx
            serial_data = pickle.dumps(data, -1)
            if self.streaming_logging:
                print("sender: sending {!r}".format(data))
            #  & send
            await client_stream.send_all(serial_data)
            await trio.sleep(self.rhythm_rate)

    # receives user data from client (typically live audio input)
    async def receiver(self, client_stream):
        print("receiver: started!")
        async for data in client_stream:
            load_data = pickle.loads(data)
            self.datadict['user_in'] = load_data
            if self.streaming_logging:
                print("receiver: got data {!r}".format(load_data))
        print("receiver: connection closed")
        sys.exit()

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
