"""main server script
will sit onboard bobot and operate as Nebula --- its dynamic soul"""

# --------------------------------------------------
#
# Embodied AI Engine Prototype v0.10
# 2021/01/25
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
        self.PORT = 65432
        self.IP_ADDR = "127.0.0.1"
        self.global_speed = speed
        self.rnd_stream = 0


        # make a default dict for the engine
        self.datadict = {'move_rnn': 0,
                         'affect_rnn': 0,
                         'move_affect_conv2': 0,
                         'affect_move_conv2': 0,
                         'master_output': 0,
                         'user_in': 0,
                         'rnd_poetry': 0,
                         'rhythm_rnn': 0,
                         'affect_net': 0,
                         'self_awareness': 0,
                         'affect_decision': 0,
                         'rhythm_rate': 0.1}

        # name list for nets
        self.netnames = ['move_rnn',
                         'affect_rnn',
                         'move_affect_conv2',
                         'affect_move_conv2',
                         'self_awareness',  # Net name for self-awareness
                         'master_output']  # input for self-awareness

        # names for affect listening
        self.affectnames = ['user_in',
                            'rnd_poetry',
                            'affect_net',
                            'self_awareness']

        self.rhythm_rate = 0.1
        self.affect_listen = 0

        # fill with random values
        self.dict_fill()
        print(self.datadict)

        # instantiate nets as objects and make  models
        self.move_net = MoveRNN()
        self.affect_net = AffectRNN()
        self.move_affect_net = MoveAffectCONV2()
        self.affect_move_net = AffectMoveCONV2()
        self.affect_perception = MoveAffectCONV2()

        # logging on/off switches
        self.net_logging = False
        self.master_logging = False
        self.streaming_logging = False
        self.affect_logging = False

    # --------------------------------------------------
    #
    # prediction and rnd num gen zone
    #
    # --------------------------------------------------

    # makes a prediction for a given net and defined input var
    async def make_data(self):
        while self.running:

            # calc rhythmic intensity based on self-awareness factor & global speed
            intensity = self.datadict.get('self_awareness')
            self.rhythm_rate = (self.rhythm_rate * intensity) * self.global_speed
            self.datadict['rhythm_rate'] = self.rhythm_rate

            # get input vars from dict (NB not always self)
            in_val1 = self.get_in_val(0) # move RNN as input
            in_val2 = self.get_in_val(1) # affect RNN as input
            in_val3 = self.get_in_val(2) # move - affect as input
            in_val4 = self.get_in_val(1) # affect RNN as input

            # send in vals to net object for prediction
            pred1 = self.move_net.predict(in_val1)
            pred2 = self.affect_net.predict(in_val2)
            pred3 = self.move_affect_net.predict(in_val3)
            pred4 = self.affect_move_net.predict(in_val4)

            # special case for self awareness stream
            self_aware_input = self.get_in_val(5) # main movement as input
            self_aware_pred = self.affect_perception.predict(self_aware_input)

            if self.net_logging:
                print(f"  'move_rnn' in: {in_val1} predicted {pred1}")
                print(f"  'affect_rnn' in: {in_val2} predicted {pred2}")
                print(f"  move_affect_conv2' in: {in_val3} predicted {pred3}")
                print(f"  'affect_move_conv2' in: {in_val4} predicted {pred4}")
                print(f"  'self_awareness' in: {self_aware_input} predicted {self_aware_pred}")

            # put predictions back into the dicts and master
            self.put_pred(0, pred1)
            self.put_pred(1, pred2)
            self.put_pred(2, pred3)
            self.put_pred(3, pred4)
            self.put_pred(4, self_aware_pred)

            # outputs a stream of random poetry
            rnd_poetry = random()
            self.datadict['rnd_poetry'] = random()
            if self.streaming_logging:
                print(f'random poetry = {rnd_poetry}')

            await trio.sleep(self.rhythm_rate)

    # function to get input value for net prediction from dictionary
    def get_in_val(self, which_dict):
        # get the current value and reshape ready for input for prediction
        input_val = self.datadict.get(self.netnames[which_dict])
        input_val = np.reshape(input_val, (1, 1, 1))
        return input_val

    # function to put prediction value from net into dictionary
    def put_pred(self, which_dict, pred):
        # randomly chooses one of te 4 predicted outputs
        out_pred_val = pred[0][randrange(4)]
        if self.master_logging:
            print(f"out pred val == {out_pred_val},   master move output == {self.datadict['master_output']}")
        # save to data dict and master move out ONLY 1st data
        self.datadict[self.netnames[which_dict]] = out_pred_val
        self.datadict['master_output'] = out_pred_val

    # fills the dictionary with rnd values for each key of data dictionary
    def dict_fill(self):
        for key in self.datadict.keys():
            rnd = random()
            self.datadict[key] = rnd

    # --------------------------------------------------
    #
    # affect and streaming methods
    #
    # --------------------------------------------------

    # define which feed to listen to, and duration
    # and a course of affect response
    async def affect(self):
        # daddy cycle = is the master running on?
        while self.running:
            if self.affect_logging:
                print('\t\t\t\t\t\t\t\t=========HIYA - DADDY cycle===========')

            # flag for breaking on big affect signal
            self.interrupt_bang = True

            # calc master cycle before a change
            master_cycle = randrange(6, 26) * self.global_speed
            loop_dur = time() + master_cycle
            if self.affect_logging:
                print(f"                 interrupt_listener: started! sleeping now for {loop_dur}...")

            # refill the dicts?????
            self.dict_fill()

            # child cycle - waiting for interrupt  from master clock
            while time() < loop_dur:
                if self.affect_logging:
                        print('\t\t\t\t\t\t\t\t=========Hello - child cycle 1 ===========')

                # if a major break out then go to Daddy cycle
                if not self.interrupt_bang:
                    break

                # randomly pick an input stream for this cycle
                rnd = randrange(4)
                self.rnd_stream = self.affectnames[rnd]
                self.datadict['affect_decision'] = rnd
                print(self.rnd_stream)
                if self.affect_logging:
                    print(self.rnd_stream)

                # hold this stream for 1-4 secs, unless interrupt bang
                end_time = time() + (randrange(1000, 4000) / 1000)
                if self.affect_logging:
                    print('end time = ', end_time)

                # baby cycle 2 - own time loops
                while time() < end_time:
                    if self.affect_logging:
                        print('\t\t\t\t\t\t\t\t=========Hello - baby cycle 2 ===========')

                    # go get the current value from dict
                    affect_listen = self.datadict[self.rnd_stream]
                    if self.affect_logging:
                        print('current value =', affect_listen)

                    # make the master output the current value of the stream
                    self.datadict['master_output'] = affect_listen
                    print(f'\t\t ==============  master move output = {affect_listen}')

                    # calc affect on behaviour
                    # if input stream is LOUD then smash a random fill and break out to Daddy cycle...
                    if affect_listen > 0.60:
                        if self.affect_logging:
                            print('interrupt > HIGH !!!!!!!!!')

                        # A - refill dict with random
                        self.dict_fill()

                        # B - jumps out of this loop into daddy
                        self.interrupt_bang = False
                        if self.affect_logging:
                            print('interrupt bang = ', self.interrupt_bang)

                        # C break out of this loop, and next (cos of flag)
                        break

                    # if middle loud fill dict with random, all processes norm
                    elif 0.30 < affect_listen < 0.59:
                        if self.affect_logging:
                            print('interrupt MIDDLE -----------')
                            print('interrupt bang = ', self.interrupt_bang)

                        # refill dict with random
                        self.dict_fill()

                    elif affect_listen <= 0.30:
                        if self.affect_logging:
                            print('interrupt LOW_______________')
                            print('interrupt bang = ', self.interrupt_bang)

                    # and wait for a cycle
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
                # self.interrupt_bang = True
                async with trio.open_nursery() as nursery:
                    # spawning all the data making
                    print("parent: spawning making data ...")
                    nursery.start_soon(self.make_data)

                    # spawning affect listener and master clocks
                    print("parent: spawning affect listener and clocks ...")
                    nursery.start_soon(self.affect)

                    # spawning listening port for user input
                    print("parent: spawning receiver...")
                    nursery.start_soon(self.receiver, client_stream)

                    # spawning sending port for output data
                    print("parent: spawning sender...")
                    nursery.start_soon(self.sender, client_stream)

    # --------------------------------------------------
    #
    # user accessible methods
    #
    # --------------------------------------------------

    # returns the live output from the class to user
    async def sender(self, client_stream):
        print("sender: started!")
        while self.running:
            # data = {'master_AI_output': self.datadict.get('master_output'),
            #         'intensity/rhythm': self.datadict.get('self_awareness'),
            #         'affect stream': self.rnd_stream,
            #         'individual NN outs':
            #             {'move RNN': self.datadict.get('move_rnn'),
            #              'affect RNN': self.datadict.get('affect_rnn'),
            #              'move_affect_conv2': self.datadict.get('move_affect_conv2'),
            #              'affect_move_conv2': self.datadict.get('affect_move_conv2'),
            #              'self_awareness': self.datadict.get('self_awareness')
            #              }
            #         }

            # serialise dict into json for Tx
            serial_data = pickle.dumps(self.datadict, -1)
            if self.streaming_logging:
                print("sender: sending {!r}".format(serial_data))
            #  & send
            await client_stream.send_all(serial_data)
            await trio.sleep(self.rhythm_rate)

    # receives user data from client (typically live audio input)
    async def receiver(self, client_stream):
        print("receiver: started!")
        while self.running:
            async for data in client_stream:
                load_data = pickle.loads(data)
                self.parse_got_dict(load_data)

                # self.datadict['user_in'] = load_data['mic_level']
                if self.streaming_logging:
                    print("receiver: got data {!r}".format(load_data))
            print("receiver: connection closed")
        sys.exit()

    def parse_got_dict(self, got_dict):
        self.datadict['user_in'] = got_dict['mic_level']

        # user change the overall speed of the engine
        self.global_speed = got_dict['speed']

        # user change tempo of outputs and parsing
        self.rhythm_rate = got_dict['tempo']

    # stop start methods
    def go(self):
        self.running = True
        trio.run(self.flywheel)
        print('I got here daddy')

    def quit(self):
        self.running = False


if __name__ == '__main__':
    engine = AiDataEngine()
    engine.go()
