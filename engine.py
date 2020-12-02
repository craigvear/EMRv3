
import sys
import trio
from random import randrange, random
from time import sleep

class Scheduler:
    def __init__(self, speed=1):
        self.global_speed = speed
        self.interrupt_bang = False
        self.running = False

        # make a default dict and populate with random
        self.datadict = {'master_move_output': 0,
                         'move_in': 0,
                         'user_input': 0}

        # todo: build models and put into list (replace strings with models)
        self.netlist = ['move_rnn',
                        'affect_rnn',
                        'move-affect_conv2',
                        'affect-move_conv2']

    async def net1(self, iden):
        iden = self.netlist[iden]
        while self.interrupt_bang:
            localvalue = self.datadict.get('move_in')
            rnd = random() * self.global_speed
            print(f"  {iden} in: {localvalue} predicted {rnd}")
            self.datadict.update({'move_in': rnd})
            await trio.sleep(rnd)
            print(f"  {iden}: looping!")

    # controls master scheduling
    async def interrupt_listener(self):
        loop_dur = randrange(6, 26) * self.global_speed
        print(f"                 interrupt_listener: started! sleeping now for {loop_dur}...")
        await trio.sleep(loop_dur)
        self.interrupt_bang = False
        print(" ###################   restarting ######################")

    async def parent(self):
        print("parent: started!")
        while self.running:
            self.interrupt_bang = True
            async with trio.open_nursery() as nursery:
                # spawning all the nets
                print("parent: spawning net1...")
                nursery.start_soon(self.net1, 0)

                print("parent: spawning net2...")
                nursery.start_soon(self.net1, 1)

                print("parent: spawning net3...")
                nursery.start_soon(self.net1, 2)

                print("parent: spawning net4...")
                nursery.start_soon(self.net1, 3)

                # spawning scheduling methods
                print("parent: spawning master cog...")
                nursery.start_soon(self.interrupt_listener)

    # user accessible methods
    # returns the live output from the class to user
    def queries(self):
        return self.datadict.get('master_move_output')

    # live input of user-data into class (0-1)
    # called and scheduled by user class
    def put(self, user_data):
        self.user_data = user_data
        # if < 30 do nothing
        # if 30 <> 60 fill dict with random
        # if > 60 trigger intrrupt bang

        self.datadict.update({'user_input': self.user_data})

    def go(self):
        self.running = True
        trio.run(self.parent)

    def quit(self):
        self.running = False

if __name__ == '__main__':
    sched = Scheduler()
    sched.go()
