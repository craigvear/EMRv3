import trio




class Numb:


    def __init__(self):
        print('on it')
        self.class_dict = {'a': 0,
                      'b': 0,
                      'c': 0}

    async def adder(self):
        while True:
            for key in self.class_dict:
                v = self.class_dict[key]
                v += 1
                self.class_dict[key] = v
                # print(self.class_dict[key])
            await trio.sleep(0.2)

    async def showr(self):
        while True:
            for key in self.class_dict:
                print(self.class_dict[key])
            await trio.sleep(2)

    async def parent(self):
        while True:
            async with trio.open_nursery() as nursery:
                # spawning all the nets
                nursery.start_soon(self.adder)

                nursery.start_soon(self.showr)


go = Numb()
trio.run(go.parent)