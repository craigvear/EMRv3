from engine import Scheduler
from time import sleep
import trio


engine = Scheduler()


async def child1():
    for i in range(100):
        result = engine.queries()
        engine.put(result)
        print(result)
        sleep(0.1)

async def child2():
    engine.go()

async def parent():
    print("parent: started!")
    async with trio.open_nursery() as nursery:
        print("parent: spawning child1...")
        nursery.start_soon(child1)

        print("parent: spawning child2...")
        nursery.start_soon(child2)

        print("parent: waiting for children to finish...")
        # -- we exit the nursery block here --
    print("parent: all done!")

engine.quit()
trio.run(parent)



