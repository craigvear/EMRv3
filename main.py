from engine import Scheduler
from time import sleep
import trio

engine = Scheduler()
engine.go()

for i in range(100):
    result = engine.queries()
    engine.put(result)
    print(result)
    sleep(0.1)

engine.quit()
