from threading import Thread

import time

from grab import grab
from keygrab import start_keys_grab
# from player import GTAPlayer

# player = GTAPlayer()

# thread = Thread(target=grab)
# thread.setDaemon(True)
# thread.start()

# thread = Thread(target=start_keys_grab)
# thread.setDaemon(True)
# thread.start()

from getkeys import key_check

def keys_to_output(keys):
    #[A,W,D]
    outputs = [0,0,0]

    if 'A' in keys: outputs[0] = 1
    elif 'D' in keys: outputs[1] = 1
    else: outputs[2] = 1

grab()

# for i in list(range(4))[::-1]:
#     print(i+1)
#     time.sleep(1)