import win32api as wapi
import time
from threading import Thread

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890., 'APS/\\":
    keyList.append(char)


def key_check():
    keys = []

    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)

        if wapi.GetAsyncKeyState(17):
            keys.append('CTRL')
    return keys

#https://www.youtube.com/watch?v=F4y4YOpUcTQ&list=PLQVvvaa0QuDeETZEOy4VdocT7TOjfSA8a&index=9

pressed_keys = []

def grab_pressed_keys():
    global pressed_keys

    while True:
        keys = key_check()
        if 'A' in keys: pressed_keys.append("A")
        if 'W' in keys: pressed_keys.append("W")
        if 'D' in keys: pressed_keys.append("D")
        if 'S' in keys: pressed_keys.append("S")
        time.sleep(0.01)


def get_pressed_keys():
    global pressed_keys
    return pressed_keys


def clear_pressed_keys():
    global pressed_keys
    pressed_keys = []

thread = Thread(target = grab_pressed_keys)
thread.setDaemon(True)
thread.start()