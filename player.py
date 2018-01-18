from threading import Thread

from directkeys import ReleaseKey, PressKey, W, A, S, D


class GTAPlayer():#Thread
    def left(self):
        PressKey(A)

    def right(self):
        PressKey(D)

    def forward(self):
        PressKey(W)

    def backwrd(self):
        PressKey(S)

    def release(self, key):
        ReleaseKey(key)

    def releaseForward(self):
        ReleaseKey(W)

    def releaseLeft(self):
        ReleaseKey(A)

    def releaseRight(self):
        ReleaseKey(D)

    def getPressedKeys(self):
        return []