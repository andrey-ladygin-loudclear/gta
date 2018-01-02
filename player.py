from directkeys import ReleaseKey, PressKey, W, A, S, D


class GTAPlayer:
    def __init__(self):
        pass

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

    def getPressedKeys(self):
        return []