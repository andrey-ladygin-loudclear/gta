import tkinter as tk
# from msvcrt import getch
import getch

def start_keys_grab():
    while True:
        key = ord(getch.getche())
        print('key', key)

def start_keys_grab2():
    root = tk.Tk()
    print('start_keys_grab')

    def keyevent(event):
        print('event.keycode', event.keycode)

        if event.keycode == 67:             #Check if pressed key has code 67 (character 'c')
            print("Hello World")

    root.bind("<Control - Key>", keyevent) #You press Ctrl and a key at the same time
    root.mainloop()