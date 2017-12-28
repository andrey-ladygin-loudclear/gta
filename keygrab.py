import tkinter as tk
root = tk.Tk()

def keyevent(event):
    print('event.keycode', event.keycode)

    if event.keycode == 67:             #Check if pressed key has code 67 (character 'c')
        print("Hello World")

root.bind("<Control - Key>", keyevent) #You press Ctrl and a key at the same time
root.mainloop()