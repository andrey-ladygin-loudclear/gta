import numpy as np
from PIL import ImageGrab
import cv2
import os
import time
import pyautogui

from player import GTAPlayer

player = GTAPlayer()

def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)

# print('down')
# PressKey(W)
# time.sleep(3)
# print('up')
# ReleaseKey(W)



from getkeys import key_check

def keys_to_output(keys):
    #[A,W,D]
    outputs = [0,0,0]

    if 'A' in keys: outputs[0] = 1
    elif 'D' in keys: outputs[1] = 1
    else: outputs[2] = 1


file_name = 'training_data.npy'

if os.path.isfile(file_name):
    training_data = list(np.load(file_name))
else:
    training_data = []



def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    return processed_img

i = 0
last_time = time.time()

# while(True):
#     screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
#     new_screen = process_img(screen)
#     #printscreen_numpy = np.array(printscreen_pil.getdata(),dtype='uint8')
#     #printscreen_numpy = np.array(printscreen_pil.getdata(),dtype='uint8').reshape((printscreen_pil.size[1],printscreen_pil.size[0],3))
#
#     # cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
#     cv2.imshow('window', new_screen)
#     #cv2.imwrite("imgs/im-" + str(i) + ".jpg", printscreen_numpy)
#     print('Loop took {}'.format(time.time() - last_time))
#     last_time = time.time()
#     i+=1
#
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break