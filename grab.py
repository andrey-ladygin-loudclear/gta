import gzip
from threading import Thread

import tables
import numpy as np
import cv2
import time
import os
import pyautogui
import zlib
from PIL import ImageGrab
from getkeys import key_check, get_pressed_keys, clear_pressed_keys
from player import GTAPlayer
import datetime
import pickle

player = GTAPlayer()
start = False

X_train = []
Y_train = []

last_img = [int(folder) for folder in os.listdir('imgs')]
if len(last_img) == 0: last_img = [0]
last_saved_image = (max(last_img) + 1) * 1000

def grab():
    global start, last_saved_image
    last_time = time.time()
    time_sum = 0

    while(True):
        check_commands()

        if(start):
            screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
            image = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            cv2.imshow('window', image)

            if last_saved_image % 250 == 0:
                mean_time_diff = time_sum / 100
                time_sum = 0
                print('Loop took {}, Len is {}, shape is {}'.format(mean_time_diff, len(Y_train), image.shape))
            time_sum += time.time() - last_time

            addToData(image)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

            last_time = time.time()
            #time.sleep(0.3)

isFirstCheckCommands = True
def check_commands():
    global start, isFirstCheckCommands, X_train, Y_train
    pressed_keys = key_check()

    if isFirstCheckCommands:
        isFirstCheckCommands = False
        return

    if 'CTRL' in pressed_keys:
        if 'T' in pressed_keys:
            start = True
        # if 'S' in pressed_keys:
        #     saveTrainingData()
        if 'E' in pressed_keys:
            start = False
        if 'C' in pressed_keys:
            X_train = []
            Y_train = []


def addToData(image):
    global Y_train, last_saved_image
    pressed_keys = get_pressed_keys()
    clear_pressed_keys()

    y = [pressed_keys.count('A'),
        pressed_keys.count('W'),
        pressed_keys.count('D'),
        pressed_keys.count('S')]

    # if 'A' in pressed_keys: y[0] = 1
    # if 'W' in pressed_keys: y[1] = 1
    # if 'D' in pressed_keys: y[2] = 1
    # if 'S' in pressed_keys: y[3] = 1

    #X_train.append(image)
    Y_train.append(y)

    folder = last_saved_image // 1000

    if not os.path.isdir("imgs/"+str(folder)):
        os.makedirs("imgs/"+str(folder))

    cv2.imwrite("imgs/"+str(folder)+"/"+str(last_saved_image)+".jpg", image)

    if len(Y_train) == 1000:
        print('SAVE', len(Y_train), last_saved_image)
        saveTrainingData()

    #print(y)
    last_saved_image += 1

def getFileName(date, iteration):
    return 'data/training_{}-{}.hdf'.format(date, iteration)

def saveTrainingData():
    global X_train, Y_train
    name = str(last_saved_image // 1000) + "-labels-" + str(len(Y_train))

    if len(Y_train) > 0:
        print('SaveTrainingData: ', len(Y_train), 'Total:', last_saved_image)
        np.save('labels/'+name, Y_train)
        Y_train = []
    return

    date = datetime.datetime.now()
    date = date.strftime('%m-%d-%Y')
    file_name = getFileName(date, i)

    while os.path.isfile(file_name):
        i += 1
        file_name = getFileName(date, i)

    print(file_name)

    thread = Thread(target=save_file, args=(file_name, X_train, Y_train))
    thread.setDaemon(True)
    thread.start()
    time.sleep(0.5)

    i += 1
    X_train = []
    Y_train = []

Cobj = zlib.compressobj(level=-1)

def save_file(file_name, X_train, Y_train):
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    print('saving training data', len(X_train))

    # data = zlib.compress(np.array(training_data).tobytes())
    # np.save(file_name+'.c', data)
    # np.save(file_name+'.b', np.array(training_data).tobytes())
    # np.save(file_name, X_train)

    f = tables.openFile(file_name, 'w')
    filters = tables.Filters(complib='blosc', complevel=5)

    atom1= tables.Atom.from_dtype(X_train.dtype)
    ds1 = f.createCArray(f.root, 'X_train', atom1, X_train.shape, filters=filters)
    ds1[:] = X_train

    atom2 = tables.Atom.from_dtype(X_train.dtype)
    ds2 = f.createCArray(f.root, 'Y_train', atom2, Y_train.shape, filters=filters)
    ds2[:] = Y_train
    f.close()

    # import gzip
    # content = "Lots of content here"
    # with gzip.open('file.txt.gz', 'wb') as f:
    #     f.write(content)

    print('Training Data saved!')


# screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
# new_screen = process_img(screen)
# #printscreen_numpy = np.array(printscreen_pil.getdata(),dtype='uint8')
# #printscreen_numpy = np.array(printscreen_pil.getdata(),dtype='uint8').reshape((printscreen_pil.size[1],printscreen_pil.size[0],3))
#
# # cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
# cv2.imshow('window', new_screen)
# #cv2.imwrite("imgs/im-" + str(i) + ".jpg", printscreen_numpy)
# print('Loop took {}'.format(time.time() - last_time))
# last_time = time.time()
# i+=1
#
# if cv2.waitKey(25) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()
#     break